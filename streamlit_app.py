# ============================================================
# streamlit_app.py – Ultramafic ML (direct pipeline, no API)
# ============================================================

import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import __main__

from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------
# RAW FEATURE ORDER (EXACT training input)
# ------------------------------------------------------------
RAW_20_COLS = [
    "SiO2", "Al2O3", "TiO2", "FeO", "MgO", "CaO", "Na2O",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Yb", "Lu"
]

# ------------------------------------------------------------
# REQUIRED: Custom transformers for joblib compatibility
# ------------------------------------------------------------
class HFSE_REE_Ratios(BaseEstimator, TransformerMixin):
    def __init__(self, candidates=None):
        self.candidates = candidates

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = (
            pd.DataFrame(X, columns=RAW_20_COLS)
            if not isinstance(X, pd.DataFrame)
            else X.copy()
        )

        # Ratios actually used by the trained pipeline
        df["Ce_Yb"] = df["Ce"] / df["Yb"].replace(0, pd.NA)
        df["La_Ce"] = df["La"] / df["Ce"].replace(0, pd.NA)

        return df.fillna(0.0)


class PivotILRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, comp_cols=(), zero_replace_factor=1e-6):
        self.comp_cols = comp_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rename_map = {
            "SiO2": "ilr_SiO2_vs_rest",
            "Al2O3": "ilr_Al2O3_vs_rest",
            "TiO2": "ilr_TiO2_vs_rest",
            "FeO": "ilr_FeO_vs_rest",
            "MgO": "ilr_MgO_vs_rest",
            "CaO": "ilr_CaO_vs_rest",
        }
        if isinstance(X, pd.DataFrame):
            return X.rename(columns=rename_map)
        return X


# ------------------------------------------------------------
# Expose transformers to __main__ (CRITICAL for joblib.load)
# ------------------------------------------------------------
__main__.HFSE_REE_Ratios = HFSE_REE_Ratios
__main__.PivotILRTransformer = PivotILRTransformer

#==================================================================

# ==========================
# CONFIG & MODEL LOAD (ULTRAMAFIC – STREAMLIT)
# ==========================

from pathlib import Path
import joblib

# ------------------------------------------------------------
# Model path (direct, no params.yaml, no FastAPI coupling)
# ------------------------------------------------------------
MODEL_PATH = Path("models/best_pipeline_XGB_SMOTE_OUT_Fixed.joblib")

def load_model_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    obj = joblib.load(path)

    # Training saved a single sklearn/imb pipeline
    if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
        return obj

    raise ValueError("Unrecognized model artifact format.")

try:
    model = load_model_artifact(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    load_error = str(e)

# ------------------------------------------------------------
# Fixed class labels (training order is stable)
# ------------------------------------------------------------
CLASS_LABELS = [
    "RIFT_VOLCANICS",
    "CONVERGENT_MARGIN",
    "PLUME_LITHOSPHERE",
]

# ------------------------------------------------------------
# Sanity info for UI
# ------------------------------------------------------------
N_RAW_FEATURES = 20
N_INTERNAL_FEATURES = 21  # ratios + ILR generated inside pipeline
#===========================================================================
# ==========================
# PREPROCESSING HELPERS
# (ULTRAMAFIC – STREAMLIT SAFE, USER-FRIENDLY)
# ==========================

RAW_20_COLS = [
    "SiO2", "Al2O3", "TiO2", "FeO", "MgO", "CaO", "Na2O",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Yb", "Lu"
]

def preprocess_df_for_model(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # --------------------------------------------------
    # Detect missing required columns
    # --------------------------------------------------
    missing_cols = [c for c in RAW_20_COLS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in RAW_20_COLS]

    if missing_cols:
        st.warning(
            "⚠️ Missing required geochemical columns:\n\n"
            f"{', '.join(missing_cols)}\n\n"
            "These will be filled with 0.0 for prediction. "
            "Results should be interpreted cautiously."
        )

    if extra_cols:
        st.info(
            "ℹ️ Extra columns detected and ignored:\n\n"
            f"{', '.join(extra_cols)}"
        )

    # --------------------------------------------------
    # Keep ONLY required raw features (order enforced)
    # --------------------------------------------------
    df = df.reindex(columns=RAW_20_COLS)

    # Convert to numeric safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaN (pipeline imputer also exists)
    df = df.fillna(0.0)

    return df


def run_model(df_raw: pd.DataFrame) -> pd.DataFrame:
    if not model_loaded:
        raise RuntimeError("Model not loaded.")

    X = preprocess_df_for_model(df_raw)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    results = []
    for i in range(len(preds)):
        row = {
            "Predicted_Class": CLASS_LABELS[int(preds[i])],
            "Confidence": float(np.max(probs[i]))
        }
        for j, cname in enumerate(CLASS_LABELS):
            row[f"P_{cname}"] = float(probs[i, j])
        results.append(row)

    return pd.DataFrame(results)
#====================================================================
# ==========================
# PARSERS FOR SINGLE-SAMPLE INPUT
# (ROBUST, USER-PROOF)
# ==========================

def parse_key_value_block(text: str) -> pd.DataFrame:
    """
    Parse key=value lines into a single-row DataFrame.
    Ignores empty lines, comments, malformed entries.
    """
    record = {}

    for line in text.splitlines():
        line = line.strip()

        # Skip blanks / comments
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()

        # Allow bdl, blank, NA etc.
        if val.lower() in {"", "bdl", "na", "n/a"}:
            record[key] = None
        else:
            record[key] = val

    if not record:
        raise ValueError("No valid key=value pairs found.")

    return pd.DataFrame([record])


def parse_single_line_csv_block(text: str) -> pd.DataFrame:
    """
    Parse header + single data row CSV pasted into text box.
    """
    buf = io.StringIO(text.strip())
    df = pd.read_csv(buf)

    if df.empty:
        raise ValueError("CSV block is empty.")

    if len(df) > 1:
        st.warning(
            "Multiple rows detected in single-sample CSV. "
            "Only the first row will be used."
        )
        df = df.iloc[[0]]

    return df


def read_uploaded_file(f) -> pd.DataFrame:
    """
    Read uploaded CSV / Excel file.
    """
    name = f.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(f)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(f)
        else:
            raise ValueError
    except Exception:
        raise ValueError(
            "Unsupported or corrupted file. "
            "Please upload a valid CSV, XLSX, or XLS file."
        )
#==========================================================================
ULTRAMAFIC_LEGEND_MD = """
### 🌋 Ultramafic Tectonic Probability Space

| End Member | Interpretation |
|-----------|----------------|
| **RIFT VOLCANICS** | MORB-like / extensional ultramafic sources |
| **CONVERGENT MARGIN** | Arc–SSZ–related ultramafics |
| **PLUME–LITHOSPHERE** | OIB / plume-influenced mantle sources |

⚠️ samples are **projected into this probability space**.  
Predictions represent **relative tectonic affinities**, not definitive class labels.
"""
#========================================================================================
# ==========================
# STREAMLIT LAYOUT (light theme)
# ==========================
st.set_page_config(
    page_title="Ultramafic Tectonic Affinity ML – Probability Space Explorer",
    layout="wide",
)

# Light background override
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f7f9 !important;
        color: #111827 !important;
    }
    .main {
        background-color: #f7f7f9 !important;
    }
    body {
        background-color: #f7f7f9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col_left, col_mid, col_right = st.columns([1.3, 3.0, 2.2])

# ----- LEFT: MODEL INFO -----
with col_left:
    st.markdown("### Model Info")

    st.markdown(
        f"""
        <div style="
            border-radius: 0.75rem;
            padding: 0.9rem 1rem;
            background-color: #DBEAFE;
            border: 1px solid #3B82F6;
            font-size: 0.9rem;
            word-break: break-all;">
            <strong>Model path:</strong><br>
            {MODEL_PATH}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    if model_loaded:
        n_feats = len(EXPECTED_FEATURES) if EXPECTED_FEATURES else "unknown"
        st.success(f"Model loaded ✅  (features: {n_feats})")
    else:
        st.error("Model failed to load ❌")
        st.caption(load_error if "load_error" in globals() else "Check MODEL_PATH.")

    st.markdown("---")
    st.caption(
        "This app loads the trained pipeline directly (no API). "
        "Raw geochemical data are preprocessed in the same way as in the FastAPI backend."
    )

#===================================================================================================
# ----- MIDDLE: BATCH PREDICTION -----
with col_mid:
    st.markdown("## Ultramafic Tectonic Affinity ML – Probability Space Explorer")
    st.markdown(
        "Upload CSV / Excel files with **whole-rock ultramafic geochemistry** to obtain "
        "**tectonic affinity probabilities**. "
        "Non-numeric entries (e.g. `bdl`, blanks) are handled safely."
    )

    st.markdown("### Upload CSV / Excel for batch projection")

    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Limit 200MB per file • CSV, XLSX, XLS",
    )

    if uploaded_files:
        all_dfs = []
        meta = []

        for f in uploaded_files:
            try:
                df_u = read_uploaded_file(f)
            except Exception as e:
                st.error(f"Failed to read `{f.name}`: {e}")
                continue
            df_u.columns = df_u.columns.astype(str)
            all_dfs.append(df_u)
            meta.append((f.name, df_u.shape))

        if not all_dfs:
            st.stop()

        st.markdown("### Uploaded files")
        for fname, (n_rows, n_cols) in zip([m[0] for m in meta], meta):
            st.markdown(f"**File:** `{fname}` &nbsp;&nbsp; Shape: `{n_rows} × {n_cols}`")

        st.info(
            "Extra elements or oxides are **ignored safely**. "
            "If required major oxides or REEs are missing, the model will still run "
            "but predictions should be interpreted cautiously. "
            "Engineered features (e.g. `Ce_Yb`, `La_Ce`, ILR components) are derived internally."
        )

        df_all = pd.concat(all_dfs, axis=0, ignore_index=True)
        max_rows = len(df_all)

        n_rows_total = st.number_input(
            "Number of rows (total) to project",
            min_value=1,
            max_value=max_rows,
            value=min(50, max_rows),
            step=1,
        )
        df_subset = df_all.iloc[:n_rows_total].copy()

        st.markdown("#### Preview of input")
        st.dataframe(df_subset.head(), use_container_width=True)

        if st.button("🌋 Run tectonic affinity projection", type="primary"):
            if not model_loaded:
                st.error("Model not loaded. Check MODEL_PATH.")
            else:
                with st.spinner("Projecting samples into tectonic probability space..."):
                    try:
                        preds_df = run_model(df_subset)
                    except Exception as e:
                        st.error("Error during projection:")
                        st.error(str(e))
                    else:
                        combined = pd.concat(
                            [df_subset.reset_index(drop=True), preds_df], axis=1
                        )
                        st.markdown("### 🧭 Tectonic affinity probabilities")
                        st.dataframe(combined, use_container_width=True)

                        st.markdown("---")
                        st.markdown(ULTRAMAFIC_LEGEND_MD)

                        # Download options
                        csv_bytes = combined.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download results as CSV",
                            data=csv_bytes,
                            file_name="ultramafic_tectonic_affinity_predictions.csv",
                            mime="text/csv",
                        )

                        excel_buf = io.BytesIO()
                        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                            combined.to_excel(writer, index=False, sheet_name="Predictions")
                        st.download_button(
                            "⬇️ Download results as Excel (.xlsx)",
                            data=excel_buf.getvalue(),
                            file_name="ultramafic_tectonic_affinity_predictions.xlsx",
                            mime=(
                                "application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet"
                            ),
                        )
#====================================================================================================
# ----- RIGHT: SINGLE-SAMPLE INPUT -----
with col_right:
    st.markdown("### Single-sample input")
    st.caption(
        "Enter one sample manually, either as key=value lines or as a single-row CSV. "
        "Only **major oxides (SiO2–Na2O)** and **REEs (La–Lu)** are used. "
        "All other elements (e.g. K2O, P2O5, MnO, LOI, HFSEs) are safely ignored."
    )

    st.markdown(
        """
        **Accepted inputs**
        - Major oxides: `SiO2, Al2O3, TiO2, FeO, MgO, CaO, Na2O`
        - REEs only: `La–Lu`
        
        Other oxides or trace elements will be dropped automatically.
        """
    )

    mode = st.radio(
        "Input mode:",
        ["Key=Value lines", "Single-line CSV"],
        index=0,
    )

    if mode == "Key=Value lines":
        example_text = (
            "SiO2=48.6\n"
            "TiO2=0.9\n"
            "Al2O3=15.2\n"
            "FeO=10.1\n"
            "MgO=9.8\n"
            "CaO=11.4\n"
            "Na2O=2.1\n"
            "La=3.2\n"
            "Ce=6.8\n"
            "Pr=0.9\n"
            "Nd=4.1\n"
            "Sm=1.1\n"
            "Eu=0.35\n"
            "Gd=1.2\n"
            "Tb=0.18\n"
            "Dy=1.0\n"
            "Ho=0.21\n"
            "Er=0.55\n"
            "Tm=0.08\n"
            "Yb=0.52\n"
            "Lu=0.08\n"
        )

        text = st.text_area(
            "Paste sample here",
            value=example_text,
            height=320,
        )

        if st.button("🌋 Project single sample", key="single_kv"):
            if not model_loaded:
                st.error("Model not loaded. Check MODEL_PATH.")
            else:
                rec = parse_key_value_block(text)
                if not rec:
                    st.error("No valid `key=value` lines detected.")
                else:
                    df_single = pd.DataFrame([rec])
                    try:
                        preds_df = run_model(df_single)
                    except Exception as e:
                        st.error("Error during projection:")
                        st.error(str(e))
                    else:
                        st.markdown("#### Tectonic affinity result")
                        st.json(preds_df.iloc[0].to_dict())
                        st.markdown("---")
                        st.markdown(ULTRAMAFIC_LEGEND_MD)

    else:  # Single-line CSV
        example_csv = (
            "SiO2,TiO2,Al2O3,FeO,MgO,CaO,Na2O,La,Ce,Pr,Nd,Sm,Eu,Gd,Tb,Dy,Ho,Er,Yb,Lu\n"
            "48.6,0.9,15.2,10.1,9.8,11.4,2.1,3.2,6.8,0.9,4.1,1.1,0.35,1.2,0.18,1.0,0.21,0.55,0.52,0.08\n"
        )

        text = st.text_area(
            "Paste header + single data row",
            value=example_csv,
            height=260,
        )

        if st.button("🌋 Project single sample", key="single_csv"):
            if not model_loaded:
                st.error("Model not loaded. Check MODEL_PATH.")
            else:
                try:
                    df_single = parse_single_line_csv_block(text)
                except Exception as e:
                    st.error(f"Could not parse CSV block: {e}")
                else:
                    try:
                        preds_df = run_model(df_single)
                    except Exception as e:
                        st.error("Error during projection:")
                        st.error(str(e))
                    else:
                        st.markdown("#### Tectonic affinity result")
                        st.json(preds_df.iloc[0].to_dict())
                        st.markdown("---")
                        st.markdown(ULTRAMAFIC_LEGEND_MD)

# Footer
st.write("---")
st.caption(
    "Predictions represent **probability-space tectonic affinity** "
    "among rift-related, convergent-margin, and plume–lithosphere ultramafic systems. "
    "Results should be interpreted comparatively, not as absolute classifications."
)
