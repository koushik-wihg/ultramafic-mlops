\# Ultramafic\_MLOPS



This repository contains a reproducible machine-learning workflow for classifying ultramafic rocks using whole-rock geochemistry, with a specific focus on tectonic affinity and out-of-domain interpretation of Archean greenstone samples.



The project is designed for \*\*research transparency\*\*, \*\*geochemical interpretability\*\*, and \*\*manuscript support\*\*, not as a black-box production classifier.



---



\## Scope and Objectives



\- Train a supervised ML model on \*\*Phanerozoic–Proterozoic ultramafic rocks\*\*

\- Predict tectonic affinity using \*\*major oxides + REE\*\*

\- Apply the trained model to \*\*Archean greenstones as out-of-domain data\*\*

\- Interpret Archean samples via \*\*probability space\*\*, not forced labels

\- Provide \*\*explainable ML outputs\*\* (SHAP) for the trained domain



---



\## Dataset Overview



\### Training domain

\- Dominated by Phanerozoic–Proterozoic samples

\- Classes used in training:

&nbsp; - Rift volcanics

&nbsp; - Convergent margin

&nbsp; - Plume–lithosphere

\- “OTHERS” and Archean samples are excluded from training



\### Archean samples

\- Never used for training

\- Projected into the trained probability space

\- Interpreted via overlap with training-domain probability centroids



---



\## Feature Engineering



Input data are \*\*raw geochemical values\*\* supplied by the user:



\- Major oxides (e.g., SiO₂, MgO, FeO, CaO, Na₂O)

\- Rare earth elements (La–Lu)



All transformations are handled internally by the pipeline:

\- Median imputation

\- HFSE / REE ratios

\- ILR transformation of major oxides

\- Optional PCA on ILR space

\- Feature selection



\*\*Users never need to supply ILR values manually.\*\*



---



\## Model Architecture



\- Algorithm: \*\*XGBoost (multi-class)\*\*

\- Class imbalance handled via \*\*SMOTE (applied outside the pipeline)\*\*

\- Nested cross-validation

\- Hyperparameter optimization with \*\*Optuna\*\*

\- Final model saved as a single `.joblib` pipeline



---



\## Explainability



\### In-domain (training data)

\- Global SHAP summaries

\- Feature importance and directionality

\- Class-specific patterns



\### Out-of-domain (Archean)

\- No global SHAP aggregation

\- Interpreted via:

&nbsp; - Probability distributions

&nbsp; - Ternary probability diagrams

&nbsp; - Comparison with training-domain centroids

\- Emphasis on \*\*overlap, ambiguity, and uncertainty\*\*



---



\## Probability-Space Analysis



Key figures produced by this workflow include:

\- Ternary probability plots for Archean samples

\- Training-domain probability centroids

\- Side-by-side comparison of Archean projections and training centroids



These figures form the primary interpretive framework for Archean mantle processes.



---



\## Project Structure





