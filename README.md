# FI-Lab and In-Hospital Mortality Risk in Mechanically Ventilated COPD Patients

An interactive web application that implements the **multivariable logistic regression**
model from the two-cohort study:

> *Association Between Laboratory Frailty Index and In-Hospital Mortality in
> Mechanically Ventilated COPD Patients: A Two-Cohort Study (MIMIC-IV and eICU-CRD)*

---

## What the app does

| Page | Content |
|---|---|
| **🫁 Risk Prediction** | Enter a patient's FI-Lab, SOFA, OASIS, cerebrovascular disease and CRRT status; get the predicted probability of in-hospital death, a risk gauge, a three-tier clinical recommendation, and a per-patient SHAP explanation |
| **📈 Model Performance** | Training / test / external-validation AUC and accuracy, side-by-side ROC curves (MIMIC-IV vs eICU-CRD) and the model's odds ratios |

## Model specification

| Item | Value |
|---|---|
| Algorithm | Multivariable logistic regression (`statsmodels` Logit, MLE) |
| Predictors | FI-Lab score, SOFA, OASIS, cerebrovascular disease, CRRT |
| Intercept | −3.9643 |
| Coefficients | FI-Lab +3.2588 · CRRT +1.2067 · cerebrovascular +0.9916 · OASIS +0.0262 · SOFA +0.0128 |
| Development cohort | MIMIC-IV, n = 807 (70 % train / 30 % test, stratified, seed 2025) |
| External validation | eICU-CRD, n = 209 |
| Discrimination | Train AUC **0.723** · Test AUC **0.702** · External AUC **0.683** |
| Preprocessing | **None** — the model was trained on raw, non-standardised values |

## Files

```
app.py                Streamlit application (UI + inference + SHAP)
logistic_model.pkl    Trained statsmodels Logit result object  ← required by app.py
app_assets.pkl        Feature order, synthetic SHAP background, ROC curves, metrics
requirements.txt      Pinned dependencies (must match the training environment)
README.md             This file
```

## Run locally

```bash
# 1) install dependencies (Python 3.12 recommended)
pip install -r requirements.txt

# 2) launch
streamlit run app.py
```

The app opens at <http://localhost:8501>.

> **Windows note:** if `streamlit` fails to bind to localhost, clear any proxy
> environment variables before launching:
> ```powershell
> $env:HTTP_PROXY=""; $env:HTTPS_PROXY=""; $env:ALL_PROXY=""
> streamlit run app.py
> ```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload **all five files** (`app.py`,
   `logistic_model.pkl`, `app_assets.pkl`, `requirements.txt`, `README.md`).
   > The `.pkl` files are mandatory — the app raises `FileNotFoundError` without them.
2. Go to <https://share.streamlit.io> → **New app** → select the repository.
3. Set **Main file path** to `app.py`, **Branch** to `main`.
4. Click **Deploy**.

`statsmodels` must be present in `requirements.txt` — `logistic_model.pkl` is a
statsmodels result object and cannot be unpickled without it.

## A note on the SHAP background

SHAP values are computed with a `KernelExplainer` using 25 **k-means centroids**
over the training feature space. These are synthetic summary points, not real
patient records, so the asset file can be published without exposing PHI.

## Interpretation of the inputs

- **FI-Lab (Laboratory Frailty Index)** — the ratio of abnormal laboratory tests to
  the total number of tests performed during the first ICU day. Higher = frailer.
  Cohort range 0.09–0.93, median 0.55.
- **SOFA** — Sequential Organ Failure Assessment. Cohort range 1–18, median 8.
- **OASIS** — Oxford Acute Severity of Illness Score. Cohort range 22–58, median 39.
- **Cerebrovascular disease** — pre-existing ischaemic stroke / TIA.
- **CRRT** — continuous renal replacement therapy administered during the ICU stay.

## Disclaimer

This application is intended for **research and educational purposes only**. It was
developed on retrospective ICU registry data, has not been prospectively validated,
and must not be used as a substitute for clinical judgement.
