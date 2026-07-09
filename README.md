# Disposable Active Socks — Analytics Dashboard (Group 3)

Full descriptive → diagnostic → predictive → prescriptive analytics dashboard, built
in response to the instructor's feedback on the original dashboard:
https://dataanalytics-grouppbl-group3.streamlit.app/

## What changed vs. the old dashboard
1. **Clustering** now uses a curated demographic + behavioral feature subset
   (not all 31 raw survey columns), avoiding the overlap/noise problem flagged
   in review.
2. **Association Rule Mining / Market Basket Analysis** has been added, showing
   which of 9 candidate bundle products (gym bag, shaker bottle, sweat towel,
   body wipes, deodorant sticks, disposable underwear, foot powder insert,
   blister plasters, muscle recovery balm) are most strongly associated with
   sock purchases, with a recommended **3–4 item starter bundle** (rather than
   bundling all 9 at once).

## Project structure
```
dashboard_project/
├── app.py                          # Streamlit app (multi-section dashboard)
├── requirements.txt                # Python dependencies
├── data/
│   └── disposable_socks_data.csv   # Enriched synthetic dataset (500 respondents, 44 columns)
└── colab_notebook.ipynb            # Standalone Google Colab notebook (EDA + full ML pipeline)
```

## Dashboard sections
- 🏠 Overview
- 📊 Descriptive Analytics — summary stats, distributions, cross-tabulation
- 🔍 Diagnostic Analytics — correlation heatmap, group comparison, bias/fairness check
- 🧩 Segmentation — K-Means (elbow + silhouette + 3D PCA view), Hierarchical
  clustering (single/complete/average/Ward dendrograms), RFM analysis,
  Regression (Linear vs Lasso vs Ridge)
- 🤖 Classification — KNN, Decision Tree, Random Forest, Gradient Boosting,
  with train/test accuracy, precision, recall, F1, ROC curve, confusion matrix,
  feature importance
- 🚨 Anomaly Detection — Isolation Forest on behavioral features
- 🛒 Association Rule Mining — Apriori algorithm, bundle recommendation with
  support/confidence/lift, tiered bundle suggestion
- 💡 Prescriptive Insights — actionable recommendations synthesized from all
  of the above

## Performance notes
The heavy computations (K-Means elbow search, classifier training, anomaly
detection, association rule mining) are wrapped in `@st.cache_data`. Streamlit
reruns the entire script on every widget interaction — without caching, moving
one unrelated slider would silently retrain every model on the page. Now a
computation only reruns when its actual inputs (selected features, k, alpha,
support/confidence thresholds, etc.) change; unrelated widget changes reuse the
cached result instantly. Random Forest / Gradient Boosting estimator counts
were also reduced (200→100, 150→100) with negligible accuracy impact on this
500-row dataset, and Random Forest runs multi-threaded (`n_jobs=-1`).

If the dashboard still feels slow after this, it's most likely Streamlit
Community Cloud's cold start (apps spin down after inactivity — the first load
after idle time can take 30–60s while it wakes back up); that's expected and
unrelated to the app code.

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Community Cloud
1. Push this folder to a new GitHub repository (keep `app.py`, `requirements.txt`,
   and the `data/` folder at the structure shown above).
2. Go to https://share.streamlit.io → "New app" → connect your GitHub repo.
3. Set **Main file path** to `app.py`.
4. Deploy. Streamlit Cloud will install `requirements.txt` automatically.

## Colab notebook
`colab_notebook.ipynb` mirrors the dashboard's analysis (EDA, clustering, RFM,
classification, anomaly detection, association rules) as a linear notebook —
useful for your write-up/report, since it shows the analysis step-by-step with
inline explanations, independent of the Streamlit UI.

## Data note
`disposable_socks_data.csv` is synthetic survey data (500 respondents). Two
known deterministic/near-deterministic relationships exist by design of the
original survey generation (`Workout_Type` and `Gym_Duration_Years` almost
perfectly determine `WTP_Disposable_Socks`) — the dashboard's classification
tab deliberately excludes these two from the model features to avoid data
leakage and inflated (100%) accuracy scores. This is documented inline in
`app.py`.
