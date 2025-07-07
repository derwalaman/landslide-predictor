# 🏔️ Landslide Susceptibility Detection using Machine Learning

This project aims to detect and classify **landslide susceptibility levels** using a variety of **geospatial, terrain, climatic, and vegetation features**. It explores **multiclass classification** (0–4) using **Logistic Regression, Random Forest**, and **XGBoost**, enhanced with **SMOTE oversampling** to tackle class imbalance.

---

## 📌 Problem Statement

Landslides are rare but catastrophic events. Predicting where they might occur is challenging due to:
- Severe class imbalance (most areas have no landslides)
- Complex nonlinear relationships between features (e.g., precipitation, slope, vegetation)
- Real-world variability and noise in geospatial data

---

## 📊 Dataset

The dataset contains over **1 million rows** and the following features:

| Feature     | Description                                |
|-------------|--------------------------------------------|
| Aspect      | Terrain aspect (compass direction)         |
| DF, DR, DW  | Distance from Faults, Roads, Rivers        |
| LULC        | Land Use Land Cover class                  |
| Lithology   | Rock and soil type                         |
| NDVI        | Vegetation index                           |
| PC, Precip  | Precipitation-related measures             |
| TPI, TWI    | Topographic indices                        |
| Temp        | Temperature                                |
| landslides  | Target (0 = No, 1 = Minor, ..., 4 = Other) |

> Rare classes (4–9) were grouped into class `4 = Other` to simplify prediction.

---

## 📈 Exploratory Data Analysis (EDA)

- ✅ Value counts & class distribution
- ✅ Correlation heatmap
- ✅ Feature distributions by landslide class (histograms & boxplots)
- ✅ Outlier analysis using `RobustScaler`

<p align="center">
  <img src="images/eda_heatmap.png" width="600" alt="Correlation Heatmap">
</p>

---

## 🧪 Data Preprocessing

- 🚫 Removed extreme rare classes (grouped into class 4)
- ✅ Feature Scaling using `RobustScaler` (resistant to outliers)
- ✅ Train-test split with `stratify` to preserve class balance
- ✅ Used **SMOTE** to handle extreme class imbalance

---

## 🧠 Model Training & Evaluation

Three classifiers were trained on balanced data and evaluated on **imbalanced real-world test data**:

| Model           | Accuracy | Strengths                           | Weaknesses                      |
|-----------------|----------|-------------------------------------|---------------------------------|
| Logistic        | 73%      | Fast, interpretable                 | Weak for non-linear patterns    |
| Random Forest   | 97%      | Non-linear, better than Logistic    | Still biased toward class 0     |
| XGBoost         | 97%      | Best minority class recall          | Still poor on very rare classes |

### Metrics used:
- ✅ Classification Report (Precision, Recall, F1)
- ✅ Confusion Matrix
- ⚠️ Accuracy alone is misleading due to imbalance

---

## 📉 Confusion Matrix Example (XGBoost)

| Actual \ Predicted | 0     | 1   | 2  | 3  | 4  |
|--------------------|-------|-----|----|----|----|
| 0 (No Landslide)   | 202k  | 4.5k|1k |198 | 34 |
| 1 (Minor)          | 741   | 342 | 98 | 27 |  2 |
| 2 (Medium)         | 77    | 55  | 17 |  2 |  1 |

---

## 🛠️ Technologies Used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)

---

## 📌 Key Takeaways

- Class imbalance heavily skews performance
- SMOTE significantly improves model training
- Ensemble models like **XGBoost** are best suited for this task
- Still, real-world deployment requires:
  - Better sampling
  - Domain tuning
  - Possibly region-specific models

---

## 🚀 Future Work

- Use deep learning with spatial image patches
- Incorporate DEM (Digital Elevation Model) data
- Build a web dashboard for real-time risk analysis

---

## 🧑‍💻 Author

**Aman Derwal**  
NIT Delhi | Landslide ML Project 