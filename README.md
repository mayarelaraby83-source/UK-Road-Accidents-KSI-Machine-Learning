# UK Road Accidents KSI Prediction — Machine Learning

## 📌 Project Overview

This project analyzes UK road traffic accident data from STATS19 (2018) to build a Machine Learning model that predicts whether an accident is likely to result in a **Killed or Seriously Injured (KSI)** outcome.

The project follows a complete Machine Learning pipeline, starting from data cleaning and merging, through Exploratory Data Analysis, Feature Engineering, model comparison and tuning, and finally translating the model results into practical road-safety recommendations.

The main goal was not simply to achieve the highest score, but to build a **realistic and trustworthy model** using information that would actually be available at prediction time.

## 🎯 Problem Statement

Road accidents vary widely in severity, while road-safety resources are limited.

The goal of this project is to predict whether an accident will be:

* **Slight**
* **Serious/Fatal (KSI)**

This was framed as a **binary classification problem**.

**Target:** `is_KSI`

* `1` → Fatal or Serious
* `0` → Slight

## 📊 Dataset

The project uses three official UK STATS19 datasets:

* `Accidents_2018.csv` — 122,635 accidents
* `Vehicles_2018.csv` — 226,409 vehicle records
* `Casualties_2018.csv` — 160,597 casualty records

The Vehicles dataset was aggregated to the accident level before merging with the Accident dataset.

The Casualties dataset was **excluded from modeling** to prevent data leakage, since casualty severity is closely related to the target outcome.

## 🔧 Data Processing

The main preprocessing steps included:

* Data cleaning
* Handling missing values
* Converting unknown `-1` values to `NaN`
* Aggregating vehicle-level data to accident level
* Merging datasets using `Accident_Index`
* Removing duplicate and redundant features
* Handling unrealistic engine capacity and driver-age values
* Iterative Imputation using a Decision Tree estimator
* Train/Test split with stratification

## 🧠 Feature Engineering

Several features were created to improve the model:

* `Hour`
* `Month`
* `Is_Weekend`
* Missing-value indicators for driver age, engine capacity and vehicle age
* Target Encoding for `Local_Authority_(Highway)`

Two feature versions were prepared:

* Scaled features for Logistic Regression
* Raw features for tree-based models

## 🔍 Exploratory Data Analysis

The EDA revealed several interesting patterns:

* KSI rates were higher during late-night and early-morning hours.
* Weekends showed higher KSI rates than weekdays.
* Rural accidents were less frequent but generally more severe.
* Darkness, fog and flooded roads were associated with higher KSI rates.
* Single carriageways showed higher KSI rates.
* Speed limit showed a non-linear relationship with KSI severity.
* Geographic accident density was concentrated around urban areas, while KSI and non-KSI accidents were spatially intermixed.

These findings suggest that accident severity depends on **non-linear interactions between multiple factors** rather than simple linear relationships.

## 🚨 Data Leakage Prevention

Data leakage was one of the biggest challenges in the project.

Several potentially leaking features were removed, including:

* `Casualty_Severity`
* `Accident_Severity`
* `Did_Police_Officer_Attend_Scene_of_Accident`

Preprocessing steps such as imputation, encoding and scaling were also fitted only on the training data to prevent information from the test set leaking into the model.

## 🤖 Models Tested

Several Machine Learning algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* LightGBM

Additional experiments included:

* K-Means
* PCA
* One-Class SVM
* Random Undersampling

A total of **18 experiments** were conducted across different stages.

## 🏆 Final Model

The final model was:

**XGBoost**

Best parameters:

* `n_estimators = 300`
* `max_depth = 6`
* `learning_rate = 0.03`
* `min_child_weight = 10`
* `subsample = 0.8`

The model achieved:

* **F1:** 0.411
* **ROC-AUC:** 0.686
* **PR-AUC:** 0.354

## 🎚️ Threshold Tuning

Instead of using the default probability threshold of `0.5`, the decision threshold was tuned for the road-safety context.

The selected threshold was:

**0.432**

This resulted in approximately:

* **Recall: 75%**
* **Precision: 27.7%**

The recall-focused threshold was chosen because, in road safety, missing a genuinely severe accident can be more costly than generating an additional false alarm.

## 💡 Business Value

The model is designed as a **Decision-Support Tool**, rather than an automatic decision-maker.

Potential applications include:

* Speed management
* Improving lighting in high-risk areas
* Prioritizing rural and high-risk roads
* Improving road maintenance and drainage
* Targeting safety campaigns
* Prioritizing limited road-safety budgets
* Supporting emergency-response planning

The predicted KSI probability can be used to **rank locations and conditions by risk**, helping authorities prioritize interventions.

## ⚠️ Limitations

* The model was trained using only 2018 UK data.
* The findings represent associations, not proven causation.
* The recall-focused threshold results in more false alarms.
* The model should be revalidated on other years and regions.
* The model should support expert judgment rather than replace it.

## 📁 Project Structure

```text
UK-Road-Accidents-KSI-Machine-Learning/
│
├── Accidents_2018.csv
├── Vehicles_2018.csv
├── Casualties_2018.csv
├── merged_accidents_vehicles.csv
│
├── Merge_data.ipynb
├── Project_2.ipynb
├── main.py
├── ksi_frontend.html
│
├── model_artifacts/
│   ├── ksi_xgboost_final.pkl
│   ├── iterative_imputer.pkl
│   ├── target_encoder.pkl
│   ├── threshold_config.json
│   └── model_comparison.csv
│
├── requirements.txt
├── report.pdf
└── UK-Road-Accidents-KSI-Presentation.pptx
```

##  Team

This project was developed as part of the **Samsung Innovation Campus** program.

**Special thanks to Samsung Innovation Campus for this experience.**
