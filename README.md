# Titanic Survival Prediction with R

A data science project built in R as part of the **Virtual Data Science with R Apprentice Internship** at YuvaIntern (October to November 2026). The project predicts whether a passenger on the RMS Titanic survived, based on passenger characteristics.

## Project Overview

- **Problem:** Binary classification, predicting `Survived` (Yes / No).
- **Dataset:** Public Titanic dataset (891 passengers) from Kaggle / datasciencedojo.
- **Language and tools:** R, RStudio, tidyverse, ggplot2, caret, randomForest, pROC.
- **Models:** Logistic Regression and Random Forest, compared on a held-out test set.

## Repository Structure

```
titanic-survival-r/
├── data/
│   └── titanic.csv                  # Dataset
├── scripts/
│   └── week4_model_building.R       # Cleaning, modeling and evaluation code
├── reports/
│   ├── Week1_Project_Plan.docx      # Project plan and strategy
│   ├── Week2_EDA_Report.docx        # Exploratory data analysis
│   ├── Week3_Data_Wrangling.docx    # Data wrangling and preprocessing
│   ├── Week4_Model_Building.docx    # Model building and evaluation
│   └── Week5_Final_Report.docx      # Final report and insights
├── outputs/
│   ├── roc_curves.png
│   ├── variable_importance.png
│   └── model_results.csv
└── README.md
```

Add or rename files to match what you actually upload.

## Workflow

| Week | Task | Summary |
|------|------|---------|
| 1 | Project planning | Objectives, scope, timeline, risks |
| 2 | Exploratory data analysis | Summary statistics and visualizations |
| 3 | Data wrangling | Missing values, feature engineering, encoding |
| 4 | Model building | Logistic regression and random forest, validation |
| 5 | Reporting | Final insights and recommendations |

## Data Preparation

- Missing `Age` values filled with the median age of each title group.
- Missing `Embarked` values filled with the most common port (S).
- New features: `Title` (extracted from name) and `FamilySize` (SibSp + Parch + 1).
- Categorical variables converted to factors.

## Modeling Approach

1. Stratified 80/20 train-test split (`set.seed(42)` for reproducibility).
2. 5-fold cross-validation on the training set, using AUC as the selection metric.
3. Random forest tuned over `mtry` = 2, 3, 4, 6 with 300 trees.
4. Final evaluation on the unseen test set.

## Results

Replace these with your own values after running the script (`model_results.csv`).

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Logistic Regression | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Random Forest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Key findings:** (write 2 or 3 sentences about which model performed better and which features mattered most.)

## How to Run

1. Install R and RStudio.
2. Install the required packages:
   ```r
   install.packages(c("tidyverse", "caret", "randomForest", "pROC"))
   ```
3. Clone this repository and open the project folder in RStudio.
4. Make sure `titanic.csv` is available (update the path in the script if you keep it in `data/`, for example `read_csv("data/titanic.csv")`).
5. Run `scripts/week4_model_building.R`.

## Limitations

- Small dataset, so results can vary with a different random split.
- Survival patterns reflect historical bias (sex, age and class priorities in 1912).
- Some age values were estimated.

## Future Work

- Tune more hyperparameters and try gradient boosting (xgboost).
- Add more engineered features (deck, ticket group size).
- Explain predictions with SHAP or partial dependence plots.

## Author

**Sabareeswaran M**
GitHub: [sabareeswaran-hash](https://github.com/sabareeswaran-hash)
LinkedIn: [sabareeswaran-m](https://linkedin.com/in/sabareeswaran-m-82116329a)
