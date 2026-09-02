# Uber Fare Prediction

## The Decision

How accurately can the business estimate a customer's fare before a trip, and which trip characteristics should drive that estimate?

The analysis shows that fare prediction can be improved by prioritizing **trip distance and geographic characteristics**, with a tuned **LightGBM model achieving the strongest predictive performance at approximately 3.72 RMSE**.

## What the Business Needs to Know

| Best Model | Final RMSE | Records Analyzed | Strongest Predictor |
|---|---:|---:|---|
| LightGBM | 3.72 | 150K+ | Trip Distance |

Trip distance emerged as the **dominant predictor of fare**, while pickup/drop-off location and temporal variables provided additional predictive value.

Boosting models also substantially outperformed linear approaches, indicating that fare behavior contains nonlinear relationships that simpler models do not capture as effectively.

## Recommended Actions

### 1. Use Trip Distance as the Primary Fare Predictor

Trip distance was the strongest predictive feature in the analysis.

Use calculated trip distance as a core input in fare-estimation systems, while supplementing it with geographic and temporal information to improve prediction accuracy.

![XGBoost Feature Importance](https://github.com/namvien94/uber-fare-prediction/blob/main/images/xgboost-feature-importance.png?raw=true)

### 2. Incorporate pickup and drop-off geography

Pickup and drop-off coordinates contributed predictive information beyond distance alone.

Use trip origin and destination characteristics alongside distance when estimating fares rather than assuming trips of similar distances should receive identical estimates.

### 3. Use nonlinear models for fare prediction

LightGBM, XGBoost, and Gradient Boosting outperformed Linear, Ridge, and Lasso Regression, with **LightGBM producing the strongest baseline performance at 3.766 RMSE**.

For this dataset, prioritize boosting-based models over basic linear approaches when prediction accuracy is the primary objective.

| Model | RMSE |
|---|---:|
| **LightGBM** | **3.7660** |
| XGBoost | 3.8604 |
| Gradient Boosting | 4.0257 |
| Random Forest | 4.0366 |
| Ridge Regression | 5.1079 |
| Linear Regression | 5.1082 |
| Lasso Regression | 5.3102 |
| Decision Tree | 6.2858 |

### 4. Keep the predictive feature set focused

Reducing the feature set improved LightGBM RMSE from approximately **3.766 to 3.752**, before hyperparameter tuning reduced it further to approximately **3.717**.

Prioritize features that contribute measurable predictive value rather than automatically retaining every available variable. This can simplify the model while maintaining or improving prediction accuracy.

| Model Stage | RMSE |
|---|---:|
| Baseline LightGBM | 3.766 |
| Selected-Feature LightGBM | 3.752 |
| **Tuned LightGBM** | **3.717** |

![Correlation Matrix](https://github.com/namvien94/uber-fare-prediction/blob/main/images/correlation-matrix.png?raw=true)

## How I Built It

- **Python & Pandas:** Cleaned and validated 200K trip records, handled missing values and geographic outliers, and prepared the data for modeling.
- **Feature Engineering:** Calculated trip distance using the Haversine formula and extracted year, month, day, hour, and day-of-week features from pickup timestamps.
- **Machine Learning:** Compared Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM regression models using RMSE.
- **Model Optimization:** Evaluated feature importance, performed feature selection, and tuned LightGBM hyperparameters to improve test RMSE from 3.766 to approximately 3.717.
- **Visualization:** Used Matplotlib, Seaborn, and Folium to analyze feature relationships, model importance, temporal patterns, and geographic trip distributions.
- **Tools:** Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn, Folium, Google Colab
