# 🛒 E-Commerce Machine Learning Capstone Project

## 📌 Project Overview

This project is an end-to-end Machine Learning capstone project built on an e-commerce dataset.

The main objective was to transform raw e-commerce data into meaningful analytical features and build Machine Learning models that can support real-world business decisions.

The project covers the complete Machine Learning workflow:

- Data Understanding
- Data Cleaning
- Data Integration
- Data Aggregation
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Selection
- Data Leakage Checking
- Train/Test Split
- Encoding & Preprocessing
- Model Training
- Model Comparison
- Hyperparameter Tuning
- Model Evaluation
- Business Insights

---

# 🎯 Business Objectives

The project focuses on two main predictive tasks:

### Model 2 — Delivery Delay Prediction

A binary classification model that predicts whether an order will be delivered late.

**Target:**

```text
is_late
0 → On Time
1 → Late

The goal is to identify high-risk deliveries early and help the business improve logistics planning and delivery performance.

Model 3 — Customer Rating Prediction

A regression model that predicts the customer rating an order will receive.

Target:

rating

The goal is to identify orders that may result in lower customer satisfaction and understand the factors associated with customer ratings.

📂 Dataset

The project uses an e-commerce dataset containing information about:

Customers
Orders
Order items
Payments
Products
Sellers
Reviews
Delivery information

Multiple tables were integrated using common identifiers such as:

order_id
customer_id
product_id
seller_id

The final order-level dataset was created after merging and aggregating the required information.

🔄 Project Workflow
Raw E-Commerce Data
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Data Integration / Merge
        ↓
Aggregation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Leakage Check
        ↓
Train / Test Split
        ↓
Encoding & Scaling
        ↓
Model Training
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Final Evaluation
        ↓
Business Insights
🧹 1. Data Cleaning

The initial datasets were inspected for:

Missing values
Duplicate records
Incorrect data types
Invalid values
Inconsistent timestamps
Unnecessary columns

Date columns were converted into proper datetime format to allow time-based analysis and feature engineering.

Examples:

order_purchase_timestamp
order_approved_at
order_delivered_carrier_date
order_estimated_delivery_date
🔗 2. Data Integration

The different e-commerce tables were merged together to create a unified dataset.

The main objective was to move from multiple transaction-level tables to an order-level analytical dataset.

This allowed information from:

Orders
Customers
Order Items
Payments
Products
Sellers
Reviews

to be analyzed together.

📊 3. Data Aggregation

Because one order can contain:

Multiple products
Multiple sellers
Multiple payments

aggregation was required before building the final order-level dataset.

Examples of aggregated features include:

total_items
total_price
total_freight
avg_item_price
avg_freight
unique_products
unique_sellers
payment_count
total_payment_value
max_installments
avg_installments

This transformed multiple rows per order into a single analytical row per order.

🧠 4. Feature Engineering

Several business-oriented features were created from the raw data.

Time Features
purchase_hour
purchase_day
purchase_month
purchase_weekday
is_weekend
Delivery Features
approval_delay_hours
estimated_delivery_days
delivery_delay_days
is_late
Pricing Features
total_order_value
freight_ratio
avg_item_price
Order Complexity
order_complexity
unique_seller_states
multiple_seller_states
seller_count

These features were designed to represent customer behavior, order complexity, pricing, payment behavior, and delivery characteristics.

📈 5. Exploratory Data Analysis

EDA was performed to understand the structure and behavior of the final dataset.

The analysis included:

Target distribution
Descriptive statistics
Feature distributions
Correlation analysis
Comparison between late and on-time orders
Feature relationships
Highly correlated features

Several strong correlations were identified.

For example:

total_order_value ↔ total_payment_value
Correlation ≈ 1.00


total_price ↔ total_order_value
Correlation ≈ 0.996


max_installments ↔ avg_installments
Correlation ≈ 0.997


total_items ↔ order_complexity
Correlation ≈ 0.885

This analysis helped identify redundant features and understand relationships between variables.

🔍 6. Feature Selection & Leakage Check

Features were reviewed before model training to avoid using information that would not realistically be available at prediction time.

For the delivery delay model, the target was:

is_late

Features related to actual delivery outcomes were excluded from the predictive feature set.

The goal was to ensure that the model learns from information available before or around the order-processing stage rather than directly observing the future outcome.

✂️ 7. Train / Test Split

The data was divided into:

80% Training
20% Testing

For the delivery classification model:

X_train: 77,182 rows
X_test : 19,296 rows

Stratified splitting was used to preserve the target class distribution.

⚙️ 8. Encoding & Preprocessing
Numerical Features

Numerical features were processed using:

Median Imputation
        ↓
StandardScaler
Categorical Features

Categorical variables were processed using:

Most-Frequent Imputation
        ↓
OneHotEncoder

The preprocessing and model were combined using a Scikit-Learn Pipeline.

This helped prevent data leakage during preprocessing and cross-validation.

🤖 9. Model 2 — Delivery Delay Prediction
Problem Type

Binary Classification

The model predicts:

0 → On Time
1 → Late
Algorithms Compared

The following models were evaluated:

Logistic Regression
Logistic Regression with Class Weight Balancing
Decision Tree
Balanced Decision Tree
Random Forest
Balanced Random Forest
KNN
📊 Model Comparison
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression Balanced	0.6609	0.1398	0.6173	0.2280	0.6885
Decision Tree Balanced	0.8653	0.2055	0.2307	0.2173	0.5760
Decision Tree	0.8701	0.2125	0.2224	0.2173	0.5748
KNN	0.9137	0.2984	0.0473	0.0816	0.6061
Random Forest	0.9198	0.8462	0.0141	0.0277	0.7434
Random Forest Balanced	0.9195	0.9231	0.0077	0.0152	0.7424
Logistic Regression	0.9189	0.5000	0.0038	0.0076	0.6962
Key Observation

The classification problem is highly imbalanced.

Accuracy alone is therefore misleading.

For example, Random Forest achieved approximately:

Accuracy = 91.98%

but its recall for late orders was only:

Recall = 1.41%

This means the model classified most orders as on-time and missed most late orders.

The balanced Logistic Regression model provided substantially higher recall:

Recall = 61.73%
F1 = 0.228
ROC-AUC = 0.688

Therefore, recall and F1 are more informative than accuracy for this business problem.

🎛️ 10. Hyperparameter Tuning — Model 2

Hyperparameter tuning was performed on the selected Logistic Regression model using GridSearchCV.

Search Space
C:
[0.01, 0.1, 1]


Penalty:
l2


Solver:
lbfgs
liblinear

A stratified 5-fold cross-validation strategy was used.

Best Parameters
C = 1
Penalty = l2
Solver = liblinear
Class Weight = balanced
Tuned Test Performance
Precision = 0.1388
Recall    = 0.6128
F1        = 0.2263
ROC-AUC   = 0.6878

The tuning confirmed that the balanced Logistic Regression model was able to identify a significantly larger proportion of late orders than the unbalanced alternatives.

📉 11. Model 3 — Customer Rating Prediction
Problem Type

Regression

The target variable is:

rating

Rows with missing ratings were removed.

Algorithms Compared
Linear Regression
Decision Tree Regressor
Random Forest Regressor
KNN Regressor
📊 Model Comparison
Model	MAE	RMSE	R²
Random Forest Regressor	0.8640	1.1338	0.2149
Linear Regression	0.9026	1.1839	0.1439
Decision Tree Regressor	0.8862	1.1871	0.1394
KNN Regressor	0.9178	1.2529	0.0413
Best Model

The initial best-performing model was:

Random Forest Regressor

with:

MAE  = 0.8640
RMSE = 1.1338
R²   = 0.2149
🎛️ 12. Hyperparameter Tuning — Model 3

Random Forest Regressor was selected for hyperparameter tuning.

Parameters Tuned
n_estimators
max_depth
min_samples_split
min_samples_leaf

A small grid with 16 combinations and 3-fold cross-validation was used.

Best Parameters
n_estimators = 200
max_depth = 10
min_samples_split = 2
min_samples_leaf = 2
Tuned Performance
MAE  = 0.8662
RMSE = 1.1319
R²   = 0.2175

The tuned model slightly improved RMSE and R² compared with the original Random Forest.

📊 13. Model Evaluation

The final models were evaluated using metrics appropriate for each task.

Classification
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
ROC Curve

Because of the class imbalance in delivery delays, particular attention was given to:

Recall
F1 Score
ROC-AUC

rather than relying only on accuracy.

Regression

The rating prediction model was evaluated using:

MAE
RMSE
R²

Additional evaluation included predicted-vs-actual analysis and residual/error analysis.

🧩 14. Business Insights
Delivery Delay
Insight 1 — Class imbalance is a major challenge

The high overall accuracy of several models does not indicate good late-delivery detection.

Models with high accuracy can still miss the majority of late orders.

Business implication:
The company should evaluate delivery-risk models using recall and F1 rather than accuracy alone.

Insight 2 — Balanced Logistic Regression detects more risky deliveries

The balanced Logistic Regression model achieved approximately:

61% Recall

for late orders.

Business implication:
This model can be used as an early-warning system to flag potentially late orders.

Insight 3 — False negatives are costly

A false negative occurs when an order is predicted to be on time but actually arrives late.

These cases are particularly important because they represent customers who may experience unexpected delays.

Business implication:
The company could prioritize high-risk orders for proactive monitoring or customer communication.

Insight 4 — Model selection should depend on business cost

Random Forest achieved high precision and ROC-AUC but extremely low recall.

Business implication:
If the goal is to catch as many late orders as possible, the balanced Logistic Regression model is more useful than simply choosing the model with the highest accuracy.

⭐ Customer Rating — Business Story
Insight 1 — Customer rating is difficult to predict

The tuned Random Forest achieved:

R² ≈ 0.218

This means the available order-level features explain only a limited portion of the variation in customer ratings.

Insight 2 — Rating is influenced by factors beyond the available order data

Customer satisfaction can depend on factors that may not be fully represented in the dataset, such as:

Product quality
Customer expectations
Review sentiment
Seller behavior
Customer-specific preferences

Business implication:
Additional customer and product-level information could improve future rating prediction.

Insight 3 — Random Forest performed best

Random Forest achieved the lowest RMSE among the tested regression algorithms.

Business implication:
The relationship between order characteristics and customer ratings is likely non-linear, making tree-based models more suitable than simple linear relationships.

Insight 4 — Model performance leaves room for improvement

The relatively modest R² indicates that the model should be considered a supporting decision tool rather than a precise rating predictor.

Business implication:
The business can use the model to identify broad risk patterns rather than relying on the prediction as an exact customer rating.

💼 Overall Business Story

This project demonstrates how Machine Learning can transform raw e-commerce transactions into actionable business intelligence.

The delivery model can help the business identify potentially late orders before delivery problems become visible.

The rating model can help identify orders where customer satisfaction may be at risk.

Together, the models can support:

Early Risk Detection
        ↓
Proactive Customer Communication
        ↓
Better Logistics Management
        ↓
Improved Customer Experience
        ↓
Better Business Decisions
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-Learn
Jupyter Notebook
GridSearchCV
OneHotEncoder
StandardScaler
Random Forest
Logistic Regression
Decision Trees
KNN
📁 Project Structure
E-Commerce-ML-Capstone/
│
├── data/
│   └── dataset files
│
├── notebooks/
│   └── E-Commerce_ML_Capstone.ipynb
│
├── models/
│   ├── delivery_delay_model.pkl
│   └── customer_rating_model.pkl
│
├── reports/
│   └── project_report.pdf
│
├── figures/
│   ├── eda/
│   ├── confusion_matrix/
│   ├── roc_curve/
│   └── regression/
│
└── README.md
🚀 Key Takeaways
Raw e-commerce tables were integrated into an analytical dataset.
Aggregation transformed transaction-level data into order-level features.
Feature engineering created meaningful business variables.
EDA revealed important relationships and highly correlated features.
Leakage checks were performed before modeling.
Multiple Machine Learning algorithms were compared.
Class imbalance significantly affected delivery-delay classification.
Hyperparameter tuning was applied to the selected best-performing algorithms.
The tuned Logistic Regression model provided strong recall for late orders.
Random Forest Regressor achieved the best performance for customer rating prediction.
Model results were translated into practical business recommendations.
👨‍💻 Project Type

Machine Learning Capstone Project

End-to-end E-Commerce Analytics & Predictive Modeling

📌 Future Improvements

Potential future improvements include:

Advanced feature engineering
Better handling of class imbalance
Threshold optimization for late-order detection
Gradient Boosting models
XGBoost / LightGBM
Sentiment analysis of customer reviews
Customer-level behavioral features
Product-level quality features
Model explainability using SHAP
Deployment through an API or dashboard
🏷️ Tags

Machine Learning Python Scikit-Learn E-Commerce Classification Regression Data Science Feature Engineering EDA Random Forest Logistic Regression Hyperparameter Tuning
