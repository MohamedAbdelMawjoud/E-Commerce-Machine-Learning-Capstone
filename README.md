# 🛒 E-Commerce Machine Learning Capstone Project

## 📌 Project Overview

This project is a complete Machine Learning capstone project built on an e-commerce dataset.

The main objective was to transform raw e-commerce transactional data into a clean, analysis-ready dataset, perform Exploratory Data Analysis (EDA), engineer meaningful features, and build Machine Learning models to solve real-world business problems.

The project focuses on two main predictive tasks:

- **Delivery Delay Prediction** — Classification
- **Customer Rating Prediction** — Regression

The project follows a complete end-to-end Machine Learning workflow, starting from raw data ingestion and data integration and ending with model evaluation, hyperparameter tuning, visualization, and business interpretation.

---

## 🎯 Business Objectives

The project addresses two important business questions:

### 1. Delivery Delay Prediction

**Can we predict whether an order will be delivered late?**

This can help an e-commerce company:

- Identify high-risk deliveries early
- Improve logistics planning
- Monitor delivery performance
- Allocate resources more efficiently
- Improve customer satisfaction

### 2. Customer Rating Prediction

**Can we predict the rating a customer is likely to give to an order?**

This can help the business:

- Identify orders at risk of receiving low ratings
- Understand factors affecting customer satisfaction
- Prioritize problematic deliveries
- Improve customer experience

---

## 📂 Dataset

The project uses multiple e-commerce datasets containing information about:

- Customers
- Orders
- Order Items
- Payments
- Products
- Sellers
- Reviews

The datasets were integrated using common identifiers such as:

- `customer_id`
- `order_id`
- `product_id`
- `seller_id`

The final master dataset was created at the order level, allowing customer, product, payment, seller, pricing, and delivery information to be analyzed together.

---

## 🔄 Project Workflow

```text
Raw Data
   |
   v
Data Loading
   |
   v
Data Inspection
   |
   v
Data Cleaning
   |
   v
Data Integration / Merging
   |
   v
Feature Engineering
   |
   v
Exploratory Data Analysis
   |
   v
Feature Selection
   |
   v
Leakage Check
   |
   v
Train / Test Split
   |
   v
Encoding & Preprocessing
   |
   v
Model Training
   |
   v
Model Comparison
   |
   v
Hyperparameter Tuning
   |
   v
Final Evaluation
   |
   v
Business Insights
```

---

## 🧹 1. Data Preparation

The first stage was loading and inspecting all available datasets.

The following steps were performed:

- Loaded the individual CSV datasets
- Inspected dataset shapes
- Checked data types
- Examined missing values
- Checked duplicated records
- Investigated unique identifiers
- Converted date columns into appropriate datetime formats
- Checked relationships between the datasets
- Identified the correct keys required for merging

The datasets were then merged progressively to create a unified order-level dataset.

---

## 🔗 2. Data Integration

Several datasets were combined using relational keys.

```text
Customers
   ↓ customer_id
Orders
   ↓ order_id
Order Items
   ↓ product_id / seller_id
Products
Payments
Reviews
```

The final merged dataset allowed us to connect:

- Customer information
- Order information
- Product information
- Seller information
- Payment behavior
- Delivery information
- Customer reviews

into a single analytical dataset.

---

## ⚙️ 3. Feature Engineering

Several new features were created to make the raw data more useful for analysis and Machine Learning.

**Time-Based Features** (from `order_purchase_timestamp`)

- `purchase_hour`
- `purchase_day`
- `purchase_month`
- `purchase_weekday`
- `is_weekend`

**Delivery Features**

- `approval_delay_hours`
- `delivery_delay_days`
- `is_late`
- `estimated_delivery_days`

**Pricing Features**

- `total_price`
- `total_freight`
- `total_order_value`
- `avg_item_price`
- `avg_freight`
- `freight_ratio`

**Order Complexity Features**

- `total_items`
- `unique_products`
- `unique_sellers`
- `unique_seller_states`
- `multiple_seller_states`
- `order_complexity`
- `seller_count`

**Payment Features**

- `payment_count`
- `total_payment_value`
- `max_installments`
- `avg_installments`

These engineered features provide more meaningful representations of customer orders than the raw transactional fields alone.

---

## 📊 4. Exploratory Data Analysis

EDA was performed to understand the structure and behavior of the data before applying Machine Learning.

The analysis included:

- Distribution analysis
- Missing-value investigation
- Target distribution
- Correlation analysis
- Comparison between late and on-time orders
- Numerical feature distributions
- Identification of highly correlated features
- Investigation of relationships between order value, freight, payments, and delivery behavior

### Important Correlations

Several highly correlated feature pairs were identified:

| Feature Pair | Correlation |
|---|---:|
| `total_order_value` ↔ `total_payment_value` | ≈ 1.000 |
| `total_price` ↔ `total_order_value` | ≈ 0.996 |
| `total_price` ↔ `total_payment_value` | ≈ 0.996 |
| `max_installments` ↔ `avg_installments` | ≈ 0.997 |
| `total_price` ↔ `avg_item_price` | ≈ 0.933 |
| `total_items` ↔ `order_complexity` | ≈ 0.885 |

These relationships were considered during feature selection to reduce redundancy and improve model interpretability.

---

## 🔍 5. Feature Selection & Leakage Check

Feature selection was performed separately for each Machine Learning task.

For the delivery delay model, features that directly reveal the actual delivery outcome were excluded. For example:

- Actual delivery duration
- Delivery status
- Other post-delivery information

were not used as predictive inputs.

This is important because the model is supposed to make predictions using information available before the delivery outcome is known.

---

## 🧪 6. Train / Test Split

The datasets were divided into training and testing sets.

**For the Delivery Delay Classification task:**

```
X_train: 77,182 rows
X_test : 19,296 rows

y_train: 77,182
y_test : 19,296
```

A test size of 20% was used.

Because the classification target was imbalanced, a stratified split was used to preserve the class distribution between training and testing sets.

---

## 🔤 7. Encoding & Preprocessing

The preprocessing pipeline handled numerical and categorical variables separately.

**Numerical Features**

- Median imputation for missing values
- `StandardScaler` for normalization

**Categorical Features**

- Most-frequent imputation
- One-Hot Encoding

```python
OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=True
)
```

All preprocessing steps were integrated into Scikit-Learn Pipelines to prevent data leakage between training and testing data.

---

## 🤖 8. Model 2 — Delivery Delay Prediction

**Problem Type:** Binary Classification

**Target:** `is_late`

Where:
- `1` → Late
- `0` → On Time

The following algorithms were compared:

- Logistic Regression
- Logistic Regression with Class Weights
- Decision Tree
- Decision Tree with Class Weights
- Random Forest
- Random Forest with Class Weights
- KNN

### 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression Balanced | 0.6609 | 0.1398 | 0.6173 | 0.2280 | 0.6885 |
| Decision Tree Balanced | 0.8653 | 0.2055 | 0.2307 | 0.2173 | 0.5760 |
| Decision Tree | 0.8701 | 0.2125 | 0.2224 | 0.2173 | 0.5748 |
| KNN | 0.9137 | 0.2984 | 0.0473 | 0.0816 | 0.6061 |
| Random Forest | 0.9198 | 0.8462 | 0.0141 | 0.0277 | 0.7434 |
| Random Forest Balanced | 0.9195 | 0.9231 | 0.0077 | 0.0152 | 0.7424 |
| Logistic Regression | 0.9189 | 0.5000 | 0.0038 | 0.0076 | 0.6962 |

Because the target variable was highly imbalanced, accuracy alone was not considered sufficient.

The balanced Logistic Regression model provided substantially higher recall for late deliveries, making it more useful when the business objective is to identify as many potentially late orders as possible.

---

## 🎛️ 9. Hyperparameter Tuning — Delivery Delay Model

Hyperparameter tuning was performed on the selected Logistic Regression model.

A stratified 5-fold cross-validation strategy was used.

The tuning grid included:

```
C = [0.01, 0.1, 1]
penalty = ['l2']
solver = ['lbfgs', 'liblinear']
```

The model was optimized using **F1-macro**, because the classification problem is imbalanced.

**Best Parameters**

- C = 1
- Penalty = L2
- Solver = liblinear
- Class Weight = balanced

**Cross-Validation Result**

- Best F1-macro (CV): 0.5054

**Test Performance**

- Precision: 0.1388
- Recall: 0.6128
- F1: 0.2263
- ROC-AUC: 0.6878

The relatively high recall indicates that the tuned model is much better at identifying late deliveries than models that optimize mainly for overall accuracy.

---

## 📉 10. Confusion Matrix & ROC Curve

The final classification model was evaluated using:

- Confusion Matrix
- ROC Curve
- ROC-AUC

The confusion matrix provides a direct view of:

- True Positives
- True Negatives
- False Positives
- False Negatives

The ROC curve evaluates the model's ability to distinguish between late and on-time orders across different classification thresholds.

---

## ⭐ 11. Model 3 — Customer Rating Prediction

**Problem Type:** Regression

**Target:** `rating`

The target represents the customer review score from 1 to 5. Rows with missing ratings were removed before training.

The rating dataset contained **95,832 observations**.

After preprocessing and splitting:

```
X_train: 76,665
X_test : 19,167

y_train: 76,665
y_test : 19,167
```

### 🤖 Models Compared

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNN Regressor

### 📊 Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest Regressor | 0.8640 | 1.1338 | 0.2149 |
| Linear Regression | 0.9026 | 1.1839 | 0.1439 |
| Decision Tree Regressor | 0.8862 | 1.1871 | 0.1394 |
| KNN Regressor | 0.9178 | 1.2529 | 0.0413 |

Random Forest Regressor achieved the best overall performance based on both MAE and RMSE.

---

## 🎛️ 12. Hyperparameter Tuning — Rating Model

Since Random Forest Regressor was the best-performing regression algorithm, hyperparameter tuning was applied only to this model.

The tuning grid included:

```
n_estimators = [100, 200]
max_depth = [10, 15]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
```

A 3-fold cross-validation strategy was used.

The scoring metric was **Negative Root Mean Squared Error**.

**Best Parameters**

- n_estimators = 200
- max_depth = 10
- min_samples_split = 2
- min_samples_leaf = 2

**Tuned Model Performance**

- CV RMSE: 1.1348
- Test MAE: 0.8662
- Test RMSE: 1.1319
- Test R²: 0.2175

The tuned Random Forest achieved a slightly lower test RMSE and a slightly higher R² than the original model.

---

## 💡 13. Key Business Insights

### Delivery Delay

**Insight 1 — Class Imbalance Is Critical**
The delivery-delay target is highly imbalanced. Models that achieved very high accuracy often had extremely low recall for late deliveries. This shows that accuracy alone can be misleading in logistics prediction.

*Business implication:* The company should prioritize recall and F1-score when the goal is to proactively identify late orders.

**Insight 2 — Balanced Logistic Regression Is More Useful for Risk Detection**
The balanced Logistic Regression model achieved approximately **61% Recall** on the late-delivery class, meaning it can identify a significantly larger proportion of potentially late orders than the unbalanced models.

*Business implication:* The model could be used as an early-warning system for potentially delayed orders.

**Insight 3 — High Accuracy Does Not Necessarily Mean a Useful Model**
Random Forest achieved approximately **92% Accuracy**, but its recall for late orders was only around **1.4%**. This means the model mostly predicts the majority class.

*Business implication:* Operational decisions should not rely on accuracy alone. Recall and F1-score are more informative when missing a late order has a business cost.

### Customer Rating

**Insight 4 — Delivery Experience Is Important for Customer Satisfaction**
Customer ratings are strongly connected to the overall order and delivery experience. Features related to delivery timing and order characteristics provide useful information for estimating customer satisfaction.

*Business implication:* Reducing delivery problems can potentially improve customer satisfaction and ratings.

**Insight 5 — Random Forest Captures Non-Linear Relationships**
Random Forest outperformed Linear Regression, Decision Tree, and KNN, indicating that customer ratings are influenced by more complex and non-linear interactions between order characteristics.

*Business implication:* Tree-based models can provide better predictive performance for customer satisfaction than simple linear relationships.

**Insight 6 — Rating Prediction Is a Difficult Problem**
The tuned Random Forest achieved MAE = 0.8662, RMSE = 1.1319, R² = 0.2175. The relatively low R² indicates that many factors affecting customer ratings are not available in the dataset.

*Business implication:* Additional information such as customer sentiment, review text, product quality, customer service interactions, and complaint history could improve future rating prediction models.

---

## 📌 14. Business Story

This project demonstrates how an e-commerce company can use Machine Learning to move from reactive decision-making to proactive business operations.

For delivery operations, the model can act as an early-warning system by identifying orders that are more likely to arrive late. Instead of waiting until a delivery becomes late, the company could prioritize high-risk orders, contact customers proactively, adjust logistics resources, and investigate operational bottlenecks.

For customer satisfaction, the rating prediction model can help identify orders that may receive lower customer ratings.

Combining both models could allow the business to identify high-risk orders from both an operational and customer-experience perspective:

```text
High probability of late delivery
              +
   High risk of low customer rating
              ↓
        High-priority order
              ↓
     Proactive intervention
```

This can support better logistics management, improved customer experience, and more data-driven decision-making.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook
- Machine Learning
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Classification
- Regression
- Hyperparameter Tuning

---

## 📁 Project Structure

```
E-Commerce-ML-Capstone/
│
├── data/
│   ├── customers.csv
│   ├── orders.csv
│   ├── order_items.csv
│   ├── order_payments.csv
│   ├── products.csv
│   └── reviews.csv
│
├── notebooks/
│   └── E-Commerce_ML_Capstone.ipynb
│
├── models/
│   ├── delivery_delay_model.pkl
│   └── rating_model.pkl
│
├── figures/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── rating_predictions.png
│
└── README.md
```

---

## 🚀 Conclusion

This capstone project demonstrates a complete Machine Learning workflow for an e-commerce business.

The project covered:

✅ Data loading and inspection
✅ Data cleaning
✅ Missing-value analysis
✅ Data integration and merging
✅ Feature engineering
✅ Exploratory Data Analysis
✅ Feature selection
✅ Data leakage prevention
✅ Train/Test splitting
✅ Encoding and preprocessing
✅ Classification
✅ Regression
✅ Model comparison
✅ Hyperparameter tuning
✅ Confusion Matrix
✅ ROC Curve
✅ Model evaluation
✅ Business interpretation

The main lesson from the project is that model performance should always be evaluated in the context of the business problem.

For delivery prediction, a model with high accuracy but extremely low recall is not necessarily useful. For customer rating prediction, the relatively low R² demonstrates that predictive performance depends heavily on the quality and completeness of the available business data.

Overall, the project shows how Machine Learning can transform transactional e-commerce data into actionable business insights.

---

If you are interested in discussing the project, Machine Learning, Data Science, or e-commerce analytics, feel free to connect with me on LinkedIn.

`#MachineLearning` `#Python` `#ScikitLearn` `#DataScience` `#DataAnalytics` `#Classification` `#Regression` `#Ecommerce`
