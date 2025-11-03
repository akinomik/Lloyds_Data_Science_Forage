# Customer Churn Prediction Analysis

A comprehensive machine learning project for predicting customer churn using multi-source customer data with advanced feature engineering and model comparison.

## 🎯 Project Overview

This project analyzes customer churn patterns using a multi-dimensional dataset that includes:
- Customer demographics
- Transaction history
- Customer service interactions
- Online activity patterns
- Historical churn status

The analysis implements multiple machine learning algorithms and provides detailed feature importance analysis to identify key churn indicators.

## 📊 Dataset Structure

The project works with an Excel file (`data/cust.xlsx`) containing 5 sheets:

| Sheet | Description | Key Features |
|-------|-------------|--------------|
| `Customer_Demographics` | Basic customer info | Gender, MaritalStatus, IncomeLevel |
| `Transaction_History` | Purchase behavior | TransactionDate, AmountSpent, ProductCategory |
| `Customer_Service` | Support interactions | InteractionType, ResolutionStatus, complaints |
| `Online_Activity` | Digital engagement | LastLoginDate, LoginFrequency, ServiceUsage |
| `Churn_Status` | Target labels | ChurnStatus (binary classification) |

**Sample Size**: 1,000 customers with complete labeling

## 🔧 Technical Implementation

### Data Processing Pipeline
1. **Data Integration**: Merges 5 data sources into customer-level features
2. **Feature Engineering**: Creates 60+ predictive features including:
   - Recency metrics (days since last transaction/login/service)
   - Frequency patterns (transaction counts, service interactions)
   - Monetary features (spending amounts, averages, volatility)
   - Temporal windows (30/60/90-day activity)
   - Behavioral flags (complaints, unresolved issues)

3. **Data Quality**: Comprehensive validation with conservation checks
4. **Missing Data**: Intelligent imputation with missingness indicators

### Machine Learning Models

| Model | Type | Performance Metric | Key Features |
|-------|------|-------------------|--------------|
| **Random Forest** | Tree-based ensemble | ROC-AUC, PR-AUC | Handles mixed data types, feature importance |
| **LightGBM** | Gradient boosting | ROC-AUC, PR-AUC | Fast training, categorical handling |
| **CatBoost** | Gradient boosting | ROC-AUC, PR-AUC | Robust to overfitting, automatic categoricals |
| **Logistic Regression** | Linear baseline | ROC-AUC | Interpretable coefficients |

### Feature Engineering Highlights
- **Recency Analysis**: Days since last activity across all channels
- **Frequency Metrics**: Transaction/service interaction counts
- **Monetary Patterns**: Spending statistics with outlier handling
- **Temporal Windows**: Activity in recent 30/60/90 days
- **Behavioral Flags**: Complaint patterns, unresolved issues
- **Derived Ratios**: Average spend per transaction, activity gaps

## 📈 Key Results & Insights

### Model Performance
```
Cross-validation results (ROC-AUC):
- Random Forest: [Your CV score] ± [std]
- LightGBM: [Your test score]
- CatBoost: [Your test score]
- Logistic Regression: [Your baseline score]
```

### Top Predictive Features
Based on feature importance analysis across models:
1. **Recency metrics** (days since last transaction/login)
2. **Recent activity patterns** (90-day windows)
3. **Service interaction patterns** (complaints, unresolved issues)
4. **Spending behavior** (transaction frequency, amounts)
5. **Channel preferences** (Mobile App vs Website vs Online Banking)

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn lightgbm catboost matplotlib seaborn shap
```

### Installation
```bash
git clone [your-repo-url]
cd customer-churn-prediction
```

### Usage
```bash
python churn2.py
```

**Expected Input**: `data/cust.xlsx` with the 5 required sheets

## 📁 Project Structure
```
├── churn2.py              # Main analysis script
├── data/
│   └── cust.xlsx         # Input dataset (5 sheets)
├── README.md             # This file
└── requirements.txt      # Dependencies (optional)
```

## 🔍 Analysis Workflow

1. **Data Loading & Validation**
   - Load 5 Excel sheets
   - Data type conversion and validation
   - Duplicate detection and conservation checks

2. **Exploratory Data Analysis**
   - Missing data analysis
   - Distribution visualizations
   - Temporal pattern analysis
   - Churn rate by categorical variables

3. **Feature Engineering**
   - Customer-level aggregation from event data
   - Recency/frequency/monetary metrics
   - Temporal window features
   - One-hot encoding for categorical variables

4. **Model Training & Evaluation**
   - Train/test split with stratification
   - Cross-validation for robust estimates
   - Threshold optimization for F1-score
   - Comprehensive evaluation metrics

5. **Feature Importance Analysis**
   - Multiple importance methods (gain, split, permutation)
   - SHAP values for model interpretation
   - Feature ranking comparison across models

## 📊 Visualizations

The analysis generates several visualization types:
- **Distribution plots**: Transaction amounts, time gaps
- **Time series**: Weekly transaction/service patterns
- **Box plots**: Feature distributions by churn status
- **Feature importance**: Bar charts for top predictors
- **SHAP plots**: Model interpretation visualizations

## 🎯 Business Applications

### Churn Prevention Strategy
- **High-risk identification**: Score customers using trained models
- **Intervention timing**: Use recency features for proactive outreach
- **Channel optimization**: Leverage service usage patterns
- **Service improvement**: Address complaint patterns and resolution issues

### Key Business Metrics
- **Precision/Recall trade-offs**: Optimize for business cost/benefit
- **Feature-driven insights**: Focus on controllable factors
- **Temporal patterns**: Understand churn timing patterns

## ⚡ Performance Notes

- **Training time**: ~2-5 minutes for all models
- **Memory usage**: Optimized for datasets up to 100K customers
- **Scalability**: Modular design for larger datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -am 'Add new analysis'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Create Pull Request

## 📋 Future Enhancements

- [ ] **Model deployment**: REST API for real-time scoring
- [ ] **Feature store**: Automated feature pipeline
- [ ] **A/B testing framework**: Intervention effectiveness measurement
- [ ] **Advanced time series**: Sequential pattern analysis
- [ ] **Ensemble methods**: Model stacking/blending
- [ ] **Hyperparameter tuning**: Automated optimization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

**Author**: [Your Name]  
**Email**: [your.email@domain.com]  
**LinkedIn**: [your-linkedin-profile]

---

⭐ **Star this repository if you found it helpful!**
