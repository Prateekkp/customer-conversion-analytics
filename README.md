# 🎯 Customer Conversion Analytics: Optimizing Digital Marketing ROI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Project Overview

A comprehensive data science project that leverages **predictive analytics** and **A/B testing** to optimize digital marketing campaigns and maximize customer conversion rates. This analysis demonstrates industry-standard practices for marketing analytics, featuring advanced machine learning techniques, statistical experimentation, and actionable business insights.

### 🎯 Business Impact
- **11% improvement** in campaign efficiency through predictive targeting
- **9.11% uplift** in conversion rates using optimized campaign types
- **96.4% PR-AUC** model performance for conversion prediction
- **Significant cost optimization** through data-driven channel allocation

## 🔍 Key Features

### 📈 Advanced Analytics Pipeline
- **Predictive Modeling**: CatBoost, XGBoost, and LightGBM with hyperparameter optimization
- **A/B Testing Framework**: Statistical significance testing for campaign optimization
- **Feature Engineering**: Custom metrics for engagement and behavioral analysis
- **SHAP Interpretability**: Model explainability for business stakeholders

### 🛠 Technical Implementation
- **Imbalanced Learning**: SMOTE for handling conversion rate imbalance
- **Automated Hyperparameter Tuning**: Optuna optimization framework
- **Cross-Validation**: Stratified K-Fold for robust model evaluation
- **Threshold Optimization**: Precision-Recall curve analysis for business metrics

## 📁 Project Structure

```
customer-conversion-analytics/
│
├── 📓 customer_conversion_analysis.ipynb    # Main analysis notebook
├── 📊 dataset/
│   └── digital_marketing_campaign_dataset.csv
├── 📄 README.md                            # Project documentation
├── 📋 requirements.txt                     # Dependencies
└── 🔄 outputs/                            # Generated predictions and reports
    └── predictions_amex_style.csv
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
8GB+ RAM (recommended)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/customer-conversion-analytics.git
cd customer-conversion-analytics

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook customer_conversion_analysis.ipynb
```

### Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
optuna>=3.0.0
shap>=0.40.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
imbalanced-learn>=0.8.0
statsmodels>=0.13.0
```

## 📋 Methodology

### 1. 🔍 Data Exploration & Validation
- **Dataset**: 8,000 customer records with 20+ features
- **Target Variable**: Binary conversion (87.6% imbalance)
- **Feature Analysis**: Demographics, behavior, campaign metrics
- **Data Quality**: Zero missing values, no duplicates

### 2. ⚙️ Feature Engineering
```python
# Advanced engagement metrics
df['TotalPagesViewed'] = df['WebsiteVisits'] * df['PagesPerVisit']
df['EmailCTR'] = df['EmailClicks'] / df['EmailOpens'].replace(0,1)
df['IncomeBin'] = pd.qcut(df['Income'], q=4, labels=['L','M','H','VH'])
```

### 3. 🤖 Machine Learning Pipeline
- **Preprocessing**: StandardScaler + OneHotEncoder
- **Resampling**: SMOTE for imbalanced data
- **Models**: CatBoost (best), XGBoost, LightGBM
- **Optimization**: Optuna hyperparameter tuning (50 trials)

### 4. 📊 Model Performance
| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.844 |
| **PR-AUC** | 0.964 |
| **F1-Score** | 0.955 |
| **Precision** | 0.92 |
| **Recall** | 0.99 |

### 5. 🧪 A/B Testing Results
```python
# Campaign Type Comparison
Awareness Campaigns:   87.33% conversion rate
Conversion Campaigns:  95.29% conversion rate
Statistical Uplift:    9.11% (p < 0.001)
```

## 📊 Key Insights & Business Recommendations

### 🎯 Strategic Recommendations

1. **Campaign Optimization**
   - Prioritize Conversion-type campaigns (9.11% higher conversion rate)
   - Allocate 70% budget to high-performing channels
   - Implement predictive scoring for customer targeting

2. **Cost Efficiency**
   - Focus on PPC channels (lowest CPA: $5,600)
   - Target top 10% predicted converters (11% efficiency gain)
   - Optimize threshold at 0.516 for balanced precision-recall

3. **Customer Segmentation**
   - High-income segments show higher conversion propensity
   - Email engagement strongly correlates with conversion
   - Website behavior patterns predict purchase intent

### 📈 ROI Impact
- **Campaign Efficiency**: 11% improvement through predictive targeting
- **Cost Reduction**: Optimized CPA across channels
- **Revenue Uplift**: 9.11% increase in conversion rates

## 🔬 Technical Deep Dive

### Model Architecture
```python
pipeline = ImbPipeline([
    ('preprocess', ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])),
    ('smote', SMOTE(random_state=42)),
    ('model', CatBoostClassifier(**optimized_params))
])
```

### Feature Importance (SHAP Analysis)
The model identifies key conversion drivers:
- Customer engagement metrics (website visits, email CTR)
- Demographic factors (age, income level)
- Campaign characteristics (channel, type, spend)

### Hyperparameter Optimization
```python
# Best CatBoost Parameters
{
    'n_estimators': 276,
    'max_depth': 6,
    'learning_rate': 0.0508,
    'subsample': 0.8719
}
```

## 📚 Business Applications

### Marketing Teams
- **Campaign Planning**: Data-driven budget allocation
- **Customer Targeting**: Predictive scoring for lead prioritization
- **Performance Monitoring**: Real-time conversion tracking

### Data Science Teams
- **Model Deployment**: Production-ready prediction pipeline
- **Experimentation**: A/B testing framework
- **Feature Engineering**: Behavioral metric creation

### Leadership
- **ROI Reporting**: Quantified marketing impact
- **Strategic Planning**: Channel optimization insights
- **Resource Allocation**: Evidence-based budget decisions

## 🔄 Next Steps & Roadmap

### Phase 1: Enhancement
- [ ] Customer Lifetime Value (CLV) modeling
- [ ] Multi-touch attribution analysis
- [ ] Real-time prediction API deployment

### Phase 2: Advanced Analytics
- [ ] Causal inference for campaign impact
- [ ] Time series forecasting for seasonal trends
- [ ] Recommendation engine for campaign optimization

### Phase 3: MLOps Integration
- [ ] Model monitoring and drift detection
- [ ] Automated retraining pipeline
- [ ] A/B testing automation framework

---

### 🙏 Acknowledgments
- Dataset provided by kaggle
- Inspired by industry best practices in marketing analytics
- Built with open-source ML frameworks

---

**⭐ If you found this project helpful, please give it a star!**
