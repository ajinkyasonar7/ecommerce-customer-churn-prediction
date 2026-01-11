# E-commerce Customer Churn Prediction

A machine learning project that predicts which customers are likely to stop purchasing from an e-commerce platform. The model helps businesses identify at-risk customers early and take preventive action to retain them.

## What is Customer Churn?

Customer churn occurs when customers stop doing business with a company. In e-commerce:
- A customer who hasn't made a purchase in 3+ months is considered "churned"
- High churn rates mean losing customers faster than acquiring new ones
- Predicting churn allows businesses to intervene before customers leave

## Problem Statement

This project addresses a critical business challenge: identifying customers who are likely to stop purchasing before they actually do. By predicting churn in advance, companies can:
- Send targeted retention offers to at-risk customers
- Reduce customer acquisition costs
- Improve customer lifetime value
- Maintain stable revenue streams

## Dataset

The dataset contains e-commerce customer behavior data with 29 features including:
- Customer demographics (age, gender, country, city)
- Engagement metrics (login frequency, session duration, pages viewed)
- Shopping behavior (cart abandonment rate, wishlist items, purchase frequency)
- Customer service interactions
- Lifetime value and credit balance
- Target variable: Churned (0 = Active, 1 = Churned)

## Model Performance

- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: 92.0%
- **Training Data**: 80% of dataset
- **Testing Data**: 20% of dataset

The model successfully identifies customers who are likely to churn with high accuracy, enabling businesses to take proactive retention measures.

## Technical Details

### Libraries Used
- pandas - Data manipulation and analysis
- numpy - Numerical computing
- matplotlib & seaborn - Data visualization
- scikit-learn - Machine learning implementation

### Preprocessing Steps
1. Data cleaning and missing value handling
2. Feature scaling using MinMaxScaler
3. Categorical encoding using OneHotEncoder
4. Train-test split (80-20)

### Model Pipeline
The project implements a complete machine learning pipeline:
- Column transformation for numerical and categorical features
- Gradient Boosting Classifier with optimized hyperparameters
- Confusion matrix and accuracy evaluation

## Project Structure

```
.
├── customer-behavior-churn-prediction.ipynb    # Main Jupyter notebook
├── ecommerce_customer_churn_dataset.csv       # Dataset
└── README.md                                   # This file
```

## Business Impact

Understanding churn prediction can help businesses:
- **Reduce Costs**: Retaining existing customers costs 5-25x less than acquiring new ones
- **Increase Revenue**: A 5% increase in retention can boost profits by 25-95%
- **Target Marketing**: Focus retention efforts on high-value at-risk customers
- **Improve Service**: Identify pain points causing customer dissatisfaction

## How to Use

1. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Open the Jupyter notebook:
```bash
jupyter notebook customer-behavior-churn-prediction.ipynb
```

3. Run all cells to:
   - Load and explore the dataset
   - Preprocess the data
   - Train the model
   - Evaluate performance
   - Generate predictions

## Key Insights

The analysis reveals important factors that indicate churn risk:
- Low login frequency
- High cart abandonment rates
- Decreased session duration
- Reduced purchase frequency
- Lower email engagement

## Future Improvements

Potential enhancements to the project:
- Feature engineering to create more predictive variables
- Testing additional algorithms (Random Forest, XGBoost, Neural Networks)
- Hyperparameter tuning for improved performance
- Deployment as a web service for real-time predictions
- Integration with customer relationship management systems

## Requirements

- Python 3.7+
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## License

This project is available for educational and commercial use.

## Author

Data Science Project - E-commerce Customer Churn Analysis
