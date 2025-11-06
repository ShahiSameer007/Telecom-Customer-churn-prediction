# main.py

from data_cleaning import load_and_clean_data
from EDA_and_visualizations import (
    plot_churn_distribution,
    plot_bill_vs_churn,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_feature_importance,
)
from prediction import train_and_evaluate

# Step 1: Load and clean data
data = load_and_clean_data()

# Step 2: EDA and Visualization (these now save automatically to /images)
plot_churn_distribution(data)
plot_bill_vs_churn(data)
plot_correlation_heatmap(data)

# Step 3: Model Training and Evaluation
model, features, coeffs, cm, accuracy = train_and_evaluate(data)

# Step 4: Visualization of Results (auto-saved)
plot_confusion_matrix(cm)
plot_feature_importance(features, coeffs)

print("\n📊 Insights:")
print("- Low service rating and high call drop rate → more churn.")
print("- Better internet speed → lower churn probability.")
print("- Monthly bill interacts with service satisfaction in predicting churn.")
