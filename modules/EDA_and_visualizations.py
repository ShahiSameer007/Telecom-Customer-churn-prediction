# import necessary libraries

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

IMAGE_DIR = r"D:\Telecom_customer_churn\images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def save_plot(fig_name):
    # Helper function to save plots in the images folder.
    file_path = os.path.join(IMAGE_DIR, f"{fig_name}.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Saved: {file_path}")

def plot_churn_distribution(data):
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Churn", data=data)
    plt.title("Churn Distribution")
    save_plot("churn_distribution")

def plot_bill_vs_churn(data):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y="Monthly_Bill", data=data)
    plt.title("Monthly Bill vs Churn")
    save_plot("bill_vs_churn")

def plot_correlation_heatmap(data):
    plt.figure(figsize=(6, 4))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    save_plot("correlation_heatmap")

def plot_confusion_matrix(cm):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot("confusion_matrix")

def plot_feature_importance(features, coefficients):
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Coefficient": abs(coefficients)
    }).sort_values(by="Coefficient", ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(x="Coefficient", y="Feature", data=feature_importance, palette="mako")
    plt.title("Feature Importance (Logistic Regression)")
    save_plot("feature_importance")
