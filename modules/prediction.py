# modules/prediction.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def train_and_evaluate(data):
    """Trains Logistic Regression model and returns predictions, metrics, and confusion matrix."""
    X = data[["Monthly_Bill", "Call_Drop_Rate", "Internet_Speed_Mbps", "Customer_Service_Rating"]]
    y = data["Churn"].map({"Yes": 1, "No": 0})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"✅ Model Trained Successfully!\nAccuracy: {accuracy}%\n")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return model, X.columns, model.coef_[0], cm, accuracy
