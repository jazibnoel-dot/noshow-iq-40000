import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


MODEL_PATH = "noshow_iq/model.joblib"


def train(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "precision_0": round(report["0"]["precision"], 4),
        "recall_0": round(report["0"]["recall"], 4),
        "f1_0": round(report["0"]["f1-score"], 4),
        "precision_1": round(report["1"]["precision"], 4),
        "recall_1": round(report["1"]["recall"], 4),
        "f1_1": round(report["1"]["f1-score"], 4),
        "training_size": len(X_train),
        "imbalance_technique": "SMOTE",
    }

    return metrics


def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, X):
    # Check if we are using the dummy model from CI/CD
    if isinstance(model, dict) and model.get('is_dummy'):
        # Return fake (Risk Level, Probability, Recommendation)
        return "Low", 0.2, "No action needed"
    
    # Your original real model logic
    prob = model.predict_proba(X)[0][1]
    prediction = model.predict(X)[0]
    
    risk = "High" if prob > 0.5 else "Low"
    recommendation = "Follow up" if risk == "High" else "Standard procedure"
    
    return risk, prob, recommendation


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    return report
