
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.pipeline import DataPreprocessor

def predict(data):
    """Predict using the trained model"""
    model_ = joblib.load('logistic_regression_model.pkl')
    scaler_ = joblib.load('scaler.pkl')
    model_features_ = joblib.load('model_features.pkl')
    data = scaler_.transform(data)
    predictions = model_.predict_proba(data[model_features_])[:, 1]
    return predictions

# Evaluating the model based on real results to check its robustness
def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc


if __name__ == "__main__":
    x_test = pd.read_csv('x_test.csv')
    prediction = predict(x_test)
    print(f'Here are the prediction results : {prediction}')