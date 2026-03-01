
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.pipeline import DataPreprocessor

def train_model(data_path):
    """Load data, preprocess, and train model"""
    
    # Load and preprocess data
    df = pd.read_csv(data_path)
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df, fit=True)
    
    # Select features
    df_key_attributes = df[['Delinquent_Account', 'Month_1', 'Month_4', 'Month_6', 
                             'Loan_Balance_Missing_Flag', 'Trend_Score', 
                             'Missed_to_Tenure_Ratio', 'Low_Credit_Score']]
    
    x, y = df_key_attributes.drop('Delinquent_Account', axis=1), df_key_attributes['Delinquent_Account']
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Train model
    model = LogisticRegression(class_weight='balanced', max_iter=300, random_state=42)
    model.fit(x_train_scaled, y_train)
    
    return x_test_scaled, y_test, model, scaler, preprocessor

def test_model(model, x_test_scaled, y_test, threshold=0.49):
    y_pred_proba = model.predict_proba(x_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, precision, recall

def cross_validate_model(model, x, y):
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score)
    }
    cv_results = cross_validate(model, x, y, cv=5, scoring=scoring)
    return cv_results

def save_model(model, scaler, preprocessor, model_path='model.pkl', scaler_path='scaler.pkl', preprocessor_path='preprocessor.pkl'):
    import joblib
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(preprocessor, preprocessor_path)


if __name__ == "__main__":
    x_test, y_test, model, scaler, preprocessor = train_model('1 Delinquency_prediction_dataset.csv')
    print("Model training completed successfully.")
    accuracy, precision, recall = test_model(model, x_test, y_test, threshold=0.49)
    print("Model evaluation completed successfully. Here are the results : ")
    print(f"Accuracy: {accuracy} \n Precision: {precision} \n Recall: {recall}")
    cv_results = cross_validate_model(model, x_test, y_test)
    print("Cross-validation completed successfully. Here are the results : ")
    print(f"Cross-validated Accuracy: {cv_results['test_accuracy'].mean()} \n Cross-validated Precision: {cv_results['test_precision'].mean()} \n Cross-validated Recall: {cv_results['test_recall'].mean()} \n Cross-validated F1 Score: {cv_results['test_f1'].mean()} \n Cross-validated ROC AUC: {cv_results['test_roc_auc'].mean()}")

