from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import joblib
import os

def train_rf_model(dataset_final: pd.DataFrame, test_size: float = 0.20, random_state: int = 42, output_model_path: str = "models/model.joblib"):
    """
    Separa atributos/target, treina o Random Forest, exibe métricas
    e serializa o modelo treinado em disco.
    """
    X = dataset_final[['total_spent', 'avg_item_price', 'avg_freight_value', 'total_items', 'avg_review_score']]
    y = dataset_final['is_churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("=== Relatório de Avaliação do Pipeline (src/train.py) ===")
    print(classification_report(y_test, y_pred, target_names=['Ativo (0)', 'Churn (1)']))
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")

   # Garante que a pasta existe sem gerar erro
    folder_path = os.path.dirname(output_model_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
    joblib.dump(rf_model, output_model_path)
    print(f"Modelo serializado com sucesso em: {output_model_path}")

    return rf_model, X.columns