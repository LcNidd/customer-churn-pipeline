from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

def train_rf_model(dataset_final: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Separa a matriz de atributos (X) e a target (y), realiza a divisão treino/teste
    e treina o modelo Random Forest.
    """
    # Matriz X e Target y (sem recency_days para evitar data leakage)
    X = dataset_final[['total_spent', 'avg_item_price', 'avg_freight_value', 'total_items', 'avg_review_score']]
    y = dataset_final['is_churn']

    # Divisão Treino / Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Treino do Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Avaliação
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("=== Relatório de Avaliação do Pipeline (src/train.py) ===")
    print(classification_report(y_test, y_pred, target_names=['Ativo (0)', 'Churn (1)']))
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")

    return rf_model, X.columns