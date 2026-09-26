from src.data_loader import load_olist_data
from src.features import create_churn_dataset
from src.train import train_rf_model

def main():
    print("1. A carregar os datasets da Olist...")
    customers_df, orders_df, items_df, reviews_df = load_olist_data(data_path="data")

    print("2. A processar features e a gerar a target de churn...")
    dataset_final = create_churn_dataset(customers_df, orders_df, items_df, reviews_df, churn_days_threshold=90)

    print("3. A treinar e avaliar o modelo Random Forest...")
    model, feature_names = train_rf_model(dataset_final)

    print("Pipeline executado com sucesso!")

if __name__ == "__main__":
    main()