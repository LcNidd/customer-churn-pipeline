import os
import pandas as pd

def load_olist_data(data_path: str = "data"):
    """
    Carrega os datasets essenciais da Olist a partir da pasta de dados.
    """
    customers_file = os.path.join(data_path, "olist_customers_dataset.csv")
    orders_file = os.path.join(data_path, "olist_orders_dataset.csv")
    items_file = os.path.join(data_path, "olist_order_items_dataset.csv")
    reviews_file = os.path.join(data_path, "olist_order_reviews_dataset.csv")
    
    customers_df = pd.read_csv(customers_file)
    orders_df = pd.read_csv(orders_file)
    items_df = pd.read_csv(items_file)
    reviews_df = pd.read_csv(reviews_file)
    
    # Converter timestamps para datetime
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    
    return customers_df, orders_df, items_df, reviews_df