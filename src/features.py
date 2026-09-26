import pandas as pd

def create_churn_dataset(customers_df, orders_df, items_df, reviews_df, churn_days_threshold: int = 90):
    """
    Consolida as tabelas da Olist, calcula a recência por cliente único,
    define a target de churn e extrai as features comportamentais/RFM.
    """
    # 1. Unir pedidos com clientes e filtrar apenas entregues
    orders_with_customers = orders_df.merge(customers_df, on='customer_id', how='inner')
    delivered_orders = orders_with_customers[orders_with_customers['order_status'] == 'delivered'].copy()
    
    # 2. Definir a data snapshot e calcular a recência
    snapshot_date = orders_df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    
    customer_recency = delivered_orders.groupby('customer_unique_id').agg(
        last_purchase_date=('order_purchase_timestamp', 'max')
    ).reset_index()
    
    customer_recency['recency_days'] = (snapshot_date - customer_recency['last_purchase_date']).dt.days
    customer_recency['is_churn'] = (customer_recency['recency_days'] > churn_days_threshold).astype(int)
    
    # 3. Unir itens e avaliações para calcular features comportamentais
    items_orders = items_df.merge(delivered_orders, on='order_id', how='inner')
    full_df = items_orders.merge(reviews_df, on='order_id', how='left')
    full_df['total_item_price'] = full_df['price'] + full_df['freight_value']
    
    features_df = full_df.groupby('customer_unique_id').agg(
        total_spent=('total_item_price', 'sum'),
        avg_item_price=('price', 'mean'),
        avg_freight_value=('freight_value', 'mean'),
        total_items=('order_item_id', 'count'),
        avg_review_score=('review_score', 'mean')
    ).reset_index()
    
    # 4. Dataset consolidado final
    dataset_final = customer_recency[['customer_unique_id', 'recency_days', 'is_churn']].merge(
        features_df, on='customer_unique_id', how='inner'
    )
    
    # Preenchimento de nulos no review_score
    dataset_final['avg_review_score'] = dataset_final['avg_review_score'].fillna(dataset_final['avg_review_score'].median())
    
    return dataset_final