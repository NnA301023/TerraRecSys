import pandas as pd
from typing import List, Dict


def recommend_top_n_product(
    customer_id: int, 
    merged_data: pd.DataFrame, knn_model: object, n: int = 5,
    features: List[str] = ["category", "price", "ratings", "page_views", "time_spent"],
    map_category: Dict[str, str] = {'Electronics': 1, 'Clothing': 2, 'Home & Kitchen': 3, 'Beauty': 4}
    ) -> List[int]:
    """
    
    """
    top_n_prods = []
    cust_features = merged_data[merged_data['customer_id']!=customer_id][features]
    cust_features['category'] = cust_features['category'].replace(map_category)
    cust_features = cust_features.drop_duplicates().reset_index(drop=True)
    _, indices = knn_model.kneighbors(cust_features.values)
    for i in indices.flatten():
        if merged_data.iloc[i]['product_id'] not in top_n_prods:
            top_n_prods.append(merged_data.iloc[i]['product_id'])
        if len(top_n_prods) >= n:
            break
    return top_n_prods