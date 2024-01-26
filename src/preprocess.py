import pandas as pd
from typing import List, Dict


def recommend_top_n_product(
    customer_id: int, 
    merged_data: pd.DataFrame, knn_model: object, n: int = 5,
    features: List[str] = ["category", "price", "ratings", "page_views", "time_spent"],
    map_category: Dict[str, str] = {'Electronics': 1, 'Clothing': 2, 'Home & Kitchen': 3, 'Beauty': 4}
    ) -> List[int]:
    """
    Recommends the top N products for a given customer based on a KNN model.

    Parameters:
    - customer_id (int): The ID of the customer for whom recommendations are generated.
    - merged_data (pd.DataFrame): A DataFrame containing merged data of customer and product features.
    - knn_model (object): A trained KNN model used for recommendation.
    - n (int, optional): The number of top products to recommend. Defaults to 5.
    - features (List[str], optional): A list of feature names used for recommendation. 
      Defaults to ["category", "price", "ratings", "page_views", "time_spent"].
    - map_category (Dict[str, str], optional): A dictionary mapping category names to numerical values. 
      Defaults to {'Electronics': 1, 'Clothing': 2, 'Home & Kitchen': 3, 'Beauty': 4}.

    Returns:
    - List[int]: A list of product IDs recommended for the given customer.

    Note:
    - Ensure that the `merged_data` DataFrame contains the necessary features such as 'customer_id', 'product_id', 
      and features listed in the `features` parameter.
    - The `knn_model` should be a trained KNN model compatible with scikit-learn's KNeighborsClassifier
    - The `map_category` parameter maps category names to numerical values for better processing.
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