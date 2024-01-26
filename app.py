import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from src.preprocess import recommend_top_n_product


pd.set_option('future.no_silent_downcasting', True)
st.set_page_config(page_title="Terra Store Web AI", layout="wide")


@st.cache(allow_output_mutation=True)
def load_data(dataset_path: str = "./src/agg_table.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(dataset_path)
    except:
        print("[*] Joining Table on Server...")
        df_customer = pd.read_csv("../dataset/customer_interactions_synt.csv")
        df_product  = pd.read_csv("../dataset/product_details_synt.csv")
        df_purchase = pd.read_csv("../dataset/purchase_history_synt.csv")
        merged_data = pd.merge(df_product, df_purchase, on='product_id', how='inner')
        merged_data = pd.merge(merged_data, df_customer, on='customer_id', how='inner')
        merged_data = merged_data.drop_duplicates().reset_index(drop=True)
        merged_data.to_csv(dataset_path, index=False)
        return merged_data


@st.cache(allow_output_mutation=True)
def load_model(model_path: str = "./src/model.pkl") -> object:
    try:
        return joblib.load(model_path)
    except:
        print("[*] Train KNN on Server...")
        merged_data = pd.read_csv("./src/agg_table.csv")
        features = ["category", "price", "ratings", "page_views", "time_spent"]
        X = merged_data[features]
        map_category = {
            'Electronics': 1, 'Clothing': 2, 
            'Home & Kitchen': 3, 'Beauty': 4
        }
        X['category'] = X['category'].replace(map_category)
        knn_model = NearestNeighbors(n_neighbors=5, metric="euclidean")
        knn_model.fit(X)
        joblib.dump(knn_model, model_path)


def cust_behaviour_viz(df: pd.DataFrame, cust_id: int) -> None:
    st1, st2 = st.columns(2)
    scope = df[df['customer_id'] == cust_id]
    with st1:
        st.text("Customer Purchase Category")
        scope_data_1 = scope['category'].value_counts()
        st.bar_chart(scope_data_1)
    with st2:
        st.text("Day to Day User Purchase")
        scope_data_2 = scope.groupby("purchase_date").agg({"total_purchase":['sum']}).reset_index()
        scope_data_2.columns = ['purchase_date', 'total_purchase']
        st.bar_chart(scope_data_2, x="purchase_date", y="total_purchase")


def main():
    
    data = load_data()
    model = load_model()
    
    st.title("Terra Store's Recomendation System")
    customer_id = st.selectbox("Select Customer ID", list(range(1, 6)))

    if st.button("Recommend Product"):
        with st.spinner("Please Wait..."):
            recommend_product = recommend_top_n_product(
                customer_id=customer_id, merged_data=data,
                knn_model=model
            )
        st.write("Top 5 Recommended Products", recommend_product)
        cust_behaviour_viz(df=data, cust_id=customer_id)

if __name__ == "__main__":
    main()