import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from src.preprocess import recommend_top_n_product


pd.set_option('future.no_silent_downcasting', True)
st.set_page_config(page_title="Terra Store Web AI", layout="wide")


@st.cache(allow_output_mutation=True)
def load_data(dataset_path: str = "./src/agg_table.csv") -> pd.DataFrame:
    return pd.read_csv(dataset_path)


@st.cache(allow_output_mutation=True)
def load_model(model_path: str = "./src/model.pkl") -> object:
    return joblib.load(model_path)


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