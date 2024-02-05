import streamlit as st
st.set_page_config(layout="wide")

st.title('Product reviews sentiment analysis')

import pandas as pd
import plotly.express as px

@st.cache_data
def load_data(path="./categorized.csv"):
    import os
    if os.path.exists(path):
        categorized = pd.read_csv(path, index_col=[0])
        categorized['categories'] = categorized['categories'].apply(lambda x: x.split(","))
        categorized = categorized.explode('categories')
    return categorized

categorized = load_data(path="./preprocessed_reviews.csv")

@st.cache_data
def ratings_by_category(category):
    fig = px.pie(data_frame=categorized[categorized['categories']==category], 
                # values='categories', 
                names='rating', 
                title=f'Ratings for {category}', width=300, height=300)
    return fig

@st.cache_data
def sentiment_by_category(category):
    fig = px.pie(data_frame=categorized[categorized['categories']==category], 
                # values='categories', 
                names='hypothesis_label', 
                title=f'Sentiment for {category}', width=300, height=300,
                color_discrete_sequence=['lightgreen', 'pink']) # [positive, negative]
    return fig

@st.cache_data
def low_rating_categories():
    fig = px.histogram(data_frame=categorized[categorized['rating']==1], 
                       x='categories', color='categories',
                       title="Frequency of low rating products by category")
    return fig

@st.cache_data
def top_products_by_category(category):
    df = categorized[(categorized['categories']==category) & ((categorized['rating']==4) | (categorized['rating']==5))]
    df = pd.DataFrame(df.groupby('product').count()['source']).reset_index()
    df = df.sort_values(by='source', ascending=False)[:5]
    fig = px.bar(data_frame=df, x='product', y='source', title=f"Top rated products from {category}")
    fig.update_traces(marker_color='green')
    return fig

@st.cache_data
def least_rating_products_by_category(category):
    df = categorized[(categorized['categories']==category) & ((categorized['rating']==1) | (categorized['rating']==2))]
    df = pd.DataFrame(df.groupby('product').count()['source']).reset_index()
    df = df.sort_values(by='source', ascending=True)[:5]
    fig = px.bar(data_frame=df, x='product', y='source', title=f"Least rated products from {category}")
    fig.update_traces(marker_color='red')
    return fig


category: str = st.selectbox(label="Select category of product", options=categorized['categories'].unique())

col1, col2 = st.columns(2)

height = 600
width = 600
with col1:
    best_prod = top_products_by_category(category)
    best_prod.update_layout(height=400, width=width, yaxis_title="Number of products")
    st.plotly_chart(best_prod, use_container_width=True)

    fig1 = sentiment_by_category(category)
    fig1.update_layout(height=height, width=width)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    worst_prod = least_rating_products_by_category(category)
    worst_prod.update_layout(height=400, width=width, yaxis_title="Number of products")
    st.plotly_chart(worst_prod, use_container_width=True)

    fig2 = ratings_by_category(category)
    fig2.update_layout(height=height, width=width)
    st.plotly_chart(fig2, use_container_width=True)


# fig3 = low_rating_categories()
# fig3.update_layout(height=800, width=1500, showlegend=False)
# fig3.update_xaxes(categoryorder="total descending")
# st.plotly_chart(fig3)