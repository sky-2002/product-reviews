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

data = pd.read_csv("./preprocessed_reviews.csv", index_col=[0])
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

@st.cache_data
def review_length_by_category_by_label(category):
    data = categorized[categorized['categories']==category]
    fig = px.histogram(data_frame=data, x='review_word_count', 
                   title=f"Review length distribution by label for {category}", 
                   color='hypothesis_label', barmode='overlay',
                   color_discrete_sequence=['blue', 'red'])
    fig.update_xaxes(range=[0,100])
    # fig.update_yaxes(range=[0, 10])
    fig.add_vline(x=data[data['hypothesis_label']=='positive']['review_word_count'].mean(), 
                line_color='blue', annotation_text='Mean positive', 
                annotation_textangle = 90)
    fig.add_vline(x=data[data['hypothesis_label']=='negative']['review_word_count'].mean(), 
                line_color='red', annotation_text='Mean negative', 
                annotation_textangle = 90)

    fig.add_vline(x=data[data['hypothesis_label']=='positive']['review_word_count'].median(), 
                line_color='purple', annotation_text='Median positive', 
                annotation_textangle = 90)
    fig.add_vline(x=data[data['hypothesis_label']=='negative']['review_word_count'].median(), 
              line_color='orange', annotation_text="Median negative", 
              annotation_textangle = 90, annotation_position='left')
    return fig

# @st.cache_data
# def rating_distribution_of_category_over_time(category):
#     data = categorized[categorized['categories']==category]
#     gb = pd.DataFrame(data.groupby('year')).reset_index()
#     # fig = px.bar(data_frame=gb, x='year', y='rating')
#     return gb

@st.cache_data
def review_length_average_over_time():
    v = data.groupby(['hypothesis_label', 'year'])['review_word_count'].mean().reset_index().rename(columns={'hypothesis_label':'sentiment'})
    fig = px.line(v, x='year', y='review_word_count', color='sentiment', color_discrete_sequence=['red', 'blue'])
    return fig

@st.cache_data
def negative_reviews_by_category(category, num_reviews):
    df = categorized[categorized['categories']==category]
    wr = df[df['hypothesis_label']=='negative'].sort_values(by='review_word_count', ascending=False)[:num_reviews]['reviews']
    return wr

category: str = st.selectbox(label="Select category of product", options=categorized['categories'].unique())
st.write(f"Number of product in our dataset for this category - {len(categorized[categorized['categories']==category])}")

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

st.write("Please autoscale from top right corner of figure because figure is getting squeezed for some categories")
fig3 = review_length_by_category_by_label(category)
fig3.update_layout(height=height, width=width, 
                   xaxis_title="Number of words in review", yaxis_title="Frequency")
st.plotly_chart(fig3, use_container_width=True)

fig4 = review_length_average_over_time()
fig4.update_layout(yaxis_title="Average number of words in review")
st.plotly_chart(fig4, use_container_width=True)

nr = st.number_input(label="Enter number of reviews to check", min_value=0, max_value=len(categorized[(categorized['categories']==category) & (categorized['hypothesis_label']=='negative')]))
st.dataframe(negative_reviews_by_category(category, nr).values, use_container_width=True)


# fig3 = low_rating_categories()
# fig3.update_layout(height=800, width=1500, showlegend=False)
# fig3.update_xaxes(categoryorder="total descending")
# st.plotly_chart(fig3)