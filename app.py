import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
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


@st.cache_data
def review_length_average_over_time():
    v = data.groupby(['hypothesis_label', 'year'])['review_word_count'].mean().reset_index().rename(columns={'hypothesis_label':'sentiment'})
    fig = px.line(v, x='year', y='review_word_count', color='sentiment', color_discrete_sequence=['red', 'blue'])
    return fig

@st.cache_data
def reviews_by_category(category, sentiment, num_reviews):
    df = categorized[categorized['categories']==category]
    wr = df[df['hypothesis_label']==sentiment].sort_values(by='review_word_count', ascending=False)[:num_reviews]['reviews']
    return wr

@st.cache_resource
def get_summarization_pipeline():
    from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
    import torch
    
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float32)

    return pipeline(task="summarization", model=model, tokenizer=tokenizer, max_length=200, min_length=50)

@st.cache_data
def generate_summary(reviews):
    pipe = get_summarization_pipeline()

    summaries_result = pipe(reviews)
    return [sum_res['summary_text'] for sum_res in summaries_result]


category: str = st.selectbox(label="Select category of product", options=categorized['categories'].unique())
st.write(f"Number of product in our dataset for this category - {len(categorized[categorized['categories']==category])}")

col1, col2 = st.columns(2)

# Adding a simple text splitter initially, working on a better approach 
# for longer reviews
def splitter(texts):
    res = []
    for t in texts:
        if len(t.split())<512:
            res.append(t)
        else:
            res.append(" ".join(t.split()[:512]))
    return res


height = 600
width = 600
with col1:
    best_prod = top_products_by_category(category)
    best_prod.update_layout(height=400, width=width, yaxis_title="Number of products")
    st.plotly_chart(best_prod, use_container_width=True)

    fig1 = sentiment_by_category(category)
    fig1.update_layout(height=height, width=width)
    st.plotly_chart(fig1, use_container_width=True)

    nr1 = st.number_input(label="Enter number of negative reviews to check", 
                          min_value=0, 
                          max_value=len(categorized[(categorized['categories']==category) & (categorized['hypothesis_label']=='negative')]),
                          key=1)
    negative_reviews = reviews_by_category(category, 'negative', nr1).values
    st.dataframe(negative_reviews, use_container_width=True)

   
    nbt = st.checkbox(label="Generate summary of negative reviews")
    if nbt:
        # Display summaries in a more organized way
        if "negative_summaries" not in st.session_state:
            st.session_state["negative_summaries"] = generate_summary(splitter(list(negative_reviews)))
        st.session_state.negative_summaries = generate_summary(splitter(list(negative_reviews)))
        st.write("## Summaries")
        
        container1 = st.container()
        
        c1, c2 = st.columns(2)

        # Display summaries in a more organized way
        with container1:
            for i, summary in enumerate(st.session_state.negative_summaries, start=1):
                if i%2==1:
                    with c1:
                        st.text_area(label=f"### Review {i} Summary", value=f"{summary}")

                elif i%2==0:
                    with c2:
                        st.text_area(label=f"### Review {i} Summary", value=f"{summary}")
        

with col2:
    worst_prod = least_rating_products_by_category(category)
    worst_prod.update_layout(height=400, width=width, yaxis_title="Number of products")
    st.plotly_chart(worst_prod, use_container_width=True)

    fig2 = ratings_by_category(category)
    fig2.update_layout(height=height, width=width)
    st.plotly_chart(fig2, use_container_width=True)

    nr2 = st.number_input(label="Enter number of positive reviews to check", 
                          min_value=0, 
                          max_value=len(categorized[(categorized['categories']==category) & (categorized['hypothesis_label']=='positive')]),
                          key=2)
    positive_reviews = reviews_by_category(category, 'positive', nr2).values
    st.dataframe(positive_reviews, use_container_width=True)


    pbt = st.checkbox(label="Generate summary of positive reviews")
    if pbt:
        # Display summaries in a more organized way
        if "positive_summaries" not in st.session_state:
            st.session_state["positive_summaries"] = generate_summary(splitter(list(positive_reviews)))
        st.session_state.positive_summaries = generate_summary(splitter(list(positive_reviews)))
        st.write("## Summaries")
        
        container2 = st.container()
        
        c3, c4 = st.columns(2)

        # Display summaries in a more organized way
        with container2:
            for i, summary in enumerate(st.session_state.positive_summaries, start=1):
                if i%2==1:
                    with c3:
                        st.text_area(label=f"### Review {i} Summary", value=f"{summary}")

                elif i%2==0:
                    with c4:
                        st.text_area(label=f"### Review {i} Summary", value=f"{summary}")
        


st.write("Please autoscale from top right corner of figure because figure is getting squeezed for some categories")
fig3 = review_length_by_category_by_label(category)
fig3.update_layout(height=height, width=width, xaxis_title="Number of words in review", yaxis_title="Frequency")
st.plotly_chart(fig3, use_container_width=True)

fig4 = review_length_average_over_time()
fig4.update_layout(yaxis_title="Average number of words in review")
st.plotly_chart(fig4, use_container_width=True)