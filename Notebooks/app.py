import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Preprocess import LoadModels
from Preprocess.Preprocess import  preprocess_text_lemmatize
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import psycopg2
from dotenv import load_dotenv

import plotly.express as px

# Load environment variables from .env file
load_dotenv()



if 'keyword_results' not in st.session_state:
    st.session_state['keyword_results'] = "Extracted keywords."
if 'topic_results' not in st.session_state:
    st.session_state['topic_results'] = "Predicted Topics"
    ## top N
if 'top_n_topics' not in st.session_state:
    st.session_state['top_n_topics'] = 10
if 'top_n_keywords' not in st.session_state:
    st.session_state['top_n_keywords'] = 10


# top_n_keywords
# top_n_keywords

nmf_vector_path = '../Models/nmf_tfidf_vectorizer.joblib'
nmf_model_path = '../Models/nmf_model.joblib'

# paths
lsi_model_path = '../Models/lsi_model.gensim'
lsi_vector_path = '../Models/lsi_dictionary.gensim'

lda_model_path = '../Models/lda_model.joblib'
lda_vector_path =  '../Models/lda_tfidf_vectorizer.joblib'

## read data for lda pairs
ndf = pd.read_csv('../data/lda_pair.csv')


# 1. App Heading
st.title("Topic Modelling Dashboard")

# 2. File Upload Option
uploaded_file = st.file_uploader("Upload text file", type=["txt"], key='uploaded_file')
if uploaded_file is not None:
    article_text = uploaded_file.read().decode("utf-8")
    st.session_state['article_text'] = article_text
else:
    article_text = "No file uploaded yet."
    st.session_state['article_text'] = article_text

# 3. Keyword Extraction Subheading
st.subheader("Keyword Extraction")

# 4. Top N Entry Box
top_n_keywords = st.number_input("Top N", min_value=1, max_value=100, value=st.session_state['top_n_keywords'], step=1, key='top_n_keywords_1')

# 5. Model Dropdown for Keyword Extraction
keyword_model = st.selectbox("Model", ["TF-IDF", "YAKE", ], key='keyword_model')

# 6. Scrollable Text Box for Keyword Extraction Results
st.text_area("Keywords Extracted", value= st.session_state['keyword_results'] , height=150, max_chars=None, key='keyword_results_1', disabled=True)

# 7. Topic Prediction Subheading
st.subheader("Topic Prediction")

# 8. Top N Entry Box for Topic Prediction
top_n_topics = st.number_input("Top N Topics", min_value=1, max_value=100, value=st.session_state['top_n_topics'], step=1, key='top_n_topics_1')

# 9. Model Dropdown for Topic Prediction
topic_model = st.selectbox("Model", ["NMF", "LDA", "LSI"], key='topic_model')

# 10. Scrollable Text Box for Topic Prediction Results
st.text_area("Topics Predicted", height=150, max_chars=None, value =st.session_state['topic_results'],   key='topic_results_1', disabled=True)

# Predict Button
if st.button("Predict", key='predict_button'):
    # function for Keyword model - for now, just updating the text areas
    keyword_model = st.session_state['keyword_model']
    topN = st.session_state['top_n_keywords_1']
    if keyword_model == "YAKE":
        m_path = '../Models/yake_params.json'
        extractor = LoadModels.YAKEExtractor(m_path,topN)
        
    else:
        m_path = '../Models/keyword_extractor_tfidf_vectorizer.joblib'
        extractor = LoadModels.TfidfExtractor(m_path,topN)

    # text = st.session_state['article_text']
    # text = """ In recent years, the rise of esports has revolutionized the concept of sports. 
    #     Competitive video gaming has attracted millions of viewers and participants, with tournaments 
    #     offering substantial prize money and professional teams forming around the world. While traditional
    #     sports emphasize physical prowess, esports demand strategic thinking, teamwork, and lightning-fast
    #     reflexes. The growth of esports underscores the evolving nature of sports in the digital age. """
    text = st.session_state['article_text']
    cleaned_text = preprocess_text_lemmatize(text)
    keywords = extractor.extract_keywords(cleaned_text)
    ktx = "\n".join([f"{i+1}. {v.title()}" for i,v in enumerate(keywords)])

    # function for Keyword model - for now, just updating the text areas
    chosen_topic_model = st.session_state['topic_model']

    if chosen_topic_model == 'LSI':
        topic_model_path = lsi_model_path
        topic_vector_path = lda_vector_path
        topic_model_name = 'lsi'
        disp_funct = lambda x: '\n'.join([f"{i+1}. {v.strip().title()}" for i, v in enumerate(x.split('+'))])

    # current model wasnt saved well
    # elif chosen_topic_model == 'LDA':
    #     topic_model_path = lda_model_path
    #     topic_vector_path = lda_vector_path
    #     topic_model_name = 'lda'
    
    else : #chosen_topic_model == 'NMF':
        topic_model_path = nmf_model_path
        topic_vector_path = nmf_vector_path
        topic_model_name = 'nmf'
        disp_funct = lambda x: '\n'.join([f"{i+1}. {v.strip().title()}" for i, v in enumerate(x.split(','))])

    no_top_words = st.session_state['top_n_topics_1']
    tp_model = LoadModels.TopicModelPredictor(topic_model_path, topic_vector_path, topic_model_name, no_top_words=no_top_words)

    ttx = tp_model.predict(cleaned_text)
    ttx = disp_funct(ttx)






    st.session_state['keyword_results'] = ktx
    st.session_state['topic_results'] =  ttx

    st.session_state['top_n_keywords'] = st.session_state['top_n_keywords_1']
    st.session_state['top_n_topics'] = st.session_state['top_n_topics_1']


## Plots


# Get PostgreSQL credentials from environment variables
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
database = os.getenv("POSTGRES_DB")

# Connect to PostgreSQL Database and retrieve all data
def get_data_from_db(query):
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Query to get all data from the table
query_all_data = "SELECT * FROM ml_features;"
df = get_data_from_db(query_all_data)

# Step 1: Create select box for domains
unique_domains = df['domain'].unique()

# 11. Domain Selection
st.subheader("Select Domain")
selected_domain = st.selectbox("Domain", unique_domains, key='selected_domain')
filtered_df = df[df['domain'] == selected_domain]

st.subheader(f"Tags Distribution for Domain: {selected_domain}")

# 12. Bar Graph of Topics
tag_counts = filtered_df['tags'].str.split(',', expand=True).stack().value_counts()
fig_tags = px.bar(tag_counts, x=tag_counts.index, y=tag_counts.values, labels={'x': 'Tags', 'y': 'Count'})
st.plotly_chart(fig_tags)


# st.subheader("Bar Graph of Topics")
# st.bar_chart(pd.DataFrame({
#     'Topics': [5, 7, 8, 10],
#     'Count': [3, 1, 2, 1]
# }))




# 13. Bar Graph of Sentiments

# Step 13: Create and display the bar chart of LDA topics number distribution
st.subheader(f"LDA Topics Number Distribution for Domain: {selected_domain}")

lda_counts = filtered_df['lda_topics_n'].value_counts()
fig_lda = px.bar(lda_counts, x=lda_counts.index, y=lda_counts.values, labels={'x': 'LDA Topics Number', 'y': 'Count'})
st.plotly_chart(fig_lda)


# st.subheader("Bar Graph of Sentiments")
# st.bar_chart(pd.DataFrame({
#     'Sentiments': ['Positive', 'Negative', 'Neutral'],
#     'Count': [3, 1, 1]
# }))

# Step 15: Display legend in Table
# ndf = pd.read_csv('../data/lda_pair.csv')
unique_lda_topics = ndf
st.subheader(f"{selected_domain} Unique Number topic Table")
unique_lda_df = ndf[['lda_topics_n', 'lda_topics']].drop_duplicates()
st.write(unique_lda_df)


# Display in two columns
# col1, col2 = st.columns(2)

# # with col1:
#     st.subheader("LDA Topics Numbers")
#     st.write(unique_lda_df[['lda_topics_n']].drop_duplicates())

# with col2:
#     st.subheader("LDA Topics Descriptions")
#     st.write(unique_lda_df[['lda_topics']].drop_duplicates())


# Footer
st.write("Developed by otto")
