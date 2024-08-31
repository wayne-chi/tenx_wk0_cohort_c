import streamlit as st
import pandas as pd

# 1. App Heading
st.title("Topic Modelling Dashboard")

# 2. File Upload Option
uploaded_file = st.file_uploader("Upload text file", type=["txt"])
if uploaded_file is not None:
    article_text = uploaded_file.read().decode("utf-8")
else:
    article_text = "No file uploaded yet."

# 3. Keyword Extraction Subheading
st.subheader("Keyword Extraction")

# 4. Top N Entry Box
top_n_keywords = st.number_input("Top N", min_value=1, max_value=100, value=10, step=1)

# 5. Model Dropdown for Keyword Extraction
keyword_model = st.selectbox("Model", ["TF-IDF", "YAKE", "Dummy Model 1"])

# 6. Scrollable Text Box for Keyword Extraction Results
st.text_area("Keywords Extracted", value="Results will be displayed here...", height=150, max_chars=None, key=None)

# 7. Topic Prediction Subheading
st.subheader("Topic Prediction")

# 8. Top N Entry Box for Topic Prediction
top_n_topics = st.number_input("Top N Topics", min_value=1, max_value=100, value=10, step=1)

# 9. Model Dropdown for Topic Prediction
topic_model = st.selectbox("Model", ["LDA", "NMF", "Dummy Model 2"])

# 10. Scrollable Text Box for Topic Prediction Results
st.text_area("Topics Predicted", value="Results will be displayed here...", height=150, max_chars=None, key=None)

# Predict Button
if st.button("Predict", key='predict_button'):
    # Dummy function for prediction
    st.write("Prediction functionality is not implemented yet.")

# 11. Domain Selection
st.subheader("Select Domain")
domain_selected = st.selectbox("Domain", ["forbes.com", "bbc.com", "cnn.com"])

# 12. Bar Graph of Topics
st.subheader("Bar Graph of Topics")
st.bar_chart(pd.DataFrame({
    'Topics': [5, 7, 8, 10],
    'Count': [3, 1, 2, 1]
}))

# 13. Bar Graph of Sentiments
st.subheader("Bar Graph of Sentiments")
st.bar_chart(pd.DataFrame({
    'Sentiments': ['Positive', 'Negative', 'Neutral'],
    'Count': [3, 1, 1]
}))
