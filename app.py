# app.py

import streamlit as st
import pandas as pd
import numpy as np
import sweetviz as sv
import tempfile
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --- Streamlit Settings ---
st.set_page_config(page_title="📚 Online Course Recommender", page_icon="🚀", layout="wide")

# --- Function Definitions ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def preprocess(df):
    df['course_name'] = df['course_name'].fillna('')
    df['instructor'] = df['instructor'].fillna('Unknown')
    difficulty_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
    df['difficulty_encoded'] = df['difficulty_level'].map(difficulty_map)
    df['combined_features'] = df['course_name'] + " " + df['instructor'] + " " + df['difficulty_level']
    return df

@st.cache_resource
def build_models(df):
    user_item_matrix = df.pivot_table(index='user_id', columns='course_id', values='rating').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(tfidf_matrix)
    
    return user_item_matrix, user_similarity_df, model_knn, tfidf_matrix

def hybrid_recommendation(df, user_item_matrix, user_similarity_df, model_knn, tfidf_matrix,
                           user_id, course_title, n_recommendations=5, alpha=0.5):
    course_idx = df[df['course_name'].str.lower() == course_title.lower()].index
    if len(course_idx) == 0:
        return []

    distances, indices = model_knn.kneighbors(tfidf_matrix[course_idx], n_neighbors=20)
    content_scores = {}
    for i, idx in enumerate(indices.flatten()):
        if idx == course_idx[0]:
            continue
        course_id = df.iloc[idx]['course_id']
        content_scores[course_id] = 1 - distances.flatten()[i]

    if user_id not in user_similarity_df.index:
        return list(df[df['course_name'].str.lower() != course_title.lower()]['course_name'].sample(n=n_recommendations))

    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]

    collaborative_scores = {}
    for sim_user_id, similarity in similar_users.items():
        sim_user_ratings = user_item_matrix.loc[sim_user_id]
        for course_id, rating in sim_user_ratings.items():
            if user_ratings[course_id] == 0:
                collaborative_scores[course_id] = collaborative_scores.get(course_id, 0) + similarity * rating

    final_scores = {}
    all_course_ids = set(content_scores.keys()) | set(collaborative_scores.keys())
    for course_id in all_course_ids:
        content_score = content_scores.get(course_id, 0)
        collab_score = collaborative_scores.get(course_id, 0)
        final_scores[course_id] = alpha * content_score + (1 - alpha) * collab_score

    top_courses = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    recommended_course_names = df[df['course_id'].isin([c[0] for c in top_courses])]['course_name'].tolist()

    return recommended_course_names

# --- Sidebar ---
st.sidebar.title("⚡ Navigation")
app_mode = st.sidebar.radio("Go to", ["📊 EDA", "🤖 Recommender"])

uploaded_file = st.sidebar.file_uploader("Upload your Dataset CSV", type=["csv"])

# --- Main App ---
if uploaded_file is not None:
    df = load_data(uploaded_file)
    df = preprocess(df)
    user_item_matrix, user_similarity_df, model_knn, tfidf_matrix = build_models(df)

    if app_mode == "📊 EDA":
        st.title("📊 Exploratory Data Analysis (EDA)")

        st.info("ℹ️ Sweetviz generates a full profiling report.")
        with st.spinner('Analyzing your data... Please wait ⏳'):
            report = sv.analyze(df)
            report_path = os.path.join(tempfile.gettempdir(), "sweetviz_report.html")
            report.show_html(report_path)
        
        st.success("✅ Sweetviz Report is ready!")

        with open(report_path, 'r', encoding='utf-8') as f:
            html_data = f.read()

        st.components.v1.html(html_data, height=800, scrolling=True)

    elif app_mode == "🤖 Recommender":
        st.title("🤖 Hybrid Course Recommendation System")

        user_id = st.sidebar.selectbox("Select User ID", df['user_id'].unique())
        course_title = st.sidebar.selectbox("Select Base Course", df['course_name'].unique())
        alpha = st.sidebar.slider("Content-Based Weight (Alpha)", 0.0, 1.0, 0.5, 0.05)
        n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

        if st.sidebar.button("Get Recommendations 🚀"):
            with st.spinner('Fetching Recommendations...'):
                recommendations = hybrid_recommendation(
                    df, user_item_matrix, user_similarity_df, model_knn, tfidf_matrix,
                    user_id=user_id, course_title=course_title,
                    n_recommendations=n_recommendations, alpha=alpha
                )
            
            if recommendations:
                st.success("✅ Here are your Recommendations:")
                for idx, course in enumerate(recommendations, 1):
                    st.write(f"{idx}. **{course}**")
            else:
                st.error("⚠️ No matching course found. Please try a different course.")

else:
    st.warning("👆 Please upload a CSV file to get started!")

