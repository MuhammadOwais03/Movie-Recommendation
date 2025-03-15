import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.title("Movie Recommendation System")

uploaded_file = st.file_uploader("Upload Merged CSV File", type=["csv"])


l = ['overview', 'cast', 'genres', 'director']
def calculate_tfidf_matrix(merged_df):
    tfidf_list = []
    tfidf =  TfidfVectorizer(stop_words='english')
    for col in l:
        print(merged_df.head())
        merged_df[col] = merged_df[col].fillna("")
        t = tfidf.fit_transform(merged_df[col])
        tfidf_list.append(t)
    return tfidf_list
        
        
def calculate_cosine_similarity(tfidf_list):
    cos_sim_list = []
    for t in tfidf_list:
        cos_sim = np.dot(t, t.T).toarray()
        cos_sim_list.append(cos_sim)
    return cos_sim_list






def recommend_movies(title, merged_df, cos_sim, weights=[0.5, 0.3, 0.2, 0.2], top_n=10):
    title_col = "title_x" if "title_x" in merged_df.columns else "title"
    
    idx = merged_df[merged_df[title_col].str.lower() == title.lower()].index
    if len(idx) == 0:
        return pd.DataFrame(columns=[title_col, 'cast', 'genres', 'director'])  
    
    idx = idx[0]  
    scaler = MinMaxScaler()
    
    total_sim = np.array((weights[0] * cos_sim[0][idx]) + 
                          (weights[1] * cos_sim[1][idx]) + 
                          (weights[2] * cos_sim[2][idx]) + 
                          (weights[3] * cos_sim[3][idx])).reshape(-1, 1)
    
    print(total_sim)
    total_sim = scaler.fit_transform(total_sim).flatten()  
    
    sim_scores = list(enumerate(total_sim))
    
    
    sim_scores = [(i, score) for i, score in sim_scores if i != idx]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]  
    
    movie_indices = [i[0] for i in sim_scores]

    return merged_df[[title_col, 'cast', 'genres', 'director']].iloc[movie_indices]




if uploaded_file is not None:
    merged_df = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully!")
    
    tfidf_list = calculate_tfidf_matrix(merged_df)
    cos_sim_list = calculate_cosine_similarity(tfidf_list)
    

    movie_title = st.text_input("Enter a movie title:")
    
    if st.button("Recommend"):
        # if "cos_sim.npy" in st.session_state:
        #     cos_sim = st.session_state["cos_sim"]
        # else:
        #     cos_sim = np.load("cos_sim.npy")
        #     st.session_state["cos_sim"] = cos_sim

        recommendations = recommend_movies(movie_title, merged_df, cos_sim_list)
        
        if recommendations.empty:
            st.write("No recommendations found!")
        else:
            st.write(recommendations)