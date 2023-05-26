import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
import joblib
from wordcloud import WordCloud

def main():
    def add_bg_from_url():
        st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.repstatic.it/content/nazionale/img/2022/11/02/175951890-afcfbe17-d41a-496e-a006-873ec202dc5a.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
        )
    
    st.write("<h1 style='color: grey;'>Classificazione multi-classe</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose your file CSV:")

    add_bg_from_url()

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        
        word = st.text_input("Inserisci la parola: ")
        
        if word == "":
            st.warning("Input text is required.")
        else:
            pipe = joblib.load('true_fake.pkl')
            
            fake = [word]
            predictions = pipe.predict(fake)
            
            if predictions[0] == "fake":
                st.write(
                    f'<div style="background-color: lightgreen; padding: 5px;">'
                    f'Sentiment: {predictions[0].upper()}'
                    '</div>',
                    unsafe_allow_html=True
                )
            else :
                st.write(
                    f'<div style="background-color: lightgrey; padding: 5px;">'
                    f'Sentiment: {predictions[0].upper()}'
                    '</div>',
                    unsafe_allow_html=True
                )
            
            
            text = df['text'].to_string(index=False)
            
            wordcloud = WordCloud(width=800, height=400, background_color='grey').generate(text)
            plt.figure(figsize=(18, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            
            st.image(wordcloud.to_array(), use_column_width=True)
            
            st.download_button(
            label="Download WordCloud",
            data=wordcloud.to_image().tobytes(),
            file_name="wordcloud.jpeg",
            mime="image/jpeg"
                )

if __name__ == "__main__":
    main()