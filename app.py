"""
Application Word2Vec - Page d'accueil
"""

import streamlit as st

st.set_page_config(page_title="Portfolio Data", page_icon="💼", layout="wide")

st.title("📊 Portfolio Data")

st.markdown(
    """
## Bienvenue !

Cette application regroupe différents petits projets de Data-Visualisation et Data-Science.

### 📚 Projets

👈 **Utilise la barre latérale** pour naviguer entre les pages :

- **Word2Vec** : Explore les similarités entre mots et réalise des analogies à partir d'un DataSet 'cinema'
- **Titanic** : Explore les données du mythique DataSet du Titanic
- **Movies** : Explore les données de [The Movie Database (TMDB)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

"""
)

st.info("💡 Astuce : Les pages sont accessibles dans le menu à gauche")
