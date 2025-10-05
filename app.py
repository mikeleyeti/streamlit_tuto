"""
Application Word2Vec - Page d'accueil
"""

import streamlit as st

st.set_page_config(page_title="Word2Vec App", page_icon="🔤", layout="wide")

st.title("🏖️BAC A SABLE🏖️")

st.markdown(
    """
## Bienvenue !

Cette application regroupe différents petits projets de Data-Visualisation et Data-Science.

### 📚 Projets

👈 **Utilise la barre latérale** pour naviguer entre les pages :

- **Word2Vec** : Explore les similarités entre mots et réalise des analogies à partir d'un DataSet 'cinema'
- **Titanic** : Explore les données du mythique DataSet du Titanic

"""
)

st.info("💡 Astuce : Les pages sont accessibles dans le menu à gauche")
