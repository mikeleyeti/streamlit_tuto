"""
Application Word2Vec - Page d'accueil
"""

import streamlit as st

st.set_page_config(page_title="Word2Vec App", page_icon="🔤", layout="wide")

st.title("🔤 Application Word2Vec")

st.markdown(
    """
## Bienvenue !

Cette application permet d'explorer les embeddings de mots grâce à un modèle Word2Vec.

### 📚 Fonctionnalités

👈 **Utilise la barre latérale** pour naviguer entre les pages :

- **Word2Vec** : Explore les similarités entre mots et réalise des analogies

### 🚀 Comment ça marche ?

Le modèle Word2Vec transforme les mots en vecteurs numériques qui capturent leur sens.
Les mots similaires sont proches dans l'espace vectoriel.

### 💡 Exemples d'analogies

- roi - homme + femme = reine
- Paris - France + Italie = Rome
- grand - petit + froid = chaud

---

**Commence par la page Word2Vec pour explorer le modèle !**
"""
)

st.info("💡 Astuce : Les pages sont accessibles dans le menu à gauche")
