"""
Interface Streamlit pour le modèle Word2Vec
"""

import streamlit as st
from Word2Vec import (
    load_dictionaries,
    load_model_and_vectors,
    get_closest_words,
    compare,
)


# Chargement des données au démarrage
@st.cache_resource
def load_data():
    """Charge les données une seule fois (mise en cache)"""
    word2idx, idx2word = load_dictionaries()
    model, vectors = load_model_and_vectors()
    return word2idx, idx2word, vectors


# Chargement
word2idx, idx2word, vectors = load_data()


def display_closest_words(word, number=10):
    """Affiche les mots les plus proches"""
    try:
        results = get_closest_words(word, word2idx, idx2word, vectors, number)
        for word_result, score in results:
            st.write(f"{word_result} -- {score:.4f}")
    except ValueError as e:
        st.error(str(e))


# Interface Streamlit
st.title("Modèle Word2Vec")

# Section 1 : Exemple fixe
st.subheader("10 mots les plus proches de Zombie")
display_closest_words("zombie")

# Section 2 : Recherche personnalisée
st.subheader("Essaye par toi même")
nb_mots = st.slider("Combien de mots proches", 0, 20, 5)
mot = st.text_input("Mot à tester", "romantic")

if mot:
    display_closest_words(mot, number=nb_mots)

# Section 3 : Analogies de mots
st.subheader("Jouons avec les mots !")
st.caption("Exemple : roi - homme + femme = ?")

with st.form("form_3_mots"):
    mot1 = st.text_input("Mot 1", placeholder="roi")
    mot2 = st.text_input("Mot 2", placeholder="homme")
    mot3 = st.text_input("Mot 3", placeholder="femme")
    submit = st.form_submit_button("Envoyer")

if submit:
    mot1 = mot1.strip().lower()
    mot2 = mot2.strip().lower()
    mot3 = mot3.strip().lower()

    if mot1 and mot2 and mot3:
        try:
            liste = compare(word2idx[mot1], word2idx[mot2], word2idx[mot3], vectors, 5)
            st.write(f"**{mot1} - {mot2} + {mot3} ≈**")
            for item in liste:
                st.write(f"• {idx2word[item[1]]} ({item[0]:.4f})")
        except KeyError as e:
            st.error(f"Mot non trouvé dans le vocabulaire : {e}")
    else:
        st.warning("Veuillez remplir les 3 champs")
