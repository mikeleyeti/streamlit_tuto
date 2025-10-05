"""
Page Word2Vec - Exploration des embeddings de mots
"""

import streamlit as st
from calculs_word2vec import (
    load_dictionaries,
    load_model_and_vectors,
    get_closest_words,
    compare,
)

st.set_page_config(page_title="Word2Vec", page_icon="📊", layout="wide")


# Chargement des données au démarrage
@st.cache_resource
def load_data():
    """Charge les données une seule fois (mise en cache)"""
    word2idx, idx2word = load_dictionaries()
    model, vectors = load_model_and_vectors()
    return word2idx, idx2word, vectors


def display_closest_words(word, number=10):
    """Affiche les mots les plus proches"""
    try:
        results = get_closest_words(word, word2idx, idx2word, vectors, number)
        for word_result, score in results:
            st.write(f"**{word_result}** -- {score:.4f}")
    except ValueError as e:
        st.error(str(e))


# Chargement des données
try:
    word2idx, idx2word, vectors = load_data()
    st.success("✅ Modèle chargé avec succès")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

# Interface
st.title("📊 Modèle Word2Vec")

# Onglets pour organiser le contenu
tab1, tab2, tab3 = st.tabs(["🔍 Recherche", "🎯 Analogies", "ℹ️ Informations"])

with tab1:
    st.header("Recherche de mots similaires")

    col1, col2 = st.columns([3, 1])

    with col1:
        mot = st.text_input(
            "Mot à analyser", "romantic", help="Entrez un mot pour trouver ses voisins"
        )

    with col2:
        nb_mots = st.slider(
            "Nombre de résultats", 1, 20, 5, help="Combien de mots similaires afficher"
        )

    if mot:
        st.subheader(f"Mots proches de '{mot}'")
        display_closest_words(mot, number=nb_mots)

with tab2:
    st.header("Analogies de mots")
    st.caption("Formule : **Mot 1 - Mot 2 + Mot 3 = ?**")
    st.caption("Exemple : roi - homme + femme = reine")

    # Exemple prédéfini
    with st.expander("📝 Voir un exemple : zombie"):
        st.subheader("10 mots les plus proches de 'zombie'")
        display_closest_words("zombie", 10)

    st.divider()

    # Formulaire d'analogie
    st.subheader("Crée ta propre analogie")

    with st.form("form_3_mots"):
        col1, col2, col3 = st.columns(3)

        with col1:
            mot1 = st.text_input(
                "Mot 1", placeholder="roi", help="Premier mot de l'analogie"
            )

        with col2:
            mot2 = st.text_input("Mot 2", placeholder="homme", help="Mot à soustraire")

        with col3:
            mot3 = st.text_input("Mot 3", placeholder="femme", help="Mot à ajouter")

        submit = st.form_submit_button(
            "🔍 Calculer l'analogie", use_container_width=True
        )

    if submit:
        mot1 = mot1.strip().lower()
        mot2 = mot2.strip().lower()
        mot3 = mot3.strip().lower()

        if mot1 and mot2 and mot3:
            try:
                with st.spinner("Calcul en cours..."):
                    liste = compare(
                        word2idx[mot1], word2idx[mot2], word2idx[mot3], vectors, 5
                    )

                st.success(f"**Résultat : {mot1} - {mot2} + {mot3} ≈**")

                for i, item in enumerate(liste, 1):
                    st.write(f"{i}. **{idx2word[item[1]]}** (score: {item[0]:.4f})")

            except KeyError as e:
                st.error(f"❌ Mot non trouvé dans le vocabulaire : {e}")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
        else:
            st.warning("⚠️ Veuillez remplir les 3 champs")

with tab3:
    st.header("Informations sur le modèle")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Taille du vocabulaire", f"{len(word2idx):,}")
        st.metric("Dimension des embeddings", vectors.shape[1])

    with col2:
        st.metric("Taille de la matrice", f"{vectors.shape[0]} × {vectors.shape[1]}")

    st.divider()

    st.markdown(
        """
    ### 🧠 Comment fonctionne Word2Vec ?
    
    Word2Vec est un modèle qui transforme les mots en vecteurs numériques.
    Les mots avec des significations similaires ont des vecteurs proches.
    
    **Similarité cosinus** : mesure l'angle entre deux vecteurs (1 = identique, 0 = orthogonal)
    
    **Analogies** : en soustrayant et ajoutant des vecteurs, on peut capturer des relations sémantiques.
    """
    )
