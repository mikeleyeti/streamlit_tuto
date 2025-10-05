import streamlit as st
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer

# Chargement du tokenizer
with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

# Chargement du modèle
vocab_size = 10000
embedding_dim = 300

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=1))
model.add(GlobalAveragePooling1D())
model.add(Dense(vocab_size, activation="softmax"))

# 🔧 Build the model so Keras knows the layer shapes
model.build(input_shape=(None, 1))
model.load_weights("word2vec.h5")
vectors = model.layers[0].trainable_weights[0].numpy()


def dot_product(vec1, vec2):
    return np.sum((vec1 * vec2))


def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2) / np.sqrt(
        dot_product(vec1, vec1) * dot_product(vec2, vec2)
    )


def find_closest(word_index, vectors, number_closest):
    if word_index >= vectors.shape[0]:
        raise ValueError(
            f"Index {word_index} exceeds vector matrix size {vectors.shape[0]}"
        )

    list1 = []
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])
    return np.asarray(sorted(list1, reverse=True)[:number_closest])


def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1 = []
    vec1 = vectors[index_word1]
    vec2 = vectors[index_word2]
    vec3 = vectors[index_word3]
    query_vector = vec1 - vec2 + vec3
    normalizer = Normalizer()
    query_vector = normalizer.fit_transform([query_vector], "l2")
    query_vector = query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])
    return np.asarray(sorted(list1, reverse=True)[:number_closest])


def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words:
        st.write(idx2word[index_word[1]], " -- ", index_word[0])
    return ""


st.title("Modèle Word2Vec")
# Exemple d'utilisation de la fonction print_closest

st.subheader("10 mots les plus proches de Zombie")
st.write(print_closest("zombie"))


st.subheader("Essaye par toi même")
nb_mots = st.slider("Combien de mots proches", 0, 20, 5)
mot = st.text_input("Mot à tester", "romantic")
st.write(print_closest(mot, number=nb_mots))


st.subheader("Jouons avec les mots !")

with st.form("form_3_mots"):
    mot1 = st.text_input("Mot 1").strip().lower()
    mot2 = st.text_input("Mot 2").strip().lower()
    mot3 = st.text_input("Mot 3").strip().lower()
    liste_mots = [mot1, mot2, mot3]
    submit = st.form_submit_button("Envoyer")

if submit:
    liste = compare(
        word2idx[liste_mots[0]],
        word2idx[liste_mots[1]],
        word2idx[liste_mots[2]],
        vectors,
        5,
    )
    for item in liste:
        st.write(idx2word[item[1]])
