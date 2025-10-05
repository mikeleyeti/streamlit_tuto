"""
calculs_word2vec.py
Module de calculs pour Word2Vec
Contient toutes les fonctions de chargement et de calcul
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
import os


def get_data_path(filename):
    """Retourne le chemin correct vers les fichiers de données"""
    # Si on est dans le dossier pages/, remonte au parent
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Vérifie si le fichier existe dans le répertoire courant
    if os.path.exists(os.path.join(current_dir, filename)):
        return os.path.join(current_dir, filename)

    # Sinon, cherche dans le dossier parent
    parent_dir = os.path.dirname(current_dir)
    return os.path.join(parent_dir, filename)


def load_dictionaries():
    """Charge les dictionnaires word2idx et idx2word"""
    word2idx_path = get_data_path("word2idx.pkl")
    idx2word_path = get_data_path("idx2word.pkl")

    with open(word2idx_path, "rb") as f:
        word2idx = pickle.load(f)
    with open(idx2word_path, "rb") as f:
        idx2word = pickle.load(f)
    return word2idx, idx2word


def load_model_and_vectors(vocab_size=10000, embedding_dim=300):
    """Charge le modèle et extrait les vecteurs d'embeddings"""
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=1))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(vocab_size, activation="softmax"))

    # Build the model so Keras knows the layer shapes
    model.build(input_shape=(None, 1))

    weights_path = get_data_path("word2vec.h5")
    model.load_weights(weights_path)

    vectors = model.layers[0].trainable_weights[0].numpy()
    return model, vectors


def dot_product(vec1, vec2):
    """Calcule le produit scalaire entre deux vecteurs"""
    return np.sum((vec1 * vec2))


def cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs"""
    return dot_product(vec1, vec2) / np.sqrt(
        dot_product(vec1, vec1) * dot_product(vec2, vec2)
    )


def find_closest(word_index, vectors, number_closest):
    """Trouve les mots les plus proches d'un mot donné"""
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
    """
    Effectue une analogie de mots : word1 - word2 + word3
    Exemple : roi - homme + femme = reine
    """
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


def get_closest_words(word, word2idx, idx2word, vectors, number=10):
    """
    Retourne les mots les plus proches d'un mot donné
    Retourne une liste de tuples (mot, score)
    """
    if word not in word2idx:
        raise ValueError(f"Le mot '{word}' n'est pas dans le vocabulaire")

    index_closest_words = find_closest(word2idx[word], vectors, number)
    results = []

    for index_word in index_closest_words:
        word_result = idx2word[index_word[1]]
        score = index_word[0]
        results.append((word_result, score))

    return results
