import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Lecture des données
df = pd.read_csv("train.csv")

st.title("🚢 Projet de classification binaire Titanic")

# Création des onglets
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Exploration", "📈 DataVizualization", "🤖 Modélisation", "🧪 Bac à sable"]
)

# ===== ONGLET 1: EXPLORATION =====
with tab1:
    st.header("Exploration des données")

    st.write("### Introduction")
    st.dataframe(df.head(10))

    st.write("**Dimensions du dataset:**")
    st.write(df.shape)

    st.write("### Statistiques descriptives")
    st.dataframe(df.describe())

    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())

# ===== ONGLET 2: DATAVIZUALIZATION =====
with tab2:
    st.header("DataVizualization")

    col1, col2 = st.columns(2)

    with col1:
        # Graphique de survie
        fig = plt.figure()
        sns.countplot(x="Survived", data=df)
        plt.title("Répartition de la survie")
        st.pyplot(fig)

        # Graphique du genre
        fig = plt.figure()
        sns.countplot(x="Sex", data=df)
        plt.title("Répartition du genre des passagers")
        st.pyplot(fig)

    with col2:
        # Graphique des classes
        fig = plt.figure()
        sns.countplot(x="Pclass", data=df)
        plt.title("Répartition des classes des passagers")
        st.pyplot(fig)

        # Distribution de l'âge
        fig = sns.displot(x="Age", data=df)
        plt.title("Distribution de l'âge des passagers")
        st.pyplot(fig)

    st.write("### Analyses croisées")

    col3, col4 = st.columns(2)

    with col3:
        # Survie par genre
        fig = plt.figure()
        sns.countplot(x="Survived", hue="Sex", data=df)
        plt.title("Survie par genre")
        st.pyplot(fig)

        # Survie par classe
        fig = sns.catplot(x="Pclass", y="Survived", data=df, kind="point")
        plt.title("Taux de survie par classe")
        st.pyplot(fig)

    with col4:
        # Survie par âge et classe
        fig = sns.lmplot(x="Age", y="Survived", hue="Pclass", data=df)
        plt.title("Survie en fonction de l'âge et de la classe")
        st.pyplot(fig)

        # Heatmap de corrélation
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            df.select_dtypes(include=[np.number]).corr(),
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
        )
        plt.title("Matrice de corrélation")
        st.pyplot(fig)

# ===== ONGLET 3: MODÉLISATION =====
with tab3:
    st.header("Modélisation")

    # Préparation des données
    df_model = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    y = df_model["Survived"]
    X_cat = df_model[["Pclass", "Sex", "Embarked"]].copy()
    X_num = df_model[["Age", "Fare", "SibSp", "Parch"]].copy()

    # Imputation des valeurs manquantes
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])

    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    # Encodage des variables catégorielles
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis=1)

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Normalisation
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    # Fonction de prédiction
    def prediction(classifier):
        if classifier == "Random Forest":
            clf = RandomForestClassifier()
        elif classifier == "SVC":
            clf = SVC()
        elif classifier == "Logistic Regression":
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf

    # Fonction de scores
    def scores(clf, choice):
        if choice == "Accuracy":
            return clf.score(X_test, y_test)
        elif choice == "Confusion matrix":
            return confusion_matrix(y_test, clf.predict(X_test))

    # Interface utilisateur
    st.write("### Choix du modèle")
    choix = ["Random Forest", "SVC", "Logistic Regression"]
    option = st.selectbox("Choix du modèle", choix)
    st.write("Le modèle choisi est :", option)

    # Entraînement du modèle
    clf = prediction(option)

    # Affichage des résultats
    st.write("### Résultats")
    display = st.radio("Que souhaitez-vous montrer ?", ("Accuracy", "Confusion matrix"))

    if display == "Accuracy":
        accuracy = scores(clf, display)
        st.metric("Accuracy", f"{accuracy:.4f}")
    elif display == "Confusion matrix":
        st.write("**Matrice de confusion:**")
        cm = scores(clf, display)
        st.dataframe(
            pd.DataFrame(
                cm,
                columns=["Prédit Non-Survivant", "Prédit Survivant"],
                index=["Réel Non-Survivant", "Réel Survivant"],
            )
        )

# ===== ONGLET 4: BAC À SABLE =====
with tab4:
    st.header("Bac à sable")

    st.write("### Génération de valeurs aléatoires")

    import random

    @st.cache_data
    def generate_random_value(x):
        return random.uniform(0, x)

    a = generate_random_value(10)
    b = generate_random_value(20)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Valeur aléatoire a (0-10)", f"{a:.2f}")

    with col2:
        st.metric("Valeur aléatoire b (0-20)", f"{b:.2f}")

    st.info(
        "💡 Ces valeurs sont mises en cache et ne changeront pas tant que vous ne relancez pas l'application."
    )
