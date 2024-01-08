import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff

# Fonction pour appliquer KMeans
def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    return labels

# Fonction pour appliquer DBScan
def apply_dbscan(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels

# Fonction pour appliquer la méthode de réduction de dimensionnalité
def apply_dimensionality_reduction(data, method, n_components=2):
    if method == "PCA":
        reducer = PCA(n_components=n_components)
        st.success("Vous avez choisi PCA")
    elif method == "Truncated SVD":
        reducer = TruncatedSVD(n_components=n_components)
        st.success("Vous avez choisi Truncated SVD")
    elif method == "Factor Analysis":
        reducer = FactorAnalysis(n_components=n_components)
        st.success(f"Vous avez choisi Factor Analysis")
    else:
        raise ValueError("Méthode de réduction de dimensionnalité non supportée.")

    st.session_state.reduced_data = reducer.fit_transform(data)
    reduced_df = pd.DataFrame(data=st.session_state.reduced_data, columns=[f"Component {i+1}" for i in range(n_components)])

    return reduced_df
# Page d'accueil
def home():
    st.title("Clustering App")
    st.write("""
    Welcome to the Clustering App!

    This application allows you to explore and visualize different clustering models using the Iris dataset.
    
    - Navigate to the Base de données section to choose a dimensionality reduction method and visualize the graph.
    - Move to the Modèle section to choose a clustering model and visualize the clustering graph.
    - Explore the "Evaluation" section to see the performance metrics of the selected clustering model.

    Enjoy exploring the fascinating world of clustering!
    """)

def createdata():
    iris = load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    return data

# Sélection du jeu de données
def dataset_selection():
    data = createdata()
    if "reduced_data" not in st.session_state:
        st.session_state.reduced_data = None
    # Sélection de la méthode de réduction de dimensionnalité
    reduction_method = st.selectbox("Méthode de Réduction de Dimensionnalité", ["PCA", "Truncated SVD", "Factor Analysis"])
    
    # Appliquer la méthode de réduction de dimensionnalité
    st.session_state.reduced_data = apply_dimensionality_reduction(data.iloc[:, :-1], method=reduction_method)  # Exclure la colonne de la cible pour la PCA
    st.markdown("<strong>Graphique de Réduction de Dimensionnalité :</strong>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(st.session_state.reduced_data['Component 1'], st.session_state.reduced_data['Component 2'], c=data['target'], cmap='viridis')
    ax.set_xlabel('Composante Principale 1 ')
    ax.set_ylabel('Composante Principale 2 ')
    st.pyplot(fig)  # Passer la figure à st.pyplot() comme argument

    # Marquer l'étape comme complétée dans la session
    st.session_state.step_data_selection = True

# Modèle de clustering
def clustering_model(data):
    st.title("Modèle de Clustering")
    st.write("Choisissez le modèle de clustering à appliquer:")

    # Vérifier si l'étape précédente a été complétée
    if "step_data_selection" not in st.session_state or not st.session_state.step_data_selection:
        st.warning("Veuillez parcourir la section Base de données avant d'accéder à cette section.")
        return

    model_option = st.selectbox("Choisissez le modèle", ["KMeans", "DBScan"])

    if model_option == "KMeans":
        n_clusters = st.slider("Nombre de clusters (KMeans)", min_value=2, max_value=10, value=3)
        labels = apply_kmeans(data, n_clusters)
    elif model_option == "DBScan":
        eps = st.slider("EPS ", min_value=0.1, max_value=2.0, value=0.5)
        min_samples = st.slider("Nombre minimal d'échantillons ", min_value=1, max_value=10, value=5)
        labels = apply_dbscan(data, eps, min_samples)

    # Stocker les labels dans la session
    st.session_state.labels = labels

    # Tracer le graphique
    fig, ax = plt.subplots()
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')

    st.pyplot(fig)

    # Marquer l'étape comme complétée dans la session
    st.session_state.step_model_selection = True

# Fonction pour évaluer les performances du clustering

# Fonction pour évaluer les performances du clustering avec Plotly
def evaluate_clustering(data):
    st.title("Évaluation du Clustering")
    
    # Vérifier si les étapes précédentes ont été complétées
    if "step_data_selection" not in st.session_state or not st.session_state.step_data_selection:
        st.warning("Veuillez parcourir la section Base de données avant d'accéder à cette section.")
        return
    elif "step_model_selection" not in st.session_state or not st.session_state.step_model_selection:
        st.warning("Veuillez parcourir la section Modèle avant d'accéder à cette section.")
        return

    # Récupérer les labels depuis la session
    labels = st.session_state.labels

    if labels is not None:
        # Calculer la matrice de confusion
        cm = confusion_matrix(data['target'], labels)

        # Afficher la matrice de confusion avec Plotly
        fig = ff.create_annotated_heatmap(cm, colorscale="Viridis", x=["Cluster " + str(i+1) for i in range(len(cm))], y=["Classe " + str(i+1) for i in range(len(cm))])
        fig.update_layout(title_text='Matrice de Confusion', xaxis_title='Clusters Prédits', yaxis_title='Classes Réelles')
        st.plotly_chart(fig)

        # Afficher le rapport de classification
        st.subheader("Rapport de Classification")
        report = classification_report(data['target'], labels)
        st.text(report)
    else:
        st.warning("Aucun modèle de clustering n'a été appliqué. Veuillez choisir un modèle dans la section 'Modèle'.")
# Fonction principale
def main():
    with st.sidebar:
        selected = option_menu("Main Menu", ["Accueil", "Base de données", "Modèle", 'Evaluation'],
                               icons=['house', 'database', 'gear', 'cast'], menu_icon="cast", default_index=0)

    if selected == "Accueil":
        home()
    elif selected == "Base de données":
        dataset_selection()
    elif selected == "Modèle":
        dataPCA = st.session_state.reduced_data
        if dataPCA is not None:
            clustering_model(dataPCA)
        else:
            st.warning("Chargez d'abord un jeu de données.")
    elif selected == "Evaluation":
        data = createdata()
        evaluate_clustering(data)

# Exécutez l'application
if __name__ == "__main__":
    main()