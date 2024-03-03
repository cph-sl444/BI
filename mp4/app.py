import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist



## 1. Data wrangling and exploration
def load_data(file_path):
    # Load the CSV file, assuming header is in the first row (index 0)
    df = pd.read_csv(file_path)
    return df

def apply_pca(data, n_components=None):
    # Ensure data is numeric
    numeric_features = data.select_dtypes(include=[np.number])

    # Adjust n_components based on the numeric features only
    if n_components is None or n_components > min(numeric_features.shape):
        n_components = min(numeric_features.shape[0], numeric_features.shape[1]) - 1 
    # Standardize the numeric data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_features)
        
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)    
    return pca, data_scaled

def main(): 
    st.title('Mini Project 4 -  Attrition Prediction')
    file_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
    data = load_data(file_path)

    # Display the data
    st.subheader('Employee Data')
    st.write(data)

    # Data cleaning - Check for missing values
    st.subheader('Missing Values')
    st.write(data.isnull().sum())

    # Discriptive statistics
    st.subheader('Descriptive Statistics')
    st.write(data.describe())

    # Display distribution of Attrition
    fig, ax = plt.subplots()  
    attrition_counts = data['Attrition'].value_counts()
    ax.bar(attrition_counts.index, attrition_counts.values, color=['skyblue', 'salmon'])
    ax.set_title('Distribution of Attrition')
    ax.set_xlabel('Attrition')
    ax.set_ylabel('Count')
    st.pyplot(fig)  

    # Assuming 'attrition' is your target variable, exclude it from PCA
    features = data.drop(columns=['Attrition'])
    pca, data_scaled = apply_pca(features)
    st.divider()
    st.subheader('PCA')

     # Correct the plot to dynamically match the number of PCA components
    n_components = pca.n_components_
    fig, ax = plt.subplots()
    ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance Ratio')
    st.pyplot(fig)

    
# 2. Supervised machine learning: classificasion 
    st.subheader('Test, Train and Validate')
    relevant_features = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'DistanceFromHome', 'YearsAtCompany', 'MonthlyIncome', 'JobLevel']
    X = data[relevant_features]
    y = data['Attrition']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Train Naïve Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # Predictions
    dt_predictions = dt_classifier.predict(X_test)
    nb_predictions = nb_classifier.predict(X_test)

    # Model Evaluation
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    nb_accuracy = accuracy_score(y_test, nb_predictions)

    # Display model results
    st.subheader('Model Results')
    st.write("Decision Tree Accuracy:", dt_accuracy)
    st.write("Naïve Bayes Accuracy:", nb_accuracy)

     # Display decision tree graph
    st.subheader('Decision Tree Graph')
    fig, ax = plt.subplots()
    plot_tree(dt_classifier, filled=True, feature_names=X.columns.tolist(), class_names=['No Attrition', 'Attrition'])
    st.pyplot(fig)
    st.subheader('Decision Tree Classification Report')
    st.write(classification_report(y_test, dt_predictions))

     # Visualize feature distributions for each class
    st.subheader('Feature Distributions for Naïve Bayes Classifier')
    for feature in relevant_features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data, x=feature, hue='Attrition', kde=True, palette='viridis')
        plt.title(f'Distribution of {feature} by Attrition')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        st.set_option('deprecation.showPyplotGlobalUse', False) #Quickffix to display the plot
        st.pyplot()  
    st.subheader('Naïve Bayes Classification Report')
    st.write(classification_report(y_test, nb_predictions))

    # Recommend the model with the highest accuracy
    if dt_accuracy > nb_accuracy:
        st.write("Decision Tree has the highest accuracy.")
    else:
        st.write("Naïve Bayes has the highest accuracy.")


# 3. Unsupervised machine learning: clustering
    st.subheader('Clustering')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define number  of clusters - range from 2 to 10
    cluster_range = range(2, 10) 

    # Evaluate silhouette scores for different cluster configurations- Elbow method
    silhouette_scores = []
    for n_clusters in cluster_range:
        model = KMeans(n_clusters=n_clusters, n_init=10)
        cluster_labels = model.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
      
    # Display the "Elbow" 
    fig, ax = plt.subplots()
    ax.plot(cluster_range, silhouette_scores, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs. Number of Clusters')
    st.pyplot(fig)
    
    # Find the cluster configuration with the highest silhouette score
    best_n_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
    st.write(f"Recommended number of clusters: {best_n_clusters}")

    # Visualize the clusters - find better way to visualize?? 
    st.subheader('Visualization of Clusters')
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Visualization of Clusters')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)
   

    # plot clusters using boundraies and steps and cluster centers 
    

if __name__ == "__main__":
    main()
