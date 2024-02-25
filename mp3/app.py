import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

def load_data(file_path, label):
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
    st.title('Mini Project 3 - Prediction by regression')
    file_path = 'house-data.csv'
    data = load_data(file_path, 'House Data')

    # Display the data
    st.write(data)

    # Display descriptive statistics for price and sqft_living
    st.write(data[['price', 'sqft_living']].describe())

    # Display box plot for price and sqft_living
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.boxplot(data['price'], ax=ax[0])
    ax[0].set_title('Price Boxplot')
    sns.boxplot(data['sqft_living'], ax=ax[1])
    ax[1].set_title('Sqft Living Boxplot')
    st.pyplot(fig)

    # Display bar char disribution of price
    fig, ax = plt.subplots()
    sns.histplot(data['price'], kde=True, ax=ax)
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Price Distribution')
    st.pyplot(fig)

    # Display bar char disribution of sqft_living
    fig, ax = plt.subplots()
    sns.histplot(data['sqft_living'], kde=True, ax=ax)
    ax.set_xlabel('Sqft Living')
    ax.set_ylabel('Frequency')
    ax.set_title('Sqft Living Distribution')
    st.pyplot(fig)

    # Assuming 'price' is your target variable, exclude it from PCA
    features = data.drop(columns=['price'])
    pca, data_scaled = apply_pca(features)
    
    # Correct the plot to dynamically match the number of PCA components
    n_components = pca.n_components_
    fig, ax = plt.subplots()
    ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance Ratio')
    st.pyplot(fig)
    
    # Update this part of your app to include PCA results visualization
    # And further analysis or processing as needed
    
    # Selecting components based on explained variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    n_important_components = (cumulative_variance < 0.95).sum() + 1  # for example, retain 95% of the variance
    
    # Transform data to n_important_components dimensions
    pca_reduced = PCA(n_components=n_important_components)
    data_reduced = pca_reduced.fit_transform(data_scaled)
    
    # Continue with your regression analysis using data_reduced
  

if __name__ == "__main__":
    main()
