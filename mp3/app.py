import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
    
    # Make new dataframe that has the first 4 principal components
    st.write('First 4 principal components')
    pca_df = pd.DataFrame(pca.transform(data_scaled)[:, :4], columns=[f'PC{i}' for i in range(1, 5)])
    st.write(pca_df)

    st.divider()

    st.subheader('Linear Regression Model Training and Prediction using PCA Components 1-4 and Price as Target Variable')

    # Use the first 4 principal components to predict the price
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(pca_df, data['price'], test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display the first 5 predictions using the first 4 principal components
    st.write('First 5 predictions')
    st.write(np.round(y_pred[:5], 2))

    # Display the first 5 actual values
    st.write('First 5 actual values')
    st.write(np.round(y_test[:5], 2))
        
    # Display the R-squared score
    r2_score = model.score(X_test, y_test)
    st.write('R-squared score')
    st.write(round(r2_score, 2))

    # Display the coefficients
    st.write('Coefficients')
    st.write(np.round(model.coef_, 2))

    # visualize the actual vs predicted values using a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    st.pyplot(fig)

    st.divider()


    # Data preprocessing and model training
    X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'yr_built', 'yr_renovated', 'zipcode', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'waterfront', 'view']]
    y = data['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    input_data = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'yr_built', 'yr_renovated', 'zipcode', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'waterfront', 'view']]
    predicted_price = model.predict(input_data)

    # Display predicted price
    st.title('House Price Predictor')
    st.write('The predicted price for the house is: $', round(predicted_price[0], 2))

if __name__ == "__main__":
    main()
