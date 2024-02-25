import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path, label):
    # Load the CSV file, assuming header is in the first row (index 0)
    df = pd.read_csv(file_path)
    return df

def main():
    st.title('Mini Project 3 - Prediction by regression')
    # Specify the file paths (adjust these paths if your files are in a different directory)
    file_path = 'house-data.csv'
    
    # Load the data once and reuse it to improve efficiency
    data = load_data(file_path, 'House Data')
    
    st.write("Aggregated House Data:")
    # Display the aggregated DataFrame
    st.dataframe(data)

    st.write("Aggregated House Data Description:")
    # Display the description of the aggregated DataFrame but only for price and sqft_living
    st.dataframe(data[['price', 'sqft_living']].describe())

    # Group the house prices into $100,000 increments and count the number of houses in each bin
    price_bins = pd.cut(data['price'], bins=range(0, 1000001, 100000), labels=[f"${i}-{i+100}K" for i in range(0, 1000, 100)])
    price_distribution = price_bins.value_counts().sort_index()

    # Plotting
    fig, ax = plt.subplots()
    price_distribution.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of House Prices')
    ax.set_xlabel('Price Range')
    ax.set_ylabel('Number of Houses')

    # Display the plot with Streamlit
    st.pyplot(fig)

    # Group data by number of bedrooms and calculate the average price
    bedroom_price_distribution = data.groupby('bedrooms')['price'].mean()

    # Plotting
    fig, ax = plt.subplots()
    bedroom_price_distribution.plot(kind='line', ax=ax, linestyle='-')
    ax.set_title('Average House Prices by Number of Bedrooms')
    ax.set_xlabel('Number of Bedrooms')
    ax.set_ylabel('Average Price')

    # Display the plot with Streamlit
    st.pyplot(fig)



if __name__ == "__main__":
    main()
