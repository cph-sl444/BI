import streamlit as st
import pandas as pd

def load_data(file_path, label):
    # Load the Excel file, setting the second row (index 1) as the header
    df = pd.read_csv(file_path, header=0)
    return df

def main():
    st.title('Mini Project 3 - Prediction by regression')
    # Specify the file paths (adjust these paths if your files are in a different directory)
    file_path = 'house-data.csv'
    load_data(file_path, 'House Data')
    st.write("Aggregated House Data:")
    # display the aggregated DataFrame
    st.dataframe(load_data(file_path, 'House Data'))
    

if __name__ == "__main__":
    main()