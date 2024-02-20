import streamlit as st
import pandas as pd

def load_data(file_path, label):
    # Load the Excel file, setting the second row (index 1) as the header
    df = pd.read_excel(file_path, header=1)
    return df