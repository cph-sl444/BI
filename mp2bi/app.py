import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(file_path, label):
    # Load the Excel file, setting the second row (index 1) as the header
    df = pd.read_excel(file_path, header=1)
    # Add a new column for wine type
    df['Type'] = label
    return df

def main():
    st.title('Wine Data Aggregator')

    # Specify the file paths (adjust these paths if your files are in a different directory)
    red_wine_file_path = 'red_wine.xlsx'
    white_wine_file_path = 'white_wine.xlsx'

    # Load and label the data
    red_wine_data = load_data(red_wine_file_path, 'Red')
    white_wine_data = load_data(white_wine_file_path, 'White')

    # Combine the dataframes
    combined_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

    # Display the aggregated DataFrame
    st.write("Aggregated Wine Data:")
    st.dataframe(combined_data)

    # Binning the pH attribute into 5 bins
    combined_data['pH_bin'] = pd.cut(combined_data['pH'], bins=5, labels=False)

    # Display the binned data
    st.write("Binned Data by pH:")
    st.dataframe(combined_data[['pH', 'pH_bin']].head())  # Show only head to avoid clutter

    # Group by the pH bins and count the number of entries in each bin
    bin_counts = combined_data['pH_bin'].value_counts().sort_index()

    # Find the bin with the highest density
    highest_density_bin = bin_counts.idxmax()
    highest_density_count = bin_counts.max()

    # Display the bin with the highest density
    st.write(f"The pH bin with the highest density is: {highest_density_bin} with {highest_density_count} entries.")

    # Optionally, show the range of pH values in the highest density bin
    pH_bin_ranges = combined_data.groupby('pH_bin')['pH'].agg(['min', 'max'])
    st.write(f"The pH range for this bin is: {pH_bin_ranges.loc[highest_density_bin]}")
   # Filter the DataFrame to only include numeric columns
    numeric_data = combined_data.select_dtypes(include=[np.number])

    # Compute the correlation matrix on just the numeric columns
    correlation_matrix = numeric_data.corr()

    # Create a heatmap of the correlation matrix
    st.write("Heatmap of the Correlation Matrix")
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

    # Identify the feature with the biggest influence on quality
    quality_correlations = correlation_matrix['quality'].drop('quality')  # Exclude self-correlation
    most_influential_feature = quality_correlations.abs().idxmax()
    st.write(f"The attribute with the biggest influence on wine quality is: {most_influential_feature} "
             f"with a correlation of {quality_correlations[most_influential_feature]:.2f}")
    # Identificer attributten med den laveste korrelation til kvalitet
    least_influential_feature = quality_correlations.abs().idxmin()

    # Fjern denne attribut fra det kombinerede datasæt
    updated_data = combined_data.drop(columns=[least_influential_feature])

    # Vis det opdaterede datasæt
    st.write(f"The attribute with the lowest influence on wine quality is: {least_influential_feature}. It has been removed from the dataset.")
    st.write("Updated Dataset after removing the least influential attribute:")
    st.dataframe(updated_data)

    # Map 'Type' column to numerical values: 'White' to 1 and 'Red' to 2
    updated_data['Type'] = updated_data['Type'].map({'White': 1, 'Red': 2})

    # Vis besked over tabellen
    st.write("Categorical 'Type' column transformed to numeric: 'White' as 1 and 'Red' as 2")

    # Vis det opdaterede datasæt
    st.write("Updated Dataset with 'Type' as Numeric Values:")
    st.dataframe(updated_data)

    # Vælg kun numeriske kolonner (sikkerhedsforanstaltning)
    numeric_data = updated_data.select_dtypes(include=[np.number])

    # Standardiserer datasættet
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Anvender PCA
    pca = PCA()
    pca.fit(scaled_data)

    # Beregner den kumulative forklarede varians ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Bestemmer det optimale antal komponenter (f.eks. for at forklare mindst 95% af variansen)
    optimal_num_components = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1
    
    # Vis det optimale antal komponenter i Streamlit
    st.markdown("### Principal Component Analysis (PCA)")
    st.write(f"Optimal number of components to retain 95% of the variance: {optimal_num_components}")

    # Valgfrit: Vis den kumulative forklarede varians ratio i Streamlit
    cumulative_variance_ratio_output = ""
    for i, ratio in enumerate(cumulative_variance_ratio):
        cumulative_variance_ratio_output += f"Component {i+1}: Cumulative explained variance: {ratio:.2f}\n"
    st.text(cumulative_variance_ratio_output)

    # Vælg ti tilfældige rækker fra det endelige datasæt
    random_rows = updated_data.sample(n=10)

    # Vis disse rækker i Streamlit
    st.write("Ten random rows from the final dataset:")
    st.dataframe(random_rows)









if __name__ == "__main__":
    main()