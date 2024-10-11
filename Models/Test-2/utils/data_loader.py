import pandas as pd

def load_data(file_path):
    """
    Load a TSV file and return it as a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the TSV file.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Successfully loaded {file_path}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

sentiment_df = load_data('Datasets/MVSA-MTS/mvsa-mts/sentiment.tsv')