# make_dataset.py
import pandas as pd
import os

class DatasetLoader:
    """
    Class to load and preprocess the Multisim dataset.
    """
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        self.path = path
        self.df = None

    def load(self) -> pd.DataFrame:
        """
        Load dataset from a parquet file, convert columns to numeric,
        and filter invalid ages.
        """
        self.df = pd.read_parquet(self.path)
        self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
        self.df['age_dev'] = pd.to_numeric(self.df['age_dev'], errors='coerce')
        self.df = self.df[self.df['age'] < 150]
        return self.df

# Helper function for easy import
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset using DatasetLoader class. Keeps the old function interface.
    """
    loader = DatasetLoader(path)
    return loader.load()

if __name__ == "__main__":
    df = load_dataset("data/multisim_dataset.parquet")
    print(df.head())
