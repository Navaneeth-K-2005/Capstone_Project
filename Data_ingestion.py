import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
  
    df = pd.read_json(file_path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df['date'] = pd.to_datetime(df['date'])
    return df
