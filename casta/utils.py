import pandas as pd
from importlib import resources
from pathlib import Path

def load_example_data():
    
    with resources.files('casta.data').joinpath('example_track.csv').open('r') as f:
            df = pd.read_csv(f)

    with resources.path('casta.data', '') as path:
            path = str(path)

    return df, path