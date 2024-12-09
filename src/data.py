import pickle as pk

def load_downsample_data(dataset_name, n=1000, random_seed=42):
    """
    Load raw dataset and perform downsampling.
    """
    with open(f'data/{dataset_name}.pkl', 'rb') as f:
        df = pk.load(f)
    return df.sample(n=n, random_state=random_seed)


def load_processed_data(dataset_name):
    """
    Load dataset already processed
    """
    with open(f'data/{dataset_name}_processed.pkl', 'rb') as f:
        df = pk.load(f)
    return df


def save_data(df, saved_name):
    with open(f'data/{saved_name}.pkl', 'wb') as f:
        pk.dump(df, f, protocol=4)