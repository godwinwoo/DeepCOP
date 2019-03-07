import pandas as pd
import numpy as np

def trim_features(features_list, cutoff=0.95):
    features = np.asarray(features_list)
    df = pd.DataFrame(features)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
    trimmed_df = df.drop(df.columns[to_drop], axis=1)
    return trimmed_df.values