# APRO/app/etl_tools/cleaning.py

import pandas as pd
import numpy as np
import re
import sqlite3
import json
import chardet
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

### ——— Data Loading / “Extractor” Functions ———

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def load_csv(file_path, chunksize=None):
    encoding = detect_encoding(file_path)
    return pd.read_csv(file_path, encoding=encoding, chunksize=chunksize)

def load_excel(file_path, sheet_name=0):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.json_normalize(data)

def load_parquet(file_path):
    return pd.read_parquet(file_path)

def load_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def load_api_response(json_response):
    return pd.json_normalize(json_response)

def infer_schema(df):
    return df.dtypes.to_dict()


### ——— Missing‐Value Handling ———

def drop_nulls(df, axis=0):
    return df.dropna(axis=axis)

def impute_missing(df, method='mean'):
    if method == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif method == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif method == 'mode':
        return df.fillna(df.mode().iloc[0])
    return df

def fill_missing(df, method='ffill'):
    return df.fillna(method=method)


### ——— Duplicate Removal ———

def remove_duplicates(df):
    return df.drop_duplicates()


### ——— Column Operations ———

def rename_columns(df, rename_map):
    return df.rename(columns=rename_map)

def drop_columns(df, columns):
    return df.drop(columns=columns)

def split_column(df, column, sep, into):
    splits = df[column].str.split(sep, expand=True)
    splits.columns = into
    return df.drop(columns=[column]).join(splits)

def combine_columns(df, columns, new_column, sep=' '):
    df[new_column] = df[columns].astype(str).agg(sep.join, axis=1)
    return df.drop(columns=columns)


### ——— String & Format Cleanup ———

def trim_whitespace(df):
    return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

def normalize_case(df, case='lower'):
    if case == 'lower':
        return df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    elif case == 'upper':
        return df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
    return df

def regex_transform(df, column, pattern, replace):
    df[column] = df[column].str.replace(pattern, replace, regex=True)
    return df


### ——— Date & Type Handling ———

def parse_dates(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def convert_dtypes(df, conversions):
    for col, dtype in conversions.items():
        df[col] = df[col].astype(dtype)
    return df


### ——— Indexing ———

def set_index(df, column):
    return df.set_index(column)

def reset_index(df):
    return df.reset_index(drop=True)


### ——— Row & Column Selection ———

def filter_rows(df, condition):
    return df.query(condition)

def select_columns(df, columns):
    return df[columns]


### ——— Sorting ———

def sort_by(df, columns, ascending=True):
    return df.sort_values(by=columns, ascending=ascending)


### ——— Joins & Concatenation ———

def join_tables(df1, df2, on, how='inner'):
    return df1.merge(df2, on=on, how=how)

def concatenate(df_list, axis=0):
    return pd.concat(df_list, axis=axis)


### ——— Reshaping ———

def pivot_table(df, index, columns, values, aggfunc='sum'):
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)

def melt_table(df, id_vars, value_vars):
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars)


### ——— Grouping & Aggregation ———

def group_and_aggregate(df, group_by, agg_dict):
    return df.groupby(group_by).agg(agg_dict).reset_index()


### ——— Custom Transforms ———

def apply_custom(df, column, func):
    df[column] = df[column].apply(func)
    return df


### ——— Categorical Encoding ———

def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns)

def label_encode(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df


### ——— Scaling & Normalization ———

def scale_columns(df, columns, method='minmax'):
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


### ——— Outlier Detection & Treatment ———

def remove_outliers_zscore(df, columns, threshold=3):
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[z_scores < threshold]
    return df

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df


### ——— Feature Engineering ———

def derive_date_parts(df, column):
    df[f"{column}_year"] = df[column].dt.year
    df[f"{column}_month"] = df[column].dt.month
    df[f"{column}_day"] = df[column].dt.day
    return df

def create_ratio(df, num_col, denom_col, new_col):
    df[new_col] = df[num_col] / df[denom_col]
    return df


### ——— Feature Selection & Reduction ———

def drop_low_variance(df, threshold=0.0):
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)
    numeric_df = df.select_dtypes(include=[np.number])
    reduced = selector.fit_transform(numeric_df)
    return pd.DataFrame(reduced, columns=numeric_df.columns[selector.get_support()])

def drop_highly_correlated(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def reduce_dimensionality(df, columns, n_components=2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df[columns])
    return pd.DataFrame(reduced, columns=[f'PC{i+1}' for i in range(n_components)])

def sample_data(df, frac=0.1, stratify_col=None):
    if stratify_col:
        return df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(frac=frac)
        )
    return df.sample(frac=frac)