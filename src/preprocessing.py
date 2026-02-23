import pandas as pd

def preprocess(df):
    df = df.drop(["Unnamed: 0", "flight"], axis=1, errors='ignore')

    stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
    class_mapping = {'Economy': 0, 'Business': 1}

    df['stops'] = df['stops'].map(stops_mapping)
    df['class'] = df['class'].map(class_mapping)

    one_hot_columns = [
        'airline', 
        'source_city', 
        'destination_city', 
        'departure_time', 
        'arrival_time'
    ]

    return pd.get_dummies(df, columns=one_hot_columns, drop_first=True)