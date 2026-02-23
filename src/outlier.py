import pandas as pd

def find_outliers_iqr(df, threshold=1.5):
    info = {}
    numeric_cols = df.select_dtypes(include=["float64","int64"]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3-Q1

        lower = Q1 - threshold*IQR
        upper = Q3 + threshold*IQR

        outliers = df[(df[col]<lower)|(df[col]>upper)]

        info[col] = {
            "count": len(outliers),
            "percentage": len(outliers)/len(df)*100
        }

    return pd.DataFrame(info).T


def remove_outliers(df, threshold=1.5):
    numeric_cols = df.select_dtypes(include=["float64","int64"]).columns
    mask = pd.Series(True,index=df.index)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3-Q1

        lower = Q1 - threshold*IQR
        upper = Q3 + threshold*IQR

        mask &= (df[col]>=lower)&(df[col]<=upper)

    return df[mask]