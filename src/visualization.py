import matplotlib.pyplot as plt
import seaborn as sns
import math

def distributions(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    ncols = 3
    nrows = math.ceil(len(numeric_cols)/ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()