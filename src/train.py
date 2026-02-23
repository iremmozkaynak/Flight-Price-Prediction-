from data_loader import load_data
from preprocessing import preprocess
from outlier import find_outliers_iqr, remove_outliers
from models import evaluate_models, tune_random_forest
from sklearn.model_selection import train_test_split

def main():
    df=load_data("data/flight_dataset.csv")

    print(find_outliers_iqr(df))

    df=preprocess(df)
    df=remove_outliers(df)

    X=df.drop("price",axis=1)
    y=df["price"]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=15)

    results=evaluate_models(X_train,X_test,y_train,y_test)
    print(results)

    best_model=tune_random_forest(X_train,y_train)
    print("Best model trained")

if __name__=="__main__":
    main()