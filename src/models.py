import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


def model_scores(y_true,y_pred):
    mae = mean_absolute_error(y_true,y_pred)
    rmse = np.sqrt(mean_absolute_error(y_true,y_pred))
    r2 = r2_score(y_true,y_pred)
    return mae,rmse,r2


def evaluate_models(X_train,X_test,y_train,y_test):

    models={
        "Linear Regression":LinearRegression(),
        "Lasso":Lasso(),
        "KNN":KNeighborsRegressor(),
        "Decision Tree":DecisionTreeRegressor(),
        "Random Forest":RandomForestRegressor(),
        "AdaBoost":AdaBoostRegressor(),
        "GradientBoost":GradientBoostingRegressor(),
        "XGBoost":XGBRegressor()
    }

    results=[]

    for name,model in models.items():
        model.fit(X_train,y_train)

        train_pred=model.predict(X_train)
        test_pred=model.predict(X_test)

        _,_,train_r2=model_scores(y_train,train_pred)
        _,_,test_r2=model_scores(y_test,test_pred)

        results.append({"Model":name,"Train R2":train_r2,"Test R2":test_r2})

    return sorted(results,key=lambda x:x["Test R2"],reverse=True)


def tune_random_forest(X_train,y_train):
    params={
        "n_estimators":[100,300],
        "max_depth":[10,20,None],
        "min_samples_split":[2,10],
        "max_features":["sqrt"]
    }

    grid=GridSearchCV(
        RandomForestRegressor(random_state=15),
        params,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train,y_train)
    return grid.best_estimator_