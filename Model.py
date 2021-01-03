#https://github.com/12190143/Black-Swan
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def drop(df):
    try:
        df = df.drop(["Unnamed: 0"], axis=1)
        df = df.drop(["Unnamed: 0.1"], axis=1)
    except:
        print("")

    return df

def mape_error(y_true, y_pred):
    sum = 0
    # print(len(y_true))
    y_true = np.nan_to_num(y_true)
    # temp = np.abs(y_true - y_pred) / y_true

    for i in range(len(y_true)):

        if y_true[i] == 0:
            continue
        else:
            sum = sum + np.abs(y_true[i] - y_pred[i]) / y_true[i]

    mape_error = sum / len(y_true)
    # print(mape_error)

    # Whether score_func is a score function (default), meaning high is good, or a loss function,
    # meaning low is good. In the latter case, the scorer object will sign-flip the outcome of the score_func.


    return mape_error

def main():


    # data load
    df_train = pd.read_csv("train.csv")
    df_date = pd.read_csv("execption_date.csv")
    df_ts = pd.read_csv("timeseries.csv")

    len_train = len(df_train)
    # df_train = df_train.append(df_test)[df_train.columns.tolist()]

    df_train = df_train.merge(df_date, on="date", how="left")
    df_train = df_train.merge(df_ts, on=["tollgate_id", "hour", "miniute", "direction"], how="left")
    df_train = drop(df_train)

    # for training _ split training dataset into test and train
    X = df_train.iloc[:len_train - 1, 7:]
    y = df_train.iloc[:len_train - 1]["volume"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    ######################random forest parameter setting ###############################

    #set MAPE as score method
    MAPE = make_scorer(mape_error,greater_is_better = False)

    #setting grid search CV parameter

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=250, num=10)]
    max_features = ['sqrt']
    max_depth = [int(x) for x in np.linspace(50, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [5, 10]
    min_samples_leaf = [4, 8]
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor(random_state=42)
    print("Start to Train RandomForestRegressor..")
    grid_rf = GridSearchCV(rf, param_grid=random_grid, scoring=MAPE, n_jobs=-1, return_train_score=True)
    grid_rf.fit(X_train, y_train)

    y_pred = grid_rf.predict(X_train)
    print('Grid RandomForestRegressor MAPE on training Data: ', mape_error(y_train, y_pred))
    y_pred = grid_rf.predict(X_test)
    print('Grid RandomForestRegressor MAPE on test Data: ', mape_error(y_test, y_pred))
    print('Grid RandomForestRegressor best parameter: ', grid_rf.best_params_)
    print('Grid RandomForestRegressor best score: ', - grid_rf.best_score_)
    
    #Whether score_func is a score function (default), meaning high is good, or a loss function, 
    #meaning low is good. In the latter case, the scorer object will sign-flip the outcome of the score_func.
    

    ######################random forest parameter setting ###############################

    MAPE = make_scorer(mape_error, greater_is_better=False)


    model = GradientBoostingRegressor(random_state=42)

    #parameter setting
    parameters = {'learning_rate': [0.01, 0.04, 0.05, 0.07],
                  'subsample': [0.9, 0.5, 0.2],
                  'n_estimators': [100, 500, 1000],
                  'max_depth': [4, 6, 8]
                  }
    print("Start to Train GradientBoostingRegressor.. ")
    grid_GBR = GridSearchCV(estimator=model, param_grid=parameters, cv=6, n_jobs=-1, scoring=MAPE)
    grid_GBR.fit(X_train, y_train)

    y_pred = grid_GBR.predict(X_train)
    print('Grid GradientBoostingRegressor MAPE on training Data: ', mape_error(y_train, y_pred))

    y_pred = grid_GBR.predict(X_test)
    print('Grid GradientBoostingRegressor MAPE on test Data: ', mape_error(y_test, y_pred))
    print('Grid GradientBoostingRegressor best parameter: ', grid_GBR.best_params_)

    # Whether score_func is a score function (default), meaning high is good, or a loss function,
    # meaning low is good. In the latter case, the scorer object will sign-flip the outcome of the score_func.
    print('Grid GradientBoostingRegressor best score: ', - grid_GBR.best_score_)

    #get grid search results
    result_df = pd.DataFrame(grid_GBR.cv_results_)
    #colums_name = ["split" + str(i) + "_test_score" for i in range(6)]
    #colums_name.append("mean_test_score")
    #result_df[colums_name] = result_df[colums_name].apply(lambda x: abs(x))
    #print(result_df)


    #analysing predic results wiht plotting
    scatter = pd.DataFrame(list(zip(y_test, y_pred)), columns=["y_true", "y_pre"]) #X_test Prediction results
    scatter["outlier"] = scatter["y_true"] - scatter["y_pre"]
    scatter["outlier"] = scatter["outlier"].apply(lambda x: abs(x) > 110)
    sns.scatterplot(x='y_true', y='y_pre', data=scatter, hue="outlier")

    sub = df_train.iloc[y_test.index, :].sort_values(by=["time_window"])


    ################################sumission.csv################################

    #load dataset for predict submission
    df_test = pd.read_csv("test.csv")
    df_test = df_test.merge(df_date, on="date", how="left")
    df_test = df_test.merge(df_ts, on=["tollgate_id", "hour", "miniute", "direction"], how="left")
    df_test = drop(df_test)

    # for submistion
    X = df_test.iloc[:, 7:]
    x = np.nan_to_num(X)

    #predict
    # need to delete
    #grid_GBR = GradientBoostingRegressor(random_state=42)
    #grid_GBR.fit(X_test,y_test)


    #from here
    y_pred_submission = grid_GBR.predict(x)

    df_test = df_test.iloc[:, :4]
    df_test["volume"] = y_pred_submission

    #get same time window as sample submission
    sample_submission = pd.read_csv("../submission_sample/submission_sample_volume.csv")
    sample_submission = sample_submission.drop(["volume"], axis=1)

    submission = df_test.merge(sample_submission, on=["tollgate_id", "time_window", "direction"], how="right")

    submission.to_csv("submission.csv", index=False)
    print("./code/submission.csv")


if __name__ == '__main__':
    print("Start Training..")
    main()
    print("Finish Training..")





