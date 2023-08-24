import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


def xgboost_regression_decision(df):
    X = df.drop('COUNT OF ACTUAL STEPS', axis=1)
    y = df['COUNT OF ACTUAL STEPS'] # Target variable (number of steps)
    X.rename(columns={"Time [sec]": "Time"}, inplace=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate the root mean squared error (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("xgboost regression RMSE:", rmse)
    error = (y_test - y_pred)

    # Plot the error
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(error)), error, color='b', s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Data Point')
    plt.ylabel('Error')
    plt.title('Deviation of Predicted Number of Steps from Actual')
    plt.show()


def random_forest_regression_decision(df):
    X = df.drop('COUNT OF ACTUAL STEPS', axis=1)
    y = df['COUNT OF ACTUAL STEPS']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 5, 10],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['sqrt']
    # }
    #
    model = RandomForestRegressor()
    #
    # # Perform cross-validation with GridSearchCV
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    # grid_search.fit(X_train, y_train)
    #
    # # Get the best parameters and the best score
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    #
    # # Print the best parameters and the best score
    # print("Best Parameters:", best_params)
    # print("Best Score:", best_score)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    # Calculate the root mean squared error (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("random forrest regression RMSE:", rmse)

    error = (y_test - y_pred)

    # Plot the error
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(error)), error, color='b', s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Data Point')
    plt.ylabel('Error')
    plt.title('Deviation of Predicted Number of Steps from Actual')
    plt.show()


def build_data(folder_path):
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    # files not by the format
    bad_files = ["6_run_3_1.csv", "6_run_4_1.csv", "6_walk_5_1.csv", "11_walk_1_1.csv", "11_walk_2_1.csv", "11_walk_3_1.csv", "11_walk_5_1.csv"]
    first_df = True
    for file in file_list:

        # read the files properly to data frame
        if file not in bad_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, skiprows=5)
        else:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, skiprows=6)
        df.dropna(inplace=True)

        # fixing wrong column names
        if 'ACC Z ' in df.columns:
            df = df.rename(columns={'ACC Z ': 'ACC Z'})
        if ' ACC X' in df.columns:
            df = df.rename(columns={' ACC X': 'ACC X'})
        if 'Time[sec]' in df.columns:
            df = df.rename(columns={'Time[sec]': 'Time [sec]'})

        # adding the activity column
        df["label"] = file.split("_")[1] + "ing"
        df_label = pd.read_csv(file_path, skiprows=3, nrows=1, header=None)
        steps = df_label[1][0]
        df["norm"] = ["" for i in range(len(df))]
        df = df.reset_index(drop=True)

        # adding the norm column and drop not float values
        for row in range(len(df)):
            x, y, z = df["ACC X"][row], df["ACC Y"][row], df["ACC Z"][row]
            if not isinstance(x, float) or not isinstance(y, float) or not isinstance(z, float):
                df.drop(row, inplace=True)
            else:
                df.loc[row, "norm"] = float(np.linalg.norm([x, y, z]))

        # aggregate the data
        df["norm"] = pd.to_numeric(df["norm"], errors='coerce')
        mean_values = df.mean(numeric_only=True).to_frame().T
        mean_values.rename(columns={"ACC X": "ACC X mean", "ACC Y": "ACC Y mean", "ACC Z": "ACC Z mean", "norm": "norm mean"}, inplace=True)
        std_values = df.std(numeric_only=True).to_frame().T
        if 'Time [sec]' in std_values.columns:
            std_values.drop(columns=['Time [sec]'], inplace=True)
        std_values.rename(columns={"ACC X": "ACC X std", "ACC Y": "ACC Y std", "ACC Z": "ACC Z std", "norm": "norm std"}, inplace=True)
        mean_std_values = pd.concat([mean_values, std_values], axis=1)
        mean_std_values["Time [sec]"] = df["Time [sec]"].max()
        mean_std_values["label"] = df["label"].max()
        mean_std_values["COUNT OF ACTUAL STEPS"] = steps
        mean_std_values["Max norm"] = df["norm"].max()
        mean_std_values["Number of samples"] = len(df)

        if mean_std_values.isnull().values.any():
            continue

        # concat the data frames(every data frame is a line)
        if first_df:
            data = mean_std_values.copy()
            first_df = False
        else:
            mean_std_values = mean_std_values.reset_index(drop=True)
            data = data.reset_index(drop=True)
            data = pd.concat([data, mean_std_values], axis=0, ignore_index=True)
            data = data.reset_index(drop=True)

    # drop extreme values
    for row in range(len(data)):
        if data["norm mean"][row] > 30:
            data.drop(row, inplace=True)

    # encode the labels
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data


def print_scatter_plot_with_trend_line(df, x, y):
    sns.set_theme(style="darkgrid")
    sns.regplot(x=x, y=y, data=df, scatter_kws={"color": "black"}, line_kws={"color": "red"})
    print("correlation is:", df[x].corr(df[y]))
    print("pearson correlation is:", pearsonr(df[x], df[y]))
    plt.show()


def correlation_matrix(data):
    # Separate the features (X) and the target variable (y)
    X = data.drop('COUNT OF ACTUAL STEPS', axis=1)  # Features
    y = data['COUNT OF ACTUAL STEPS']  # Target variable

    # Compute the correlation matrix
    correlation_matrix = X.corrwith(y)

    # Reshape the correlation matrix
    correlation_matrix = correlation_matrix.values.reshape(-1, 1)

    # Get the feature names
    feature_names = X.columns.tolist()

    # Create a DataFrame with feature names and correlation values
    correlation_df = pd.DataFrame(correlation_matrix, index=feature_names, columns=['Correlation'])

    # Plot the correlation matrix as a heatmap
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', cbar=False)
    plt.title('Correlation Matrix')
    plt.show()


def main():
    folder_path = r'C:\Users\oriba\PycharmProjects\IOTprojectPartA\data'
    data = build_data(folder_path)
    random_forest_regression_decision(data)
    # xgboost_regression_decision(data)
    # print_scatter_plot_with_trend_line(data, "norm std", "COUNT OF ACTUAL STEPS")
    # print_scatter_plot_with_trend_line(data, "Number of samples", "COUNT OF ACTUAL STEPS")



if __name__ == "__main__":
    main()