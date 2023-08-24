import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns


def random_forest_decision(data):
    X = data.drop("label", axis=1)
    print(X.columns)
    print(X)
    y = data["label"]  # Target variable (walking or running)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(y_pred, y_test)


    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", model.score(X_test, y_test))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


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
        mean_std_values['NUMBER OF ACTUAL STEPS'] = steps

        if mean_std_values.isnull().values.any():
            continue

        # concat all the data frames
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
    return data


def print_scatter_plot(x,y,data):
    sns.set()
    sns.scatterplot(x=x, y=y, hue="label", data=data)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(x + " vs. " + y)
    plt.show()


def main():
    folder_path = r'C:\Users\oriba\PycharmProjects\IOTprojectPartA\data'
    data = build_data(folder_path)
    # print_scatter_plot("norm mean", "norm std", data)
    random_forest_decision(data)


if __name__ == "__main__":
    main()
