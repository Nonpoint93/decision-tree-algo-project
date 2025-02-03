import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

TARGET = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"

def read_data() -> pd.DataFrame:
    try:
        return pd.read_csv(TARGET, delimiter=',')
    except Exception as e:
        print("Error reading the file")
        print(e)
        return None

def save_data(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print("Error writing the file")
        print(e)


def main():

    # Read the data from dataset to dataframe
    df = read_data()
    if df is not None:
        print(df.head())
        print(df.tail())
        print(df.info())
        print(df.describe())

    if df.duplicated().sum() > 0:
        print("There are duplicated rows\n")
        df = df.drop_duplicates()
    else:
        print("There are no duplicated rows\n")

    if df.isnull().sum().sum() > 0:
        print("There are missing values\n")
        df = df.dropna(axis=1)
    else:
        print("There are no missing values\n")

    outliers = []
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        num_outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))).sum()
        if num_outliers > 0:
            outliers.append(column)
            print(f"Column {column} has {num_outliers} outliers")

    if len(outliers) != 0:
        _, axi = plt.subplots(len(outliers), 1, figsize=(10, 10))
        for i, column in enumerate(outliers):
            sns.boxplot(df[column], ax=axi[i])
            plt.title(f"Boxplot de {column}")
        plt.show()
    else:    
        print("There are no outliers\n")
    
    conditions = [
        (df["Glucose"] >= 40) & (df["Glucose"] <= 300),
        (df["BloodPressure"] >= 50) & (df["BloodPressure"] <= 180),
        (df["Insulin"] <= 300),
        (df["BMI"] <= 60),
        (df["Age"] <= 100)
    ]

    df = df[np.logical_and.reduce(conditions)]
    df["DiabetesPedigreeFunction"] = np.log1p(df["DiabetesPedigreeFunction"])
    
    print("After cleaning the data\n")
    print(df.describe())

    save_data(df=df, path="./data/interim/clean_diabetes_data.csv")

    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    total_data = X
    total_data["Outcome"] = y
    pd.plotting.parallel_coordinates(total_data, "Outcome", color=('#556270', '#4ECDC4'))
    plt.title("Parallel Coordinates Plot")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.head())

    save_data(df=X_train, path="./data/processed/X_train.csv")
    save_data(df=X_test, path="./data/processed/X_test.csv")
    save_data(df=y_train, path="./data/processed/y_train.csv")
    save_data(df=y_test, path="./data/processed/y_test.csv")

if __name__ == '__main__':
    main()