import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocessed_data(file_path):
    # load the data
    data = pd.read_csv(file_path)
    # print null values
    print(f"null values: {data.isnull().sum()}")
    # handeling outliers using iqr method
    def remove_outliers(data, column):
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        IQR = q3-q1
        lower_bound = q1 - 1.5* IQR
        upper_bound = q3 _ 1.5 * IQR
        data = data[(data[column]>= lower_bound) & (data[column]<= upper_bound)]
        return data

    numeric_columns =  ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'b', 'lstat', 'medv']
    for col in numeric_columns:
        data = remove_outliers(data, col)

    # there is no need to encode categorical value beacause each and every values are numerical.

    # seperate the features and target
    X = data.drop('medv', axis = 1)
    y = data['medv')
    # standardrize the feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform()
    X_scaled = pd.Dataframe(X_scaled,columns = X.columns)
    # split data into train test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test, scaler = preprocess_data('boston_housing.csv')
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    