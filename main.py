import pandas as pd
import math


def train_data(df_data):
    # Test Data. Y = -1 : Y = +1
    # Creating data frames for: Y = 1 and Y = -1
    df_isPositive = df_data.loc[df_data['y'] == 1]
    df_isNegative = df_data.loc[df_data['y'] == -1]


def get_mean(df_data):
    numerator = 0
    for x in df_data:
        numerator += x
    return numerator / df_data.count()


def get_sigma(df_data):
    numerator = 0
    mean = get_mean(df_data)
    for x in df_data:
        numerator += (x - mean) ** 2
    return math.sqrt(numerator / df_data.count())


# Main program starts here
if __name__ == '__main__':
    # Import Files
    df_train_data = pd.read_csv("TrainData.csv")
    train_data(df_train_data)
