import math
import pandas as pd

CORRECT = 0
INCORRECT = 0

def to_the_moon(df_data, df_test):
    # to_the_moon will train data first saving all values into dicts.
    # it will then use that train data to classify and calculate the accuracy 
    # on the TestData.csv file.

    # Teset Data: Y = 1 : Y = -1
    df_isPositive = df_data.loc[df_data['y'] == 1]
    df_isNegative = df_data.loc[df_data['y'] == -1]
    
    # Features split into continues and discrete
    continues_features = ["x1", "x3", "x5", "x11", "x12", "x13"]
    discrete_features = ["x2", "x4", "x6", "x7", "x8", "x9" ,"x10", "14"]

    # Dictionary for mean and sigma
    continues_mean_pos = {}
    continues_mean_neg = {}
    continues_sigma_pos = {}
    continues_sigma_neg = {}
    discrete_pos = {}
    discrete_neg = {}

    # Getting the mean and sigma for continues features when y = 1 and y = -1
    for i in continues_features:
        if (i == "x1"):
            continues_mean_pos[i] = get_mean(df_isPositive.x1)
            continues_mean_neg[i] = get_mean(df_isNegative.x1)
            continues_sigma_pos[i] = get_sigma(df_isPositive.x1)
            continues_sigma_neg[i] = get_sigma(df_isNegative.x1)
        elif (i == "x3"):
            continues_mean_pos[i] = get_mean(df_isPositive.x3)
            continues_mean_neg[i] = get_mean(df_isNegative.x3)
            continues_sigma_pos[i] = get_sigma(df_isPositive.x3)
            continues_sigma_neg[i] = get_sigma(df_isNegative.x3)
        elif (i == "x5"):
            continues_mean_pos[i] = get_mean(df_isPositive.x5)
            continues_mean_neg[i] = get_mean(df_isNegative.x5)
            continues_sigma_pos[i] = get_sigma(df_isPositive.x5)
            continues_sigma_neg[i] = get_sigma(df_isNegative.x5)
        elif (i == "x11"):
            continues_mean_pos[i] = get_mean(df_isPositive.x11)
            continues_mean_neg[i] = get_mean(df_isNegative.x11)
            continues_sigma_pos[i] = get_sigma(df_isPositive.x11)
            continues_sigma_neg[i] = get_sigma(df_isNegative.x11)
        elif (i == "x12"):
            continues_mean_pos[i] = get_mean(df_isPositive.x12)
            continues_mean_neg[i] = get_mean(df_isNegative.x12)
            continues_sigma_pos[i] = get_sigma(df_isPositive.x12)
            continues_sigma_neg[i] = get_sigma(df_isNegative.x12)
        elif (i == "x13"):
            continues_mean_pos[i] = get_mean(df_isPositive.x13)
            continues_mean_neg[i] = get_mean(df_isNegative.x13)
            continues_sigma_pos[i] = get_sigma(df_isPositive.x13)
            continues_sigma_neg[i] = get_sigma(df_isNegative.x13)

    # Getting the mean and sigma for discrete features when y = 1 and y = -1
    for i in discrete_features:
        if (i == "x2"):
            discrete_pos[i] = get_discrete(df_isPositive.x2)
            discrete_neg[i] = get_discrete(df_isNegative.x2)
        elif (i == "x4"):
            discrete_pos[i] = get_discrete(df_isPositive.x4)
            discrete_neg[i] = get_discrete(df_isNegative.x4)
        elif (i == "x6"):
            discrete_pos[i] = get_discrete(df_isPositive.x6)
            discrete_neg[i] = get_discrete(df_isNegative.x6)
        elif (i == "x7"):
            discrete_pos[i] = get_discrete(df_isPositive.x7)
            discrete_neg[i] = get_discrete(df_isNegative.x7)
        elif (i == "x8"):
            discrete_pos[i] = get_discrete(df_isPositive.x8)
            discrete_neg[i] = get_discrete(df_isNegative.x8)
        elif (i == "x9"):
            discrete_pos[i] = get_discrete(df_isPositive.x9)
            discrete_neg[i] = get_discrete(df_isNegative.x9)
        elif (i == "x10"):
            discrete_pos[i] = get_discrete(df_isPositive.x10)
            discrete_neg[i] = get_discrete(df_isNegative.x10)
        elif (i == "x14"):
            discrete_pos[i] = get_discrete(df_isPositive.x14)
            discrete_neg[i] = get_discrete(df_isNegative.x14)
    
    
    # Y = 1 ; Continues Features ; Print
    print("-----------------------------------------------------\n")
    print("Y = 1 ; Continues")
    print("-----------------------------------------------------")
    for key,value in continues_mean_pos.items():
        print("Mean: " + str(key) + "\t" + str(value))
    for key,value in continues_sigma_pos.items():
        print("Sigma: " + str(key) + "\t" + str(value))

    # Y = -1 ; Continues Features ; Print
    print("-----------------------------------------------------\n")
    print("Y = -1 ; Continues")
    print("-----------------------------------------------------")
    for key,value in continues_mean_neg.items():
        print("Mean: " + str(key) + "\t" + str(value))
    for key,value in continues_sigma_neg.items():
        print("Sigma: " + str(key) + "\t" + str(value))

    # Y = 1 ; Discrete Features ; Print
    print("-------------------------------\n")
    print("Y = 1 ; Discrete")
    print("-------------------------------")
    for key,value in discrete_pos.items():
        print("Feature: " + str(key) + "\n" + str(value) + "\n**************")

    # Y = -1 ; Discrete Features ; Print
    print("-------------------------------\n")
    print("Y = -1 ; Discrete")
    print("-------------------------------")
    for key,value in discrete_neg.items():
        print("Feature: " + str(key) + "\n" + str(value) + "\n**************")

    print("-----------------------------------------------------\n")
    for x in continues_features:
        if (x == "x1"):
            classify_data(x, continues_mean_pos[x], continues_mean_neg[x], continues_sigma_pos[x], continues_sigma_neg[x], df_test)
        elif (x == "x3"):
            classify_data(x, continues_mean_pos[x], continues_mean_neg[x], continues_sigma_pos[x], continues_sigma_neg[x], df_test)
        elif (x == "x5"):
            classify_data(x, continues_mean_pos[x], continues_mean_neg[x], continues_sigma_pos[x], continues_sigma_neg[x], df_test)
        elif (x == "x11"):
            classify_data(x, continues_mean_pos[x], continues_mean_neg[x], continues_sigma_pos[x], continues_sigma_neg[x], df_test)
        elif (x == "x12"):
            classify_data(x, continues_mean_pos[x], continues_mean_neg[x], continues_sigma_pos[x], continues_sigma_neg[x], df_test)
        elif (x == "x13"):
            classify_data(x, continues_mean_pos[x], continues_mean_neg[x], continues_sigma_pos[x], continues_sigma_neg[x], df_test)
    print("Accuracy: %" + str((round(CORRECT / len(df_test), 2)) * 100))


def get_mean(df_data):
    # Giving a dataframe find and return the mean
    numerator = 0
    for x in df_data:
        numerator += x
    return numerator / df_data.count()


def get_sigma(df_data):
    # Given a dataframe find and return sigma
    numerator = 0
    mean = get_mean(df_data)
    for x in df_data:
        numerator += (x - mean) ** 2
    return math.sqrt(numerator / df_data.count())


def get_discrete(df_data):
    counts = df_data.value_counts(normalize=True)
    return counts


def classify_data(feature, mean_pos, mean_neg, sigma_pos, sigma_neg, df_data):
    # Function used to classify data from the test data.
    # Increments CORRECT and INCORRECT global variables to keep track of
    #   of accuracy on the class labels
    global CORRECT
    global INCORRECT
    for x in df_data[feature].unique():
        if (naive_bayes(sigma_pos, mean_pos, x) > naive_bayes(sigma_neg, mean_neg, x)):
            CORRECT += 1
        else:
            INCORRECT += 1


def naive_bayes(sigma_pos, mean_pos, x):
    # Naive bayes function 
    exponent = math.exp(-(math.pow(x - mean_pos, 2) / (2 * math.pow(sigma_pos, 2))))
    return (1 / (math.sqrt(2 * math.pi) * sigma_pos)) * exponent


# Main Method
if __name__ == '__main__':
    # Import files
    df_test_data = pd.read_csv("TestData.csv")
    df_train_data = pd.read_csv("TrainData.csv")
    to_the_moon(df_train_data, df_test_data)
