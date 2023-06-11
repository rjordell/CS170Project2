import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt
import pandas as pd


def accuracy():
    #part 2:
    #acc = leave_one_out_cross_validation()
    acc = random.random()
    return acc

def leave_one_out_cross_validation(data, current_features, k):
    pass

def feature_search_demo(data):
    current_features = []

    for i in range(data):
        print("On the", i+1,"th level of the search tree")
        feature_to_add = []
        best_so_far_accuracy = 0

        for k in range(data):
            if k not in current_features:
                print("--Considering adding the", k+1, "feature")
                acc = accuracy()

                if acc > best_so_far_accuracy:
                    best_so_far_accuracy = acc
                    feature_to_add = k
        
        current_features.append(feature_to_add)
        print("On level", i+1, "i added feature", feature_to_add, "to current set")
    


cv = LeaveOneOut()
model = LinearRegression()
scores = cross_val_score
#need to find out how to read the file
#df = pd.read_csv("small-test-dataset.txt", )

feature_search_demo(int(input("enter total number of features: ")))