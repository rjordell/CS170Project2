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
    choice = int(input("Choose number of the algorithm you want to run. \n 1. Forward Selection \n 2. Backward Elimination \n"))
    if (choice == 1):
        print("Forward Selection")
        forward_select(data)
    elif (choice == 2):
        print("Backward Elimination")
        backward_elim(data)
    
def forward_select(data):
    current_features = []
    best_so_far_accuracy = 0
    flag = False
    print("Beginning search.\n")
    for i in range(data):
        feature_to_add = []
        for k in range(1,data+1):
            if k not in current_features:
                #acc = accuracy()
                current_features.append(k)
                acc = int(accuracy()*100)
                print("\tUsing feature(s) ",current_features," accuracy is: ",acc, sep='')
                current_features.pop(-1)  
                if acc > best_so_far_accuracy:
                    best_so_far_accuracy = acc
                    feature_to_add = k
                    flag = True
        if(flag):
            current_features.append(feature_to_add)
        flag = False
        print("\nFeature set ", current_features, " was best, accuracy is:", best_so_far_accuracy,"\n")
    print("Finished Search! The best feature subset is:", current_features, "which has an accuracy of:",best_so_far_accuracy)

def backward_elim(data):
    current_features = []
    best_features = []
    for i in range(data):
        print("On the", i+1,"th level of the search tree")
        feature_to_add = []
        best_so_far_accuracy = 0

        for k in range(1,data+1):
            if k not in current_features:
                #acc = accuracy()
                acc = int(accuracy()*100)
                print("--Considering adding the", k, "feature, accuracy is:", acc)
                if acc > best_so_far_accuracy:
                    best_so_far_accuracy = acc
                    feature_to_add = k
            
        
        current_features.append(feature_to_add)
        print("On level", i+1, "i added feature", feature_to_add, "to current set ", current_features, ", accuracy is:", best_so_far_accuracy)


#cv = LeaveOneOut()
#model = LinearRegression()
#scores = cross_val_score
#need to find out how to read the file
#df = pd.read_csv("small-test-dataset.txt", )

feature_search_demo(int(input("enter total number of features: ")))