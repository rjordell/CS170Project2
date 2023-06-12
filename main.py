import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import time
import os
from operator import itemgetter
import numpy as np
from collections import Counter


def accuracy():
    #part 2:
    #acc = leave_one_out_cross_validation()
    acc = random.random()
    return acc

def leave_one_out_cross_validation(data, current_features, k):
    pass

def feature_search_demo(data):
    input_file_path = os.path.join(os.getcwd(), input_file)
    file = np.genfromtxt(input_file_path)
    choice = int(input("Choose number of the algorithm you want to run. \n 1. Forward Selection \n 2. Backward Elimination \n"))
    start = time.time()
    if (choice == 1):
        print("Forward Selection")
        forward_select(file)
    elif (choice == 2):
        print("Backward Elimination")
        backward_elim(file)
    end = time.time()
    print("Time to finish: {}".format(end-start))
    
def forward_select(file):
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

def backward_elim(file):
    current_features = list(range(1,data+1))
    best_so_far_accuracy = int(accuracy()*100)
    flag = False
    print("Beginning search.\n")
    print("Feature set ", current_features, " has accuracy:", best_so_far_accuracy, "\n")
    while (1): 
        feature_to_remove = None
        for k in current_features[:]:
            current_features.remove(k)
            acc = int(accuracy()*100)
            print("\tRemoving feature ",k," leaves us with feature set ", current_features, " accuracy is: ", acc, sep='')
            current_features.append(k)
            if acc > best_so_far_accuracy:
                best_so_far_accuracy = acc
                feature_to_remove = k
                flag = True
        if flag:
            current_features.remove(feature_to_remove)
            flag = False
        else:
            break
        print("\nFeature set ", current_features, " was best, accuracy is:", best_so_far_accuracy, "\n")
    print("\nFinished Search! The best feature subset is:", current_features, "which has an accuracy of:", best_so_far_accuracy)


#cv = LeaveOneOut()
#model = LinearRegression()
#scores = cross_val_score
#need to find out how to read the file
#df = pd.read_csv("small-test-dataset.txt", )

feature_search_demo(string(input("Please type in the name of the file to test: ")))