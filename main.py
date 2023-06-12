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

def leave_one_out_cross_validation(file):
    acc = random.random()
    return acc
    pass

def feature_search_demo():
    input_file = input("Please type in the name of the file to test: ")
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
    numOfFeats = file.shape[1]
    current_features = []
    best_feats = []
    bestGlobalAcc = 0

    print("Beginning search.\n")
    for i in range(numOfFeats):
        bestLocalAcc = 0
        feature_to_add = []
        for k in range(1,numOfFeats):
            if k not in current_features:
                #acc = accuracy()
                testFeats = [0] + current_features + [k]
                acc = leave_one_out_cross_validation(file[:, testFeats])
                print("\tUsing feature(s) ",current_features," accuracy is: ",acc, sep='')
                 
                if acc > bestLocalAcc:
                    bestLocalAcc = acc
                    feature_to_add = k
        
        if(feature_to_add):
            current_features.append(feature_to_add)
        
            if bestLocalAcc > bestGlobalAcc:
                bestGlobalAcc = bestLocalAcc
                best_feats[:] = current_features
                print("\nFeature set ", current_features, " was best, accuracy is:", bestLocalAcc,"\n")
                
            else:
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
                print("\nFeature set ", current_features, " was best, accuracy is:", bestLocalAcc,"\n")

    print("Finished Search! The best feature subset is:", best_feats, "which has an accuracy of:",bestGlobalAcc)

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

feature_search_demo()