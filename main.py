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


def accuracy(predictedClasses, classData):
    #part 2:
    cnt = 0
    
    for i,j in zip(predictedClasses, classData):
        if i == j:
            cnt += 1
            
    acc = (cnt/len(classData)) * 100
    return acc

def leave_one_out_cross_validation(file):
    projection = []
    classes = [entry[0] for entry in file]
    
    for index in range(len(file)):
        dataEval = file[index]
        dataEval = dataEval[1:]
        trainingData = []
        
        if index == 0:
            trainingData = file[1:]
            
        elif index == (len(file)-1):
            trainingData = file[:index]
            
        else:
            trainingData = np.concatenate((file[:index],file[index+1:]), axis=0)
            
        trainingClass = [i[0] for i in trainingData]
        trainingData = np.delete(trainingData, 0, axis=1)
        projectedClass = nearest_neighbor_classifier(dataEval, trainingData, trainingClass, 2)
        projection.append(projectedClass)
    
    acc = accuracy(projection, classes)
    return acc

"Euclidean distance between two vectors function implementation"
def euclidean_distance(x, y, p):
    squared_distance = np.sum(np.power(np.abs(x - y), p))
    distance = np.power(squared_distance, 1/p)
    return distance

"K nearest neighbor function implementation (gets nn for one tuple)"
def k_nearest_neighbors(trainingData, instance, trainingClass, p):
    k = 1
    totalDistances = []
    count = 0
    
    for entry in trainingData:
        generatedDistance = euclidean_distance(entry, instance, p)
        totalDistances.append((generatedDistance, trainingClass[count]))
        count += 1
        
    totalDistances = sorted(totalDistances, key=itemgetter(0))
    k_neighbors = [val[1] for val in totalDistances[:k]]
    return k_neighbors

"Nearest neighbor classifier function implementation"
def nearest_neighbor_classifier(instance, trainingData, trainingClass, p):
    predictedClasses = []
    neighbors = k_nearest_neighbors(trainingData, instance, trainingClass, p)
    most_common_class = Counter(neighbors).most_common(1)[0][0]
    predictedClasses.append(most_common_class)
    return predictedClasses

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
    numOfFeats = file.shape[1]
    current_features = list(range(1,numOfFeats))
    best_feats = list(range(1,numOfFeats))
    bestGlobalAcc = 0

    print("Beginning search.\n")

    for i in range(numOfFeats):
        bestLocalAcc = 0
        feature_to_add = []
        for k in range(1,numOfFeats):
            if k not in current_features:
                testFeats = [n for n in current_features if n != k]
                acc = leave_one_out_cross_validation(file[:, testFeats])
                print("\tUsing feature(s) ",current_features," accuracy is: ",acc, sep='')

                if acc > bestLocalAcc:
                    bestLocalAcc = acc
                    feature_to_add = k
                 
        if(feature_to_add):
            current_features = [n for n in current_features if n != feature_to_add]
        
            if bestLocalAcc > bestGlobalAcc:
                bestGlobalAcc = bestLocalAcc
                best_feats[:] = current_features
                print("\nFeature set ", current_features, " was best, accuracy is:", bestLocalAcc,"\n")
                
            else:
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
                print("\nFeature set ", current_features, " was best, accuracy is:", bestLocalAcc,"\n") 

    print("Finished Search! The best feature subset is:", best_feats, "which has an accuracy of:",bestGlobalAcc)

feature_search_demo()