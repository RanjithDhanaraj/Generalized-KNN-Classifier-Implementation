# Import packages
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import numpy.random as npr
from scipy.spatial.distance import pdist, squareform

def kNN(X,y,k):
    # Find the number of obsvervation
    n = shape(X)[0]
    y.reset_index(inplace=True,drop=True)
    # Set up return values
    y_star = []
    # Calculate the distance matrix for the observations in X
    dist = []
    dist = squareform(pdist(X, metric = 'euclidean'))
    # Make all the diagonals very large so it can't choose itself as a closest neighbour
    np.fill_diagonal(dist, 1000)
    # Loop through each observation to create predictions
    for i in range(n):
        distances = []
        label = []
        for j in range(n):    
            distances.append((dist[i][j]))
        # Find the y values of the k nearest neighbours
        for j in range(k):
            index = distances.index(sorted(distances)[j])
            label.append(y[index])
        y_star.append(round(mean(label, 0)))
    return y_star

def misclassification_rate(y, predictions):
    correct_pred = 0
    for i, val in enumerate(y):
        if predictions[i] == y[i]:
            correct_pred += 1
    return ((1 - (correct_pred / len(y))) * 100)

def kNN_select(X,y,k_vals):
    rates = []
    for k in k_vals:
        predictions = kNN(X, y, k)
        rates.append(misclassification_rate(y, predictions))
    mis_class_rates = pd.Series(rates, index = k_vals)
    return mis_class_rates


def kNN_classification(df,class_column,seed,percent_train,k_vals):
    # df            - DataFrame to 
    # class_column  - column of df to be used as classification variable, should
    #                 specified as a string  
    # seed          - seed value for creating the training/test sets
    # percent_train - percentage of data to be used as training data
    # k_vals        - set of k values to be tests for best classification
    
    # Separate X and y
    
    temp = df.drop(class_column, axis=1)
    y = df[class_column]
    X = (temp - temp.mean()) / (temp.std())
    
    # Divide into training and test
    
    X_train = X.sample(frac = percent_train, random_state = seed)
    y_train = y.sample(frac = percent_train, random_state = seed)
    
    X_test = X.drop(X_train.index)
    y_test = y.drop(y_train.index)
        
    # Compute the mis-classification rates for each for the values in k_vals
    
    rates = kNN_select(X_train, y_train, k_vals)
    #print('Best K for TRAINING is', rates.idxmin(), 
    #        ' with misclassification rate ', rates.min())
    
    # Find the best k value, by finding the minimum entry of mis_class_rates 
    best_k = rates.idxmin()
    
    # Run the classification on the test set to see how well the 'best fit'
    # classifier does on new data generated from the same source

    test_predictions = kNN(X_test, y_test, best_k)

    # Calculate the mis-classification rates for the test data
    mis_class_test = misclassification_rate(y_test, test_predictions)
    return mis_class_test
    

# Testing 1
data_1 = pd.read_csv('tunedit_genres.csv')
kNN_classification(data_1, 'RockOrNot', 123, 0.75, [1, 3, 5, 7, 9])

# Testing 2
data_2 = pd.read_csv('house_votes.csv')    
kNN_classification(data_2, 'Party', 123, 0.75, [1, 3, 5, 7, 9])