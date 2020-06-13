import os
import pickle
from random import sample
from math import sqrt,ceil
import pandas as pd
import numpy as np

GINI = "gini"
ENTROPY = "entropy"

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, (int,float))

def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

########################## decision tree metrics #############################

def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    p_x = probabilities of find element of "x" class 
    """
    
    classes_count = class_counts(rows)
    impurity = 1
    for x in classes_count: 
        p_x = classes_count[x] / float(len(rows))
        impurity -= p_x**2
    
    return impurity


def entropy(rows):
    """
    Calculate the entropy of a dataset.
    p_x = probabilities of find element of "x" class 
    """
    classes_count = class_counts(rows)
    entropy = 0
    for x in classes_count: 
        p_x = classes_count[x] / float(len(rows))
        entropy-=p_x*np.log2(p_x)

    return entropy

def info_gain(metrics,left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    if metrics == GINI:
        # Calculate the information gain from this split
        return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
               
    if metrics == ENTROPY:
        return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)
            
     
       
##############################################################################

   
def get_input_var_pos(header,var_tree):
    
    input_variables = []
    for var in var_tree:
        for idx,head in enumerate(header):
            if var == head:
                input_variables.append(idx)
                
    return input_variables

def sub_sample_and_features(df,labels,n_selected_features,sample_ratio = 0.5):
    
    n_features = df.shape[1]-1
    n_row = df.shape[0]

    sampled_row = sample(range(0,n_row), round(n_row*float(sample_ratio)))
    
    if n_selected_features == "best":
        selected_features = sample(range(0,n_features), ceil(sqrt(n_features)))
    elif n_selected_features == "all":
        selected_features = sample(range(0,n_features), n_features)
    else:
        selected_features = sample(range(0,n_features), n_selected_features)
    
    df_sel = df.iloc[sampled_row,selected_features]
    
    return pd.concat([df_sel.reset_index(drop=True),labels.reset_index(drop=True)], axis=1).dropna()
    
    
def save_model(dir_save,filename,modelName):
    full_dir = os.path.join(dir_save, filename)
    pickle_out = open(full_dir,"wb")
    pickle.dump(modelName, pickle_out)
    pickle_out.close()


def load_model(dir_save,modelName):
    full_dir = os.path.join(dir_save, modelName)
    pickle_in = open(full_dir,"rb")
    model = pickle.load(pickle_in)
    return model
    