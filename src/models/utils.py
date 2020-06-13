import os
import pickle
from random import sample
from math import sqrt,ceil
import pandas as pd

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
    return isinstance(value, int) or isinstance(value, float)

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

def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts: 
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def gini_impurity(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

   
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
    