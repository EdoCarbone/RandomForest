import sys
sys.path.append('src')

import pandas as pd
from utils import print_leaf, gini, entropy, class_counts, info_gain, is_numeric, unique_vals, partition
import logging

logger = logging.getLogger(__name__)
GINI = "gini"
ENTROPY = "entropy"

class DecisionTree: 
    
    def __init__(self,
                 question=None,
                 true_branch=None,
                 false_branch=None,
                 metrics=None, 
                 max_depth=None,
                 leafs=None,
                 final_leaf=None):
        
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.max_depth = max_depth
        self.leafs = []
        self.final_leaf = final_leaf
        
        if metrics is None:
            self.metrics = GINI  
        else:
            self.metrics = metrics  
   
              
        
    def build_tree(self,train_data,header,count=0):
        
        """Builds the tree.
        Try partitioing the dataset on each of the unique attribute,
        calculate the information gain,
        and return the question that produces the highest gain.
        """
  
        #print("n° features: {}".format(len(train_data[0]) - 1))
        gain, question = self.find_best_split(train_data,header)
        logger.info("--best question is ''{}'' with information gain: {}".format(question,round(gain,2)))
            
        print("--",count)
        print("--",gain)
        
        if isinstance(train_data,pd.DataFrame):
            train_data = train_data.values.tolist()
        
        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain < 0.001:
            print("final_gain=",gain)
            
            leafs = self.leafs.append(Leaf(train_data))
            #print(leafs)
            return DecisionTree(None,
                                self.true_branch, 
                                self.false_branch,
                                self.metrics,
                                self.max_depth,
                                leafs,
                                Leaf(train_data))
        
        if count == self.max_depth:
            print("final_depth=",count)
            self.leafs.append(Leaf(train_data))
            
            return DecisionTree(None,
                                self.true_branch, 
                                self.false_branch,
                                self.metrics,
                                self.max_depth,
                                self.leafs,
                                Leaf(train_data))
    
        true_rows, false_rows = partition(train_data, question)
        
        # Recursively build the true branch.
        logger.info("\n----TRUE BRANCH----")
        true_branch = self.build_tree(true_rows,header,count+1)
        
        # Recursively build the false branch.
        logger.info("\n----FALSE BRANCH----")
        false_branch = self.build_tree(false_rows,header,count+1)
        
        return DecisionTree(question,
                            true_branch, 
                            false_branch,
                            self.metrics,
                            self.max_depth,
                            self.leafs,
                            None)
    
    def initialize_split(self,train_data):
        
        if self.metrics == GINI:
            # Calculate the information gain from this split
            current_uncertainty = gini(train_data)
               
        if self.metrics == ENTROPY:
            
            current_uncertainty = entropy(train_data)
        
        return current_uncertainty
     
    def find_best_split(self,train_data,header):
        """Find the best question to ask by iterating over every feature / value
        and calculating the information gain."""
    
        if isinstance(train_data,pd.DataFrame):
            train_data = train_data.values.tolist()
        
        n_features = len(train_data[0]) - 1  # number of columns
        
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = self.initialize_split(train_data)
        
        for col in range(n_features):  # for each feature
            
            values = unique_vals(train_data, col)  # unique values in the column
            logger.info("values features n° {} ''{}'': {}".format(col+1,header[col],values))
            for val in values:  # for each value
        
                question = Question(header,col, val)
                
                # try splitting the dataset
                true_rows, false_rows = partition(train_data, question)
                
                # Skip this split if it doesn't divide the dataset.
                if len(true_rows) == 1 or len(false_rows) == 1:
                    continue
                
                # Calculate the information gain from this split
                gain = info_gain(self.metrics,true_rows, false_rows, current_uncertainty)
               
                if gain >= best_gain:
                    best_gain, best_question = gain, question
    
        return best_gain, best_question


    def classify(self,row):
        """See the 'rules of recursion' above."""
        
        # Base case: we've reached a leaf
        if self.true_branch == None or self.true_branch == None:
            logger.info("predictions: {} ".format(self.final_leaf.predictions))
            return self.final_leaf.predictions
    
        logger.info("tree_classifier->classify:",self.question)
       
        #print("input value {}".format(row[node.question.column]))
        
        #print(self.question.match(row))
        if self.question.match(row):
            logger.info("yes")
            return self.true_branch.classify(row)
        else:
            logger.info("no")
            return self.false_branch.classify(row)
    

    def print_tree(self, spacing=""):
        """World's most elegant tree printing function."""
    
        # Base case: we've reached a leaf
        if self.true_branch == None or self.true_branch == None:
            print (spacing + "Predict", str(print_leaf(self.final_leaf.predictions)))
            return "--- END ---"
        
        # Print the question at this node
        print (spacing + str(self.question))
    
        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.true_branch.__repr__(spacing + "  ")
    
        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.false_branch.__repr__(spacing + "  ")
        
        return "--- END ---"
    
class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self,header, column, value):
        self.header = header
        self.column = column
        self.value = value
        

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        
        val = example[self.column]
        #print("valore domanda {}".format(self.value))
        #print("::input value {}::".format(val))
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))
 
    
class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)
