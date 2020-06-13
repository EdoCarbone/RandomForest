import os
import logging
from pathlib import Path
#import click
import pandas as pd
from utils import save_model, load_model
from decision_tree import DecisionTree
from random_forest import RandomForest

project_dir = Path(__file__).resolve().parents[2]
output_filepath = os.path.join(project_dir, "models")
input_filepath = os.path.join(project_dir, "data/processed") 

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())    
def main():
    """
    N.B. Last DataFrame Column contains labels
    """
    logger = logging.getLogger(__name__)    
    logger.debug('read data')
     
    dframe_train = pd.read_excel(os.path.join(input_filepath, "train_data.xlsx"), index_col=0)
    logger.debug('train model')
    
    '''CREATE SINGLE TREE'''
    d_t = DecisionTree(metrics = 'entropy') #max_depth = 8
    #trained_dt = dt.build_tree(dframe,header)
    #prediction = classify(small_train.values[0][:-1],t0)
    
    '''CREATE RANDOM FOREST WITH TREES d_t'''
    r_f = RandomForest(decision_tree_type=d_t, n_trees=20)
    r_f = r_f.build_forest(dframe_train, n_selected_features="best", sample_ratio =.8)
    
    '''GET MODEL ACCURACY ON VALIDATION DATA'''
    logger.debug('get model accuracy')
    dframe_val = pd.read_excel(os.path.join(input_filepath, "validate_data.xlsx"), index_col=0)
    predictions_validation = r_f.get_model_accuracy(dframe_val.columns.values.tolist(), dframe_val)  
    
    #logger.debug('single prediction')
    #rf.classify_forest(dframe_val.columns.values.tolist(),dframe_val.values[0],forest) 
    
    logger.debug('save model')
    save_model(output_filepath, "model_00.npy", r_f)
    
   
if __name__ == '__main__':
    
    main()   
    output_filepath = os.path.join(project_dir, "models")
    forest_00 = load_model(output_filepath, "model_00.npy")
    