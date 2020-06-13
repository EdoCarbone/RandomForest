import click
import logging
import os
from pathlib import Path
import pandas as pd
from utils import save_model,load_model
from decision_tree import DecisionTree
from random_forest import RandomForest
 
#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(PROJECT_DIR):
    """
    N.B. Last DataFrame Column contains labels
    """

    logger = logging.getLogger(__name__)
    
    logger.debug('read data')
    input_filepath = os.path.join(PROJECT_DIR, "data/processed")
    
    dframe_train = pd.read_excel(os.path.join(input_filepath,"train_data.xlsx"), index_col=0)
    
    logger.debug('train model')
    ###SINGLE TREE
    dt = DecisionTree()
        #max_depth = 8
        
    
    #trained_dt = dt.build_tree(dframe,header)
    #prediction = classify(small_train.values[0][:-1],t0)
    
    ###FOREST
    rf = RandomForest(decision_tree_type=dt,n_trees=100)
    rf = rf.build_forest(dframe_train,n_selected_features="best",sample_ratio = .8)
    
    #pred_forest = classify_forest(header,validate_data.values[0],forest)
    global val
    logger.debug('get model accuracy')
    dframe_val = pd.read_excel(os.path.join(input_filepath,"validate_data.xlsx"), index_col=0)

    
    val = rf.get_model_accuracy(dframe_val.columns.values.tolist(),dframe_val)  
    
    
    logger.debug('PREDICTION')
    #rf.classify_forest(dframe_val.columns.values.tolist(),dframe_val.values[0],forest) 
    
    logger.debug('save model')
    output_filepath = os.path.join(PROJECT_DIR, "models")
    save_model(output_filepath,"model_00",rf)
    
   
if __name__ == '__main__':
    global PROJECT_DIR
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    main(PROJECT_DIR)
    
    output_filepath = os.path.join(PROJECT_DIR, "models")
    forest_00 = load_model(output_filepath,"model_00")