import click
import logging
import os
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np


def read_data(PROJECT_DIR,input_filename):
    
    input_dir = os.path.join(PROJECT_DIR, "data/raw")
    filepath = os.path.join(input_dir, input_filename)
    df = pd.read_csv(filepath)
    
    return df


def load_split_dataset(df,train_percent=0.8,validate_percent=0.2,seed=None):
     
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)

    train_end = int(train_percent * m)

    validate_end = int(validate_percent * m) + train_end

    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]

    return train,validate,test

    
#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(input_filename):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    
    logger.info('read data')
    dframe = read_data(PROJECT_DIR,input_filename)
    
    logger.info('split data')
    train_data, validate_data,test_data = load_split_dataset(dframe,train_percent=0.8,validate_percent=0.2,seed=None)  
    
    logger.info('save data')
    output_filepath = os.path.join(PROJECT_DIR, "data/processed")
    
    if train_data is not None:
        train_data.to_excel(os.path.join(output_filepath, "train_data.xlsx"))
        
    if validate_data is not None:
        validate_data.to_excel(os.path.join(output_filepath, "validate_data.xlsx"))
    
    if test_data is not None:
        test_data.to_excel(os.path.join(output_filepath, "test_data.xlsx"))
    

   
if __name__ == '__main__':

    main("iris.csv")
