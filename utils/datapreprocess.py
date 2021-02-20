from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

data_list = list()

def preprocess(data):
    filenum = list(set(data['filenum']))[0]
    true_speaker = list(set(data['true_speaker']))[0]
    da_token = list(data['da_token'])
    sent_id = list(set(data['sent_id']))[0]
    da_label = list(set(data['da_label']))[0]
    start_time = list(data['start_time'])
    end_time = list(data['end_time'])
    data_list.append({'sent_id':sent_id ,'filenum' : filenum, 'true_speaker' : true_speaker, 'da_token' : da_token, 'da_label' : da_label, 'start_time' : start_time, 'end_time' : end_time})
    return


if __name__=="__main__":

    # Load Training Data
    fields = ["filenum", "true_speaker", "da_token", "sent_id", "da_label", "start_time", "end_time"]
    train_data = pd.read_csv(os.path.join(config['data_dir'], "train_aligned.tsv"), sep='\t', usecols=fields) 
    train_data =  train_data.groupby(['sent_id']).apply(preprocess) 
    # === Train Data Processed ===
    train_data_processed = pd.DataFrame(data_list)
    # === Adding training data to new modified .csv ===
    train_data_processed.to_csv(os.path.join(config['data_dir'], "processed_train.csv"), index=False)
    # === Adding training data to pickle file ===
    pd.to_pickle(train_data_processed, os.path.join(config['data_dir'], "processed_train.pkl")) 
    # Process Test and Validation Data
    data_list = []
    exit(0)  
