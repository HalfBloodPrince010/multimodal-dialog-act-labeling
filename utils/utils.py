# Utils 
import os
import pandas as pd
import gc
import logging

# Data
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Models 
import torch.nn as nn
import torch

# Training and Eval
from dataset.dataset import DialogDataset


# == Dataloading ==
def load_data(config, label_dict, tokenizer):

    logging.info("Loading Data and DataLoader")
    # == Dataset and DataLoader - Test, Train and Validation==

    train_data = pd.read_pickle(os.path.join(config['data_dir'], "bert_processed_train_token.pkl"))
    speech_train = pd.read_pickle(os.path.join(config['data_dir'], "speech_summarized_train.pkl"))
    #train_data = train_data[:256]
    train_dataset = DialogDataset(tokenizer=tokenizer, data=train_data, speech=speech_train, max_len=config['max_len'], text_field=config['text_field'], label_field=config['label_field'],label_dict = label_dict)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], drop_last=True, shuffle=False, num_workers=config['num_workers'])
    
    val_data = pd.read_pickle(os.path.join(config['data_dir'], "bert_processed_val_token.pkl"))
    speech_val = pd.read_pickle(os.path.join(config['data_dir'], "speech_summarized_val.pkl"))
    #val_data = val_data[:32]
    val_dataset = DialogDataset(tokenizer=tokenizer, data=val_data, speech=speech_val, max_len=config['max_len'], text_field=config['text_field'], label_field=config['label_field'],label_dict = label_dict)
    val_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], drop_last=True, shuffle=False, num_workers=config['num_workers'])
    
    test_data = pd.read_pickle(os.path.join(config['data_dir'], "bert_processed_test_token.pkl"))
    speech_test = pd.read_pickle(os.path.join(config['data_dir'], "speech_summarized_test.pkl"))
    #test_data = test_data[:32]
    test_dataset = DialogDataset(tokenizer=tokenizer, data=test_data,speech=speech_test, max_len=config['max_len'], text_field=config['text_field'], label_field=config['label_field'],label_dict = label_dict)
    test_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], drop_last=True, shuffle=False, num_workers=config['num_workers'])
    
    return train_loader, val_loader, test_loader

def get_label_dict(config):
    set_label_data = pd.read_csv(os.path.join(config['data_dir'], "bert_processed_train.csv"),  usecols=[config['label_field']])
    label_dict = {}

    # Weights
    n_samples = len(set_label_data)
    n_classes = config['num_classes']

    # Build/Update the Label dictionary
    classes = sorted(set(set_label_data[config['label_field']]))
    label_counts = set_label_data[config['label_field']].value_counts()
    count_dict = label_counts.to_dict()

    w = list()
    for cls in classes:
        if cls not in label_dict.keys():
            w.append(round(n_samples/(count_dict[cls] + n_classes), 3))
            label_dict[cls]=len(label_dict.keys())

    weights = torch.FloatTensor(w)

    return label_dict, weights
