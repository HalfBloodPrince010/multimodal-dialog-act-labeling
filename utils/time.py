from models.DialogActClassificationProsody import DialogActClassificationProsody
from models.DialogActClassificationWE import DialogActClassificationWE
from eval import eval_net
from test import test_net
from utils.utils import load_data, get_label_dict
from matplotlib import pyplot as plt

# Utils 
import torch
import os
import pandas as pd
import gc
import logging

# Data
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Models 
import torch.nn as nn
from transformers import BertTokenizer ,RobertaTokenizer ,AutoConfig, AutoTokenizer, AutoModel

# Training and Eval
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from dataset.dataset import DialogDataset

# Configuration
from config import config

import time

def train(checkpoint_dir=None, data_dir=None, data_config=None):
   
    config = data_config 
    # Device and Loss Function
    device = config['device']
    criterion = nn.CrossEntropyLoss()

#     if checkpoint_dir:
#         model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
#         net.load_state_dict(model_state)
#         optimizer.load_state_dict(optimizer_state)
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    
    # Label Dictionary
    label_dict = get_label_dict(config)
    
    # DataLoaders - Train, Test and Validation
    train_loader, val_loader, test_loader = load_data(config, label_dict, tokenizer)
    
    # Model
    model = DialogActClassificationWE(
            model_name=config['model_name'],
            hidden_size=config['hidden_size'],
            num_classes=config['num_classes'],
            device=config['device']
        )

    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    for epoch in range(config['epochs']): 
        model.train()
        print("Run Epoch {}".format(epoch))
        start_time = time.time()
        for batch in train_loader:
            input_ids  = batch['input_ids'].to(device=device)
            attention_mask = batch['attention_mask'].to(device=device)
            targets = (batch['label'].squeeze()).to(device=device)
            seq_len = batch['seq_len'].to(device=device)
            end_time = time.time()
            print("Time to Read batch", end_time-start_time)
            start_time = time.time()

    print("==End of Time Analysis ==")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Training Begin..")
    train(checkpoint_dir="./checkpoints",data_config=config)
