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
    
    train_loss = []
    validation_loss = []
    start_time = time.time()
    n_iters = 0
    for epoch in range(config['epochs']): 
        model.train()
        print("Run Epoch {}".format(epoch))
        epoch_loss = []
        for batch in train_loader:
            input_ids  = batch['input_ids'].to(device=device)
            attention_mask = batch['attention_mask'].to(device=device)
            targets = (batch['label'].squeeze()).to(device=device)
            seq_len = batch['seq_len'].to(device=device)
            data = {'input_ids':input_ids, 'attention_mask':attention_mask, 'label':targets, 'seq_len':seq_len}
            logits = model(data)
            loss = criterion(logits, targets)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Train Loss per epoch    
        train_loss.append(sum(epoch_loss) / len(epoch_loss))

    # Validation
        val_loss, val_accuracy = eval_net(model, val_loader, device, criterion)
        validation_loss.append(val_loss)
        print("Validation accuracy :", val_accuracy)
    
        if epoch%2 == 0:
            try:
                os.mkdir(checkpoint_dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(),
                       checkpoint_dir + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        end_time = time.time()
        #print("Time difference for each Epoch: {}".format(end_time-start_time))
        start_time = time.time()

    test_accuracy, f1_score = test_net(test_loader, model, device)  
    print("f1-score: ",f1_score)
    print("Test Accuracy: ", test_accuracy)
    # == Train Loss Curves ==
    plt.figure(figsize=(15,7))
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(train_loss, '-o')
    plt.xlabel('Epoch')
    plt.savefig('./plots/train_loss.png')

# == Validation Loss Curves ==
    plt.figure(figsize=(15,7))
    plt.subplot(2, 1, 1)
    plt.title('Validation loss')
    plt.plot(validation_loss, '-o')
    plt.xlabel('Total Batch size/10')
    plt.savefig('./plots/val_score.png')

    print("==End of Training ==")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Training Begin..")
    train(checkpoint_dir="./checkpoints/",data_config=config)
