from models.DialogActClassificationProsody import DialogActClassificationProsody
from models.DialogActClassificationWE import DialogActClassificationWE
# Utils 
import torch
import os
import pandas as pd
import gc

# Data
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Models 
import torch.nn as nn
from transformers import BertTokenizer ,RobertaTokenizer ,AutoConfig, AutoTokenizer, AutoModel

# Training and Eval
import wandb
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from dataset.dataset import DialogDataset


class LightningModel(pl.LightningModule):
    
    def __init__(self, config):
        super(LightningModel, self).__init__()
        
        self.config = config
        
        self.model = DialogActClassificationWE(
            model_name=self.config['model_name'],
            hidden_size=self.config['hidden_size'],
            num_classes=self.config['num_classes'],
            device=self.config['device']
        )
        set_label_data = pd.read_csv(os.path.join(self.config['data_dir'], "bert_processed_train.csv"),  usecols=[self.config['label_field']])
        self.label_dict = {}

        # Build/Update the Label dictionary 
        classes = sorted(set(set_label_data[self.config['label_field']]))

        for cls in classes:
            if cls not in self.label_dict.keys():
                self.label_dict[cls]=len(self.label_dict.keys())

        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        
    def forward(self, batch):
        logits  = self.model(batch)
        return logits
    
    def configure_optimizers(self):
        return optim.Adam(params=self.parameters(), lr=self.config['lr'])
    
    def train_dataloader(self):
        # Loading the data from the data.
        fields = ["filenum", "true_speaker", "da_token", "sent_id", "da_label", "start_time", "end_time"] 
        train_data = pd.read_pickle(os.path.join(self.config['data_dir'], "bert_processed_train.pkl"))
        train_dataset = DialogDataset(tokenizer=self.tokenizer, data=train_data, max_len=self.config['max_len'], text_field=self.config['text_field'], label_field=self.config['label_field'],label_dict = self.label_dict)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'], drop_last=True, shuffle=False, num_workers=self.config['num_workers'])
        return train_loader
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['label'].squeeze()
        print(batch['input_ids'].shape)
        print(batch['label'].shape)
        print(batch['attention_mask'].shape)
        print(batch['input_ids'][0])
        print("Exiting here.. in Training step\n")
        exit(0)
        logits = self(batch)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        return {"loss":loss, "accuracy":acc, "f1_score":f1}
