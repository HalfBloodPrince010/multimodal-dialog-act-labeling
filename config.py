import torch
import os

config = {
    
    # Data 
    "data_dir":os.path.join(os.getcwd(), 'data'),
    "dataset":"switchboard",
    "text_field":"da_token",
    "label_field":"da_label",

    "max_len":256,
    "batch_size":64,
    "num_workers":4,
    
    # Model
    "model_name":"roberta-base",
    "hidden_size":768,
    "num_classes":43, # There are 43 classes in Switchboard Corpus
    
    # Training
    "save_dir":"./",
    "project":"dialogue-act-label-classification",
    "run_name":"context-aware-attention-dac",
    "lr":1e-5,
    "monitor":"val_accuracy",
    "min_delta":0.001,
    "filepath":"./checkpoints/{epoch}-{val_accuracy:4f}",
    "precision":32,
    "average":"micro",
    "epochs":100,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
