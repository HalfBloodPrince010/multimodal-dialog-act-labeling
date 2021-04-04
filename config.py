import torch
import os

config = {
    
    # Data 
    "data_dir":os.path.join(os.getcwd(), 'data'),
    "dataset":"switchboard",
    "text_field":"da_token",
    "label_field":"da_label",
    "pitch":os.path.join(os.getcwd(),'data/pitch_json'),

    "max_len":128,
    "batch_size":128,
    "num_workers":4,

    #Model
    "model_name":"bert-base-uncased",
    "hidden_size":786,
    "num_classes":41, # There are 41 classes in Switchboard Corpus
    
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
    "epochs":15,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
