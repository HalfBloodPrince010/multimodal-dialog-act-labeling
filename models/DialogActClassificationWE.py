import torch.nn as nn
import torch
from .UtteranceWordEmbeddings import UtteranceWordEmbeddings

"Classification with just Word Embeddings"

class DialogActClassificationWE(nn.Module):
    
    def __init__(self, model_name="bert-base-uncased", hidden_size=768, num_classes=18, device=torch.device("cpu")):
        
        super(DialogActClassificationWE, self).__init__()
        
        self.in_features = hidden_size
        
        self.device = device
        
        # Get Utterance level word embeddings using Transformers.
        self.utterance_word_embeddings = UtteranceWordEmbeddings(model_name=model_name, hidden_size=hidden_size)
        
        # Classifier on top to predict the label
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=self.in_features, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        ])
        
    
    def forward(self, batch):
        """
        DAC Word Embeddings Data shapes and sizes
        (Batch_size, max_length, hidden_size) --> (64, 128, 768)

        Just get the first vector, which is a CLS token, so (64, 128, 768) --> (64, 768)

        Pass this (Batch_size, [CLS]) to a classifier and check using the labels.

        [CLS} dim here is 768
        """
        
        word_outputs = self.utterance_word_embeddings(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], seq_len=batch['seq_len'].tolist())
       
        features = word_outputs[:, 0, :]  
        print("Exiting from the DACProsody, Feature shape", features.shape)
        exit(0)

        # Pass through a classifier

        return word_outputs        
