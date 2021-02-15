import torch.nn as nn
import torch
from .UtteranceWordEmbeddings import UtteranceWordEmbeddings

class DialogActClassificationProsody(nn.Module):
    
    def __init__(self, model_name="roberta-base", hidden_size=768, num_classes=18, device=torch.device("cpu")):
        
        super(DialogActClassificationProsody, self).__init__()
        
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
            x.shape = [batch, seq_len, hidden_size]
        """
        
        word_outputs = self.utterance_word_embeddings(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], seq_len=batch['seq_len'].tolist())
        
        # == Get Speech Features ==

        # Concatenate Word and Speech Vectors

        # Use RNN/LSTM to get a Vector representation of entire sentence


        # Pass the Vector representation obtained above to classifier.

        return word_outputs        
