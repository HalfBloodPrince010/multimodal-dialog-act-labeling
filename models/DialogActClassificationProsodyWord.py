import torch.nn as nn
import torch
from .UtteranceWordEmbeddings import UtteranceWordEmbeddings

""" Dialog Act Label Classification with Word and Speech Features at Word Level"""

class DialogActClassificationProsodyWord(nn.Module):
    
    def __init__(self, model_name="roberta-base", hidden_size=768, num_classes=18, device=torch.device("cpu")):
        
        super(DialogActClassificationProsodyWord, self).__init__()
        
        self.in_features = hidden_size
        
        self.device = device
         
        # Get Utterance level word embeddings using Transformers.
        self.utterance_word_embeddings = UtteranceWordEmbeddings(model_name=model_name, hidden_size=hidden_size, device=self.device) 
 
        # Summarize the word level features using Bi-directional LSTM
        self.summarize_word_level_features = nn.LSTM(self.in_features, hidden_size, batch_first=True, bidirectional=True)
        self.h0 = torch.rand(2, 64, hidden_size).float().to(self.device)
        self.c0 = torch.rand(2, 64, hidden_size).float().to(self.device)
         
        # Classifier on top to predict the label
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=2*self.in_features, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        ])
        
    
    def forward(self, batch):
        """
        DAC Word Embeddings Data shapes and sizes
        (Batch_size, max_length, hidden_size) --> (64, 128, 768)

        Concatenate each word embeddings with each word's speech embeddings.

        print("Utterance - throught which gradients flow")
        for name, param in self.utterance_word_embeddings.named_parameters():
            if param.requires_grad:
                print (name)
        print("Classifier - Through which gradients flow")
        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                print (name)
        """
        word_outputs = self.utterance_word_embeddings(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], seq_len=batch['seq_len'].tolist())
        word_features = word_outputs[:, 1:, :] # Get Embeddings of each word, merge it with Speech embeddings of each Word.
        word_features = word_features.double()
        feature = torch.cat((word_features,batch['pitch'],batch['freq']), dim=2).float()
        output, (hn, cn) = self.summarize_word_level_features(feature, (self.h0, self.c0))
        features = torch.cat((hn[0], hn[1]), 1)
        classifier_out = self.classifier(features)

        return classifier_out
