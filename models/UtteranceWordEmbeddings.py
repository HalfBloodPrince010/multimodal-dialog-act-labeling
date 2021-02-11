import  torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class UtteranceWordEmbeddings(nn.Module):
    
    def __init__(self, model_name="roberta-base", hidden_size=768, bidirectional=True, num_layers=1):
        super(UtteranceWordEmbeddings, self).__init__()
        
        
        # Using roBERTa's Embeddings
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        
        # Dont train the roBERTa, instead freeze the Model Parameters.
        for param in self.base.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, seq_len):
        """
        Pass the input_ids, obtained from the encode_plus/encode to the pre-trained roBERTa to obtain the embeddings.

        return : List of WordEmbeddings
        """
        
        utterance_level_word_embeddings, _ = self.base(input_ids, attention_mask) # Utterance Level Word Embeddings.
        
        return utterance_level_word_embeddings
