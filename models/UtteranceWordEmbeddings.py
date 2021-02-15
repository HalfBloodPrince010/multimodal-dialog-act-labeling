import  torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class UtteranceWordEmbeddings(nn.Module):
    
    def __init__(self, model_name="roberta-base", hidden_size=768):
        super(UtteranceWordEmbeddings, self).__init__()
        
        
        # Using roBERTa's model  (pre-trained)
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        
        # Dont train the roBERTa, freeze the Model Parameters and dont let gradients pass through.
        for param in self.base.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, seq_len):
        """
        Pass the input_ids, obtained from the encode_plus/encode to the pre-trained roBERTa to obtain the embeddings.

        return : List of WordEmbeddings
        """
        # Sequence Length? and Hidden size? - If we use RNN - # utterance_level_word_embeddings.shape = [batch, max_len, hidden_size]
        utterance_level_word_embeddings, _ = self.base(input_ids, attention_mask) # Utterance Level Word Embeddings.
        
        return utterance_level_word_embeddings
