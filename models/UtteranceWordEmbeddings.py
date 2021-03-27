import  torch.nn as nn
# from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BertModel

class UtteranceWordEmbeddings(nn.Module):
    
    def __init__(self, model_name="bert-base-uncased", hidden_size=768, device=None):
        super(UtteranceWordEmbeddings, self).__init__()
        
        # Using BERT's model  (pre-trained)
        #self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.base = BertModel.from_pretrained(model_name)
        self.base.to(device)

        # Dont train the roBERTa, freeze the Model Parameters and dont let gradients pass through.
        for param in self.base.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, seq_len):
        """
        Pass the input_ids, obtained from the encode_plus/encode to the pre-trained roBERTa to obtain the embeddings.

        return : List of WordEmbeddings
        
        print("BERT - Parameters through which gradients flow")
        for name, param in self.base.named_parameters():
            if param.requires_grad:
                print (name, param.data)
        """

        # Sequence Length? and Hidden size? - If we use RNN - # utterance_level_word_embeddings.shape = [batch, max_len, hidden_size]
        utterance_level_word_embeddings = self.base(input_ids, attention_mask) # Utterance Level Word Embeddings.
        return utterance_level_word_embeddings[0]
