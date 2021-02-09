from torch.utils.data import Dataset, DataLoader
import torch

class DialogDataset(Dataset):
    
    _label_dict = dict()
    
    def __init__(self, tokenizer, data, text_field = "clean_text", label_field="act_label_1", max_len=512):
        
        """
        Process the text, here in each row, we have a word, group them by sentenceID and then form a list[str],which can be passed to the tokenizer.
        Tokenizer, we will take list[str], and pass it to encode/encode_plus, to get the token ID's
        """

        self.text = 

        # === Process the Labels - Pick one per sentence ID ===  
        self.acts = 

        # Tokenizer, since text is alread split up into list[str], just pass it through the pretrainer BERT/roBERTa encode or enocde_plus
        self.tokenizer = tokenizer


        self.max_len = max_len
        
        
        # Build/Update the Label dictionary 
        classes = sorted(set(self.acts))
        
        for cls in classes:
            if cls not in DialogDataset._label_dict.keys():
                DialogDataset._label_dict[cls]=len(DialogDataset._label_dict.keys())
    
    def __len__(self):
        # === First group by sentenceID/or use Unique SentenceIDs ==
        return 
    
    def label_dict(self):
        return DialogDataset._label_dict
    
    def __getitem__(self, index):
        
        text = self.text[index]
        act = self.acts[index]
        label = DialogDataset._label_dict[act]
        
        input_encoding = self.tokenizer.encode_plus(
            text=text, # List[str] or str or list[int](tokens)
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt", # Returns Pytorch tensor, use tf for Tensorflow and np for numpy.
            return_attention_mask=True,
            padding="max_length", # Useful if using multiple sentences, to make them all same D
        )
        
        # Number of words in a sentence, if we have string then use -- len(self.tokenize.tokenize(text))
        seq_len = len(text)
        
        return {
            "text":text,
            "input_ids":input_encoding['input_ids'].squeeze(),
            "attention_mask":input_encoding['attention_mask'].squeeze(),
            "seq_len":seq_len,
            "act":act,
            "label":torch.tensor([label], dtype=torch.long),
        }