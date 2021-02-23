from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


class DialogDataset(Dataset):
    
    def __init__(self, tokenizer, data, text_field = "da_token", label_field="da_label", filenum="filenum", speaker="true_speaker", s_time="start_time", e_time="end_time", max_len=512, label_dict = None):
        
        """
        Process the text, here in each row, we have a word, group them by sentenceID and then form a list[str],which can be passed to the tokenizer.
        Tokenizer, we will take list[str], and pass it to encode/encode_plus, to get the token ID's
        """
        self.length =  len(data['sent_id'])
        
        self.text = data[text_field]
        
        # === Process the Labels - Pick one per sentence ID ===  
        self.acts = data[label_field]

        # Tokenizer, since text is alread split up into list[str], just pass it through the pretrainer BERT/roBERTa encode or enocde_plus
        self.tokenizer = tokenizer

        # == Other columns ==
        self.filenum = data[filenum]

        self.speaker = data[speaker]

        self.start_time = data[s_time]

        self.end_time = data[e_time]

        self.max_len = max_len
        
        self.label_dict = label_dict
        # Update the Label dictionary, which wasn't done in trainer class, Training data has only 41 labels, migth cause problem during val and test, hence update.

        #if label_dict is not None: 
        classes = sorted(set(self.acts))
        
        for cls in classes:
            if cls not in self.label_dict.keys():
                self.label_dict[cls]=len(self.label_dict.keys())
    
    def __len__(self):
        # === Unique SentenceIDs ==
        return self.length
    
    def label_dict(self):
        return self.label_dict
    
    def __getitem__(self, index):
        
        text = self.text[index]
        act = self.acts[index]
        input_encoding = self.tokenizer.encode(
            text=text, # List[str] or str or list[int](tokens)
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt", # Returns Pytorch tensor, use tf for Tensorflow and np for numpy.
            return_attention_mask=True,
            padding="max_length", # Useful if using multiple sentences, to make them all same D
        )
        
        # Number of words in a sentence, if we have string then use -- len(self.tokenize.tokenize(text))
        seq_len = len(list(text))
       
        filenum = self.filenum[index]
 
        true_speaker = self.speaker[index]

        start_time = self.start_time[index]

        end_time = self.end_time[index]
        
        return {
            "text":text,
            "input_ids":input_encoding['input_ids'].squeeze(),
            "attention_mask":input_encoding['attention_mask'].squeeze(),
            "seq_len":seq_len,
            "act":act,
            "filenum":filename,
            "true_speaker":true_speaker,
            "start_time":start_time,
            "end_time":end_time,
            "label":torch.tensor([label], dtype=torch.long),
        }
