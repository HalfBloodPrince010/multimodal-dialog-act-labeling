import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
# == Prediction ==
def test_net(loader, model, device):
    
    correct = 0
    total = 0
    label = []
    pred = []
    with torch.no_grad():  
        for batch in loader:       
            input_ids  = batch['input_ids'].to(device=device)
            attention_mask = batch['attention_mask'].to(device=device)
            labels = (batch['label'].squeeze()).to(device=device)  
            seq_len = batch['seq_len'].to(device=device)
            pitch = batch['pitch'].to(device=device)
            freq = batch['freq'].to(device=device)
            data = {'input_ids':input_ids, 'attention_mask':attention_mask, 'label':labels, 'seq_len':seq_len, 'pitch':pitch, 'freq':freq}
            label += labels.cpu()
            outputs = model(data)          
            _, predicted  = torch.max(outputs.data, 1)        
            pred += predicted.cpu()        
            total += labels.size(0)        
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct/total)
    f1 = f1_score(label, pred, average='micro')
    f1_individual = f1_score(label, pred, average=None) 
    print("Test Accuracy : ", accuracy)
    return accuracy, f1, f1_individual
