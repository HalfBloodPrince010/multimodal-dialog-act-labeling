import torch
import torch.nn.functional as F

# == Validation ==
def eval_net(model, loader, device, criterion):
    
    model.eval()    
    val_loss = []   
    total = 0    
    correct = 0    
    with torch.no_grad():       
        for batch in loader:            
            
            input_ids  = batch['input_ids'].to(device=device)
            attention_mask = batch['attention_mask'].to(device=device)
            labels = (batch['label'].squeeze()).to(device=device)
            seq_len = batch['seq_len'].to(device=device)
            data = {'input_ids':input_ids, 'attention_mask':attention_mask, 'label':labels, 'seq_len':seq_len}
            logits = model(data)            
            
            # Accuracy           
            _, predicted  = torch.max(logits.data, 1)        
            total += labels.size(0)        
            correct += (predicted == labels).sum().item()            
            
            # Loss            
            loss = criterion(logits, labels)            
            val_loss.append(loss.item())
    
    val_accuracy = correct/total
                 
    return (sum(val_loss) / len(val_loss)), val_accuracy
