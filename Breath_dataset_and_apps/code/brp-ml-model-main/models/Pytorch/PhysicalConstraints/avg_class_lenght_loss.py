import torch
import torch.nn.functional as F

def min_duration_loss(emissions, target_class, min_len):

    probs = F.softmax(emissions, dim=-1)          
    pred = probs[..., target_class]               

    loss = torch.tensor(0.0, device=emissions.device)
    batch_size = pred.shape[0]

    for seq in pred:
        current_len = 0.0
        
        i=0
        for p in seq:
            weight = torch.sigmoid((p - 0.5) * 10)
            
            if weight > 0.5:          
                current_len += p
                i+=1
            else:                     
                if current_len > 0 and current_len < min_len:
                   # print("Adding loss for segment of length", current_len.item(), "real length", i)
                    loss += (min_len - current_len)
                current_len = 0.0

        if current_len > 0 and current_len < min_len:
            loss += (min_len - current_len)

    return loss / batch_size