import torch
import torch.nn.functional as F

def inhale_exhale_correlation(emissions, factor_min, factor_max):

    probs = F.softmax(emissions, dim=-1)  
 

    loss = torch.tensor(0.0, device=emissions.device)
    batch_size, T, C = emissions.shape

    for seq_idx in range(batch_size):
        current_breath_in = 0.0
        current_breath_out = 0.0

        for t in range(T):
            p_in = probs[seq_idx, t, 2]   
            p_out = probs[seq_idx, t, 0]  

    
            if current_breath_in > 0 and current_breath_out > 0:
                correlation = current_breath_out / current_breath_in 
                if correlation < factor_min:
                    loss += (factor_min - correlation)
                elif correlation > factor_max:
                    loss += (correlation - factor_max)
                
        
                current_breath_in = 0.0
                current_breath_out = 0.0

    return loss / batch_size