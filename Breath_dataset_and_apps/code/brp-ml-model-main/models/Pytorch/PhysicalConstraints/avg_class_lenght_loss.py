import torch

def min_duration_loss(emissions, target_class, min_len):

    pred = torch.argmax(emissions, dim=-1)  # [B, T]

    loss = 0.0
    batch_size = pred.shape[0]

    for seq in pred:
        current_len = 0
        
        for label in seq:
            if label == target_class:
                current_len += 1
            else:
                if current_len > 0 and current_len < min_len:
                    loss += (min_len - current_len)
                current_len = 0

        if current_len > 0 and current_len < min_len:
            loss += (min_len - current_len)

    return loss / batch_size