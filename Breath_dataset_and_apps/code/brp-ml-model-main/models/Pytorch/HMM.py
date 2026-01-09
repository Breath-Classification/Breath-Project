import torch
from main import create_train_test
if __name__ == "__main__":
    train,test = create_train_test(30) #block = 30 :(
    
    #zlicz macierz przejsc
    trues=[]
    
    for X, y in train:
        trues.append(y.cpu())
    N=4
    A_counts = torch.zeros(N, N)


    for seq in trues:  
        seq = seq.long()  
        for t in range(len(seq)-1):
            i = seq[t].item()
            j = seq[t+1].item()
            A_counts[i, j] += 1
    print(A_counts)
    A = A_counts / A_counts.sum(dim=1, keepdim=True) #macierz znormalizowana

    print(A)