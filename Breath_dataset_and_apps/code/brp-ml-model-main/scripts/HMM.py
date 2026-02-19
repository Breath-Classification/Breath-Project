import torch
import json
import math
"https://www.geeksforgeeks.org/artificial-intelligence/viterbi-algorithm-for-hidden-markov-models-hmms"

filename = "ConfigA_PI"
def CalcA():
    from main import config_dataloaders
    train,test = config_dataloaders(block_size=30,batch_size=32,target=0)#block = 30 :(
    
    trues=[]
    
    for X, y in train:
        trues.append(y.cpu())
        
    N=4 #liczba klas
    A_counts = torch.zeros(N, N)
    
    #zlicz macierz przejsc
    for seq in trues:  
        seq = seq.long()  
        for t in range(len(seq)-1):
            i = seq[t].item()
            j = seq[t+1].item()
            A_counts[i, j] += 1
    A = A_counts / A_counts.sum(dim=1, keepdim=True) #macierz znormalizowana
    return A

def CalcPI():
    N=4 #liczba klas
    pi = torch.ones(N) / N
    return pi

def save_A_PI(A,pi):
    config= {
        "A":A,
        "pi":pi
    }
    config["A"] = A.cpu().tolist()
    config["pi"] = pi.cpu().tolist()
    with open(f"SavedData/{filename}.json","w",encoding="utf-8") as f:
        json.dump(config,f,indent=4)

def get_A_pi():
    
    with open(f"SavedData/{filename}.json","r",encoding="utf-8") as f:
        config=json.load(f)
    
    A = torch.tensor(config["A"], dtype=torch.float32)
    PI = torch.tensor(config["pi"], dtype=torch.float32)
    return A,PI

def viterbi_algorithm(preds,states=[0, 1, 2, 3], A=None, pi=None):
    
    A, pi = get_A_pi()
    
    V = [{}]
    path = {}
    T = preds.shape[0]

    for s in states:
        V[0][s] = math.log(pi[s]) + preds[0, s]
        path[s] = [s]
        
    for t in range(1, T):
        V.append({})
        newpath = {}
        for s in states:
            prob, prev_s = max(
                [(V[t-1][s0] + math.log(A[s0, s]) + preds[t, s], s0) for s0 in states]
            )
            V[t][s] = prob
            newpath[s] = path[prev_s] + [s]
        path = newpath
    prob, state = max([(V[T-1][s], s) for s in states])
    return path[state]


if __name__ == "__main__":
    A =CalcA()
    PI=CalcPI()
    save_A_PI(A,PI)