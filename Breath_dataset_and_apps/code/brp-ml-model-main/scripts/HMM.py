import torch

import math
"https://www.geeksforgeeks.org/artificial-intelligence/viterbi-algorithm-for-hidden-markov-models-hmms"
def CalcA():
    from main import create_train_test
    train,test = create_train_test(30) #block = 30 :(
    
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

_cached_A = None
_cached_pi = None

def get_A_pi():
    global _cached_A, _cached_pi
    if _cached_A is None or _cached_pi is None:
        _cached_A = CalcA()
        _cached_pi = CalcPI()
    return _cached_A, _cached_pi

def viterbi_algorithm(preds,states=[0, 1, 2, 3],A=None,pi=None):
    if A is None or pi is None:
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
    print(CalcA())