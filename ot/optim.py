import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def regularized_sinkhorn_knapp(r, c, M, reg, pre_P=None, stop_th=1e-9, check_freq=100, max_iters=1000, show_iter_info=True):
    
    if show_iter_info:
        start_time = time.time()
        curr_time = time.time()
    r = torch.tensor(r, dtype=torch.double, device=device)
    c = torch.tensor(c, dtype=torch.double, device=device)
    M = torch.tensor(M, dtype=torch.double, device=device)
    N = len(M)
    
    K = torch.exp(-reg * M)
    if pre_P != None:
        K = pre_P.to(device=device)
        
    u = torch.ones(N, dtype=torch.double, device=device)
    v = torch.ones(N, dtype=torch.double, device=device)
    for iters in range(max_iters):
        if iters % check_freq == 0:
            P = torch.matmul(torch.matmul(torch.diag(u), K), torch.diag(v))
            error = torch.sum(torch.abs(torch.sum(P, dim=0) - c)) + torch.sum(torch.abs(torch.sum(P, dim=1) - r))
            if show_iter_info:
                iter_time = time.time() - curr_time
                curr_time = time.time()
                print('iters: ' + str(iters) + '; error : ' + "{0:.4g}".format(error.item()) \
                      + "; elpse per check: " + "{0:.4g}".format(iter_time))
            if error < stop_th:
                break
        if iters % 2 == 1:
            u = torch.div(r, torch.matmul(K, v))
            u = torch.nan_to_num(u, nan=1)
        else:
            v = torch.div(c, torch.matmul(torch.t(K), u))
            v = torch.nan_to_num(v, nan=1)
        
    P = torch.matmul(torch.matmul(torch.diag(u), K), torch.diag(v))
    P = torch.div(P, torch.sum(K))
    cost = torch.sum(P * M)
    if show_iter_info:
        print("total time: " + str(time.time() - start_time))
   
    return P, cost