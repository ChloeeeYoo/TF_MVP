import numpy as np
from collections import Counter

#The num. of array size
H = 1024
W = 1024 
N = 256 #The num. of PE
M = 8 #The num. of SA
SETS = int(N / M)

ARR_MEM = 128 

#Prune rate 
P_R = 0.6
#VECT_SIZE = 16

#initialize
l = 0 #A num. of SA by PE
num_set = 0

# weight array
w_q = np.random.randint(1,100, size=(H,W))
print(f'weight array')
print(w_q)


# Input array (single batch)
in_a = np.arange(0,H+10000)
in_a = np.transpose(in_a)
print(f'input array')
print(in_a)


def rc_format(MASK_TYPE, VECT_SIZE, w_q):
    
    global W, H

    ### mask array (Threshold pruning)
    #fine-grained version
    if MASK_TYPE == "fine":
        print(f'+++++++++++ fine-grained +++++++++++++')
        #compute threshold value
        num_w = w_q.size
        kth = int(num_w * P_R)
        threshold = np.partition(w_q.ravel(), kth)[kth]
        print(threshold)
        mask = w_q.copy()
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        print(mask)
    #vector version
    if MASK_TYPE == "vect":
        print(f'+++++++++++ vector +++++++++++++')
        ma = np.transpose(w_q)
        ma = np.reshape(ma, (-1, VECT_SIZE))
        ma = np.mean(ma, axis=1, keepdims=True)
        #compute threshold value
        num_w = ma.size
        kth = int(num_w * P_R)
        threshold = np.partition(ma.ravel(), kth)[kth]
        print(threshold)
        mask = ma.copy()
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        mask = np.repeat(mask, VECT_SIZE, axis=1)
        mask = np.reshape(mask, (-1, w_q.shape[0]))
        mask = np.transpose(mask)
        print(mask)
    
    #Generate random mask version
    if MASK_TYPE == "rand":
        mask = np.random.choice([0,1], size=(H,W), p=[P_R, 1-P_R])
    
    
    w_q1 = w_q * mask
    print(f'pruned weight array')
    print(w_q1)
    
    ### Implement RCSC format
    mask_r = mask.copy()
    
    ## Reduce to 1/4 for column vector (mask)
    G_UNIT = VECT_SIZE
    if MASK_TYPE == "vect":
        temp = mask_r.copy()
        temp = np.delete(temp, np.s_[::2], axis=0)
        temp = np.delete(temp, np.s_[::2], axis=0)
        temp = np.delete(temp, np.s_[::2], axis=0)
        temp = np.delete(temp, np.s_[::2], axis=0)
        mask_r = temp.copy()
        H = int(H / G_UNIT)
    
    sum_rows = mask_r.sum(axis=1)
    
    # Descending 
    d_row = np.sort(sum_rows)[::-1]
    d_idx = np.argsort(sum_rows)[::-1]
    
    # SA-RCSC format - Flip by n size
    rc_idx = []
    new_idx = []
    print(mask_r.shape[0])
    #temp1_idx = d_idx[0:int(mask_r.shape[0]/2)-1]
    temp1_idx = d_idx[0:int(H/2)-1]
    #temp2_idx = d_idx[int(mask_r.shape[0]/2)-1:mask_r.shape[0]]
    temp2_idx = d_idx[int(H/2)-1:H]
    temp2_idx = temp2_idx[::-1]
    rc_idx.extend(temp1_idx)
    rc_idx.extend(temp2_idx)

    print()
    #print(rc_idx)
    print()

    for d in range(H):
    #for d in range(mask_r.shape[0]):
        if(d % N == 0) and (d / N % 2 == 0):
              rc1_idx = rc_idx[d:d+N]
              # SA-RCSC format - Flip by m size
              for e in range(N):
                  if(e % SETS == 0) and (e / SETS % 2 == 0):
                         sa1_idx = rc1_idx[e:e+SETS]
                         new_idx.extend(sa1_idx) 
        
                  if(e % SETS == 0) and (e / SETS % 2 == 1):
                         sa2_idx = rc1_idx[e:e+SETS]
                         sa2_idx = sa2_idx[::-1]  #flip
                         new_idx.extend(sa2_idx)
              rc1_idx = []
    
        if(d % N == 0) and (d / N % 2 == 1):
              rc2_idx = rc_idx[d:d+N]
              rc2_idx = rc2_idx[::-1] #flip
              # SA-RCSC format
              for f in range(N):
                  if(f % SETS == 0) and (f / SETS % 2 == 0):
                         sa3_idx = rc2_idx[f:f+SETS]
                         new_idx.extend(sa3_idx) 
        
                  if(f % SETS == 0) and (f / SETS % 2 == 1):
                         sa4_idx = rc2_idx[f:f+SETS]
                         sa4_idx = sa4_idx[::-1] #flip
                         new_idx.extend(sa4_idx)
    
              rc2_idx = []
    
    #print(new_idx)   
    
    ## SA-RCSC weight output
    w_q_r = w_q1.copy()
    ## Reduce to 1/4 for column vector (weight)
    G_UNIT = VECT_SIZE
    if MASK_TYPE == "vect":
        temp = w_q_r.copy()
        temp = np.delete(temp, np.s_[::2], axis=0)
        temp = np.delete(temp, np.s_[::2], axis=0)
        temp = np.delete(temp, np.s_[::2], axis=0)
        temp = np.delete(temp, np.s_[::2], axis=0)
        w_q_r = temp.copy()
    
    w_q_r = w_q_r[new_idx]
    
    print(f'rearranged weight array')
    print(w_q_r)
    prune_rate = np.count_nonzero(w_q_r) / w_q_r.size
    print(f'pruning rate: {1 - prune_rate}') 
    
    ### Implement PE indexing
    rows = []
    PE = []
    
    for a in range(N):
        PE.append([])
        #PE[a].append(0)
    
    for j in range(W):
       for i in range(H):
          #for SA type
          for a in range(N):
             l = a % M
             rows.append([])
             rows[l].append(len(PE[a]))
             #rows[l].append(PE[a].size)
             #print(f'rows')
             #print(rows[l])
          
          #print(f'++++++++++++++++++++')
          num_set = (i % N) % M
          idx = rows[num_set].index(min(rows[num_set]))
          sel = idx * M + num_set
         
          if(w_q_r[i,j] != 0):
             PE[sel].append([j, int(w_q_r[i,j])])
             #print(f'PE[{sel}]')
             #print(PE[sel])
             #PE[sel].append([j , w_q[i,j]], axis=0)
          
          #initiate rows
          for b in range(M):
             rows[b] = []
    

    ## Print PE result 
    temp2 = [] 
    print(f'########-----------------------########')
    for c in range(N):
        #print(f'PE[{c}]')
        #print(f'The num. of element by PE: {len(PE[c])}')
        #print(PE[c])
        temp2.append(len(PE[c]))
        min_len = min(temp2)

    print(f'########-----------------------########')
    
    sum_rep = 0
    cor = []
    

    ## Counting clock cycles per PE    
    call = 0
    bk = 0
    m = 0
    PE_OUT = []
    PE_temp = []

    count  = 0
    for a in range(N):
        PE_OUT.append([])
 
    while True:
        mem_input = in_a[m : m + ARR_MEM]

        for j in range(N):   # N: the num. of PE
          if PE[j] == []:   #whether NULL or not
               PE_OUT[j].append('n')
               bk += 1
               continue        
          elif np.any(mem_input == PE[j][0][0]): #There is an element
               PE_OUT[j].append(PE[j][0][0])
               PE[j].pop(0)
          else:   # stall
          #elif np.any(mem_input != PE[j][0][0]): #There is an element
               PE_OUT[j].append(-1)
               print(j)
               print(PE[j][0][0])
               print(mem_input)
               print(PE_OUT[j])
          
          if PE[j] != []:
             PE_temp.append(PE[j][0][0])  
        
        if bk == N:   #fine the timing for break (all PE has NULL)
            break
        
        try:
            if (np.min(PE_temp) != mem_input[0]   
            and np.min(PE_temp) != mem_input[1]   
            and np.min(PE_temp) != mem_input[2]   
            and np.min(PE_temp) != mem_input[3]):   #increase +1 for input array
              m += 4
        except ValueError: 
            pass
        
        call = 0
        bk = 0
        PE_temp = []
    
    temp3 = [] 
    temp4 = [] 
    for j in range(N):
        """
        print(f'total num: {len(PE_OUT[j])}')
        print(f'NULL num: {PE_OUT[j].count("n")}')
        print(f'stall num: {PE_OUT[j].count(-1)}')
        print(f'Total clock of PE[{j}]: {len(PE_OUT[j]) - PE_OUT[j].count("n")}')
        """
        temp3.append(len(PE_OUT[j]) - PE_OUT[j].count("n"))
        temp4.append(PE_OUT[j].count(-1))
        max_cycle = max(temp3)
        min_cycle = min(temp3)
        max_stall = max(temp4)
    
    print(f'$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print(f'Max clock cycle: {max_cycle}')
    print(f'Min clock cycle: {min_cycle}')
    print(f'stall num: {PE_OUT[j].count(-1)}')
    print(f'stall num: {max_stall}')


           
    """
    for i in range(min_len):
        for j in range(N):
            cor.append([])
            cor[i].append(PE[j][i][0])
        max_rep = Counter(cor[i])[max(Counter(cor[i]))]
        sum_rep += max_rep 
    cor_rate = sum_rep / (N*min_len)
    print(f'The coincidence rate of column value: {cor_rate}') 
    """

def main():
    
    #rc_format("fine", 1, w_q)
    #rc_format("vect", 4, w_q)
    #rc_format("vect", 8, w_q)
    rc_format("vect", 16, w_q)
    
if __name__ == "__main__":
    main() 
