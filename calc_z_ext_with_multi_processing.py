import numpy as np
import multiprocessing
import time
import functions
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.io as sio

global accept_rate

##################################################################################################################################

# SET HYPER-PARAMETERS
path = 'C:\\Users\\akash\\Documents\\MSc Project\\CSV_Files\\'
n=20
matrix = functions.generate_random_graph(n)
matrix = csr_matrix(matrix)
number_of_betas = 1000
number_of_samples = 125
mixing_time = 300
alpha = 0.85
initial_temperature = 0.01 # (inverse temperature)
beta_cutoff = 3.0
betas = functions.get_betas('lin', beta_cutoff = beta_cutoff , initial_temperature = initial_temperature, alpha = 0.85, number_points = number_of_betas)
start = time.time()
number_of_processes = list(range(0,8))
H_init,S_init = functions.calc_rand_H(matrix,1)
H_init = H_init[0]
global all_values
global all_values_list
all_values_list = []

all_values = {0: [],
              1: [],
              2: [],
              3: [],
              4: [],
              5: [],
              6: [],
              7: []}
##################################################################################################################################


def run_SA(betas, matrix, mixing_time, H_init, S_init, return_dict, pnum):

    for i in range(0,len(betas)):

        number_of_samples = 70
        b = betas[i]
        #number_of_samples = int(1000 - 0.995*i)
        H = functions.sample_gibbs_state_new(b, matrix, number_of_samples, mixing_time, H_init, S_init, return_dict,pnum)

        print(i)

        return_dict[pnum] = return_dict[pnum] + [H[0]]

if __name__ == '__main__':

    manager = multiprocessing.Manager()

    return_dict = manager.dict({0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]})

    processes = []

    for i in number_of_processes:

        target = run_SA
        args = [betas, matrix, mixing_time, H_init, S_init, return_dict, i]

        proc = multiprocessing.Process(target=target, args=args)
        processes.append(proc)
        proc.start()


    for p in processes:
        p.join()

    new_dict = return_dict

    values = []
    for i in range(0,8):

        values = pd.DataFrame(return_dict[i])
        values.to_csv(path + 'allvalues_{}.csv'.format(str(i)))


print(time.time() - start)