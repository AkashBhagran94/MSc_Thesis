import numpy as np
import multiprocessing
import random
import scipy.io as sio
import time
import math


def get_random_s(n):  # create an arbitrary set of spins up to n
    s = np.array([[-1, 1][np.random.randint(2)] for i in range(0, n)])
    return s


def calc_rand_H(A, n):  # Randomly configures all spins and calculates H given adjacency matrix. Completeley Random.
    H = []
    for i in range(0, n):
        s = []
        while len(s) < A.shape[0]:
            x = random.randint(-1, 1)
            if not x == 0:
                s.append(x)
        s = np.array(s)
        H.append(s.T @ A @ s)

    return np.array(H), s


def calc_rand_spin_H(A, S_init):  # Randomly configures a single spin and calculates H given adjacency matrix

    s = S_init
    r = random.randint(0, A.shape[0] - 1)
    s[r] = -s[r]

    H = s.T @ A @ s

    return H, s


def generate_random_graph(n):  # creates a random adjacency matrix with weights 0 or 1

    a = np.array([[0, 1][np.random.randint(2)] for i in range(0, n ** 2)]).reshape(n, n)
    a[np.tril_indices(n, 0)] = 0
    a = a + a.T

    return a


def find_all_combinations(n):  # returns a 2D array of all combinations of spins from n vertices

    l = list(range(0, 2 ** n))
    l = ["{0:b}".format(i) for i in l]

    b0 = []
    for i in l:
        if len(i) < n:
            for j in range(0, n - len(i)):
                i = '0' + i

            b0.append(i)
        else:
            b0.append(i)

    b2 = []
    for i in range(0, len(b0)):
        b1 = []
        for char in b0[i]:
            b1.append(int(char))
        b2.append(np.array(b1))

    b2 = np.array(b2)
    b2[b2 == 0] = -1

    return b2


def find_minimum_energy(matrix):  # finds the minimum energy of the matrix

    r = find_all_combinations(matrix.shape[0])
    E, V = (np.einsum('ij,jk,ki->i', r, matrix, r.T), r)

    return min(E), E, V


def calc_z_bruteforce(betas, matrix):  # Calculates Z brute-force

    n = matrix[0].shape[0]

    Zb = []

    for beta in betas:

        z = 0

        for j in find_all_combinations(n):
            H = j.T @ matrix @ j
            z = z + np.exp(-beta * H)

        Zb.append(z)

    return np.array(Zb)


def sample_gibbs_state(beta, A, number_of_samples, mixing_time, H_init,
                       S_init):  # simulated annealing algorithm to sample Gibbs state

    H_old = H_init
    S_old = S_init
    H_samples = []
    S_samples = []
    global accept_rate
    global glob_H

    accept_rate = []
    glob_H = []

    accept_rate_indexed = []

    for i in range(0, number_of_samples):

        count = 0
        # start = time.time()

        for j in range(0, mixing_time):

            H_new, S_new = calc_rand_spin_H(A, S_old)

            T = H_old - H_new

            if T >= 0:
                H_old = H_new
                S_old = S_new
                count = count + 1

            if T < 0 and np.random.uniform() < np.exp(-beta * (np.abs(H_old - H_new))):
                H_old = H_new
                S_old = S_new
                count = count + 1

        accept_rate_indexed.append(np.array(count / mixing_time))
        H_samples.append(H_old)
        glob_H.append(H_samples)
        S_samples.append(S_old)

    accept_rate.append((np.mean(accept_rate_indexed), beta))

    H_samples = np.array(H_samples)

    return (H_samples, S_samples)


def calc_z2z1(beta2, beta1, number_of_samples, mixing_time, A):  # Calcualtes the ratio between zbk+1 and zbk+1
    # from the average of the samples
    H_init, S_init = calc_rand_H(A, 1)
    SGS = sample_gibbs_state(beta1, A, number_of_samples, mixing_time, H_init, S_init)
    samples = SGS[0]
    spins = SGS[1]

    return np.mean(np.exp(-(beta2 - beta1) * samples)), samples, spins


def calc_Zbk(betas, number_of_samples, mixing_time,
             matrix):  # Calculate partition functions up to m, returns numpy array

    z = 1

    Zb = []

    for i in range(0, len(betas) - 1):
        ratio, samples, spins = calc_z2z1(betas[i + 1], betas[i], number_of_samples, mixing_time, matrix)
        z = z * ratio
        Zb.append(z)

    return np.array(Zb)


def get_betas(schedule, beta_cutoff, initial_temperature, alpha,
              number_points):  # produces a vector of betas from 0 to cutoff
    # given schedule and number of points

    if schedule == 'lin':
        betas = np.linspace(0, beta_cutoff, number_points)

    if schedule == 'log':
        xf = 1.0 / (np.exp(initial_temperature / beta_cutoff) - 1)
        x = np.linspace(0, xf, number_points)
        betas = initial_temperature / np.log((1 / x) + 1)

    if schedule == 'geo':
        xf = math.log(initial_temperature / beta_cutoff, alpha)  # = 3
        x = np.linspace(0, xf, number_points)
        betas = initial_temperature * (alpha ** (-x))

    return betas


def plot_LB(betas, z, n):  # plots lower bound

    x = np.linspace(0, 1, 300)

    E = []

    for i in x:

        e = []

        for k in range(0, len(betas) - 1):
            c = ((1 / betas[k]) * (-np.log(z[k]) - n * i))
            e.append(c)

        E.append((1 / n) * max(e))

    return x, E


#################################################################################################################################
# Redefined Simulated annealing algorithm which collects samples after a burn-in period, rather than collecting one sample from
# after a burn-in.
#################################################################################################################################

def sample_gibbs_state_new(beta, A, number_of_samples, mixing_time, H_init, S_init):
    # simulated annealing algorithm to sample Gibbs state. This time, samples are collected by first reaching stability after the burn
    # - in time

    H_old = H_init
    S_old = S_init
    H_samples = []
    S_samples = []

    A = csr_matrix(A)

    global accept_rate
    global glob_H

    count = 0
    iteration = 0
    t = 0

    while (len(H_samples) <= number_of_samples):

        iteration += 1

        H_new, S_new = calc_rand_spin_H(A, S_old)
        T = H_old - H_new

        if T >= 0:
            H_old = H_new
            S_old = S_new
            count = count + 1
            if iteration >= mixing_time:
                H_samples.append(H_old)
                t += 1
                # print(t)

        if T < 0 and np.random.uniform() < np.exp(-beta * (np.abs(H_old - H_new))):
            H_old = H_new
            S_old = S_new
            count = count + 1
            if iteration >= mixing_time:
                H_samples.append(H_old)
                t += 1
                # print(t)

    accept_rate.append(count / iteration)

    return (np.array(H_samples), S_samples)


def calc_z2z1_new(beta2, beta1, number_of_sampless, mixing_time,
                  matrix):  # Calcualtes the ratio between zbk+1 and zbk, uses the new sampling method

    H_init, S_init = calc_rand_H(matrix, 1)
    SGS = sample_gibbs_state_new(beta1, matrix, number_of_samples, mixing_time, H_init, S_init)
    samples = SGS[0]
    spins = SGS[1]

    return np.mean(np.exp(-(beta2 - beta1) * (samples))), samples, spins


def calc_Zbk_new(betas, number_of_samples, mixing_time,
                 matrix):  # Calculate partition functions up to m, returns numpy array

    global salvage
    salvage = []

    z = 1

    Zb = []

    for i in range(0, len(betas) - 1):
        ratio, samples, spins = calc_z2z1_new(betas[i + 1], betas[i], number_of_samples, mixing_time, matrix)
        z = z * ratio
        Zb.append(z)
        salvage = Zb
        print(i)

    return np.array(Zb)


start = time.time()

def f_sum(a, b):
    return a + b

data = [(1, 1)]

inp = [(np.linspace(0,0.2,100),)]




if __name__ == '__main__':

    with multiprocessing.Pool(processes=3) as pool:
        results = pool.starmap(calc_zbk, inp)
    print(results)

print(time.time() - start)