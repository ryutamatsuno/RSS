"""
Check the uniformity of the algorithm
"""
import os
import sys
import random
import time
import math

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import u_time
from datetime import datetime

from models.multi_RSSs import RSS, RSS2

from sampling_util import load_G, gen_all_ksub

def loss_uniform(samples,pi) -> float:
    n_samples = np.sum(samples)
    freq = samples/n_samples
    loss = np.sum( np.abs(freq - pi))/2
    return float(loss)

if __name__ == "__main__":

    # load arguments
    if len(sys.argv) < 3:
        print('please run like "python3 main.py ba10 3 RSS"')
        exit(0)

    # load
    data_name = sys.argv[1]
    k = int(sys.argv[2])
    model_name = sys.argv[3]
    mixing_time_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    # load edge file
    G = load_G(data_name)

    n = len(G)
    m = len(G.edges())

    all_ksub = gen_all_ksub(G, k)
    n_omega = len(all_ksub)

    #indexmap
    ksub2index = {}
    for i in range(n_omega):
        ksub2index[all_ksub[i]] = i


    # generate ratio for each subgraph
    generating_ratio = 100
    e = 0.05

    n_samples = generating_ratio * n_omega
    pi = 1 / n_omega

    print('data set:', data_name)
    print("n=",n,"m=",len(nx.edges(G))," k=",k)
    print("actual number of k-subgraph:", n_omega)
    print("generating_ratio :", generating_ratio)
    print("n_samples:", n_samples)
    print("e           :", e)

    model_name = sys.argv[3]
    if model_name == "RSS":
        sampler = RSS(G, e, mixing_time_ratio=mixing_time_ratio)
    elif model_name == "RSS2":
        sampler = RSS2(G, e, mixing_time_ratio=mixing_time_ratio)
    else:
        raise ValueError("%s is not implemented"%model_name)

    print("pi:",pi)

    counts = np.zeros(n_omega)

    start = time.time()
    done = 0
    # counting
    while done < n_samples:
        vs = sampler.uniform_state_sample(k, n_samples=min(int(k*(n_samples-done)),1000000),only_accepted=True)
        for v in vs:
            i = ksub2index[v]
            counts[i] += 1
            done += 1
            if done == n_samples:
                break

        t = time.time() - start
        et = t / done * n_samples
        loss = loss_uniform(counts, pi)
        print('%7d/%d %s estimated: %s loss:%7.5f' % (done, n_samples, u_time.time2str(t), u_time.time2str(et), loss))

    spent_time = time.time() - start
    print("over all time:" + u_time.time2str(spent_time))

    # freq
    freq = counts / n_samples

    # loss
    loss = loss_uniform(counts,pi)
    print("loss:",loss)
    print("should be smaller than e:",e)








