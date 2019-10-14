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

from models.buffed_RSSs import RSS, RSS2

from sampling_util import load_G, gen_all_ksub

def loss_uniform(samples,pi) -> float:
    n_samples = np.sum(samples)
    freq = samples/n_samples
    loss = np.sum( np.abs(freq - pi))/2
    return float(loss)

if __name__ == "__main__":
    # load
    data_name = sys.argv[1]
    k = int(sys.argv[2])
    model_name = sys.argv[3]
    mixing_time_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    e = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    generating_ratio = int(sys.argv[6]) if len(sys.argv) > 6 else 100

    # load edge file
    G = load_G(data_name)

    n = len(G)
    m = len(G.edges())

    all_ksub = gen_all_ksub(G, k)
    n_omega = len(all_ksub)

    # generate ratio for each subgraph

    n_samples = generating_ratio * n_omega
    pi = 1 / n_omega

    print('arguments;')
    print('data set         :', data_name)
    print("k                :", k)
    print("model_name       :", model_name)
    print("mixing_time_ratio:", mixing_time_ratio)
    print("e                :", e)
    print("generating_ratio :", generating_ratio)
    print("n=",n,"m=",len(nx.edges(G))," k=",k)
    print("actual number of k-subgraph:", n_omega)
    print("n_samples:", n_samples)

    model_name = sys.argv[3]
    if model_name == "RSS":
        sampler = RSS(G, e, preload_k=k, mixing_time_ratio=mixing_time_ratio)
    elif model_name == "RSS+" or model_name == "RSS2":
        sampler = RSS2(G, e, preload_k=k, mixing_time_ratio=mixing_time_ratio)
    else:
        raise ValueError("%s is not implemented"%model_name)

    print("pi:",pi)

    counts = np.zeros(n_omega)

    start = time.time()
    done = 0
    # counting
    while done < n_samples:
        vs = sampler.uniform_state_sample(k, n_samples=min(int(k*(n_samples-done)),1000000),only_accepted=True)
        for i in vs:
            counts[i] += 1
            done = int(np.sum(counts))
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








