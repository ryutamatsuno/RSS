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

from sampling_util import load_G

if __name__ == "__main__":
    # load
    data_name = sys.argv[1]
    k = int(sys.argv[2])
    model_name = sys.argv[3]
    mixing_time_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    e = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    n_samples = int(sys.argv[6]) if len(sys.argv) > 6 else 100

    # load edge file
    G = load_G(data_name)

    n = len(G)
    m = len(G.edges())

    print('arguments;')
    print('data set         :', data_name)
    print("k                :", k)
    print("model_name       :", model_name)
    print("mixing_time_ratio:", mixing_time_ratio)
    print("e                :", e)
    print("n_samples        :", n_samples)
    print("n=",n,"m=",len(nx.edges(G))," k=",k)

    model_name = sys.argv[3]
    if model_name == "RSS":
        sampler = RSS(G, e, mixing_time_ratio=mixing_time_ratio)
    elif model_name == "RSS+" or model_name == "RSS2":
        sampler = RSS2(G, e, mixing_time_ratio=mixing_time_ratio)
    else:
        raise ValueError("%s is not implemented"%model_name)



    start = time.time()

    # obtain sample

    # Simply
    # samples = sampler.uniform_state_sample(k, n_samples=n_samples, only_accepted=False)

    # More efficient
    done = 0
    samples = []
    while done < n_samples:
        vs = sampler.uniform_state_sample(k, n_samples=int(k * (n_samples - done)), only_accepted=True)
        for v in vs:
            done += 1
            samples.append(v)
            if done == n_samples:
                break
        if done == 0:
            continue

        t = time.time() - start
        et = t / done * n_samples
        print('%7d/%d %s estimated: %s' % (done, n_samples, u_time.time2str(t), u_time.time2str(et)))

    spent_time = time.time() - start
    print("over all time:" + u_time.time2str(spent_time))

    print('Obtained %d-subgraphs'%k)
    for v in samples[:5]:
        print(v)
    print('....etc')




