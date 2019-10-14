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

from sampling_util import load_G
from models.time_RSSs import RSS, RSS2
from models.time_MCMC import MCMCSampling
from models.time_PSRW import PSRW

if __name__ == "__main__":

    # load
    data_name = sys.argv[1]
    k = int(sys.argv[2])
    model_name = sys.argv[3]
    mixing_time_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    e = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    n_samples = int(sys.argv[6]) if len(sys.argv) > 6 else 10

    # load graph data
    G = load_G(data_name + '.edg')

    n = len(G)
    m = len(nx.edges(G))

    if model_name == "RSS":
        sampler = RSS(G, e, data_name, mixing_time_ratio=mixing_time_ratio)
    elif model_name == "RSS+" or model_name == "RSS2":
        sampler = RSS2(G, e, data_name, mixing_time_ratio=mixing_time_ratio)
    elif model_name == "MCMC":
        sampler = MCMCSampling(G, e)
    elif model_name == "PSRW":
        sampler = PSRW(G, e)
    else:
        raise ValueError("%s is not implemented" % model_name)

    print('arguments;')
    print('data set         :', data_name)
    print("k                :", k)
    print("model_name       :", model_name)
    print("mixing_time_ratio:", mixing_time_ratio)
    print("e                :", e)
    print("n_samples        :", n_samples)
    print("n=", n, ", m=", len(nx.edges(G)), ", k=", k, ", e=", e)

    # sampling start

    ts = []
    for l in range(n_samples):
        t = sampler.time(k)
        ts.append(t)
        if l % int(n_samples / 10) == 0:
            print('%7d/%d %18.6f[s]' % (l, n_samples, t))

    averagetime = np.mean(ts)
    stdv = np.std(ts)
    print("Estimated Sampling time:", averagetime, ' +-', stdv, '[s]')
    print(" ~ ", u_time.time2str(averagetime))
