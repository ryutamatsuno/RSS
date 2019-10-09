"""
Measure the sampling time
"""

import numpy as np
import networkx as nx
import os

import random
import itertools
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys
from datetime import datetime

# import u_time

from sampling_util import load_G
from model_RSSs import RecursiveSampling, RecursiveSampling2


# from Model_MCMC import MCMCSampling
# from Model_PSRW import PSRW

# from sampling import gen_all_ksub


def loss_uniform(samples, pi) -> float:
    n_samples = np.sum(samples)
    freq = samples / n_samples
    loss = np.sum(np.abs(freq - pi)) / 2
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

    # load graph data
    if not '.' in data_name:
        data_name += '.edg'
    data_head = data_name.split('.')[0]
    G = load_G(data_name)

    n = len(G)
    m = len(nx.edges(G))

    # error parameter
    e = 0.05


    # load model
    if model_name == "RSS":
        sampler = RecursiveSampling(G, e)
    elif model_name == "RSS+":
        sampler = RecursiveSampling2(G, e)
    elif model_name == "MCMC":
        sampler = MCMCSampling(G, e, use_buffer=False)
    elif model_name == "PSRW":
        sampler = PSRW(G, e, use_buffer=False)
    else:
        raise ValueError("%s is not implemented" % model_name)


    print('data set:', data_name)
    print("n=", n, ", m=", len(nx.edges(G)), ", k=", k, ", e=", e)
    print("model_name:", model_name)

    # sampling start

    fstart = time.time()
    start = time.time()
    v = sampler.uniform_state_sample(k)
    t = time.time() - start
    print('sampling_time:%12.8f[s]' % t, ' sample:', v)

