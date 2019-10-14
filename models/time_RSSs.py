import os
import math
import time
import random
import networkx as nx
import numpy as np

from models.core_RSSs import t_k, t_k2
from sampling_util import ln, binom, choose_one, RVE2, neighbor_states, degree, diff, num_edges_yields, state_merge

import u_time

from models.model_RSSs import RSS as actRSS, RSS2 as actRSS2

topdir = './samplingtime_buf'


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


class RSS:

    def __init__(self, G, e, data_name, mixing_time_ratio=1.0):
        self.G = G
        self.e = e
        self.data_name = data_name
        self.mixing_time_ratio = mixing_time_ratio
        self.delta = max([nx.degree(G, n) for n in G.nodes()])

        # buf
        self.edges = [tuple(e) if e[0] < e[1] else (e[1], e[0]) for e in G.edges()]
        self.edge_prob = [nx.degree(G, e[0]) + nx.degree(G, e[1]) - 2 for e in G.edges()]
        self.edge_prob = self.edge_prob / np.sum(self.edge_prob)
        self.edge_arange = np.arange(0, len(self.edges))

        # buffer
        self.loaded = set()
        self.tU = {}
        self.tD = {}

        self.n = len(self.G.nodes())

    def time(self, k):
        self.preload_time(k)
        return self.time_uniform_state_sample(k)

    def buf_file_name(self, k: int, uord: str) -> str:

        mkdir(topdir)

        if type(self) == RSS:
            mkdir(topdir + '/RSS')
            return topdir + '/RSS/' + self.data_name + '_' + str(self.mixing_time_ratio) + '_' + str(
                k) + uord.upper() + '.csv'
        elif type(self) == RSS2:
            mkdir(topdir + '/RSS2')
            return topdir + '/RSS2/' + self.data_name + '_' + str(self.mixing_time_ratio) + '_' + str(
                k) + uord.upper() + '.csv'

    def preload_time(self, k):
        if k in self.loaded:
            return
        for i in range(3, k):
            u_time.start('pl:%d' % k)
            self.generate_buf(i)
            print('Done:', u_time.stop('pl:%d' % k), '[s]')
            self.loaded.add(k)

    def generate_buf(self, k):
        # inside preload
        print('Preloading:', k)

        n_buf_samples = 100

        if k == 3:

            # actual time
            if type(self) == RSS:
                sampler = actRSS(self.G, self.e, mixing_time_ratio=self.mixing_time_ratio)
            elif type(self) == RSS2:
                sampler = actRSS2(self.G, self.e, mixing_time_ratio=self.mixing_time_ratio)
            else:
                raise ValueError()

            # U
            fname = self.buf_file_name(k, 'U')
            if os.path.exists(fname):
                self.tU[k] = np.loadtxt(fname, delimiter=',').tolist()
            else:
                ts = []
                for _ in range(n_buf_samples):
                    start = time.time()
                    sampler.uniform_state_sample(k)
                    t = time.time() - start
                    ts.append(t)
                np.savetxt(fname, np.array(ts), delimiter=',')
                self.tU[k] = ts

            # D2
            fname = self.buf_file_name(2, 'D')
            if os.path.exists(fname):
                self.tD[2] = np.loadtxt(fname, delimiter=',').tolist()
            else:
                ts = []
                for _ in range(n_buf_samples):
                    start = time.time()
                    sampler.degree_prop_state_sample(2)
                    t = time.time() - start
                    ts.append(t)
                np.savetxt(fname, np.array(ts), delimiter=',')
                self.tD[2] = ts

            # D3
            fname = self.buf_file_name(k, 'D')
            if os.path.exists(fname):
                self.tD[k] = np.loadtxt(fname, delimiter=',').tolist()
            else:
                ts = []
                for l in range(n_buf_samples):
                    t = self.time_degree_prop_state_sample(k)
                    ts.append(t)
                np.savetxt(fname, np.array(ts), delimiter=',')
                self.tD[k] = ts
        else:
            # U
            fname = self.buf_file_name(k, 'U')

            if os.path.exists(fname):
                self.tU[k] = np.loadtxt(fname, delimiter=',').tolist()
            else:
                ts = []
                for l in range(n_buf_samples):
                    t = self.time_uniform_state_sample(k)
                    ts.append(t)
                np.savetxt(fname, np.array(ts), delimiter=',')
                self.tU[k] = ts

            # D
            fname = self.buf_file_name(k, 'D')

            if os.path.exists(fname):
                self.tD[k] = np.loadtxt(fname, delimiter=',').tolist()
            else:
                ts = []
                for l in range(n_buf_samples):
                    t = self.time_degree_prop_state_sample(k)
                    ts.append(t)
                np.savetxt(fname, np.array(ts), delimiter=',')
                self.tD[k] = ts

        print('UniformSampling(%d)   :' % k, np.mean(self.tU[k]))
        print('DegreePropSampling(%d):' % k, np.mean(self.tD[k]))

    def t_k(self, k):
        return t_k(self.n, k, self.e, self.delta, self.mixing_time_ratio)

    def time_uniform_state_sample(self, k) -> int:
        if k in self.tU:
            return np.random.choice(self.tU[k], 1)[0]

        key = 'U' + str(k)

        if k == 2:
            u_time.start(key)
            choose_one(self.edges)
            return u_time.stop(key)

        rt = 0
        u_time.start(key)
        while True:
            u_time.pause(key)
            s = RVE2(self.G, k - 1)  # self.degree_prop_state_sample(k - 1)
            rt += self.time_degree_prop_state_sample(k - 1)
            u_time.resume(key)
            s_neighbor = neighbor_states(self.G, s)
            n = choose_one(s_neighbor)
            m = num_edges_yields(s, n, s_neighbor)
            #rt += u_time.stop()
            if random.random() < 1 / m:
                return rt + u_time.stop(key)

    def time_degree_prop_state_sample(self, k) -> float:
        if k in self.tD:
            return np.random.choice(self.tD[k], 1)[0]

        key = 'D' + str(k)
        if k == 2:
            u_time.start(key)
            x = self.edges[np.random.choice(self.edge_arange, 1, p=self.edge_prob)[0]]
            return u_time.stop(key)

        rt = 0

        # curr_s = self.uniform_state_sample(k)
        curr_s = RVE2(self.G, k)
        rt += self.time_uniform_state_sample(k)

        u_time.start(key)
        curr_d = degree(self.G, curr_s)
        rt += u_time.stop(key)

        n_samples = 100
        mixing_time = self.t_k(k)

        y = 0
        u_time.start(key)
        for _ in range(n_samples):
            if random.random() < 1/2:
                continue

            u_time.pause(key)
            # next_s = self.uniform_state_sample(k)
            next_s = RVE2(self.G, k)
            y += self.time_uniform_state_sample(k)
            u_time.resume(key)

            next_d = degree(self.G, next_s)

            if random.random() < next_d / curr_d:
                # accept
                curr_s = next_s
                curr_d = next_d

        x = u_time.stop(key) + y
        rt += (x / n_samples) * mixing_time
        return rt


class RSS2(RSS):
    """
    Only use degree_prop_sampling
    """

    def t_k(self, k):
        return t_k2(self.n, k, self.e, self.delta, self.mixing_time_ratio)

    def estimate_degree(self, s, u, v, neighbors):
        """
        estimate degree of e in G_k
        :param e:
        :param neighbors: neighbors of e[0]
        :return:
        """
        return degree(self.G, s) / num_edges_yields(u, v, neighbors)

    def time_degree_prop_state_sample(self, k) -> float:
        if k in self.tD:
            return np.random.choice(self.tD[k], 1)[0]

        key = 'D' + str(k)
        if k == 2:
            u_time.start(key)
            x = self.edges[np.random.choice(self.edge_arange, 1, p=self.edge_prob)[0]]
            return u_time.stop(key)

        rt = 0
        n_samples = 100

        # u = self.degree_prop_state_sample(k - 1)

        u = RVE2(self.G, k - 1)
        rt += self.time_degree_prop_state_sample(k - 1)

        u_time.start(key)
        neighbor_of_u = neighbor_states(self.G, u)
        v = choose_one(neighbor_of_u)
        curr_s = state_merge(u, v)
        curr_f = self.estimate_degree(curr_s, u, v, neighbor_of_u)
        rt += u_time.stop(key)



        mixing_time = self.t_k(k)
        # MH Sampling

        y = 0
        u_time.start(key)
        for _ in range(n_samples):
            if random.random() < 1/2:
                continue

            u_time.pause(key)
            u = RVE2(self.G, k - 1)
            y += self.time_degree_prop_state_sample(k - 1)
            u_time.resume(key)

            neighbor_of_u = neighbor_states(self.G, u)
            v = choose_one(neighbor_of_u)
            next_s = state_merge(u, v)
            next_f = self.estimate_degree(next_s, u, v, neighbor_of_u)

            if random.random() < min(1, next_f / curr_f):
                # accept
                curr_s = next_s
                curr_f = next_f

        x = u_time.stop(key) + y
        rt += (x / n_samples) * mixing_time
        return rt
