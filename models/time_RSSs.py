import os
import math
import time
import random
import networkx as nx
import numpy as np


from sampling_util import ln, binom, choose_one, RVE2, neighbor_states, degree, diff, num_edges_yields

import u_time

from models.model_RSSs import RSS as actRSS, RSS2 as actRSS2


def state_marge(x, y):
    """
    :param x:
    :param y:
    :return: tuple
    """
    if isinstance(x, int):
        return (x, y) if x < y else (y, x)
    l = set(x).union(set(y))
    return tuple(sorted(l))


topdir = './samplingtime_buf'


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


class RSS:

    def __init__(self, G, e, data_name):
        self.G = G
        self.e = e
        self.data_name = data_name
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

    def time(self, k):
        self.preload_time(k)
        return self.time_uniform_state_sample(k)

    def buf_file_name(self, k: int, uord: str) -> str:

        mkdir(topdir)

        if type(self) == RSS:
            mkdir(topdir + '/RSS')
            return topdir + '/RSS/' + self.data_name + '_' + str(k) + uord.upper() + '.csv'
        elif type(self) == RSS2:
            mkdir(topdir + '/RSS2')
            return topdir + '/RSS2/' + self.data_name + '_' + str(k) + uord.upper() + '.csv'

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
            # U
            fname = self.buf_file_name(k, 'U')
            if os.path.exists(fname):
                self.tU[k] = np.loadtxt(fname, delimiter=',').tolist()
            else:
                # actual time
                if type(self) == RecursiveSampling:
                    sampler = actRSS(self.G, self.e)
                elif type(self) == RecursiveSampling2:
                    sampler = actRSS2(self.G, self.e)
                ts = []
                for l in range(n_buf_samples):
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
                # actual time
                if type(self) == RecursiveSampling:
                    sampler = actRSS(self.G, self.e)
                elif type(self) == RecursiveSampling2:
                    sampler = actRSS2(self.G, self.e)
                ts = []
                for l in range(n_buf_samples):
                    start = time.time()
                    sampler.degree_prop_state_sample(2)
                    t = time.time() - start
                    ts.append(t)
                np.savetxt(fname, np.array(ts), delimiter=',')
                self.tD[2] = ts

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

        print('UniformSampling(%d)   :'%k, np.mean(self.tU[k]))
        print('DegreePropSampling(%d):'%k, np.mean(self.tD[k]))

    def t_k(self, k):

        e = self.e
        delta = self.delta
        n = len(self.G.nodes())
        rho = 2 * delta * k
        tau = rho * (ln(binom(n, k)) + ln(k) + ln(delta) + ln(1 / e))
        t = int(math.ceil(tau))
        return t

    def time_uniform_state_sample(self, k) -> int:
        if k in self.tU:
            return np.random.choice(self.tU[k], 1)[0]

        if k == 2:
            u_time.start()
            choose_one(self.edges)
            return u_time.stop()

        G = self.G
        rt = 0
        while True:
            s = RVE2(G, k - 1)  # self.degree_prop_state_sample(k - 1)
            rt += self.time_degree_prop_state_sample(k - 1)
            u_time.start()
            s_neighbor = neighbor_states(self.G, s)
            n = choose_one(s_neighbor)
            m = num_edges_yields(s, n, s_neighbor)
            rt += u_time.stop()
            if random.random() < 1 / m:
                return rt

    def time_degree_prop_state_sample(self, k) -> float:
        if k in self.tD:
            return np.random.choice(self.tD[k], 1)[0]

        if k == 2:
            u_time.start()
            x = self.edges[np.random.choice(self.edge_arange, 1, p=self.edge_prob)[0]]
            return u_time.stop()

        rt = 0

        # curr_s = self.uniform_state_sample(k)
        curr_s = RVE2(self.G, k)
        rt += self.time_uniform_state_sample(k)
        u_time.start()
        curr_d = degree(self.G, curr_s)
        rt += u_time.stop()

        n_samples = 100
        mixing_time = self.t_k(k)

        y = 0
        u_time.start()
        for _ in range(n_samples):
            # if random.random() < 1/2:
            #     continue

            u_time.pause()
            # next_s = self.uniform_state_sample(k)
            next_s = RVE2(self.G, k)
            y += self.time_uniform_state_sample(k)
            u_time.resume()

            next_d = degree(self.G, next_s)

            if random.random() < next_d / curr_d:
                # accept
                curr_s = next_s
                curr_d = next_d

        x = u_time.stop() + y
        rt += (x / n_samples) * mixing_time / 2
        return rt


class RSS2(RSS):
    """
    Only use degree_prop_sampling
    """

    def t_k(self, k):

        e = self.e
        delta = self.delta
        n = len(self.G.nodes())
        rho = 2 * delta * k
        tau = rho * (ln(binom(n, k)) + 3 * ln(k) + ln(delta) + ln(1 / e))
        t = int(math.ceil(tau))
        return t

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

        # if k == 1:
        #     u_time.start()
        #     x = np.random.choice(self.nodes, 1, p=self.node_prob)[0]
        #     return u_time.stop()
        if k == 2:
            u_time.start()
            x = self.edges[np.random.choice(self.edge_arange, 1, p=self.edge_prob)[0]]
            return u_time.stop()

        rt = 0
        n_samples = 100

        # u = self.degree_prop_state_sample(k - 1)

        u = RVE2(self.G, k - 1)
        rt += self.time_degree_prop_state_sample(k - 1)

        u_time.start()
        neighbor_of_u = neighbor_states(self.G, u)
        v = choose_one(neighbor_of_u)
        curr_s = state_marge(u, v)
        curr_f = self.estimate_degree(curr_s, u, v, neighbor_of_u)
        u_time.stop()

        mixing_time = self.t_k(k)
        # MH Sampling

        y = 0
        u_time.start()
        for _ in range(n_samples):
            # if random.random() < 1/2:
            #     continue
            u_time.pause()
            # u = self.degree_prop_state_sample(k - 1)
            u = RVE2(self.G, k - 1)
            y += self.time_degree_prop_state_sample(k - 1)
            u_time.resume()

            neighbor_of_u = neighbor_states(self.G, u)
            v = choose_one(neighbor_of_u)
            next_s = state_marge(u, v)
            next_f = self.estimate_degree(next_s, u, v, neighbor_of_u)

            if random.random() < min(1, next_f / curr_f):
                # accept
                curr_s = next_s
                curr_f = next_f

        x = u_time.stop() + y
        rt += (x / n_samples) * mixing_time / 2
        return rt
