import math
import random
import networkx as nx

import u_time
from sampling_util import ln, binom, RVE2, choose_one, neighbor_states, degree, num_edges_yields, state_merge
from models.mixing_time import tPSRW_k

class PSRW:

    def __init__(self, G, e=0.01):
        self.G = G
        self.e = e

        self.n = len(self.G.nodes())
        self.delta = max([nx.degree(G, n) for n in G.nodes()])
        self.dia = nx.diameter(self.G)

    def t_k(self, k):
        return tPSRW_k(self.n, k, self.e, self.delta, self.dia)


    def uniform_state_sample(self, k):

        while True:
            s = self.degree_prop_state_sample(k - 1)
            s_neighbor = neighbor_states(self.G, s)
            n = choose_one(s_neighbor)
            m = num_edges_yields(s, n, s_neighbor)
            if random.random() < 1 / m:
                return state_merge(s, n)


    def time_degree_prop_state_sample(self, k)-> int:
        # running time
        rt = 0
        n_times = 1000
        mixingtime = self.t_k(k)

        u_time.start()
        curr_s = RVE2(self.G, k)
        curr_neighbors = neighbor_states(self.G, curr_s)
        rt += u_time.stop()

        u_time.start()
        for _ in range(n_times):
            if random.random() < 0.5:
                continue
            curr_s = choose_one(curr_neighbors)
            curr_neighbors = neighbor_states(self.G, curr_s)
        rt += u_time.stop() * mixingtime / n_times

        return rt

    def time(self, k):
        """
        :param G:
        :param k:
        :return: tuple
        """
        rt = 0
        key = 'U'
        u_time.start(key)
        while True:
            u_time.pause(key)
            s = RVE2(self.G, k-1)
            rt += self.time_degree_prop_state_sample(k-1)
            u_time.resume(key)
            s_neighbor = neighbor_states(self.G, s)
            n = choose_one(s_neighbor)
            m = num_edges_yields(s, n, s_neighbor)
            if random.random() < 1 / m:
                return rt + u_time.stop(key)
