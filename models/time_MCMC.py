import math
import random
import networkx as nx

# from sampling import binom, ln, RVE2, choose_one, boundVk
from sampling_util import ln, binom, RVE2, choose_one, boundVk, neighbor_states
from models.mixing_time import tMCMC_k


class MCMCSampling:

    def __init__(self, G, e=0.01):

        self.G = G
        self.e = e

        self.n = len(self.G.nodes())
        self.delta = max([nx.degree(G, n) for n in G.nodes()])
        self.dia = nx.diameter(self.G)

    def t_k(self, k):
        return tMCMC_k(self.n, k, self.e, self.delta, self.dia)

    def time(self, k):
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
            if random.random() < 1 / 2:
                continue

            next_s = choose_one(curr_neighbors)
            next_neighbors = neighbor_states(self.G, next_s)

            current_degree = len(curr_neighbors)
            next_degree = len(next_neighbors)

            if random.random() < current_degree / next_degree:
                curr_s = next_s
                curr_neighbors = next_neighbors
        rt += u_time.stop() * mixingtime / n_times

        return rt
