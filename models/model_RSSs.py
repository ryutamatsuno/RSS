import math
import random
import networkx as nx
import numpy as np

from models.mixing_time import t_k, t_k2
from sampling_util import ln, binom, choose_one, degree, neighbor_states, diff, state_merge, num_edges_yields


class RSS:

    def __init__(self, G, e=0.01, mixing_time_ratio=1.0):
        self.G = G
        self.e = e
        self.mixing_time_ratio = mixing_time_ratio

        self.delta = max([nx.degree(G, n) for n in G.nodes()])

        # for uniform 2-subgraph sampling
        self.edges = [tuple(e) if e[0] < e[1] else (e[1], e[0]) for e in G.edges()]

        # for degree-prop 2-subgraph sampling
        self.edge_prob = [nx.degree(G, e[0]) + nx.degree(G, e[1]) - 2 for e in G.edges()]
        self.edge_prob = self.edge_prob / np.sum(self.edge_prob)
        self.edge_arange = np.arange(0, len(self.edges))

        self.n = len(self.G.nodes())

    def t_k(self, k):
        return t_k(self.n, k, self.e, self.delta, self.mixing_time_ratio)

    def degree_prop_state_sample(self, k):
        if k == 2:
            return self.edges[np.random.choice(self.edge_arange, 1, p=self.edge_prob)[0]]

        curr_s = self.uniform_state_sample(k)
        curr_d = degree(self.G, curr_s)

        # MH Sampling
        for _ in range(self.t_k(k)):
            if random.random() < 1 / 2:
                continue

            next_s = self.uniform_state_sample(k)
            next_d = degree(self.G, next_s)

            if random.random() < next_d / curr_d:
                # accept
                curr_s = next_s
                curr_d = next_d
        return curr_s

    def uniform_state_sample(self, k):
        if k == 2:
            return choose_one(self.edges)

        while True:
            s = self.degree_prop_state_sample(k - 1)
            # print(s)
            s_neighbor = neighbor_states(self.G, s)
            n = choose_one(s_neighbor)
            m = num_edges_yields(s, n, s_neighbor)
            if random.random() < 1 / m:
                return state_merge(s, n)


class RSS2(RSS):

    def t_k(self, k):
        return t_k2(self.n, k, self.e, self.delta, self.mixing_time_ratio)

    def estimate_degree(self, s, u, v, neighbors):
        return degree(self.G, s) / num_edges_yields(u, v, neighbors)

    def degree_prop_state_sample(self, k):
        if k == 2:
            return self.edges[np.random.choice(self.edge_arange, 1, p=self.edge_prob)[0]]

        u = self.degree_prop_state_sample(k - 1)
        neighbor_of_u = neighbor_states(self.G, u)
        v = choose_one(neighbor_of_u)
        curr_s = state_merge(u, v)
        curr_f = self.estimate_degree(curr_s, u, v, neighbor_of_u)

        # MH Sampling
        for _ in range(self.t_k(k)):
            if random.random() < 1 / 2:
                continue

            u = self.degree_prop_state_sample(k - 1)
            neighbor_of_u = neighbor_states(self.G, u)
            v = choose_one(neighbor_of_u)
            next_s = state_merge(u, v)
            next_f = self.estimate_degree(next_s, u, v, neighbor_of_u)

            if random.random() < min(1, next_f / curr_f):
                # accept
                curr_s = next_s
                curr_f = next_f
        return curr_s
