"""
RSS that can obtain multiple samples in a single run

Full scratch
"""

import math
import numpy as np
import networkx as nx

import random

from sampling_util import ln, binom, neighbor_states, degree, num_edges_yields, state_merge


class RSS:

    def __init__(self, G, e=0.01, mixing_time_ratio=1.0):
        self.G = G
        self.e = e
        self.mixing_time_ratio = mixing_time_ratio

        # buf
        self.edges = [tuple(e) if e[0] < e[1] else (e[1], e[0]) for e in G.edges()]
        self.edge_prob = [nx.degree(G, e[0]) + nx.degree(G, e[1]) - 2 for e in G.edges()]
        self.edge_prob = self.edge_prob / np.sum(self.edge_prob)
        self.edge_arange = np.arange(0, len(self.edges))

        self.n = len(G)
        self.delta = max([nx.degree(G, n) for n in G.nodes()])


    def t_k(self, k):
        e = self.e
        delta = self.delta
        n = self.n
        rho = 2 * delta * k
        tau = rho * (ln(binom(n, k)) + ln(k) + ln(delta) + ln(1 / e))
        t = int(math.ceil(tau * self.mixing_time_ratio))
        return t

    def degree_prop_state_sample(self, k, n_samples) -> list:
        if k == 2:
            ids = np.random.choice(self.edge_arange, n_samples, p=self.edge_prob)
            return [self.edges[i] for i in ids]

        curr_s = self.uniform_state_sample(k, n_samples)
        curr_d = [degree(self.G, s) for s in curr_s]

        # MH Sampling
        for _ in range(self.t_k(k)):
            loop_mask = np.where(np.random.rand(n_samples) < 0.5, 1, 0)  # index 0 do not change

            next_s = self.uniform_state_sample(k, n_samples)
            next_d = [degree(self.G, next_s[i]) if loop_mask[i] == 1 else 0 for i in range(n_samples)]

            accept_mask = np.where(np.random.rand(n_samples) < np.divide(next_d, curr_d), 1, 0)
            mask = accept_mask

            ids = np.where(mask == 1)[0]

            def update(i):
                curr_s[i] = next_s[i]
                curr_d[i] = next_d[i]
                return None

            list(map(update, ids))

        return curr_s

    def uniform_state_sample(self, k: int, n_samples: int, only_accepted=False) -> [tuple]:
        if k == 2:
            ids = np.random.choice(self.edge_arange, n_samples)
            return [self.edges[i] for i in ids]

        accepted_samples = []
        while True:
            num_need = n_samples - len(accepted_samples)
            s = self.degree_prop_state_sample(k - 1, n_samples=num_need)
            s_neighbor = [neighbor_states(self.G, x) for x in s]
            n = [ns[np.random.randint(0, len(ns), 1)[0]] for ns in s_neighbor]
            H = [state_merge(s[i], n[i]) for i in range(num_need)]
            m = [num_edges_yields(s[i], n[i], s_neighbor[i]) for i in range(num_need)]

            accept_mask = np.where(np.random.rand(num_need) < np.reciprocal(m), 1, 0)

            for i in range(num_need):
                if accept_mask[i] == 1:
                    accepted_samples.append(H[i])

            if only_accepted or len(accepted_samples) == n_samples:
                return accepted_samples


class RSS2(RSS):

    def t_k(self, k):
        e = self.e
        delta = self.delta
        n = self.n
        rho = 2 * delta * k
        tau = rho * (ln(binom(n, k)) + 3 * ln(k) + ln(delta) + ln(1 / e))
        t = int(math.ceil(tau * self.mixing_time_ratio))
        if t == 0:
            t = 1
        return t

    def degree_prop_state_sample(self, k, n_samples, top_mask=None) -> [tuple]:
        if k == 2:
            ids = np.random.choice(self.edge_arange, n_samples, p=self.edge_prob)
            return [self.edges[i] for i in ids]

        if top_mask is None:
            top_mask = np.ones(n_samples)

        s = self.degree_prop_state_sample(k - 1, n_samples=n_samples, top_mask=top_mask)
        s_neighbor = [neighbor_states(self.G, s[i]) if top_mask[i] == 1 else 0 for i in range(n_samples)]
        n = [s_neighbor[i][np.random.randint(0, len(s_neighbor[i]), 1)[0]] if top_mask[i] == 1 else 0 for i in
             range(n_samples)]

        curr_s = [state_merge(s[i], n[i]) if top_mask[i] == 1 else 0 for i in range(n_samples)]
        curr_f = [degree(self.G, curr_s[i]) / num_edges_yields(s[i], n[i], s_neighbor[i]) if top_mask[i] == 1 else 1 for
                  i in range(n_samples)]

        # MH Sampling
        for _ in range(self.t_k(k)):
            loop_mask = np.where(np.random.rand(n_samples) < 0.5, 1, 0)  # index 0 do not change
            loop_mask = np.multiply(loop_mask, top_mask)

            s = self.degree_prop_state_sample(k - 1, n_samples=n_samples, top_mask=loop_mask)

            s_neighbor = [neighbor_states(self.G, s[i]) if loop_mask[i] == 1 else 0 for i in range(n_samples)]
            n = [s_neighbor[i][np.random.randint(0, len(s_neighbor[i]), 1)[0]] if loop_mask[i] == 1 else 0 for i in
                 range(n_samples)]

            next_s = [state_merge(s[i], n[i]) if loop_mask[i] == 1 else 0 for i in range(n_samples)]
            next_f = [
                degree(self.G, next_s[i]) / num_edges_yields(s[i], n[i], s_neighbor[i]) if loop_mask[i] == 1 else 0 for
                i in range(n_samples)]

            accept_mask = np.where(np.random.rand(n_samples) < np.divide(next_f, curr_f), 1, 0)

            mask = accept_mask
            ids = np.where(mask == 1)[0]

            def f(i):
                curr_s[i] = next_s[i]
                curr_f[i] = next_f[i]
                return None

            list(map(f, ids))

        return curr_s
