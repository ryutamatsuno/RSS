import math
import random
import networkx as nx

from sampling_util import ln, binom, RVE2, choose_one, neighbor_states, degree, num_edges_yields, state_merge


def tPSRW_k(n, k, e, delta, dia, mixing_time_ratio=1.0):
    if mixing_time_ratio == 0:
        return 0
    rho = 1 / 2 * math.factorial(k - 1) * (k - 1) * delta ** k * (dia + k - 2) * n
    tau = rho * (ln(k - 1) + ln(delta) + ln(binom(n, k - 1)) + ln(1 / e))
    t = int(math.ceil(tau * mixing_time_ratio))
    if t == 0:
        t = 1
    return t


class PSRW:

    def __init__(self, G, e=0.01):
        self.G = G
        self.e = e

        self.n = len(self.G.nodes())
        self.delta = max([nx.degree(G, n) for n in G.nodes()])
        self.dia = nx.diameter(self.G)

    def t_k(self, k):
        return tPSRW_k(self.n, k, self.e, self.delta, self.dia)

    def degree_prop_state_sample(self, k):
        G = self.G

        curr_s = RVE2(G, k)
        curr_neighbors = neighbor_states(self.G, curr_s)

        for _ in range(self.t_k(k)):
            if random.random() < 0.5:
                continue
            curr_s = choose_one(curr_neighbors)
            curr_neighbors = neighbor_states(self.G, curr_s)

        return curr_s

    def uniform_state_sample(self, k):

        while True:
            s = self.degree_prop_state_sample(k - 1)
            s_neighbor = neighbor_states(self.G, s)
            n = choose_one(s_neighbor)
            m = num_edges_yields(s, n, s_neighbor)
            if random.random() < 1 / m:
                return state_merge(s, n)
