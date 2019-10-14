import math
import random
import networkx as nx

#from sampling import binom, ln, RVE2, choose_one, boundVk
from sampling_util import ln, binom, RVE2, choose_one, boundVk



def tMCMC_k(n, k, e, delta, dia, mixing_time_ratio=1.0):
    if mixing_time_ratio == 0:
        return 0
    rho = 1/2 * math.factorial(k) * delta ** k * (dia + k - 1) * n
    tau = rho * (ln(binom(n, k)) + ln(1 / e))
    t = int(math.ceil(tau * mixing_time_ratio))
    if t == 0:
        t = 1
    return t

class MCMCSampling:

    def __init__(self, G, e = 0.01):
        """
        :param G:
        :param k:
        :param num_pararell:
        :param start_state:
        :param all_states:
        """

        self.G = G
        self.e = e

        self.n = len(self.G.nodes())
        self.delta = max([nx.degree(G,n) for n in G.nodes()])
        self.dia = nx.diameter(self.G)



    def t_k(self,k):
        return tMCMC_k(self.n, k, self.e, self.delta, self.dia)


    def uniform_state_sample(self, k):

        curr_s = RVE2(self.G,k)
        curr_neighbors = neighbor_states(self.G, curr_s)


        for _ in range(self.t_k(k)):
            if random.random() < 0.5:
                continue

            next_s = choose_one(curr_neighbors)
            next_neighbors = neighbor_states(self.G, next_s)

            current_degree = len(curr_neighbors)
            next_degree = len(next_neighbors)

            if random.random() < current_degree / next_degree:
                curr_s = next_s
                curr_neighbors = next_neighbors

        return curr_s

