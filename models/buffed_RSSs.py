
import math
import time
import numpy as np
import networkx as nx
import itertools

import random

from models.core_RSSs import t_k, t_k2
from sampling_util import binom, ln, state_merge, gen_all_ksub, gen_gm

def num_all_ksub(G, k)-> int:
    if k == 1:
        return len(nx.nodes(G))
    if k == 2:
        return len(nx.edges(G))

    N = len(G)
    ite = itertools.combinations(np.arange(N), k)
    S = 0
    nodes = np.array(G.nodes(),dtype=int)
    for v in ite:
        x = nodes[np.array(v)]
        H = G.subgraph(x)
        if nx.is_connected(H):
            S += 1
    return S

class Gkp:
    def __init__(self,G:nx.DiGraph):
        self.G = G
        n = len(G)

        self.neighbors = [None for _ in range(n)]
        self.degrees = [None for _ in range(n)]
        self.yields = [None for _ in range(n)]

        self.merges = [None for _ in range(n)]

        self.__s2i = [None for _ in range(n)]
        self.__i2s = [None for _ in range(n)]

        self.loaded = set()

    def s2i(self,s):
        k = len(s)
        return self.__s2i[k][s]

    def i2s(self,i,k):
        return self.__i2s[k][i]

    def preload(self, k:int):
        if k in self.loaded:
            return

        print("pre-loading:",k)
        start = time.time()
        G = self.G
        G_M = gen_gm(G, k)

        nodes = list(nx.nodes(G_M))
        n_nodes = len(nodes)

        # indexing
        s2i = {}
        i2s = {}
        for i,s in enumerate(nodes):
            s2i[s] = i
            i2s[i] = s
        self.__s2i[k] = s2i
        self.__i2s[k] = i2s

        # neighbors and neighbors
        nbrs = [None for _ in range(n_nodes)]
        degs = - np.ones(n_nodes,dtype=np.int)
        for i, n in enumerate(nodes):
            b = list(nx.neighbors(G_M, n))
            nbrs[i] = np.array(list(map(lambda x:s2i[x],b)))
            degs[i] = len(b)
        # list
        self.neighbors[k] = nbrs
        # np.array
        self.degrees[k] = degs

        # num_edge_yields
        yld = - np.ones(n_nodes,dtype=np.float)
        for i,s in enumerate(nodes):
            H = nx.induced_subgraph(G,s)
            m = num_all_ksub(H, k-1)
            yld[i] = m*(m-1)/2
        self.yields[k] = yld


        print('...time:%8.3f[s]'%(time.time()-start))
        self.loaded.add(k)



    def state_merge(self,k:int,e:(int,int)) -> int:
        if e[0] > e[1]:
            e = (e[1],e[0])

        if not self.merges[k] is None:
            return self.merges[k][e]

        G = self.G
        G_M = gen_gm(G, k)

        # num_edge_yields
        mrg = {}
        for f in G_M.edges():
            x = state_merge(f[0], f[1])
            x = self.s2i(x)
            e0 = self.s2i(f[0])
            e1 = self.s2i(f[1])
            key = (e0, e1) if e0 < e1 else (e1, e0)
            mrg[key] = x
        self.merges[k] = mrg

        return mrg[e]

    def num_edge_yields(self,k:int,i) -> int:
        yld = self.yields[k]
        return yld[i]

    def random_neighbor(self, k:int, i, top_mask):
        if isinstance(i,np.ndarray):
            #y = np.zeros(len(i))
            y = np.array([ np.random.choice(self.neighbors[k][i[j]],1)[0] if top_mask[j] == 1 else -2 for j in range(len(i))])
            return y

            #return np.array(list(map(lambda j: np.random.choice(self.neighbors[k][j],1)[0] if top_mask[j] == 1 else -2, i)), dtype=np.int)
            # nbs = self.neighbor_states(k,i)
            # return np.array(list(map(lambda i: np.random.choice(i, 1)[0],nbs)), dtype=np.int)
        return np.random.choice(self.neighbors[k][i],1)[0]

    def neighbor_states(self,k:int, i):
        if isinstance(i,np.ndarray):
            nbs = self.neighbors[k]
            return list(map(lambda i: nbs[i],i))
        return self.neighbors[k][i]

    def degree(self,k:int, i) -> int:
        return self.degrees[k][i]



class RSS:

    def __init__(self,G, e = 0.01, preload_k = 5, mixing_time_ratio = 1.0):
        """
        :param G:
        :param o: the number of samples to obtain in a single run
        :param e:
        """
        self.G = G
        self.Gk = Gkp(G)
        self.e = e
        self.mixing_time_ratio = mixing_time_ratio

        # preload
        self.Gk = Gkp(G)
        for i in range(2, preload_k+1):
            self.Gk.preload(i)

        edges = [tuple(e) if e[0] < e[1] else (e[1], e[0]) for e in G.edges()]
        edges = list(map(self.Gk.s2i, edges))
        self.edges = np.array(edges)
        self.edge_prob = np.array([nx.degree(G,e[0]) + nx.degree(G,e[1]) - 2 for e in G.edges()])
        self.edge_prob = self.edge_prob/ np.sum(self.edge_prob)

        self.n = len(G)
        self.delta = max([nx.degree(G,n) for n in G.nodes()])


    def t_k(self, k):
        return t_k(self.n, k, self.e, self.delta, self.mixing_time_ratio)


    def degree_prop_state_sample(self, k, n_samples) -> np.ndarray:
        # if k == 1:
        #     return np.random.choice(self.nodes, n_samples, p=self.node_prob)
        if k == 2:
            return np.random.choice(self.edges, n_samples, p=self.edge_prob)


        curr_s = self.uniform_state_sample(k, n_samples)
        curr_d = self.Gk.degree(k, curr_s)

        # MH Sampling
        for _ in range(self.t_k(k)):
            loop_mask = np.where(np.random.rand(n_samples) < 0.5, 1, 0) # index 0 do not change

            next_s = self.uniform_state_sample(k, n_samples)
            next_d = self.Gk.degree(k, next_s)

            accept_mask = np.where(np.random.rand(n_samples) < np.divide(next_d, curr_d), 1, 0) # index 0 do no change
            mask = np.multiply(loop_mask,accept_mask)

            ids = (mask==1)
            curr_s[ids] = next_s[ids]
            curr_d[ids] = next_d[ids]

        return curr_s


    def uniform_state_sample(self, k:int, n_samples:int, only_accepted=False) -> np.ndarray:
        """
        :param k:
        :param n_samples:
        :param only_accepted:
        :return:
        """
        if k == 2:
            return np.random.choice(self.edges, n_samples)

        accepted_samples = np.empty(0,dtype=np.int)
        while True:
            num_need = n_samples - len(accepted_samples)
            s = self.degree_prop_state_sample(k - 1, n_samples=num_need)
            n = self.Gk.random_neighbor(k-1,s, np.ones(num_need))
            x = np.array([self.Gk.state_merge(k-1, (s[i],n[i])) for i in range(num_need)],dtype=np.int) # x = s U n
            m = self.Gk.num_edge_yields(k, x)

            accept_mask = np.where(np.random.rand(num_need) < np.reciprocal(m),1,0) # 1 is accept

            accepted_samples = np.hstack((accepted_samples, x[accept_mask==1]))
            if only_accepted or len(accepted_samples) == n_samples:
                return accepted_samples



class RSS2(RSS):
    def t_k(self, k):
        return t_k2(self.n, k, self.e, self.delta, self.mixing_time_ratio)

    def degree_prop_state_sample(self, k, n_samples, top_mask=None) -> np.ndarray:
        if k == 2:
            return np.random.choice(self.edges, n_samples, p=self.edge_prob)

        if top_mask is None:
            top_mask = np.ones(n_samples)

        u = self.degree_prop_state_sample(k-1, n_samples, top_mask=top_mask)
        v = self.Gk.random_neighbor(k-1, u, top_mask)

        curr_s = np.array([self.Gk.state_merge(k-1, (u[i],v[i])) if top_mask[i] == 1 else -1 for i in range(n_samples)],dtype=np.int)
        curr_f = self.Gk.degree(k, curr_s) / self.Gk.num_edge_yields(k, curr_s)


        # MH Sampling
        for _ in range(self.t_k(k)):

            loop_mask = np.where(np.random.rand(n_samples) < 0.5, 1, 0) # index 0 do not change
            loop_mask = np.multiply(loop_mask, top_mask)

            u = self.degree_prop_state_sample(k - 1, n_samples, top_mask=loop_mask)
            v = self.Gk.random_neighbor(k - 1, u, loop_mask)

            next_s = np.array([self.Gk.state_merge(k - 1, (u[i], v[i])) if loop_mask[i] == 1 else -1 for i in range(n_samples)], dtype=np.int)
            next_f = self.Gk.degree(k, next_s) / self.Gk.num_edge_yields(k, next_s)



            accept_mask = np.where(np.random.rand(n_samples) < np.divide(next_f, curr_f), 1, 0)
            # index 0 do no change

            mask = np.multiply(loop_mask, accept_mask)
            ids = mask==1
            curr_s[ids] = next_s[ids]
            curr_f[ids] = next_f[ids]

        return curr_s