import numpy as np
import networkx as nx

import random
import itertools
import math

"""
Data loading
"""


def load_G(data_name):
    G = nx.read_edgelist('data_set/' + data_name, nodetype=int, create_using=nx.Graph())

    # G.remove_edges_from(G.selfloop_edges())

    # remap the node names from 0,1,2,....,|V|-1
    mapping = {}
    nodes = sorted(nx.nodes(G))
    for i in range(len(nodes)):
        mapping[nodes[i]] = i
    G = nx.relabel_nodes(G, mapping)

    return G


"""
Basic functions
"""



def ln(x):
    return math.log(x)


def boundVk(G, k):
    """
    Upper bound of V^k
    :param G: nx.Graph
    :param k: integer, 1<k<|V|
    :return:
    """
    n = len(G)
    delta = max([nx.degree(G, n) for n in nx.nodes(G)])
    return math.factorial(k - 1) * delta ** (k - 1) * n


def binom(n, r):
    if r == 0:
        return 1
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))


def choose_one(l):
    return l[np.random.randint(0, len(l))]



def diff(x, y):
    """
    :return one element that is in x but not in y
    :param x:
    :param y:
    :return:
    """
    for u in x:
        if not u in y:
            return u


def state_merge(x, y):
    """
    return a tuple of union of elements in x and y
    :param x:
    :param y:
    :return: tuple
    """
    l = set(x).union(set(y))
    return tuple(sorted(l))


def num_edges_yields(x, y, neighbor_of_x):
    """
    number of edges that yield the state (x U y)
    :param x:
    :param y:
    :param neighbor_of_x:
    :return:
    """

    df = diff(y, x)
    m = 1
    for an in neighbor_of_x:
        if df in an:
            m += 1

    # return m * (m - 1) / 2
    return binom(m, 2)


"""
Subgraph sampling utilities
"""


def neighbor_nodes(G, s) -> set:
    """
    return the set of nodes that is adjacent to at least one node in s
    :param G: nx.Graph
    :param s: tuple of nodes
    :return: set of nodes
    """

    n = len(G)

    nb = set()
    for v in s:
        vn = nx.neighbors(G, v)
        nb = nb.union(set(vn))
        if len(nb) == n:
            break

    nb = nb.difference(set(s))
    return nb


def degree(G, s):
    """
    state degree in |s|-state graph G^{|s|}
    :param G: nx.Graph
    :param s: tuple of nodes
    :return:
    """
    d = 0
    temp = set(s)
    for x in neighbor_nodes(G, s):
        temp.add(x)
        for y in s:
            temp.discard(y)
            H = G.subgraph(temp)
            if nx.is_connected(H):
                d += 1
            temp.add(y)
        temp.discard(x)
    return d


def neighbor_states(G, s):
    """
    return neighbor state in |s|-state graph G^{|s|}
    :param s: tuple
    :return:
    """
    nei_states = []
    k = len(s)
    for x in neighbor_nodes(G, s):
        for j in range(k):
            news = list(s)
            news.append(x)
            del news[j]
            H = G.subgraph(news)
            if nx.is_connected(H):
                nei_states.append(tuple(sorted(news)))
    return nei_states


def random_next_state(neighbor_states):
    next_state = choose_one(neighbor_states)
    return next_state


def neighbor_edges(G, s):
    """
    return the set of nodes that is adjacent a node in s
    :param G:
    :param s:
    :return:
    """
    nb = []
    for v in s:
        vn = nx.neighbors(G, v)
        for n in vn:
            if n in s:
                continue
            nb.append((v, n))
    return nb


def RVE(G, k):
    """
    Random Vertex Expansion
    Sampling k-subgraph with some bias
    :param G:
    :param k:
    :return:
    """
    e = choose_one(list(G.edges()))

    s = [e[0], e[1]]

    while len(s) < k:
        ne = neighbor_edges(G, s)
        e = choose_one(ne)
        s.append(e[1])

    s = sorted(s)
    return s


def RVE2(G, k):
    """
    A variant of Random Vertex Expansion
    Sampling k-subgraph with some bias
    :param G:
    :param k:
    :return:
    """
    s = [choose_one(list(G.nodes()))]
    while len(s) < k:
        nei = list(neighbor_nodes(G, s))
        s.append(choose_one(nei))
    return tuple(sorted(s))
