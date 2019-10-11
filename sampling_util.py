import numpy as np
import networkx as nx

import random
import itertools
import math

"""
Data loading
"""


def load_G(data_name):
    if data_name[-4:] != '.edg':
        data_name += '.edg'
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
    if r == 0 or n == r:
        return 1
    x = n
    for i in range(max(r,n-r)+1,n):
        x *= i
    x /= math.factorial(min(r,n-r))
    return x


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

def __removable(G, s):
    """
    return removable nodes do no think about the adding node
    :param G:
    :param s:
    :return:
    """
    rem = set()
    news = set(s)
    for y in s:
        news.discard(y)
        H = G.subgraph(news)
        if nx.is_connected(H):
            rem.add(y)
        news.add(y)
    unrem = news.difference(rem)
    return rem, unrem

def degree(G,s):
    d = 0
    news = set(s)
    do_remove, mynot_remove = __removable(G, s)

    for x in neighbor_nodes(G, s):

        # check if
        nei_x = set(nx.neighbors(G,x))
        node_connected2x = news.intersection(nei_x)

        news.add(x)
        for y in s:
            if len(node_connected2x) == 1 and (y in node_connected2x or y in mynot_remove):
                continue
            if y in do_remove:
                d += 1
                continue

            news.discard(y)
            H = G.subgraph(news)
            if nx.is_connected(H):
                d += 1
            news.add(y)
        news.discard(x)
    return d




def neighbor_states(G, s):
    """
    :param s: tuple return list of tuple
    :return: [next_s] : next states
    """
    states = []
    news = set(s)
    do_remove, mynot_remove = __removable(G, s)

    for x in neighbor_nodes(G, s):

        # check if
        nei_x = set(nx.neighbors(G, x))
        node_connected2x = news.intersection(nei_x)

        news.add(x)
        for y in s:
            if len(node_connected2x) == 1 and (y in node_connected2x or y in mynot_remove):
                continue
            if y in do_remove:
               news.discard(y)
               states.append(tuple(sorted(news)))
               news.add(y)
               continue

            news.discard(y)
            H = G.subgraph(news)
            if nx.is_connected(H):
                states.append(tuple(sorted(news)))
            news.add(y)
        news.discard(x)
    return states



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



def gen_all_ksub(G, k):
    """

    :param G:
    :param k:
    :return: list of touples
    """
    if k == 1:
        return [(n,) for n in G.nodes()]
    if k == 2:
        return [tuple(e if e[0] < e[1] else (e[1],e[0])) for e in nx.edges(G)]

    N = len(G)
    ite = itertools.combinations(np.arange(N), k)
    S = []
    nodes = np.array(G.nodes(),dtype=int)
    for v in ite:
        x = nodes[np.array(v)]
        H = G.subgraph(x)
        if nx.is_connected(H):
            S.append(tuple(sorted(x)))
    return S
