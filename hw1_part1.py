import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import random

import Timer
random.seed(0)


class ModelData:
    """The class reads 5 files as specified in the init function, it creates basic containers for the data.
    See the get functions for the possible options, it also creates and stores a unique index for each user and movie
    """

    def __init__(self, dataset, time):
        """Expects data set file with index column (train and test) """
        self.init_graph = pd.read_csv(dataset)

        self.train = self.init_graph[(self.init_graph['time'] <= time)]
        self.test = self.init_graph[(self.init_graph['time'] > time)]

        self.train_x = self.train[['source', 'target']]
        self.train_y = self.train[['rating']]
        self.test_x = self.test[['source', 'target']]
        self.test_y = self.test[['rating']]

# Part A1


def compute_histogram(dataset):
    data = pd.read_csv("data_students.txt")
    df = data[['source', 'target']].groupby('source').nunique()
    df['target'].plot.hist(ylim=(0, 50))

    df = data[['source', 'target']].groupby('source')['target'].nunique().reset_index(name='count')
    df_hist = df.groupby('count')['source'].count().reset_index(name='hist_value')
    df_hist.plot(x='count', y='hist_value', kind='scatter', logx=True, logy=True)
    plt.show()

    x = np.log(np.array(df_hist['count']))
    y = np.log(np.array(df_hist['hist_value']))
    a = np.polyfit(x, y, 1)
    y_1 = [(a[0]*i + a[1]) for i in x]
    fig = plt.figure()

    ax = fig.gca()
    ax.scatter(np.exp(x), np.exp(y), c='blue')
    ax.plot(np.exp(x), np.exp(y_1), c='red')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()

    # bow tie part
    data_source = data['source'].tolist()
    data_target = data['target'].tolist()
    data_intersection = list(set(data_source) & set(data_target))
    difference_left = set(data_source) - set(data_target)
    difference_right = set(data_target) - set(data_source)

    print(len(difference_left))
    print(len(data_intersection))
    print(len(difference_right))
    print(len(difference_left.union(data_intersection, difference_right)))
    print('a value =', np.exp(a[1]))
    print('c value =', -a[0])

# for Part A2


def create_unweighted_G_t(train, time):
    return nx.DiGraph(train)

# Part A2


def G_features(G, time):
    G_t = create_unweighted_G_t(G.train, 0)
    biggest_scc_graph = nx.DiGraph(max(nx.strongly_connected_component_subgraphs(G_t), key=len))
    graph_size = biggest_scc_graph.number_of_nodes()
    reversed_scc_graph = biggest_scc_graph.reverse()

    paths = {}
    closeness = {}
    betweenness = dict.fromkeys(biggest_scc_graph, 0.0)

    for v in reversed_scc_graph.nodes():
        paths[v] = nx.single_source_shortest_path_length(biggest_scc_graph, v)
        closeness[v] = (graph_size - 1) / sum(nx.single_source_shortest_path_length(reversed_scc_graph, v).values())

    for v in biggest_scc_graph.nodes():
        S, P, paths = all_shortest_paths_bfs(biggest_scc_graph, v)
        betweenness = compute_betweenness(betweenness, S, P, paths, v)

    n = 1.0 / ((graph_size - 1) * (graph_size - 2))

    for v in betweenness:
        betweenness[v] *= n

    return {'a': closeness, 'b': betweenness}

# for Part A2


def compute_betweenness(betweenness, S, P, paths, source):
    delta = dict.fromkeys(S, 0)
    while S:
        v = S.pop()
        coeff = (1.0 + delta[v]) / paths[v]
        for u in P[v]:
            delta[u] += paths[u] * coeff
        if v != source and v in betweenness:
            betweenness[v] += delta[v]
    return betweenness

# Part B


def run_k_iterations(graph, N, mode='undirected unweighted'):
    edges = []
    for i in range(N):
        edge_probabilities = compute_probabilities(graph, mode)
        for v, value in edge_probabilities.items():
            for u, prob in value.items():
                if should_probability(prob):
                    edges.append((v, u))
                    if mode == 'undirected weighted':
                        graph.add_edge(v, u, weight='weak')
                    else:
                        graph.add_edge(v, u)
    return edges

# for Part B


def create_unweighted_H_t(train, time):
    return nx.Graph(train)

# for Part B


def create_weighted_H_t(train, time):
    H_t = nx.Graph()
    directed_graph = nx.DiGraph(train)
    for edge in directed_graph.edges():
        if H_t.has_edge(edge[0], edge[1]):
            H_t.add_edge(edge[0], edge[1], weight='strong')
        else:
            H_t.add_edge(edge[0], edge[1], weight='weak')
    return H_t


# for part B
def calc_error(predictions, test, mode='undirected unweighted'):
    precision, recall = 0, 0
    data_list = tuple(map(tuple, test.values.tolist()))
    intersections = set(predictions) & set(data_list)
    recall = len(intersections) / len(test)
    precision = len(intersections) / len(predictions)
    return (precision, recall)


# for Part B
def should_probability(prob):
    return prob > random.random()

# for Part B


def compute_probabilities(graph, mode='undirected unweighted'):
    edge_probabilities = defaultdict(dict)
    if mode == 'directed':
        shortest_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    for v in graph.nodes():
        if mode == 'directed':
            S, P, shortest_paths = all_shortest_paths_bfs(graph, v)
        for u in nx.non_neighbors(graph, v):
            if mode == 'undirected unweighted':
                common_neighbors_size = len(list(nx.common_neighbors(graph, v, u)))
                edge_probabilities[v][u] = 1 - pow(float(0.97), common_neighbors_size)
            elif mode == 'undirected weighted':
                m = 0
                n = 0
                for common_neighbor in nx.common_neighbors(graph, v, u):
                    if graph[v][common_neighbor]['weight'] == 'strong':
                        m += 1
                    else:
                        n += 1
                edge_probabilities[v][u] = 1 - (pow(float(0.96), m) * pow(float(0.98), n))
            elif mode == 'directed':
                if u in shortest_lengths[v]:
                    shortest_length = shortest_lengths[v][u]
                    if shortest_length <= 4:
                        num_of_paths = shortest_paths[u]
                        edge_probabilities[v][u] = min(1, num_of_paths / (math.pow(5, shortest_length)))
    return edge_probabilities


def all_shortest_paths_bfs(graph, source):
    S = []
    P = {}
    for v in graph:
        P[v] = []
    # we put sigma[v]=0 for every v in the graph
    sigma = dict.fromkeys(graph, 0.0)
    path_lengths = {}
    sigma[source] = 1.0
    path_lengths[source] = 0
    Q = [source]
    while Q:  # we use BFS to find the shortest paths
        v = Q.pop(0)
        S.append(v)
        v_length = path_lengths[v]
        sigma_v = sigma[v]
        for u in graph[v]:
            if u not in path_lengths:
                Q.append(u)
                path_lengths[u] = v_length + 1
            if path_lengths[u] == v_length + 1:  # count shortest path if found
                sigma[u] += sigma_v
                P[u].append(v)  # track predecessors
    return S, P, sigma
