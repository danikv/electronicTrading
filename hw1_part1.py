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


def create_unweighted_H_t(train, time):
    return nx.Graph(train)


def create_weighted_H_t(train, time):
    H_t = nx.Graph()
    directed_graph = nx.DiGraph(train)
    for edge in directed_graph.edges():
        # add edges with weights to H_t
        if H_t.has_edge(edge[0], edge[1]):
            H_t.add_edge(edge[0], edge[1], weight='strong')
        else:
            H_t.add_edge(edge[0], edge[1], weight='weak')
    return H_t


def create_unweighted_G_t(train, time):
    return nx.DiGraph(train)


def calc_error(predictions, test, mode='undirected unweighted'):
    precision, recall = 0, 0
    data_list = tuple(map(tuple, test.values.tolist()))
    intersections = set(predictions) & set(data_list)
    recall = len(intersections) / len(test)
    precision = len(intersections) / len(predictions)
    return (precision, recall)


def G_features(G, time):
    G_t = create_unweighted_G_t(G.train, 0)
    biggest_scc = nx.DiGraph(max(nx.strongly_connected_component_subgraphs(G_t), key=len))
    size = biggest_scc.number_of_nodes()
    reversed_scc = biggest_scc.reverse()

    shortest_paths = {}
    closeness_dict = {}
    betweenness_dict = dict.fromkeys(biggest_scc, 0.0)

    for node in reversed_scc.nodes():
        shortest_paths[node] = nx.single_source_shortest_path_length(biggest_scc, node)
        closeness_dict[node] = (size - 1) / sum(nx.single_source_shortest_path_length(reversed_scc, node).values())

    for node in biggest_scc.nodes():
        S, P, sigma = find_shortest_paths(biggest_scc, node)
        betweenness_dict = compute_betweenness(betweenness_dict, S, P, sigma, node)

    scale = 1.0 / ((size - 1) * (size - 2))
    for node in betweenness_dict:
        betweenness_dict[node] *= scale

    return {'a': closeness_dict, 'b': betweenness_dict}


def compute_betweenness(betweenness_dict, S, P, sigma, source):
    delta = dict.fromkeys(S, 0)
    while S:
        node = S.pop()
        coefficients = (1.0 + delta[node]) / sigma[node]
        for second_node in P[node]:
            delta[second_node] += sigma[second_node] * coefficients
        if node != source and node in betweenness_dict:
            betweenness_dict[node] += delta[node]
    return betweenness_dict


def calculate_probabilities(graph, mode='undirected unweighted'):
    probabilities_to_add_edge = defaultdict(dict)
    if mode == 'directed':
        shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    for node in graph.nodes():
        if mode == 'directed':
            shortest_paths = find_shortest_paths(graph, node)[2]
        for second_node in nx.non_neighbors(graph, node):
            if mode == 'undirected unweighted':
                common_neighbors_size = len(list(nx.common_neighbors(graph, node, second_node)))
                probabilities_to_add_edge[node][second_node] = 1 - pow(float(0.97), common_neighbors_size)
            elif mode == 'undirected weighted':
                m = 0
                n = 0
                for neighbor in nx.common_neighbors(graph, node, second_node):
                    if graph[node][neighbor]['weight'] == 'strong':
                        m += 1
                    else:
                        n += 1
                probabilities_to_add_edge[node][second_node] = 1 - (pow(float(0.96), m) * pow(float(0.98), n))
            elif mode == 'directed':
                if second_node in shortest_path_lengths[node]:
                    shortest_path_distance = shortest_path_lengths[node][second_node]
                    if shortest_path_distance <= 4:
                        all_paths_count = shortest_paths[second_node]
                        probabilities_to_add_edge[node][second_node] = min(1, all_paths_count / (math.pow(5, shortest_path_distance)))
    return probabilities_to_add_edge


def run_k_iterations(graph, N, mode='undirected unweighted'):
    edges_to_add = []
    for i in range(N):
        probabilities_to_add_edge = calculate_probabilities(graph, mode)
        for node, value in probabilities_to_add_edge.items():
            for second_node, probability in value.items():
                if probability > random.random():
                    edges_to_add.append((node, second_node))
                    if mode == 'undirected weighted':
                        graph.add_edge(node, second_node, weight='weak')
                    else:
                        graph.add_edge(node, second_node)
    return edges_to_add


def compute_distribution(dataset):
    data = pd.read_csv("data_students.txt")
    dataframe = data[['source', 'target']].groupby('source').nunique()
    dataframe['target'].plot.hist(ylim=(0, 50))

    dataframe = data[['source', 'target']].groupby('source')['target'].nunique().reset_index(name='count')
    dataframe_hist = dataframe.groupby('count')['source'].count().reset_index(name='hist_value')
    dataframe_hist.plot(x='count', y='hist_value', kind='scatter', logx=True, logy=True)
    plt.show()

    x = np.log(np.array(dataframe_hist['count']))
    y = np.log(np.array(dataframe_hist['hist_value']))
    a = np.polyfit(x, y, 1)
    y_1 = [(a[0]*i + a[1]) for i in x]
    fig = plt.figure()

    ax = fig.gca()
    ax.scatter(np.exp(x), np.exp(y), c='blue')
    ax.plot(np.exp(x), np.exp(y_1), c='red')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()

    # bow tie
    source_data = data['source'].tolist()
    target_data = data['target'].tolist()
    intersection = list(set(source_data) & set(target_data))
    left_data = set(source_data) - set(target_data)
    right_data = set(target_data) - set(source_data)

    print(len(left_data))
    print(len(intersection))
    print(len(right_data))
    print(len(left_data.union(intersection, right_data)))


def find_shortest_paths(graph, source):
    S = []
    P = {}
    for node in graph:
        P[node] = []
    sigma = dict.fromkeys(graph, 0.0) # sigma[node]=0 for every node in G
    distances = {}
    sigma[source] = 1.0
    distances[source] = 0
    Q = [source]
    while Q: # we use BFS to find the shortest paths
        node = Q.pop(0)
        S.append(node)
        Dv = distances[node]
        sigma_node = sigma[node]
        for second_node in graph[node]:
            if second_node not in distances:
                Q.append(second_node)
                distances[second_node] = Dv + 1
            if distances[second_node] == Dv + 1:  # if we found the shortest path, we count it
                sigma[second_node] += sigma_node
                P[second_node].append(node)  # we save the predecessors
    return S, P, sigma
