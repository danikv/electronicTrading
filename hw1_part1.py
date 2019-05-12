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


def update_node_betweenness(node, graph, shortest_paths_length, size):
    betweenness = 0
    for x in graph.nodes():
        if node != x:
            for y in graph.nodes():
                if x != y and node != y:
                    shortest_paths_with_node = 0
                    shortest_paths = 0
                    shortest_path_length_without_node = shortest_paths_length[x][y]
                    shortest_path_length_with_node = shortest_paths_length[x][node] + shortest_paths_length[node][y]
                    all_shortest_paths = list(nx.all_shortest_paths(graph, x, y))
                    if shortest_path_length_with_node == shortest_path_length_without_node:
                        for path in all_shortest_paths:
                            if node in path:
                                shortest_paths += 1
                                shortest_paths_with_node += 1
                            else:
                                shortest_paths += 1
                    else:
                        shortest_paths += len(list(all_shortest_paths))
                    betweenness += (shortest_paths_with_node / shortest_paths)
    return (1 / ((size - 1) * (size - 2))) * betweenness


def G_features(G, time):
    G_t = create_unweighted_G_t(G.train, 0)
    biggest_scc = nx.DiGraph(max(nx.strongly_connected_component_subgraphs(G_t), key=len))
    closeness_dict = {}
    betweenness_dict = {}
    size = biggest_scc.number_of_nodes()
    shortest_paths = {}
    reversed_scc = biggest_scc.reverse()
    for node in reversed_scc.nodes():
        shortest_paths[node] = nx.single_source_shortest_path_length(biggest_scc, node)
        closeness_dict[node] = (size - 1) / sum(nx.single_source_shortest_path_length(reversed_scc, node).values())

    for node in biggest_scc.nodes():
        betweenness_dict[node] = update_node_betweenness(node, biggest_scc, shortest_paths, size)

    return {'a': closeness_dict, 'b': betweenness_dict}


def calculate_probabilities(graph, mode='undirected unweighted'):
    probabilities_to_add_edge = defaultdict(dict)
    if mode == 'directed':
        shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    for node in graph.nodes():
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
                        all_paths_count = len(list(nx.all_shortest_paths(graph, node, second_node)))
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
    data = pd.read_csv(dataset)
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

    # bow tie