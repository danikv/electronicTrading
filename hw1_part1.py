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
		
		self.train=self.init_graph[(self.init_graph['time'] <= time)]
		self.test=self.init_graph[(self.init_graph['time'] > time)]
		

		self.train_x = self.train[['source', 'target']]
		self.train_y = self.train[['rating']]
		self.test_x  = self.test[['source', 'target']]
		self.test_y  = self.test[['rating']]

def create_unweighted_H_t(train, time):
	return nx.Graph(train)

def create_weighted_H_t(train, time):
	h_t = nx.Graph()
	direct_graph = nx.DiGraph(train)
	for edge in direct_graph.edges():
		if h_t.has_edge(edge[0], edge[1]) :
			h_t.add_edge(edge[0], edge[1], weight='strong')
		else :
			h_t.add_edge(edge[0], edge[1], weight='weak')
	return h_t

def create_unweighted_G_t(train, time):
	return nx.DiGraph(train)
		
def calc_error(predictions, test, mode='undirected unweighted'):
	precision, recall=0,0
	data_list = tuple(map(tuple, test.values.tolist()))
	intersections = set(predictions) & set(data_list)
	recall = len(intersections) / len(test)
	precision = len(intersections) / len(predictions)
	return (precision, recall)

def update_node_betweenness(node, graph, shortest_paths_length , size):
	betweenes = 0
	for x in graph.nodes() :
		if node != x :
			for y in graph.nodes() :
				if x != y and node != y:
					shortest_paths_with_node = 0
					shortest_paths = 0
					shortest_path_length_without_node = shortest_paths_length[x][y]
					shortest_path_length_with_node = shortest_paths_length[x][node] + shortest_paths_length[node][y]
					all_shortest_paths = list(nx.all_shortest_paths(graph, x, y))
					if shortest_path_length_with_node == shortest_path_length_without_node :
						for path in all_shortest_paths :
							if node in path :
								shortest_paths += 1
								shortest_paths_with_node += 1
							else:
								shortest_paths += 1
					else:
						shortest_paths += len(list(all_shortest_paths))
					betweenes += (shortest_paths_with_node / shortest_paths)
	return (1 / ((size - 1) *(size -2 ))) * betweenes


def G_features(G, time):
	# section a is closeness centrality and b is betweenes centrality
	g_t  = create_unweighted_G_t(G.train, 0)
	biggest_scc = nx.DiGraph(max(nx.strongly_connected_component_subgraphs(g_t), key=len))
	a_dict = {}
	b_dict = {}
	size = biggest_scc.number_of_nodes()
	shortest_paths_lengths = {}
	reversed_scc = biggest_scc.reverse()
	for node in  reversed_scc.nodes():
		shortest_paths_lengths[node] = nx.single_source_shortest_path_length(biggest_scc, node)
		a_dict[node] = (size - 1) / sum(nx.single_source_shortest_path_length(reversed_scc,node).values())

	for node in biggest_scc.nodes() :
		b_dict[node] = update_node_betweenness(node, biggest_scc, shortest_paths_lengths, size)
	
	return {'a': a_dict, 'b': b_dict }

def should_add_edge(probability) :
	return probability > random.random()

def calculate_undirected_weigthed_probability(graph, node, second_node):
	number_of_strong_connections = 0
	number_of_weak_connections = 0
	for neighbor in nx.common_neighbors(graph, node, second_node) :
		if graph[node][neighbor]['weight'] == 'strong' :
			number_of_strong_connections += 1
		else:
			number_of_weak_connections += 1
	return 1 - (pow(float(0.96), number_of_strong_connections) * pow(float(0.98),number_of_weak_connections))

def calculate_probabilities(graph, mode='undirected unweighted') :
	probabilities_to_add_edge = defaultdict(dict)
	if mode == 'directed':
		shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
	for node in graph.nodes() :
		for second_node in nx.non_neighbors(graph, node):
			if mode == 'undirected unweighted' :
				coomon_neighbors_size = len(list(nx.common_neighbors(graph, node, second_node)))
				probabilities_to_add_edge[node][second_node] = 1 - pow(float(0.97), coomon_neighbors_size)
			elif mode == 'undirected weighted' :
				probabilities_to_add_edge[node][second_node] = calculate_undirected_weigthed_probability(graph, node, second_node)
			elif mode == 'directed':
				if second_node in shortest_path_lengths[node]:
					L = shortest_path_lengths[node][second_node]
					if L <= 4:
						M = len(list(nx.all_shortest_paths(graph, node, second_node)))
						probabilities_to_add_edge[node][second_node] = min(1, M / (math.pow(5, L)))
	return probabilities_to_add_edge

def run_k_iterations(graph, N, mode='undirected unweighted'):
	new_edges = []
	while(N > 0):
		probabilities = calculate_probabilities(graph, mode)
		for node, node_probabilities in probabilities.items() :
			for second_node, probability in node_probabilities.items():
				if should_add_edge(probability) :
					new_edges.append((node, second_node))
					if mode == 'undirected weighted' :
						graph.add_edge(node, second_node, weight='weak')
					else:
						graph.add_edge(node, second_node)
		N -= 1
	return new_edges
	

def partA_q1(dataset):
	#plot
	data = pd.read_csv(dataset)
	df = data[['source', 'target']].groupby('source')['target'].nunique().reset_index(name='count')
	histogram = df.groupby('count')['source'].count().reset_index(name='hist_value')
	histogram.plot(x='count', y='hist_value', kind='scatter', logx=True, logy=True)
	plt.show()

	x_values = np.log(np.array(histogram['count']))
	y_values = np.log(np.array(histogram['hist_value']))
	coef = np.polyfit(x_values, y_values, 1)
	line = [(coef[0]*i + coef[1]) for i in x_values]
	fig = plt.figure()
	ax = fig.gca()
	ax.scatter(np.exp(x_values), np.exp(y_values), c='blue')
	ax.plot(np.exp(x_values), np.exp(line), c='red')
	ax.set_yscale('log')
	ax.set_xscale('log')
	plt.show()

	### bow tie -------------------------------
	#answer = data['source'].tolist()
	#ask = data['target'].tolist()
	#middle = list(set(answer) & set(ask))
	#left = set(answer) - set(ask)
	#right = set(ask) - set(answer)
	#print(len(left))
	#print(len(middle))
	#print(len(right))
	#print(len(left.union(middle, right)))
