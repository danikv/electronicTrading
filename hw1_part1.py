import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import random
import concurrent.futures



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
		#add wieght to h_t
		if h_t.has_edge(edge[0], edge[1]) :
			h_t.add_edge(edge[0], edge[1], weight='strong')
		else :
			h_t.add_edge(edge[0], edge[1], weight='weak')
	return h_t

def create_unweighted_G_t(train, time):
	return((1,2),(1,3), (3,1), (1,4))
		
		
def calc_error(predictions, test, mode='undirected unweighted'):
	precision, recall=0,0
	#get list intersection
	if mode == 'undirected unweighted':
		graph = nx.Graph(test)
	elif mode == 'undirected weighted':
		graph = create_weighted_H_t(test, 0)
	intersections = 0
	for edge in predictions :
		if graph.has_edge(edge[0], edge[1]) :
			#if mode == 'undirected unweighted':
			intersections += 1
			#if mode == 'undirected weighted':
			#if graph[edge[0]][edge[1]]['weight'] == edge[3]:
			#intersections += 1
	recall = intersections / len(test)
	precision = intersections / len(predictions)
	return (precision, recall)

def update_node_betweenness(node, shortest_paths, b_dict, size):
	shortest_path_with_node = 0
	shortest_path_without_node = 0
	for path_from, paths in shortest_paths.items() :
		if path_from != node :
			shortest_path_to_node = paths[node]
			for path_to, path in paths.items() :
				if path_to != node :
					if shortest_path_to_node + shortest_paths[node][path_to] == path:
						shortest_path_without_node += 1
						shortest_path_with_node += 1
					else:
						shortest_path_without_node += 1
	b_dict[node] = (1 / ((size - 1) *(size -2 ))) * \
		(shortest_path_with_node / shortest_path_without_node)


def G_features(G, time):
	# section a is closeness centrality and b is betweenes centrality
	g_t  = nx.DiGraph(G.train_x)
	biggest_scc = nx.DiGraph(max(nx.strongly_connected_component_subgraphs(g_t), key=len))
	a_dict = {}
	b_dict = {}
	size = biggest_scc.number_of_nodes()
	reversed_scc = biggest_scc.reverse()
	shortest_paths_length = {}
	for node in  reversed_scc.nodes():
		shortest_paths_length[node] = nx.single_source_shortest_path_length(biggest_scc,node)
		a_dict[node] = (size - 1) / sum(shortest_paths_length[node])

	for node in biggest_scc :
		update_node_betweenness(node, shortest_paths_length, b_dict, size)

	print('finished')
	#a2_dict = nx.closeness_centrality(biggest_scc, wf_improved=False)
	b2_dict = nx.betweenness_centrality(biggest_scc)
	#if a2_dict == a_dict :
	#	print("equal")
	for key, value in b_dict.items() :
		if b2_dict[key] != value :
			print("key : " + str(key) + " , value b1 : " + str(value) + ", value b2 : " + str(b2_dict[key]))
	return {'a': a_dict, 'b': b2_dict }

def should_add_edge(probability, mode='undirected unweighted') :
	if mode == 'undirected unweighted':
		return probability > random.random()
	elif mode == 'undirected weighted':
		for prob in probability:
			if prob > random.random() :
				return True
		return False

def calculate_probabilities(graph, mode='undirected unweighted') :
	probabilities_to_add_edge = defaultdict(dict)
	for node in graph.nodes() :
		for second_node in graph.nodes():
			if node != second_node and second_node not in graph.neighbors(node):
				if mode == 'undirected unweighted' :
					coomon_neighbors_size = len(list(nx.common_neighbors(graph, node, second_node)))
					probability = 1 - pow(float(0.97), coomon_neighbors_size)
					probabilities_to_add_edge[node][second_node] = probability
				elif mode == 'undirected weighted' :
					probability = []
					for neighbor in nx.common_neighbors(graph, node, second_node) :
						if graph[node][neighbor]['weight'] == 'strong' :
							probability.append(0.04)
						else:
							probability.append(0.02)
					probabilities_to_add_edge[node][second_node] = probability
	return probabilities_to_add_edge

def run_k_iterations(graph, N, mode='undirected unweighted'):
	added_edges = []
	while(N > 0):
		probabilities_to_add_edge = calculate_probabilities(graph, mode)
		for node , value in probabilities_to_add_edge.items() :
			for second_node, probability in value.items():
				if should_add_edge(probability, mode) :
					if mode == 'undirected unweighted' :
						added_edges.append([node, second_node])
						graph.add_edge(node, second_node)
					elif mode == 'undirected weighted' :
						#if graph.has_edge(node, second_node) :
						#	added_edges.remove([second_node, node, 'weak'])
						#	added_edges.append([node, second_node, 'strong'])
						#	graph.add_edge(node, second_node, weight='strong')
						#else:
						added_edges.append([node, second_node, 'weak'])
						graph.add_edge(node, second_node, weight='weak')
		N -= 1
	return added_edges
	

def partA_q1(dataset):
	#plot
	data = pd.read_csv(dataset)
	df = data[['source', 'target']].groupby('source')['target'].nunique().reset_index(name='count')
	df_hist = df.groupby('count')['source'].count().reset_index(name='hist_value')
	df_hist.plot(x='count', y='hist_value', kind='scatter', logx=True, logy=True)
	plt.show()

	x = np.log(np.array(df_hist['count']))
	y = np.log(np.array(df_hist['hist_value']))
	a = np.polyfit(x, y, 1)
	print('c value is - ', -a[0])
	print('a value is - ', np.exp(a[1]))
	y_1 = [(a[0]*i + a[1]) for i in x]
	fig = plt.figure()
	ax = fig.gca()
	ax.scatter(np.exp(x), np.exp(y), c='blue')
	ax.plot(np.exp(x), np.exp(y_1), c='red')
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
