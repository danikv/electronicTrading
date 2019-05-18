import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import random
import csv

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
	graph = nx.DiGraph(train)
	for e in graph.edges():
		if h_t.has_edge(e[0], e[1]) :
			h_t.add_edge(e[0], e[1], weight='strong')
		else :
			h_t.add_edge(e[0], e[1], weight='weak')
	return h_t

def create_unweighted_G_t(train, time):
	return nx.DiGraph(train)
		
def calc_error(predictions, test, mode='undirected unweighted'):
	precision, recall=0,0
	data = tuple(map(tuple, test.values.tolist()))
	correct_predictions = set(predictions) & set(data)
	recall = len(correct_predictions) / len(test)
	precision = len(correct_predictions) / len(predictions)
	return (precision, recall)

def G_features(G, time):
	# section a is closeness centrality and b is betweenes centrality
	a = {}
	g_t  = create_unweighted_G_t(G.train, 0)
	scc = g_t.subgraph(max(nx.strongly_connected_components(g_t), key=len))
	reversed_graph = scc.reverse()
	for node in  reversed_graph.nodes():
		a[node] = closeness_centrality(reversed_graph, node)
	b = betweenness_centrality(scc)
	return {'a': a, 'b': b }

def closeness_centrality(graph, node):
	size = graph.number_of_nodes()
	return (size - 1) / sum(nx.single_source_shortest_path_length(graph,node).values())

def betweenness_centrality(graph):
	betweenness = dict.fromkeys(graph, 0)
	size = graph.number_of_nodes()
	for node in graph.nodes() :
		nodes_queue, predecessors, number_of_shortest_paths = Extra_Data_BFS(graph, node)
		betweenness = update_betweenness(betweenness, nodes_queue, predecessors, number_of_shortest_paths, node)
	for node in graph.nodes() :
		betweenness[node] = betweenness[node] * (1 / ((size - 1) * (size -2)))
	return betweenness

def Extra_Data_BFS(G, source):
    nodes_queue = []
    predecessors = {}
    number_of_shortest_paths = dict.fromkeys(G, 0)
    number_of_shortest_paths[source] = 1
    D = {}
    D[source] = 0
    for node in G:
        predecessors[node] = []
    Q = [source]
    while Q:
        node = Q.pop(0)
        nodes_queue.append(node)
        for neighbor in G[node]:
            if neighbor not in D:
                Q.append(neighbor)
                D[neighbor] = D[node] + 1
            if D[neighbor] == D[node] + 1:
                number_of_shortest_paths[neighbor] +=  number_of_shortest_paths[node]
                predecessors[neighbor].append(node)
    return nodes_queue, predecessors, number_of_shortest_paths

def update_betweenness(betweenness, bfs_nodes_queue, predecessors, number_of_shortest_paths, source):
    delta = dict.fromkeys(bfs_nodes_queue, 0)
    while bfs_nodes_queue:
        node = bfs_nodes_queue.pop()
        coeff = (1 + delta[node]) / number_of_shortest_paths[node]
        for predecessor in predecessors[node]:
            delta[predecessor] += number_of_shortest_paths[predecessor] * coeff
        if node != source:
            betweenness[node] += delta[node]
    return betweenness

def should_add_edge(probability) :
	return probability > random.random()

def calculate_probabilities(graph, mode='undirected unweighted') :
	probabilities_to_add_edge = defaultdict(dict)
	node_shortests_paths_count = defaultdict(dict)
	if mode == 'directed':
		shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
		for node in graph.nodes() :
			node_shortests_paths_count[node] = Extra_Data_BFS(graph, node)[2]
	for node in graph.nodes() :
		for non_neighbor in nx.non_neighbors(graph, node):
			if mode == 'undirected unweighted' :
				probabilities_to_add_edge[node][non_neighbor] = 1 - pow(float(0.97), len(list(nx.common_neighbors(graph, node, non_neighbor))))
			elif mode == 'undirected weighted' :
				probabilities_to_add_edge[node][non_neighbor] = calculate_undirected_weigthed_probability(graph, node, non_neighbor)
			elif mode == 'directed':
				probability = calculate_directed_probability(graph, node, non_neighbor, shortest_path_lengths, node_shortests_paths_count)
				if probability is not None:
					probabilities_to_add_edge[node][non_neighbor] = probability
	return probabilities_to_add_edge

def calculate_undirected_weigthed_probability(graph, node, non_neighbor):
	number_of_strong_connections = 0
	number_of_weak_connections = 0
	for common_neighbor in nx.common_neighbors(graph, node, non_neighbor) :
		if graph[node][common_neighbor]['weight'] == 'strong' :
			number_of_strong_connections += 1
		else:
			number_of_weak_connections += 1
	return 1 - (pow(float(0.96), number_of_strong_connections) * pow(float(0.98), number_of_weak_connections))

def calculate_directed_probability(graph, node, second_node, shortest_path_lengths, node_shortests_paths_count):
	if second_node in shortest_path_lengths[node]:
		L = shortest_path_lengths[node][second_node]
		if L <= 4:
			M = node_shortests_paths_count[node][second_node]
			return min(1, M / (math.pow(5, L)))

def run_k_iterations(graph, N, mode='undirected unweighted'):
	new_edges = []
	for i in range(N):
		probabilities = calculate_probabilities(graph, mode)
		for node, node_probabilities in probabilities.items() :
			for second_node, probability in node_probabilities.items():
				if should_add_edge(probability) :
					new_edges.append((node, second_node))
					if mode == 'undirected weighted' :
						graph.add_edge(node, second_node, weight='weak')
					else:
						graph.add_edge(node, second_node)
	return new_edges

def partc(dataset, prediction_time):
	graph = create_unweighted_G_t(dataset.train, 0)
	new_size = number_of_new_edges(dataset, prediction_time, len(list(graph.edges())))
	train_source_average_data, train_target_average_data, train_rating, test_source_average_data, test_target_average_data,\
		test_rating, source_average_rating, target_average_rating = build_dataset(dataset)
	hypotesis = linear_regression(train_source_average_data, train_target_average_data, train_rating, test_source_average_data, test_target_average_data, test_rating)
	print(new_size)
	new_edges = add_edges(graph, new_size, hypotesis, source_average_rating, target_average_rating)

	with open('hw1_part2.csv', mode='w' ,newline='') as csv_file:
		fieldnames = ['source','target','rating','time']
		data_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		for edge in new_edges :
			data_writer.writerow({'source': edge[0], 'target': edge[1], 'rating': edge[2], 'time': 1453438800})

def build_dataset(dataset):
	train_data = dataset.train[['time', 'source', 'target', 'rating']].sort_values(by=['time'])
	train_source_average_data = []
	train_target_average_data = []
	train_rating = []
	source_average_rating = defaultdict(int)
	target_average_rating = defaultdict(int)
	for index, data in train_data.iterrows():
		train_source_average_data.append(source_average_rating[data['source']])
		train_target_average_data.append(target_average_rating[data['target']])
		train_rating.append(data['rating'])
		source_average_rating[data['source']] += data['rating'] 
		target_average_rating[data['source']] += data['rating']

	test_data = dataset.test[['time', 'source', 'target', 'rating']].sort_values(by=['time'])
	test_source_average_data = []
	test_target_average_data = []
	test_rating = []
	for index, data in test_data.iterrows():
		test_source_average_data.append(source_average_rating[data['source']])
		test_target_average_data.append(target_average_rating[data['target']])
		test_rating.append(data['rating'])
		source_average_rating[data['source']] += data['rating'] 
		target_average_rating[data['source']] += data['rating']

	return train_source_average_data, train_target_average_data, train_rating, test_source_average_data, test_target_average_data, test_rating, source_average_rating, target_average_rating

def linear_regression(train_source_average_data, train_target_average_data, train_rating, test_source_average_data, test_target_average_data, test_rating):
	m = len(train_rating)
	Y = np.array(train_rating)
	X = np.hstack((np.matrix(np.ones(m).reshape(m, 1)), np.matrix(train_source_average_data).T,  np.matrix(train_target_average_data).T))
	theta = normal_equation(X, Y)

	test_Y = np.matrix(test_rating).T
	test_M = len(test_rating)
	test_X = np.hstack((np.matrix(np.ones(test_M).reshape(test_M, 1)), np.matrix(test_source_average_data).T,  np.matrix(test_target_average_data).T))
	print(theta)
	print(cost(np.matrix(theta).T, test_X, test_Y))
	return np.matrix(theta).T

def predict_rating(source, target, source_average_rating, target_average_rating, hypotesis):
	data = np.hstack((1, source_average_rating[source], target_average_rating[target])).T
	prediction = np.matmul(data, hypotesis)
	rating = int(prediction.item(0))
	source_average_rating[source] += rating
	target_average_rating[target] += rating
	return rating

def add_edges(graph, added_edges_size, hypotesis, source_average_rating, target_average_rating):
	added_edges = []
	while(added_edges_size > 0):
		probabilities = calculate_last_question_probabilities(graph)
		for source , value in probabilities.items() :
			for target, probability in value.items():
				if should_add_edge(probability):
					prediction = predict_rating(source, target, source_average_rating, target_average_rating, hypotesis)
					graph.add_edge(source, target, rating=prediction)
					added_edges.append((source, target, prediction))
					added_edges_size -= 1
				if added_edges_size == 0 :
					return added_edges
	return added_edges

def calculate_last_question_probabilities(graph) :
	shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
	number_of_shortest_paths = defaultdict(dict)
	for node in graph.nodes() :
		number_of_shortest_paths[node] = Extra_Data_BFS(graph, node)[2]
	probabilities_to_add_edge = defaultdict(dict)
	for node in graph.nodes() :
		for second_node in nx.non_neighbors(graph, node):
			if second_node in shortest_path_lengths[node]:
				L = shortest_path_lengths[node][second_node]
				if L <= 4:
					M = number_of_shortest_paths[node][second_node]
					probabilities_to_add_edge[node][second_node] = min(1, (M / (math.pow(7, 1 * L))))
	return probabilities_to_add_edge

def number_of_new_edges(dataset, prediction_time, edges_currently_in_graph):
	X , Y = features(dataset.train[['time', 'target']])
	predictor = np.poly1d(np.polyfit(X, Y, 1))
	return int(predictor(prediction_time) - edges_currently_in_graph)

def features(dataset):
	X = dataset.groupby('time')['target'].count().to_dict()
	time_values = []
	count_values = []
	last_value = 0
	for key, value in X.items() :
		time_values.append(key)
		count_values.append(value + last_value)
		last_value += value
	return time_values, count_values

def normal_equation(X, Y):
	step1 = np.dot(X.T, X)
	step2 = np.linalg.pinv(step1)
	step3 = np.dot(step2, X.T)
	theta = np.dot(step3, Y) # if y is m x 1.  If 1xm, then use y.T

	return theta

def cost(theta, X, Y):
    m = Y.size
    hx = np.matmul(X, theta)
    return np.sum(np.power(np.subtract(hx, Y), 2)) / (2 * m)
	
#answer to question A.1
def q1_plot_data(dataset):
	data = pd.read_csv(dataset)
	graph = data[['source', 'target']].groupby('source').nunique()
	graph['target'].plot.hist(ylim=(0, 100))
	plt.show()

	df = data[['source', 'target']].groupby('source')['target'].nunique().reset_index(name='count')
	df_hist = df.groupby('count')['source'].count().reset_index(name='hist_value')
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
	print('c value is - ', -a[0])
	print('a value is - ', np.exp(a[1]))
	plt.show()
