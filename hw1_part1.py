import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import random
import numpy.polynomial.polynomial as poly
from math import sqrt, floor
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
	direct_graph = nx.DiGraph(train)
	for edge in direct_graph.edges():
		#add wieght to h_t
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

def BFS_With_Shortest_Paths(G, source):
    nodes_queue = []
    predecessors = {}
    D = {}
    D[source] = 0
    number_of_shortest_paths = dict.fromkeys(G, 0.0)
    number_of_shortest_paths[source] = 1.0
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


def G_features(G, time):
	# section a is closeness centrality and b is betweenes centrality
	g_t  = create_unweighted_G_t(G.train, 0)
	biggest_scc = g_t.subgraph(max(nx.strongly_connected_components(g_t), key=len))
	a_dict = {}
	b_dict = dict.fromkeys(biggest_scc, 0.0)
	size = biggest_scc.number_of_nodes()
	reversed_scc = biggest_scc.reverse()
	for node in  reversed_scc.nodes():
		a_dict[node] = (size - 1) / sum(nx.single_source_shortest_path_length(reversed_scc,node).values())

	for node in biggest_scc.nodes() :
		nodes_queue, predecessors, number_of_shortest_paths = BFS_With_Shortest_Paths(biggest_scc, node)
		b_dict = update_betweenness(b_dict, nodes_queue, predecessors, number_of_shortest_paths, node)
	
	for node in biggest_scc.nodes() :
		b_dict[node] = b_dict[node] * (1 / ((size - 1) * (size -2)))

	return {'a': a_dict, 'b': b_dict }

def should_add_edge(probability) :
	return probability > random.random()

def calculate_probabilities(graph, mode='undirected unweighted') :
	probabilities_to_add_edge = defaultdict(dict)
	number_of_shortest_paths = {}
	if mode == 'directed':
		shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
		for node in graph.nodes() :
			number_of_shortest_paths[node] = BFS_With_Shortest_Paths(graph, node)[2]
	for node in graph.nodes() :
		for second_node in nx.non_neighbors(graph, node):
			if mode == 'undirected unweighted' :
				coomon_neighbors_size = len(list(nx.common_neighbors(graph, node, second_node)))
				probabilities_to_add_edge[node][second_node] = 1 - pow(float(0.97), coomon_neighbors_size)
			elif mode == 'undirected weighted' :
				m = 0
				n = 0
				for neighbor in nx.common_neighbors(graph, node, second_node) :
					if graph[node][neighbor]['weight'] == 'strong' :
						m += 1
					else:
						n += 1
				probabilities_to_add_edge[node][second_node] = 1 - (pow(float(0.96), m) * pow(float(0.98),n))
			elif mode == 'directed':
				if second_node in shortest_path_lengths[node]:
					L = shortest_path_lengths[node][second_node]
					if L <= 4:
						M = number_of_shortest_paths[node][second_node]
						probabilities_to_add_edge[node][second_node] = min(1, M / (math.pow(5, L)))
	return probabilities_to_add_edge

def run_k_iterations(graph, N, mode='undirected unweighted'):
	added_edges = []
	while(N > 0):
		probabilities_to_add_edge = calculate_probabilities(graph, mode)
		for node , value in probabilities_to_add_edge.items() :
			for second_node, probability in value.items():
				if should_add_edge(probability) :
					added_edges.append((node, second_node))
					if mode == 'undirected weighted' :
						graph.add_edge(node, second_node, weight='weak')
					else:
						graph.add_edge(node, second_node)
		N -= 1
	return added_edges

def last_question(dataset, test_time, prediction_time):
	#predicting number of edges
	added_edges_size = calculate_new_added_edges_size(dataset, test_time, prediction_time)
	print(added_edges_size)
	#predicting new edges
	points = dataset.train[['source', 'target', 'rating']].values
	points[:,2] *= 10000
	test = dataset.test[['source', 'target', 'rating']].values
	centroids, centroids_to_rating, best_mse = k_means(points, test, 9, 100)
	graph = create_unweighted_G_t(dataset.train, 0)
	added_edges = add_edges(graph, added_edges_size, centroids, centroids_to_rating)
	#write data
	with open('hw1_part2.csv', mode='w' ,newline='') as csv_file:
		fieldnames = ['source','target','rating','time']
		data_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		for edge in added_edges :
			data_writer.writerow({'source': edge[0], 'target': edge[1], 'rating': edge[2], 'time': 1453438800})

def add_edges(graph, added_edges_size, centroids, centroids_to_rating):
	added_edges = []
	while(added_edges_size > 0):
		probabilities = probabilities_for_new_edges(graph)
		for node , value in probabilities.items() :
			for second_node, probability in value.items():
				if should_add_edge(probability):
					closest = predict_closest_centroid((node,second_node), centroids)
					graph.add_edge(node, second_node, rating=centroids_to_rating[closest])
					added_edges.append((node, second_node, centroids_to_rating[closest]))
					added_edges_size -= 1
				if added_edges_size == 0 :
					return added_edges
	return added_edges

def calculate_new_added_edges_size(dataset, test_time, prediction_time):
	X , Y = calculate_features(dataset.train[['time', 'target']])
	test_X , test_Y = calculate_features(dataset.test[['time', 'target']])
	mse = 0
	best_mse = 0
	best_prediction = 0
	for counter in range(1, 2):
		z = np.poly1d(np.polyfit(X, Y, counter))
		for j in range(len(test_Y)):
			mse += np.power(np.subtract(z(test_X[j]), test_Y[j]), 2)
		mse = mse / (2 * len(test_Y))
		if best_mse > mse or best_mse == 0:
			best_mse = mse
			best_prediction = z
	return int(best_prediction(prediction_time) - best_prediction(test_time))

def calculate_features(dataset):
	X = dataset.groupby('time')['target'].count().to_dict()
	time_values = []
	count_values = []
	last_value = 0
	for key, value in X.items() :
		time_values.append(key)
		count_values.append(value + last_value)
		last_value += value
	return time_values, count_values

def probabilities_for_new_edges(graph) :
	shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
	number_of_shortest_paths = defaultdict(dict)
	for node in graph.nodes() :
		number_of_shortest_paths[node] = BFS_With_Shortest_Paths(graph, node)[2]
	probabilities_to_add_edge = defaultdict(dict)
	for node in graph.nodes() :
		for second_node in nx.non_neighbors(graph, node):
			if second_node in shortest_path_lengths[node]:
				L = shortest_path_lengths[node][second_node]
				if L <= 4:
					M = number_of_shortest_paths[node][second_node]
					probabilities_to_add_edge[node][second_node] = min(1, (M / (math.pow(10, 1.05 * L))) * get_average_rating_neighbors(graph, node))
	return probabilities_to_add_edge

def k_means(points, test, k, iterations) :
	best_mse = 0
	best_centroids = []
	best_centroids_to_rating = {}
	for i in range(0,iterations) :
		try:
			centroids = initialize_centroids(points, k)

			closest = closest_centroid(points, centroids)
			centroids = move_centroids(points, closest, centroids)
			old_centroids = []
			while np.array_equal(centroids, old_centroids):
				closest = closest_centroid(points, centroids)
				old_centroids = centroids
				centroids = move_centroids(points, closest, centroids)
			centroids_to_rating = {}
			for i in range(k):
				centroids_to_rating[i] = get_average_rating_in_centroid(centroids, centroids[i], points)
			mse = 0
			points2 = test
			for point in points2:
				closest = predict_closest_centroid((point[0],point[1]), centroids)
				mse += pow(centroids_to_rating[closest] - point[2], 2)
			print(best_centroids_to_rating)
			if mse < best_mse or best_mse == 0:
				best_mse = mse
				best_centroids = centroids
				best_centroids_to_rating = centroids_to_rating
		except(Exception):
			pass
	return best_centroids, best_centroids_to_rating, best_mse

def get_average_rating_neighbors(graph, node) :
	neighbors = list(graph.neighbors(node))
	return max(1, sum(map(lambda x : graph[node][x]['rating'], neighbors)) / len(neighbors))

def move_centroids(points, closest, centroids):
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def get_average_rating_in_centroid(centroids, centroid, points):
	average_rating = 0
	points_size = 0
	for point in points :
		if np.array_equal(centroid, centroids[closest_centroid(point, centroids)[0]]):
			points_size += 1
			average_rating += point[2] / 10000
	rating = int(average_rating / points_size)
	if rating == 0 :
		return -1
	return rating

def closest_centroid(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def predict_closest_centroid(points, centroids):
	distances = np.sqrt(((points - centroids[:, [0,1]])**2).sum(axis=1))
	return np.argmin(distances, axis=0)

def initialize_centroids(points, k):
	centroids = points.copy()
	np.random.shuffle(centroids)
	return centroids[:k]

def q1(dataset):
	data = pd.read_csv(dataset)
	graph = data[['source', 'target']].groupby('source').nunique()
	graph['target'].plot.hist(ylim=(0, 300), bins=[0,1,2,10,20,50,100,200,300])
	plt.xlabel("Answers")
	plt.ylabel("Users")
	plt.show()

	df = data[['source', 'target']].groupby('source')['target'].nunique().reset_index(name='count')
	df_hist = df.groupby('count')['source'].count().reset_index(name='hist_value')
	x = np.log(np.array(df_hist['count']))
	y = np.log(np.array(df_hist['hist_value']))
	a = np.polyfit(x, y, 1)
	print('c value is - ', -a[0])
	print('a value is - ', np.exp(a[1]))
	print(a)
	y_1 = [(a[0]*i + a[1]) for i in x]
	fig = plt.figure()
	ax = fig.gca()
	ax.scatter(np.exp(x), np.exp(y), c='blue')
	ax.plot(np.exp(x), np.exp(y_1), c='red')
	ax.set_yscale('log')
	ax.set_xscale('log')
	plt.show()
