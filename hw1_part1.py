import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import random
import numpy.polynomial.polynomial as poly

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

def BFS_With_Number_Of_Shortest_Paths(G, source):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[source] = 1.0
    D[source] = 0
    Q = [source]
    while Q:   # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:   # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma

def update_betweenness(betweenness, bfs_nodes_queue, predecessors, number_of_shortest_paths, source):
    delta = dict.fromkeys(bfs_nodes_queue, 0)
    while bfs_nodes_queue:
        w = bfs_nodes_queue.pop()
        coeff = (1 + delta[w]) / number_of_shortest_paths[w]
        for v in predecessors[w]:
            delta[v] += number_of_shortest_paths[v] * coeff
        if w != source:
            betweenness[w] += delta[w]
    return betweenness


def G_features(G, time):
	# section a is closeness centrality and b is betweenes centrality
	g_t  = create_unweighted_G_t(G.train, 0)
	biggest_scc = nx.DiGraph(max(nx.strongly_connected_component_subgraphs(g_t), key=len))
	a_dict = {}
	b_dict = dict.fromkeys(biggest_scc, 0.0)
	size = biggest_scc.number_of_nodes()
	reversed_scc = biggest_scc.reverse()
	for node in  reversed_scc.nodes():
		a_dict[node] = (size - 1) / sum(nx.single_source_shortest_path_length(reversed_scc,node).values())

	for node in biggest_scc.nodes() :
		nodes_queue, predecessors, number_of_shortest_paths = BFS_With_Number_Of_Shortest_Paths(biggest_scc, node)
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
			number_of_shortest_paths[node] = BFS_With_Number_Of_Shortest_Paths(graph, node)[2]
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

def last_question(dataset):
	#first try linear regression
	points = dataset.train[['source', 'rating']].values
	centroids = initialize_centroids(points, 20)

	closest = closest_centroid(points, centroids)
	centroids = move_centroids(points, closest, centroids)
	old_centroids = []
	while np.array_equal(centroids, old_centroids):
		closest = closest_centroid(points, centroids)
		old_centroids = centroids
		centroids = move_centroids(points, closest, centroids)

	points = dataset.train[['target', 'rating']].values
	centroids = initialize_centroids(points, 10)

	closest = closest_centroid(points, centroids)
	centroids2 = move_centroids(points, closest, centroids)
	old_centroids = []
	while np.array_equal(centroids, old_centroids):
		closest = closest_centroid(points, centroids)
		old_centroids = centroids
		centroids = move_centroids(points, closest, centroids)

	points = dataset.test[['source', 'rating']].values
	for point in points :
		print(point)
		print(closest_centroid(point, centroids))
	plt.subplot(122)
	plt.scatter(points[:, 0], points[:, 1])
	plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
	plt.show()

	points = dataset.test[['target', 'rating']].values
	for point in points :
		print(point)
		print(closest_centroid(point, centroids2))
	plt.subplot(122)
	plt.scatter(points[:, 0], points[:, 1])
	plt.scatter(centroids2[:, 0], centroids2[:, 1], c='r', s=100)
	plt.show()

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def linear_regression(dataset):
	X = dataset.train_x.values
	Y = dataset.train_y.values
	m = Y.size
	t = X
	X = np.hstack((np.matrix(np.ones(m).reshape(m, 1)), t))
	theta = np.matrix(np.ones(3).reshape(3, 1))
	theta = GradientDissent(X, Y, theta, 0.0001, 1000)
	print (cost(theta, X, Y))

	X = dataset.test_x.values
	Y = dataset.test_y.values
	m = Y.size
	t = X
	X = np.hstack((np.matrix(np.ones(m).reshape(m, 1)), t))
	print(cost(theta, X, Y))

def GradientDissent(X, Y, theta, alpha, num_iters):
	m = np.size(Y)
	hx = np.matmul(X, theta)
	c1 = cost(theta, X, Y)
	for i in range(0, num_iters): # 3 X 15 * 15
		temp = (alpha / m) * np.matmul(X.T, (hx - Y))
		c2 = cost(theta - temp, X, Y)

		if c1 > c2:
			c1 = c2
			theta = theta - temp

	return theta

def cost(theta, X, Y):
    m = Y.size
    hx = np.matmul(X, theta)
    return np.sum(np.power(np.subtract(hx, Y), 2)) / (2 * m)

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
