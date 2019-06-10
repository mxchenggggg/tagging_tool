import overpy
import pickle
from tagging_tool import get_distance
from collections import deque
from math import acos, pi

def get_angle(in_node, vertex, out_node, nodes):
	a = get_distance([ float(nodes[in_node][0]), float(nodes[in_node][1]) ], [ float(nodes[vertex][0]), float(nodes[vertex][1]) ])
	b = get_distance([ float(nodes[out_node][0]), float(nodes[out_node][1]) ], [ float(nodes[vertex][0]), float(nodes[vertex][1]) ])
	c = get_distance([ float(nodes[in_node][0]), float(nodes[in_node][1]) ], [ float(nodes[out_node][0]), float(nodes[out_node][1]) ])
	temp = (a**2 + b**2 - c**2)/(2*a*b)
	# print(temp)
	return acos(temp)

def get_type(angle):
	if angle <= pi/4:
		return 'branch_in'
	if angle >= 3*pi/4:
		return 'branch_out'
	return 'intersection'

class Overpass_api:
	def __init__(self, gps_data):
		self.api = overpy.Overpass()
		self.gps_data = gps_data
		way_pts = ""	
		for i in range(len(gps_data)):
			g = gps_data[i]
			way_pts +=  str(g[0]) + "," + str(g[1]) + ","
		way_pts = way_pts[:-1]
		# self.result = api.query("""
		# way[highway]["highway"!="service"]["highway"!="footway"]["highway"!="cycleway"]
		# (around:15,{});
		# (._;>;);
		# out body;
		# """.format(way_pts))
		# nodes = {}
		# for n in result.nodes:
		# 	nodes[n.id] = (n.lat, n.lon)
		# ways = {}
		# for w in result.ways:
		# 	n_list = []
		# 	for n in w.nodes:
		# 		n_list.append(n.id)
		# 	ways[w.id] = n_list
		# 
		# route = [nodes, ways]
		# pickle.dump(route, open('route.p', 'wb'))

		self.nodes, self.ways = pickle.load(open('route.p', 'rb'))
	
	def auto_tag(self):
		# find all junctions
		junctions = {}
		# case 1: end to end
		for w, n_list in self.ways.items():
			end1, end2 = n_list[0], n_list[-1]
			if end1 not in junctions:
				junctions[end1] = [w]
			else:
				junctions[end1].append(w)
			if end2 not in junctions:
				junctions[end2] = [w]
			else:
				junctions[end2].append(w)
		# case 2: end to mid
		way_with_mid_junc = {}
		mid_junc = set()
		for w, n_list in self.ways.items():
			for n in n_list[1:-1]:
				if n in junctions:
					mid_junc.add(n)
					junctions[n].append(w)
					if w not in way_with_mid_junc:
						way_with_mid_junc[w] = [n]
					else:
						way_with_mid_junc[w].append(n)
		
		# delete irrelevant ways
		delete_ways = []
		for w, n_list in self.ways.items():
			end1, end2 = n_list[0], n_list[-1]
			if len(junctions[end1]) == 1 and len(junctions[end2]) == 1:
				delete_ways.append(w)
				junctions.pop(end1)
				junctions.pop(end2)
		for w in delete_ways:
			n_list = self.ways[w]
			for n in n_list:
				self.nodes.pop(n, None)
			self.ways.pop(w, None)

		# find start and end junctions:
		start_min, end_min = float('inf'), float('inf')
		start_j, end_j = -1, -1
		for j in junctions:
			if len(junctions[j]) == 1:
				dis1 = get_distance(self.gps_data[0][0:2], [float(self.nodes[j][0]), float(self.nodes[j][1])])
				dis2 = get_distance(self.gps_data[-1][0:2], [float(self.nodes[j][0]), float(self.nodes[j][1])])
				if dis1 < start_min:
					start_min = dis1
					start_j = j
				if dis2 < end_min:
					end_min = dis2
					end_j = j

		# find main route
		adj_list = {}
		for j, w_list in junctions.items():
			adj_list[j] = []
			for w in w_list:
				if w not in way_with_mid_junc:
					if j != self.ways[w][0]:
						adj_list[j].append((self.ways[w][0], w))
					if j != self.ways[w][-1]:
						adj_list[j].append((self.ways[w][-1], w))
				else:
					if j == self.ways[w][0]:
						adj_list[j].append((way_with_mid_junc[w][0], w))
					elif j == self.ways[w][-1]:
						adj_list[j].append((way_with_mid_junc[w][-1], w))
					else:
						temp_list = [self.ways[w][0]] + way_with_mid_junc[w] + [self.ways[w][-1]]
						ind = 1
						while temp_list[ind] != j:
							ind += 1
						adj_list[j].append((temp_list[ind-1], w))
						adj_list[j].append((temp_list[ind+1], w))
		path = {}
		queue = deque()
		queue.append(start_j)
		visited = set()
		while len(queue) != 0:
			cur_junc = queue.popleft()
			visited.add(cur_junc)
			for pair in adj_list[cur_junc]:
				j, w = pair
				if j not in visited:
					queue.append(j)
					path[j] = (cur_junc, w)
					if self.ways[w].index(cur_junc) > self.ways[w].index(j):
						self.ways[w].reverse()
		cur_junc = end_j
		main_route = [cur_junc]
		main_route_ways = set()
		while cur_junc != start_j:
			main_route.append(path[cur_junc][0])
			main_route_ways.add(path[cur_junc][1])
			cur_junc = path[cur_junc][0]
		main_route.reverse()

		string = ''
		for n in main_route[1:]:
			string += str(path[n][1])+','
		# print(string)

		# deleted unvisited ways
		delete_juncs = []
		for j in junctions:
			if j not in visited:
				delete_juncs.append(j)
				for w in junctions[j]:
					if w in self.ways:
						n_list = self.ways[w]
						for n in n_list:
							self.nodes.pop(n, None)
						self.ways.pop(w, None) 
		for j in delete_juncs:
			junctions.pop(j)

		string = ''
		count = 0
		for j in main_route:
			if len(junctions[j]) == 2:
				in_node = self.ways[path[j][1]][self.ways[path[j][1]].index(j)-1]
				if junctions[j].index(path[j][1]) == 0:
					out_node = self.ways[junctions[j][1]][1]
				else:
					out_node = self.ways[junctions[j][0]][1]
				angle = get_angle(in_node, j, out_node, self.nodes)
				if j in mid_junc:
					print(j, get_type(angle))
				elif angle > pi/3 and angle < 2*pi/3:
					print(j, 'intersection!')
				else:
					print(j, 'none')
			elif len(junctions[j]) == 3:
				in_node = self.ways[path[j][1]][-2]
				for w in junctions[j]:
					if w not in main_route_ways:
						out_node = self.ways[w][1]
				angle = get_angle(in_node, j, out_node, self.nodes)
				print(j, get_type(angle))
			else:
				print(j, 'intersection!')
		# print(string)


