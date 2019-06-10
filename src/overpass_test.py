from overpass_api import Overpass_api
import pickle
import overpy
import numpy as np

def main():
	gps_data = np.array(pickle.load(open('../hd_map/vehicle_gps.txt', 'rb')))
	o_api = Overpass_api(gps_data)
	o_api.auto_tag()

	# count = 0
	# string = ''
	# for j, w_list in junctions.items():
	# 	if len(w_list) == 1:
	# 		count += 1
	# 		string += str(w_list[0]) + ','
	# print(string)
	# print(count)

if __name__ == "__main__":
	main()