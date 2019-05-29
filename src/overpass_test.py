from overpass_api import Overpass_api
import pickle
import overpy
import numpy as np

def main():
	gps_data = np.array(pickle.load(open('../hd_map/vehicle_gps.txt', 'rb')))
	b_box = [np.min(gps_data[:, 0]), np.min(gps_data[:, 1]), np.max(gps_data[:, 0]), np.max(gps_data[:, 1])]
	# api = Overpass_api(b_box)	
	print(b_box)
	
if __name__ == "__main__":
	main()