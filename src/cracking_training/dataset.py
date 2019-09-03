from torch.utils.data import Dataset, DataLoader
from scipy.misc import imread
import pickle
import torch 
import os, glob
import numpy as np


def get_train_val_test_loaders(batch_size):
	tr = CrackedDataset('train')
	va = CrackedDataset('val')
	# te = CrackedDataset('test')

	tr.X = tr.X.transpose(0,3,1,2)
	va.X = va.X.transpose(0,3,1,2)

	tr_loader =  DataLoader(tr, batch_size=batch_size, shuffle=True)
	va_loader =  DataLoader(va, batch_size=batch_size, shuffle=False)
	# te_loader =  DataLoader(te, batch_size=batch_size, shuffle=False)

	return tr_loader, va_loader

class CrackedDataset(Dataset):
	def __init__(self, partition):
		super().__init__()
		self.partition = partition
		self.X, self.y = self._load_data()

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

	def _load_data(self):
		print("Loading {} data...".format(self.partition))

		# img_folder = '/media/ac12/Data/tagging_tool/resized_new'
		img_folder = '/media/ac12/Data/tagging_tool/resized_shadow'

		img_names = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
		# cracked = pickle.load(open(os.path.join(img_folder, 'cracked.p'), 'rb'))
		shadow = pickle.load(open(os.path.join(img_folder, 'shadow.p'), 'rb'))
		X_c, y_c = [], []
		X_nc, y_nc = [], []
		# c_names, nc_names = [], []
		for img_name in img_names:
			basename = os.path.basename(img_name)
			img = imread(img_name, mode='L').astype(float)
			img = (img - np.min(img)) / (np.max(img) - np.min(img))

			if basename in shadow:
				X_c.append(img)
				# c_names.append(img_name)
				y_c.append(1)
			else:
				X_nc.append(img)
				# nc_names.append(img_name)
				y_nc.append(0)
		
		# X = X_c[0:1900] + X_nc[0:1900]  + X_c[1900:] + X_nc[1900:]
		# y = y_c[0:1900] + y_nc[0:1900]  + y_c[1900:] + y_nc[1900:]
		# X_file = os.path.join('data_X.p')
		# y_file = os.path.join('data_y.p')

		X = X_c[0:1250] + X_nc[0:1250]  + X_c[1250:] + X_nc[1250:]
		y = y_c[0:1250] + y_nc[0:1250]  + y_c[1250:] + y_nc[1250:]	
		X_file = os.path.join('data_X_shadow.p')
		y_file = os.path.join('data_y_shadow.p')
		
		###################
		# pre = pickle.load(open('pre_output_8.p', 'rb'))
		# pre = pre[1:]
		# print(pre.shape)
		# ind = 0
		# for n in c_names[1900:]:
		# 	bn = os.path.basename(n)
		# 	print("{}, gt: true, pre: {}".format(bn, pre[ind]==1.0))
		# 	ind += 1
		# 
		# for n in nc_names[1900:]:
		# 	bn = os.path.basename(n)
		# 	print("{}, gt: false, pre: {}".format(bn, pre[ind]==1.0))
		# 	ind += 1
		#######################
		
		pickle.dump(np.array(X), open(X_file, 'wb'))
		pickle.dump(np.array(y), open(y_file, 'wb')) 

		X, y = pickle.load(open(X_file, 'rb')), pickle.load(open(y_file, 'rb'))
		X = np.expand_dims(X, axis=3)
		# print(X.shape)
		if self.partition == 'train':
			# return X[0:3800], y[0:3800]
			return X[0:2500], y[0:2500]
		if self.partition == 'val':
			# return X[3800:], y[3800:]
			return X[2500:], y[2500:]

val = CrackedDataset('val')