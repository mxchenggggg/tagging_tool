import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, 5)
		self.conv1_bn = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 16, 5)
		# self.conv2_bn = nn.BatchNorm2d(16)
		# self.conv3 = nn.Conv2d(16, 16, 5)
		self.maxpool1 = nn.MaxPool2d(2, stride=2)
		self.fc1 = nn.Linear(434816, 500)
		# self.fc1_bn = nn.BatchNorm1d(500)
		self.fc2 = nn.Linear(500, 1)
		self.sigmoid = nn.Sigmoid()

		self.init_weights()

	def init_weights(self):
		for conv in [self.conv1]:# , self.conv2, self.conv3]:
			C_in = conv.weight.size(1)
			nn.init.normal_(conv.weight, 0.0, 0.01 / sqrt(5*5*C_in))
			nn.init.constant_(conv.bias, 0.0)

		for fc in [self.fc1, self.fc2]:
			F_in = fc.weight.size(1)
			nn.init.normal_(fc.weight, 0.0, 0.01/sqrt(F_in))
			nn.init.constant_(fc.bias, 0.0)
        
	def forward(self, x):
		N, C, H, W = x.shape
		z = F.relu(self.conv1(x))
		z = self.conv1_bn(z)
		z = F.relu(self.conv2(z))
		# z = self.conv2_bn(z)
		# z = F.relu(self.conv3(z))
		z = self.maxpool1(z)
		z = z.view(-1, z.size(1)*z.size(2)*z.size(3))
		z = F.relu(self.fc1(z))
		# z = self.fc1_bn(z)
		z = self.sigmoid(self.fc2(z))
		return z