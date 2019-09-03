import torch
import numpy as np 
from dataset import get_train_val_test_loaders
from model import CNN
import pickle

def predictions(output):
	return (output >= 0.5).view(output.size(0)).long()


def _train_epoch(data_loader, model, device, criterion, optimizer):
	for i, (X, y) in enumerate(data_loader):
		X, y = X.to(device), y.to(device)
		optimizer.zero_grad()
		# forward + backward + optimize
		output = model(X)
		loss = criterion(output, y.view(-1, 1).float())
		loss.backward()
		optimizer.step()


def _evaluate_epoch(tr_loader, val_loader, model, device, criterion, best_test_acc):
	y_true, y_pred = [], []
	correct, total = 0, 0
	running_loss = []
	print('validation......')
	for X, y in tr_loader:
		with torch.no_grad():
			X, y = X.to(device), y.to(device)
			output = model(X)
			predicted = predictions(output.data)
			y_true.append(y)
			y_pred.append(predicted)
			total += y.size(0)
			correct += (predicted == y).sum().item()
			running_loss.append(criterion(output, y.view(-1, 1).float()).item())
	train_loss = np.mean(running_loss)
	train_acc = correct / total
	print('training acc: {}, training loss: {}'.format(train_acc, train_loss))

	y_true, y_pred = [], []
	correct, total = 0, 0
	running_loss = []
	pre_output = np.empty((1))
	for X, y in val_loader:
		with torch.no_grad():
			X, y = X.to(device), y.to(device)
			output = model(X)
			predicted = predictions(output.data)
			y_true.append(y)
			y_pred.append(predicted)
			pre_output = np.hstack((pre_output, predicted.cpu().numpy()))
			total += y.size(0)
			correct += (predicted == y).sum().item()
			running_loss.append(criterion(output, y.view(-1, 1).float()).item())
	pickle.dump(pre_output,open('pre_output.p', 'wb'))
	val_loss = np.mean(running_loss)
	val_acc = correct / total
	print('validation acc: {}, validation loss: {}'.format(val_acc, val_loss))
	if val_acc > best_test_acc:
		best_test_acc = val_acc
		network_to_save = {}
		network_to_save["conv1_w"] = model.conv1.weight
		network_to_save["conv1_b"] = model.conv1.bias
		network_to_save["fc1_w"] = model.fc1.weight
		network_to_save["fc1_b"] = model.fc1.bias
		network_to_save["fc2_w"] = model.fc2.weight
		network_to_save["fc2_b"] = model.fc2.bias
		pickle.dump(network_to_save, open('best_cnn_shadow.p', 'wb'))
	return best_test_acc
	# stats.append([val_acc, val_loss, train_acc, train_loss])

def main():
	tr_loader, va_loader = get_train_val_test_loaders(50)

	use_cuda = torch.cuda.is_available()
	# use_cuda = 'False'
	device = torch.device("cuda" if use_cuda else "cpu")
	print('use_cuda: ', use_cuda)

	model = CNN().to(device)

	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)

	num_epochs = 85

	# Evaluate the randomly initialized model
	_evaluate_epoch(tr_loader, va_loader, model, device, criterion, 0)
	print('Started Training')
	best_test_acc = 0
	for epoch in range(0, num_epochs):
		print('epoch: {}'.format(epoch))
		# Train model
		_train_epoch(tr_loader, model, device, criterion, optimizer)
		# Evaluate model
		best_test_acc = _evaluate_epoch(tr_loader, va_loader, model, device, criterion, best_test_acc)
		print('\n')
	print('Finished Training')

if __name__ == "__main__":
	main()