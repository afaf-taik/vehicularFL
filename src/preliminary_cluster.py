#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar

import time

def flip_labels_clusters(args,dataset_train, dataset_test, n_clusters = 4 ,original_labels= [1,7,3,5], alternative_labels=[7,1,5,3]):
	dataset_train1 = []
	dataset_test1 = []
	c = 0
	i = 0
	while(i<len(dataset_train)):        
		if(i<int((c+1)*len(dataset_train)/n_clusters)):
			img,lbl = dataset_train[int(i)]
			if(lbl==original_labels[c]):
					#ds1 = list(dataset_train[int(i)])
				ds = img, alternative_labels[c]
				dataset_train1.append(tuple(ds))
			else:
				dataset_train1.append(dataset_train[int(i)])
			i = i+1        
		else:
			c = c+1

		#cluster_ids = [a for a in range(c*len(dataset_test)/n_clusters,(c+1)*len(dataset_test)/n_clusters)]

	c = 0
	i = 0
	while(i<len(dataset_test)):        
		if(i<int((c+1)*len(dataset_test)/n_clusters)):
			img,lbl = dataset_test[int(i)]
			if(lbl==original_labels[c]):
					#ds1 = list(dataset_train[int(i)])
				ds = img, alternative_labels[c]
				dataset_test1.append(tuple(ds))
			else:
				dataset_test1.append(dataset_test[int(i)])
			i = i+1        
		else:
			c = c+1

	return dataset_train1, dataset_test1


if __name__ == '__main__':
	args = args_parser()
	if args.gpu:
		torch.cuda.set_device(args.gpu)
	device = 'cuda' if args.gpu else 'cpu'

	# load datasets
	train_dataset1, test_dataset1, _ = get_dataset(args)
	train_dataset, test_dataset = flip_labels_clusters(args,train_dataset1, test_dataset1)
	# BUILD MODEL
	if args.model == 'cnn':
		# Convolutional neural netork
		if args.dataset == 'mnist':
			global_model = CNNMnist(args=args)
		elif args.dataset == 'fmnist':
			global_model = CNNFashion_Mnist(args=args)
		elif args.dataset == 'cifar':
			global_model = CNNCifar(args=args)
	elif args.model == 'mlp':
		# Multi-layer preceptron
		img_size = train_dataset[0][0].shape
		len_in = 1
		for x in img_size:
			len_in *= x
			global_model = MLP(dim_in=len_in, dim_hidden=64,
							   dim_out=args.num_classes)
	else:
		exit('Error: unrecognized model')

	# Set the model to train and send it to device.
	global_model1 = copy.deepcopy(global_model)
   
	global_model.to(device)
	global_model.train()
	print(global_model)
	global_model1.to(device)
	global_model1.train()    
	# Training
	# Set optimizer and criterion
	if args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
									momentum=0.5)
		optimizer1 = torch.optim.SGD(global_model1.parameters(), lr=args.lr,
									momentum=0.5)

	elif args.optimizer == 'adam':
		optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
									 weight_decay=1e-4)
		optimizer1 = torch.optim.Adam(global_model1.parameters(), lr=args.lr,
									 weight_decay=1e-4)


	trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	trainloader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)    
	criterion = torch.nn.NLLLoss().to(device)
	epoch_loss = []
	epoch_loss1 = []    
	times = []
	args.epochs = 100


	for epoch in tqdm(range(args.epochs)):
		batch_loss = []
		
		for batch_idx, (images, labels) in enumerate(trainloader):
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = global_model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
			if batch_idx % 50 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch+1, batch_idx * len(images), len(trainloader.dataset),
					100. * batch_idx / len(trainloader), loss.item()))
			batch_loss.append(loss.item())
		loss_avg = sum(batch_loss)/len(batch_loss)
		print('\nTrain loss:', loss_avg)
		epoch_loss.append(loss_avg)

	for epoch in tqdm(range(args.epochs)):
		batch_loss1 = []
		for batch_idx, (images, labels) in enumerate(trainloader1):
			
			images, labels = images.to(device), labels.to(device)

			optimizer1.zero_grad()
			outputs = global_model1(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer1.step()
			
			if batch_idx % 50 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch+1, batch_idx * len(images), len(trainloader1.dataset),
					100. * batch_idx / len(trainloader1), loss.item()))
			batch_loss1.append(loss.item())

		loss_avg = sum(batch_loss1)/len(batch_loss1)
		print('\nTrain loss:', loss_avg)
		epoch_loss1.append(loss_avg)

	# Plot loss
	plt.figure()
	plt.grid()
	plt.plot(range(len(epoch_loss)), epoch_loss, color = 'r', label='skewed distribution')
	plt.plot(range(len(epoch_loss1)), epoch_loss1, color = 'b',label='i.i.d vanilla distribution')
	plt.xlabel('epochs')
	plt.ylabel('Train loss')
	plt.legend()
	plt.savefig('differences_1.png'.format(args.dataset, args.model,
												 args.epochs))
#    torch.save(global_model.state_dict(), 'model_mlp_size_test.pkl')
	# testing
	test_acc, test_loss = test_inference(args, global_model, test_dataset)
	test_acc1, test_loss1 = test_inference(args, global_model1, test_dataset1)    
	print('Test on', len(test_dataset), 'samples')
	print("Test Accuracy skewed: {:.2f}%".format(100*test_acc))
	print("Test Accuracy vanilla: {:.2f}%".format(100*test_acc1))