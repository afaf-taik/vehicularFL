import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

import os

import read_results
tries = 100

experiment_num = '50u_1e250000S'
dirage = 'simulations_age_'+experiment_num
dirdiv = 'simulations_imp_'+experiment_num
resdir = 'diversity_exp/'+experiment_num
#os.mkdir(resdir)

for t in range(28,tries):
	dirname = resdir+'/exp'+str(t)
	os.mkdir(dirname)
	start_time = time.time()

	# define paths
	path_project = os.path.abspath('..')
	logger = SummaryWriter('logs')

	args = args_parser()
	exp_details(args)

	if args.gpu:
		torch.cuda.set_device(args.gpu)
	device = 'cuda' if args.gpu else 'cpu'

	# load dataset and user groups
	train_dataset, test_dataset, user_groups = read_results.load_data(t)
	args.model = 'mlp'
	args.dataset = 'mnist'	
	args.epochs = 30
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
	global_model2 = copy.deepcopy(global_model)

	global_model.to(device)
	global_model.train()

	global_model2.to(device)
	global_model2.train()
	print(global_model)

	# copy weights
	global_weights = global_model.state_dict()
	global_weights2 = global_model2.state_dict()
	# Training
	train_loss, train_accuracy = [], []
	test_lossL, test_accuracyL = [], []
	test_loss2L, test_accuracy2L = [], []

	val_acc_list, net_list = [], []
	cv_loss, cv_acc = [], []
	train_loss2, train_accuracy2 = [], []
	val_acc_list2, net_list2 = [], []
	cv_loss2, cv_acc2 = [], []
	print_every = 2

	val_loss_pre, counter = 0, 0
	for epoch in tqdm(range(args.epochs)):
		local_weights, local_losses = [], []
		print(f'\n | Global Training Round : {epoch+1} |\n')
		global_model.train()
		global_model2.train()
		#m = max(int(args.frac * args.num_users), 1)
		idxs_users_rnd = read_results.load(t,epoch,dirage)
		#modifyyy heeeeere    
		sizes=[]
		for idx in idxs_users_rnd:
			#print('Data size of user', idx , ':', len(user_groups[idx]))
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model), global_round=epoch)
			sizes.append(len(user_groups[idx]))
			local_weights.append(copy.deepcopy(w))
			local_losses.append(copy.deepcopy(loss))
   
		# update global weights
		global_weights = average_weights(local_weights,sizes)

		# update global weights
		global_model.load_state_dict(global_weights)

		loss_avg = sum(local_losses) / len(local_losses)
		train_loss.append(loss_avg)

		# Calculate avg training accuracy over all users at every epoch
		list_acc, list_loss = [], []
		global_model.eval()
		for c in range(args.num_users):
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			acc, loss = local_model.inference(model=global_model)
			list_acc.append(acc)
			list_loss.append(loss)
		train_accuracy.append(sum(list_acc)/len(list_acc))

		# print global training loss after every 'i' rounds
		if (epoch+1) % print_every == 0:
			print(f' \nAvg Training Stats after {epoch+1} global rounds:')
			print(f'Training Loss for random: {np.mean(np.array(train_loss))}')
			print('Train Accuracy for random: {:.2f}% \n'.format(100*train_accuracy[-1]))


#####################################################################################################################
		idxs_users_opt = read_results.load(t,epoch,dirdiv)
		sizes2=[]
		local_weights2, local_losses2 = [], []
		for idx in idxs_users_opt:
			print('Data size of user', idx , ':', len(user_groups[idx]))
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model2), global_round=epoch)
			sizes2.append(len(user_groups[idx]))
			local_weights2.append(copy.deepcopy(w))
			local_losses2.append(copy.deepcopy(loss))
		# update global weights
		global_weights = average_weights(local_weights2,sizes2)
	#   global_weights = average_weights(local_weights)

		# update global weights
		global_model2.load_state_dict(global_weights)

		loss_avg2 = sum(local_losses2) / len(local_losses2)
		train_loss2.append(loss_avg2)

		# Calculate avg training accuracy over all users at every epoch
		list_acc2, list_loss2 = [], []
		global_model2.eval()
		for c in range(args.num_users):
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			acc, loss = local_model.inference(model=global_model2)
			list_acc2.append(acc)
			list_loss2.append(loss)
		train_accuracy2.append(sum(list_acc2)/len(list_acc2))

		# print global training loss after every 'i' rounds
		if (epoch+1) % print_every == 0:
			print(f' \nAvg Training Stats after {epoch+1} global rounds:')
			print(f'Training Loss for opt : {np.mean(np.array(train_loss))}')
			print('Train Accuracy for opt: {:.2f}% \n'.format(100*train_accuracy[-1]))
	# Test inference after completion of training
		test_acc, test_loss = test_inference(args, global_model, test_dataset)
		test_acc2, test_loss2 = test_inference(args, global_model2, test_dataset)
		test_accuracyL.append(test_acc)
		test_lossL.append(test_loss)
		test_accuracy2L.append(test_acc2)
		test_loss2L.append(test_loss2L)



##########################################################################################################
	print(f' \n Results after {args.epochs} global rounds of training:')
	print("|---- Avg Train Accuracy for random: {:.2f}%".format(100*train_accuracy[-1]))
	print("|---- Test Accuracy for random: {:.2f}%".format(100*test_acc))

	print(f' \n Results after {args.epochs} global rounds of training:')
	print("|---- Avg Train Accuracy for opt: {:.2f}%".format(100*train_accuracy2[-1]))
	print("|---- Test Accuracy for opt: {:.2f}%".format(100*test_acc2))


	# Saving the objects train_loss and train_accuracy:
	file_name = dirname+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_random.pkl'.\
		format(args.dataset, args.model, args.epochs, args.frac, args.iid,
			   args.local_ep, args.local_bs)
	#file_name = 'save/objects/model'+ str(args.dataset)+'_'+ str(args.model)+'.p' 
	with open(file_name, 'wb') as f:
		pickle.dump([train_loss, train_accuracy,test_lossL,test_accuracyL], f)
	print('TRAIN LOSS for RND IS :')	
	print(train_loss)	

	# Saving the objects train_loss and train_accuracy:
	file_name = dirname+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_opt.pkl'.\
		format(args.dataset, args.model, args.epochs, args.frac, args.iid,
			   args.local_ep, args.local_bs)
	#file_name = 'save/objects/model'+ str(args.dataset)+'_'+ str(args.model)+'.p' 
	with open(file_name, 'wb') as f:
		pickle.dump([train_loss2, train_accuracy2,test_loss2L,test_accuracy2L], f)


	print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

	# PLOTTING (optional)
	import matplotlib
	import matplotlib.pyplot as plt
	matplotlib.use('Agg')

	# Plot Loss curve
	plt.figure()
	plt.title('Training Loss vs Communication rounds')
	plt.plot(range(len(train_loss)), train_loss, color='r',label = 'age based')
	plt.plot(range(len(train_loss2)), train_loss2, color='b',label = 'diversity based')
	plt.legend()
	plt.ylabel('Training loss')
	plt.xlabel('Communication Rounds')
	plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))
	
	# # Plot Average Accuracy vs Communication rounds
	plt.figure()
	plt.title('Average Accuracy vs Communication rounds')
	plt.plot(range(len(train_accuracy)), train_accuracy, color='r',label = 'age based')
	plt.plot(range(len(train_accuracy2)), train_accuracy2, color='b',label = 'diversity based')
	plt.ylabel('Average Accuracy')
	plt.xlabel('Communication Rounds')
	plt.legend()
	plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))
