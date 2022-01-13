import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import statistics
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, get_gini, get_order, get_entropy, get_importance
from vehicular_utils import dataRateRB, required_rate, cost_in_RBs ,allocate, capacities, create_wts_dict, create_wt_doubledict, solve_wbm,print_solution,get_selected_edges, canUploadV2V
from VehicularEnv import Freeway,V2IChannels,V2VChannels, Vehicle
from pulp import *
#Size of the model
modelsize = 160000
Power = 0.1
tries = 50
Pmax = 1 #Watt #5 Watt is a high. You can use 1 Watt as the max.
Ttrain_batch = 0.005 #seconds , time to train on a batch of SGD 
num_lanes = 6  
#We can also suppose that the width is negligible 
numEpochs = 1  
D = 2000 # Radius of 2km 
#TOTAL_RBs = 4 #Total number of RBs ( to be adjusted )
BW = 180000   # Bandwidth of a RB Hz
#BW = 1000000
sig2_dBm = -114 # additive white Gaussian noise (dB)
sig2_watts = 10 ** ((sig2_dBm-30)/10) # additive white Gaussian noise (watts)
NCHANNELS = 4
experiment_name = 'mobility_normalnoniid_new'
path = 'diversity_exp/mobility/'+experiment_name
os.mkdir(path)
Nmax = 2
for t in range(tries):
	dirname = path + '/exp'+str(t)
	os.mkdir(dirname)
	start_time = time.time()
	# define paths

	path_project = os.path.abspath('..')
	logger = SummaryWriter('logs')

	args = args_parser()
	exp_details(args)
	#m = args.num_users
	if args.gpu:
		torch.cuda.set_device(args.gpu)
	device = 'cuda' if args.gpu else 'cpu'

	# load dataset and user groups
	train_dataset, test_dataset, user_groups = get_dataset(args)				
	# BUILD MODEL

	args.model = 'mlp'
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
	global_model1.to(device)
	
	global_model.train()
	global_model1.train()

	# copy weights
	global_weights = global_model.state_dict()
	global_weights1 = global_model1.state_dict()	
	# Training

	train_loss, train_accuracy = [], []
	#train_loss1, train_accuracy1 = [], []
	# train_loss, train_accuracy = [], []
	train_loss1, train_accuracy1 = [], []		
	

	test_lossL, test_accuracyL = [], []
	#test_loss1L, test_accuracy1L = [], []
	# test_lossL, test_accuracyL = [], []
	test_loss1L, test_accuracy1L = [], []

	val_acc_list, net_list = [], []
	cv_loss, cv_acc = [], []
	train_loss, train_accuracy = [], []
	val_acc_list, net_list = [], []
	cv_loss, cv_acc = [], []


	print_every = 500

	val_loss_pre, counter = 0, 0
	R = np.ones(args.num_users)
	E = get_gini(user_groups,train_dataset)
	S = [len(user_groups[idx]) for idx in range(len(user_groups))]
	A = np.zeros(args.num_users)
	A1 = np.zeros(args.num_users)
	total_size = sum(S)
	Ttrain = [numEpochs*len(user_groups[i])*Ttrain_batch/args.local_bs for i in range(args.num_users)]

	Rep1 = [E[i]+S[i]/total_size for i in range(args.num_users)]

	#print(env.channelgains)
	for epoch in tqdm(range(args.epochs)):
		env = Freeway(num_vehicles = args.num_users, num_slots = 1, num_channels = NCHANNELS)
		env.build_environment()			
		local_weights1, local_losses1 = [], []					
		global_model1.train()
		model = copy.deepcopy(global_model)
		#cost = required_bandwidth(T, Ttrain, args)
		######################################################################################################
		div_indicator = get_importance(E,S,A1,args.epochs)
		LLT_ = env.LLT()
		print(LLT_)
		Preferences = np.ones((args.num_users,1))
############################Allocate test##########################			
		Tk = env.rate_stay()
		r_min = required_rate(Ttrain, Tk, modelsize)

		cost, rb = cost_in_RBs(r_min, args, env.channelgains, NCHANNELS)
		print(cost)
		
		size = [len(user_groups[i]) for i in range(args.num_users)]
		a,b = allocate(div_indicator, cost, rb, r_min,NCHANNELS)
		print(a)
		print(b)
		NotSelected = [i for i in range(args.num_users)]
		for x in a:
			NotSelected.remove(x)
		dict_heads, dict_followers = capacities(a, NotSelected, Nmax)
		print(dict_heads, dict_followers)
		Twait = 2
		wts_time = canUploadV2V(args,a,NotSelected, env.channelgains, Ttrain, Twait, LLT_, NCHANNELS, Nmax)
		wts = create_wts_dict(a, NotSelected, Preferences, wts_time)
		wt = create_wt_doubledict(a, NotSelected,wts)
		p = solve_wbm(a, NotSelected, wt, dict_heads , dict_followers)
		print_solution(p)    
		selected_edges = get_selected_edges(p)
		print(selected_edges)
		all_participants = a
		for x in selected_edges:
			_,s = x
			all_participants.append(int(s))

		print(all_participants)

		#ordered = allocate(Rep1,cost)
		idxs_users_opt1 = all_participants
		################################ Calculate LLT
		################################ Match and add newly joint to the list 
		################################ Save length of list 
		sizes1 = []
		local_weights1, local_losses1 = [], []
		

		for idx in idxs_users_opt1:
			print('Data size of user', idx , ':', len(user_groups[idx]))
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model1), global_round=epoch)
			sizes1.append(len(user_groups[idx]))
			local_weights1.append(copy.deepcopy(w))
			local_losses1.append(copy.deepcopy(loss))
		for U in range(args.num_users):
			if(U in idxs_users_opt1):
				A1[U] = 0
			else:
				A1[U] +=1    
		# update global weights
		global_weights1 = average_weights(local_weights1,sizes1)
	#   global_weights = average_weights(local_weights)

		# update global weights
		global_model1.load_state_dict(global_weights1)

		loss_avg1 = sum(local_losses1) / len(local_losses1)
		train_loss1.append(loss_avg1)

		# Calculate avg training accuracy over all users at every epoch
		list_acc1, list_loss1 = [], []
		global_model1.eval()
		for c in range(args.num_users):
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			acc, loss = local_model.inference(model=global_model1)
			list_acc1.append(acc)
			list_loss1.append(loss)
		train_accuracy1.append(sum(list_acc1)/len(list_acc1))

	# Test inference after completion of training
		test_acc1, test_loss1 = test_inference(args, global_model1, test_dataset)
		test_accuracy1L.append(test_acc1)	
	######################################################################################################

		local_weights, local_losses = [], []					
		global_model.train()
		model = copy.deepcopy(global_model)

		#####################insert the selection of other vehicles here 
		idxs_users_opt = np.random.choice(range(args.num_users), NCHANNELS, replace=False) # + the other ones using VV		
		sizes = []
		local_weights, local_losses = [], []
		for idx in idxs_users_opt:
			print('Data size of user', idx , ':', len(user_groups[idx]))
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model), global_round=epoch)
			sizes.append(len(user_groups[idx]))
			local_weights.append(copy.deepcopy(w))
			local_losses.append(copy.deepcopy(loss))
		for U in range(args.num_users):
			if(U in idxs_users_opt):
				A[U] = 0
			else:
				A[U] +=1			    
		# update global weights
		global_weights = average_weights(local_weights,sizes)
	#   global_weights = average_weights(local_weights)

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
	# Saving the objects train_loss and train_accuracy:
	file_name = dirname+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_proposed.pkl'.\
		format(args.dataset, args.model, args.epochs, args.frac, args.iid,
			   args.local_ep, args.local_bs)
	#file_name = 'save/objects/model'+ str(args.dataset)+'_'+ str(args.model)+'.p' 
	with open(file_name, 'wb') as f:
		pickle.dump([train_loss1, train_accuracy1,test_loss1L,test_accuracy1L], f)

	file_name = dirname+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_vanilla.pkl'.\
		format(args.dataset, args.model, args.epochs, args.frac, args.iid,
			   args.local_ep, args.local_bs)
	#file_name = 'save/objects/model'+ str(args.dataset)+'_'+ str(args.model)+'.p' 
	with open(file_name, 'wb') as f:
		pickle.dump([train_loss, train_accuracy,test_lossL,test_accuracyL], f)

# 	print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

	# PLOTTING (optional)
	import matplotlib
	import matplotlib.pyplot as plt
	matplotlib.use('Agg')

	# Plot Loss curve
	plt.figure()
	plt.title('Training Loss vs Communication rounds')
	plt.plot(range(len(train_loss1)), train_loss1, color='r')
	plt.plot(range(len(train_loss)), train_loss, color='b')

	plt.ylabel('Training loss')
	plt.xlabel('Communication Rounds')
	plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))
	
	# # # Plot Average Accuracy vs Communication rounds
	plt.figure()
	plt.title('Average Accuracy vs Communication rounds')
	#plt.plot(range(len(train_accuracy)), train_accuracy, color='r')
	plt.plot(range(len(train_accuracy1)), train_accuracy1, color='r')
	plt.plot(range(len(train_accuracy)), train_accuracy, color='b')
	plt.ylabel('Average Accuracy')
	plt.xlabel('Communication Rounds')
	plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))
	plt.figure()
	plt.title('Average Accuracy vs Communication rounds')
	#plt.plot(range(len(test_accuracyL)), test_accuracyL, color='r')
	plt.plot(range(len(test_accuracy1L)), test_accuracy1L, color='r')
	plt.plot(range(len(test_accuracyL)), test_accuracyL, color='b')
	plt.ylabel('Average Accuracy')
	plt.xlabel('Communication Rounds')
	plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc_test.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))
