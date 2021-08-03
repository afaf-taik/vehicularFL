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
from utils import get_dataset, average_weights, exp_details, get_gini, get_order, get_entropy, get_importance, get_reputation, get_measure, flip_labels

from scheduling import train_time, datarate, allocate, required_bandwidth

#Size of the model
S = 160000
#Total Bandwidth = 1Mhz
B = 1000000
#inverse of N0 
N10 = 0.25*10**15
T = 300
P = 0.2

tries = 10
m = 5
for t in range(tries):
	dirname = 'diversity_exp/priority4/exp'+str(t)
	os.mkdir(dirname)
	start_time = time.time()
	# define paths

	path_project = os.path.abspath('..')
	logger = SummaryWriter('logs')

	args = args_parser()
	exp_details(args)
	m = args.num_users
	if args.gpu:
		torch.cuda.set_device(args.gpu)
	device = 'cuda' if args.gpu else 'cpu'

	# load dataset and user groups
	# args.dataset = 'fmnist'
	train_dataset1, test_dataset, user_groups = get_dataset(args)				
	# BUILD MODEL
	Ttrain = train_time(args,user_groups)
	malicious_ids = np.random.choice(range(args.num_users), args.malicious, replace=False)
	train_dataset = flip_labels(user_groups,train_dataset1, malicious_ids)

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
	global_model2 = copy.deepcopy(global_model)
	global_model3 = copy.deepcopy(global_model)
	
	global_model.to(device)
	global_model1.to(device)
	global_model2.to(device)
	global_model3.to(device)
	
	global_model.train()
	global_model1.train()
	global_model2.train()
	global_model3.train()	

	# copy weights
	global_weights = global_model.state_dict()
	global_weights1 = global_model1.state_dict()
	global_weights2 = global_model2.state_dict()
	global_weights3 = global_model3.state_dict()	
	# Training

	train_loss, train_accuracy = [], []
	train_loss1, train_accuracy1 = [], []
	train_loss2, train_accuracy2 = [], []
	train_loss3, train_accuracy3 = [], []		
	

	test_lossL, test_accuracyL = [], []
	test_loss1L, test_accuracy1L = [], []
	test_loss2L, test_accuracy2L = [], []
	test_loss3L, test_accuracy3L = [], []

	val_acc_list, net_list = [], []
	cv_loss, cv_acc = [], []
	train_loss2, train_accuracy2 = [], []
	val_acc_list2, net_list2 = [], []
	cv_loss2, cv_acc2 = [], []


	print_every = 500

	val_loss_pre, counter = 0, 0
	R = np.ones(args.num_users)
	E = get_gini(user_groups,train_dataset)
	S = [len(user_groups[idx]) for idx in range(len(user_groups))]
	A = np.zeros(args.num_users)
	A3 = np.zeros(args.num_users)
	total_size = sum(S)

	Rep3 = [E[i]+S[i]/total_size for i in range(args.num_users)]
	r = []
	Rep1  = R   #Priority index = Reputation 
	Rep2  = get_measure(Rep3,R,0.5,0.5)   #Priority index = Reputation +diversity
	Avg  = []

	Rep1L = []
	Rep2L = []
	Rep3L = []

	for epoch in tqdm(range(args.epochs)):
		
		local_weights, local_losses = [], []
		local_weights1, local_losses1 = [], []
		local_weights2, local_losses2 = [], []
		local_weights3, local_losses3 = [], []				
		
		Rep1L.append(Rep1)
		Rep2L.append(Rep2)
		Rep3L.append(Rep3)		
		global_model.train()
		global_model1.train()
		global_model2.train()
		global_model3.train()

		model = copy.deepcopy(global_model)

		cost = required_bandwidth(T, Ttrain, args)
		######################################################################################################
		

		local_accuracies1 = []
		test_accuracies1 = []
		sizes1 = []
		ordered = allocate(Rep1,cost)
		idxs_users_opt1 = ordered
		
		for idx in idxs_users_opt1:
			#print('Data size of user', idx , ':', len(user_groups[idx]))
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model1), global_round=epoch)
			sizes1.append(len(user_groups[idx]))
			local_weights1.append(copy.deepcopy(w))
			local_losses1.append(copy.deepcopy(loss))
			#test models
			model.load_state_dict(w)
			acc, loss = local_model.inference(model=model)
			
			test_acc, test_loss = test_inference(args, model, test_dataset)
		# 	acc = local_model.train_eval(dataset= train_dataset, idxs = user_groups[idx], model=model)
			test_accuracies1.append(test_acc) 
			local_accuracies1.append(acc)		

		avg = np.average(test_accuracies1)
		Avg.append(avg)
		print(len(Rep1))
		print(test_accuracies1)
		print(local_accuracies1)
		Rep1 = get_reputation(local_accuracies1, test_accuracies1, idxs_users_opt1, Rep1)
		global_weights1 = average_weights(local_weights1,sizes1)
		# update global weights
		global_model1.load_state_dict(global_weights1)
		loss_avg1 = sum(local_losses1) / len(local_losses1)
		train_loss1.append(loss_avg1)

		# Calculate avg training accuracy over all users at every epoch
		list_acc1, list_loss1 = [], []
		global_model.eval()
		for c in range(args.num_users):
			local_model = LocalUpdate(args=args, dataset=train_dataset1,
									  idxs=user_groups[idx], logger=logger)
			acc, loss = local_model.inference(model=global_model1)
			list_acc1.append(acc)
			list_loss1.append(loss)
		train_accuracy1.append(sum(list_acc1)/len(list_acc1))

		######################################################################################################

		local_accuracies2 = []
		test_accuracies2 = []
		sizes2 = []
		print(Rep2)
		print('=====cost======',cost)
		print([Rep2[a]/cost[a] for a in range(len(Rep2))])
		ordered = allocate(Rep2,cost)
		print(ordered)

		idxs_users_opt2 = ordered
		print(idxs_users_opt2)
		for idx in idxs_users_opt2:
			local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model2), global_round=epoch)
			sizes2.append(len(user_groups[idx]))
			local_weights2.append(copy.deepcopy(w))
			local_losses2.append(copy.deepcopy(loss))
			#test models
			model.load_state_dict(w)
			acc2, loss2 = local_model.inference(model=model)
			
			test_acc2, test_loss2 = test_inference(args, model, test_dataset)
		# 	acc = local_model.train_eval(dataset= train_dataset, idxs = user_groups[idx], model=model)
			test_accuracies2.append(test_acc2) 
			local_accuracies2.append(acc2)
		print(local_accuracies2)			

		# avg = np.average(test_accuracies2)
		# Avg.append(avg)
		Rep2 = get_reputation(local_accuracies2, test_accuracies2, idxs_users_opt2, Rep2)
		global_weights2 = average_weights(local_weights2,sizes2)
		# update global weights
		global_model2.load_state_dict(global_weights2)
		loss_avg2 = sum(local_losses2) / len(local_losses2)
		train_loss2.append(loss_avg2)


		# Calculate avg training accuracy over all users at every epoch
		list_acc2, list_loss2 = [], []
		global_model.eval()
		for c in range(args.num_users):
			local_model = LocalUpdate(args=args, dataset=train_dataset1,
									  idxs=user_groups[idx], logger=logger)
			acc, loss = local_model.inference(model=global_model2)
			list_acc2.append(acc)
			list_loss2.append(loss)
			if(c in idxs_users_opt2):
				A[c] = 0
			else:
				A[c] +=1
		train_accuracy2.append(sum(list_acc2)/len(list_acc2))
	# Save reputation in an external file
		I2 = get_importance(E,S,A,args.epochs)
		print('======================================I am here=========================',I2)
		Rep2 = get_measure(I2,Rep2,0.5,0.5)



	# file_name = dirname+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_rep2.pkl'.\
	# 	format(args.dataset, args.model, args.epochs, args.frac, args.iid,
	# 		   args.local_ep, args.local_bs)
	# with open(file_name, 'wb') as f:
	# 	pickle.dump([train_loss2, train_accuracy2,test_loss2L,test_accuracy2L], f)


#####################################################################################################################
		Rep3 = get_importance(E,S,A3,args.epochs)
		ordered = allocate(Rep3,cost)
		idxs_users_opt3 = ordered
		sizes3 = []
		local_weights3, local_losses3 = [], []
		for idx in idxs_users_opt3:
			print('Data size of user', idx , ':', len(user_groups[idx]))
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model3), global_round=epoch)
			sizes3.append(len(user_groups[idx]))
			local_weights3.append(copy.deepcopy(w))
			local_losses3.append(copy.deepcopy(loss))
		for U in range(args.num_users):
			if(U in idxs_users_opt3):
				A3[U] = 0
			else:
				A3[U] +=1    
		# update global weights
		global_weights3 = average_weights(local_weights3,sizes3)
	#   global_weights = average_weights(local_weights)

		# update global weights
		global_model3.load_state_dict(global_weights3)

		loss_avg3 = sum(local_losses3) / len(local_losses3)
		train_loss3.append(loss_avg3)

		# Calculate avg training accuracy over all users at every epoch
		list_acc3, list_loss3 = [], []
		global_model3.eval()
		for c in range(args.num_users):
			local_model = LocalUpdate(args=args, dataset=train_dataset1,
									  idxs=user_groups[idx], logger=logger)
			acc, loss = local_model.inference(model=global_model3)
			list_acc3.append(acc)
			list_loss3.append(loss)
		train_accuracy3.append(sum(list_acc3)/len(list_acc3))

	# file_name = dirname+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_rep3.pkl'.\
	# 	format(args.dataset, args.model, args.epochs, args.frac, args.iid,
	# 		   args.local_ep, args.local_bs)
	# with open(file_name, 'wb') as f:
	# 	pickle.dump([train_loss3, train_accuracy3,test_loss3L,test_accuracy3L], f)


		# print global training loss after every 'i' rounds
	# Test inference after completion of training
		test_acc, test_loss = test_inference(args, global_model, test_dataset)
		test_acc1, test_loss1 = test_inference(args, global_model1, test_dataset)
		test_acc2, test_loss2 = test_inference(args, global_model2, test_dataset)
		test_acc3, test_loss3 = test_inference(args, global_model3, test_dataset)

		test_accuracyL.append(test_acc)
		test_accuracy1L.append(test_acc1)
		test_accuracy2L.append(test_acc2)
		test_accuracy3L.append(test_acc3)		#test_loss2L.append(test_loss2L)



# ##########################################################################################################
# 	print(f' \n Results after {args.epochs} global rounds of training:')
# 	print("|---- Avg Train Accuracy for random: {:.2f}%".format(100*train_accuracy[-1]))
# 	print("|---- Test Accuracy for random: {:.2f}%".format(100*test_acc))

# 	print(f' \n Results after {args.epochs} global rounds of training:')
# 	print("|---- Avg Train Accuracy for opt: {:.2f}%".format(100*train_accuracy2[-1]))
# 	print("|---- Test Accuracy for opt: {:.2f}%".format(100*test_acc2))
	

	# Saving the objects train_loss and train_accuracy:
	file_name = dirname+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_r1.pkl'.\
		format(args.dataset, args.model, args.epochs, args.frac, args.iid,
			   args.local_ep, args.local_bs)
	#file_name = 'save/objects/model'+ str(args.dataset)+'_'+ str(args.model)+'.p' 
	with open(file_name, 'wb') as f:
		pickle.dump([train_loss1, train_accuracy1,test_loss1L,test_accuracy1L,Rep1L], f)

	# Saving the objects train_loss and train_accuracy:
	file_name = dirname+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_r2.pkl'.\
		format(args.dataset, args.model, args.epochs, args.frac, args.iid,
			   args.local_ep, args.local_bs)
	#file_name = 'save/objects/model'+ str(args.dataset)+'_'+ str(args.model)+'.p' 
	with open(file_name, 'wb') as f:
		pickle.dump([train_loss2, train_accuracy2,test_loss2L,test_accuracy2L,Rep2L], f)


	# Saving the objects train_loss and train_accuracy:
	file_name = dirname+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_r3.pkl'.\
		format(args.dataset, args.model, args.epochs, args.frac, args.iid,
			   args.local_ep, args.local_bs)
	#file_name = 'save/objects/model'+ str(args.dataset)+'_'+ str(args.model)+'.p' 
	with open(file_name, 'wb') as f:
		pickle.dump([train_loss3, train_accuracy3,test_loss3L,test_accuracy3L, Rep3L], f)

# 	print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

	# PLOTTING (optional)
	import matplotlib
	import matplotlib.pyplot as plt
	matplotlib.use('Agg')

	# # Plot Loss curve
	# plt.figure()
	# plt.title('Training Loss vs Communication rounds')
	# plt.plot(range(len(train_loss)), train_loss, color='r')
	# plt.plot(range(len(train_loss2)), train_loss2, color='b')

	# plt.ylabel('Training loss')
	# plt.xlabel('Communication Rounds')
	# plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
	# 			 format(args.dataset, args.model, args.epochs, args.frac,
	# 					args.iid, args.local_ep, args.local_bs))
	
	# # # Plot Average Accuracy vs Communication rounds
	plt.figure()
	plt.title('Average Accuracy vs Communication rounds')
	#plt.plot(range(len(train_accuracy)), train_accuracy, color='r')
	plt.plot(range(len(train_accuracy1)), train_accuracy1, color='g')
	plt.plot(range(len(train_accuracy2)), train_accuracy2, color='b')
	plt.plot(range(len(train_accuracy3)), train_accuracy3, color='y')
	plt.ylabel('Average Accuracy')
	plt.xlabel('Communication Rounds')
	plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))
	plt.figure()
	plt.title('Average Accuracy vs Communication rounds')
	#plt.plot(range(len(test_accuracyL)), test_accuracyL, color='r')
	plt.plot(range(len(test_accuracy1L)), test_accuracy1L, color='g')
	plt.plot(range(len(test_accuracy2L)), test_accuracy2L, color='b')
	plt.plot(range(len(test_accuracy3L)), test_accuracy3L, color='y')
	plt.ylabel('Average Accuracy')
	plt.xlabel('Communication Rounds')
	plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc_test.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))
