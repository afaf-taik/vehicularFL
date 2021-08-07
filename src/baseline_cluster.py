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
from utils import get_dataset, average_weights, exp_details, models_similarity,\
				 flip_labels_clusters, newmodel,build_model_by_cluster,determine_preferred_model,\
				 get_gini, get_order, get_entropy , get_importance

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcl
from scipy.spatial.distance import squareform
tries = 10
modelname = 'clustering'
resdir = 'diversity_exp/' + 'baseline_'+ modelname 
dirname = resdir
os.mkdir(resdir)
test_time = 10
n_clusters_max = 3
m = 12
n = 3
for i in range (tries):
	expdir = resdir+'/exp'+str(i)
	os.mkdir(expdir)
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
	train_dataset1, test_dataset1, user_groups = get_dataset(args)
	train_dataset, test_dataset, clusters = flip_labels_clusters(args,user_groups,train_dataset1,test_dataset1)
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
	global_model.to(device)
	global_model.train()
	print(global_model)
	global_model2 = copy.deepcopy(global_model)
	global_model2.to(device)
	global_model2.train()
	# copy weights
	global_weights = global_model.state_dict()
	global_weights2 = global_model2.state_dict()
	# Training
	train_loss, train_accuracy = [], []
	test_lossL, test_accuracyL = [], []
	val_acc_list, net_list = [], []
	cv_loss, cv_acc = [], []
	print_every = 10
	val_loss_pre, counter = 0, 0
	similarities = []
	E = get_gini(user_groups,train_dataset)
	S = [len(user_groups[idx]) for idx in range(len(user_groups))]
	A = np.zeros(args.num_users)
	for epoch in tqdm(range(args.epochs)):
		local_weights, local_losses = [], []
		local_weights2, local_losses2 = [], []		
		print(f'\n | Global Training Round : {epoch+1} |\n')
		ordered = get_order(get_importance(E,S,A,args.epochs))
		idxs_users = ordered[:m]
		#m = max(int(args.frac * args.num_users), 1)    
		if (epoch<test_time):
			global_model.train()
			#idxs_users = np.random.choice(range(args.num_users), m, replace=False) 
			sizes=[]
			for idx in idxs_users:
				print('Data size of user', idx , ':', len(user_groups[idx]))
				local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
				w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model), global_round=epoch)
				sizes.append(len(user_groups[idx]))
				local_weights.append(copy.deepcopy(w))
				local_losses.append(copy.deepcopy(loss))
		
			global_weights = average_weights(local_weights,sizes)
			global_model.load_state_dict(global_weights)
			list_acc = []	    
			for c in range(args.num_users):
				local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
				acc, loss = local_model.inference(model=global_model)
				list_acc.append(acc)
			#list_loss2.append(loss)
			train_accuracy.append(sum(list_acc)/len(list_acc))		

		elif(epoch==test_time):
			global_model.train()
			#idxs_users = np.random.choice(range(args.num_users), m, replace=False) 
			sizes=[]
			for idx in idxs_users:
				print('Data size of user', idx , ':', len(user_groups[idx]))
				local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
				w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model), global_round=epoch)
				sizes.append(len(user_groups[idx]))
				local_weights.append(copy.deepcopy(w))
				local_losses.append(copy.deepcopy(loss))            
			similarity = models_similarity(local_weights)
			print(similarity)
			Z = hcl.linkage(squareform(similarity))
			print(Z)
			clusters_found = hcl.fcluster(Z,t=n_clusters_max,criterion='maxclust')
			print('here are the clusters', clusters_found)
			# dendro = hcl.dendrogram(Z)
			# plt.title('Dendrogram')
			# plt.ylabel('Cosine Similarity')
			# plt.show()
			models = [ copy.deepcopy(global_model) for j in range(n_clusters_max)]
			weights = [ models[j].state_dict() for j in range(len(models))]
			preferences = np.zeros(shape = (args.num_users, len(models)))
			for j in range(max(clusters_found)):
				print('model=====',j)
				newmodel(args,models[j],global_model)
				weights[j] = build_model_by_cluster(local_weights,sizes,clusters_found,j)
				models[j].load_state_dict(weights[j])
				models[j].eval()
				for g in range(args.num_users):
					local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[g], logger=logger)
					acc, _ = local_model.inference(model=models[j])
					preferences[g][j] = acc                    

					#list_acc.append(acc)
			print('======================PREFERENCES==================',preferences)            
			#choose cluster  
			preferred = determine_preferred_model(preferences)
			print(preferred)
			idx_users_clusters = []
			for j in range(max(clusters_found)):
				idx_users_clusters.append(np.argwhere(preferred == j))
			print(idx_users_clusters)
			idxs_users_flat = [ x.flatten() for x in idx_users_clusters]
			print(idxs_users)
			list_acc = []			
			for j in range(max(clusters_found)):
				models[j].eval()
				for a in idxs_users_flat[j]:
					local_model = LocalUpdate(args=args, dataset=train_dataset,
									   idxs=user_groups[idx], logger=logger)
					acc, _ = local_model.inference(model=models[j])
					list_acc.append(acc)							
				train_accuracy.append(sum(list_acc)/len(list_acc))			

		else:
			local_losses = []
			idx_all = get_order(get_importance(E,S,A,args.epochs))
			for j in range(max(clusters_found)):
				models[j].train()
				idx_h = np.where(idx_all in idxs_users_flat[j])[0]
				print(idx_h)
				idxs_users = np.random.choice(idxs_users_flat[j], min(n,len(idxs_users_flat[j])), replace=False) 
				if idx_h not in idxs_users:
					idxs_users = np.concatenate((idxs_users, idx_h))
				sizes = []
				local_weights = []
				for idx in idxs_users:
					print('Data size of user', idx , ':', len(user_groups[idx]))
					local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
					w, loss = local_model.update_weights(
							model=copy.deepcopy(models[j]), global_round=epoch)
					sizes.append(len(user_groups[idx]))
					local_weights.append(copy.deepcopy(w))
					local_losses.append(copy.deepcopy(loss))            
				global_weights = average_weights(local_weights,sizes)
				models[j].load_state_dict(global_weights)
		for U in range(args.num_users):
			if(U in idxs_users):
				A[U] = 0
			else:
				A[U] +=1
		loss_avg = sum(local_losses) / len(local_losses)
		train_loss.append(loss_avg)
	list_acc = []			
	for j in range(max(clusters_found)):
		models[j].eval()
		for a in idxs_users_flat[j]:
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									   idxs=user_groups[idx], logger=logger)
			acc, _ = local_model.inference(model=models[j])
			list_acc.append(acc)							
		train_accuracy.append(sum(list_acc)/len(list_acc))
	
#test all 
	list_acc_test = []
	for j in range(max(clusters_found)):
			models[j].eval()
			for a in idxs_users_flat[j]:
				local_model = LocalUpdate(args=args, dataset=train_dataset,
									   idxs=user_groups[idx], logger=logger)
				acc, _ = local_model.inference_test_local(model=models[j])
				list_acc_test.append(acc)							
			#test_accuracyL.append(sum(list_acc_test)/len(list_acc_test))		         	
		#test_acc, test_loss = test_inference(args, global_model, test_dataset)

	# Saving the objects train_loss and train_accuracy:
	file_name = expdir+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_proposed.pkl'.\
		format(args.dataset, args.model, args.epochs, args.frac, args.iid,
			   args.local_ep, args.local_bs)

	with open(file_name, 'wb') as f:
		pickle.dump([train_loss, train_accuracy, list_acc_test], f)

	# print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
	train_loss2, train_accuracy2,test_accuracyL2 = [], [], []
	for epoch in tqdm(range(args.epochs)):
		idxs_users_rnd = np.random.choice(range(args.num_users), m, replace=False)
		sizes2=[]
		local_weights2, local_losses2 = [], []
		for idx in idxs_users_rnd:
			print('Data size of user', idx , ':', len(user_groups[idx]))
			local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model2), global_round=epoch)
			sizes2.append(len(user_groups[idx]))
			local_weights2.append(copy.deepcopy(w))
			local_losses2.append(copy.deepcopy(loss))
		# update global weights
		global_weights2 = average_weights(local_weights2,sizes2)
	#   global_weights = average_weights(local_weights)

		# update global weights
		global_model2.load_state_dict(global_weights2)

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
			#list_loss2.append(loss)
		train_accuracy2.append(sum(list_acc2)/len(list_acc2))
	# Test inference after completion of training
	global_model2.eval()
	list_acc_test2 = []
	for j in range(args.num_users):
		local_model = LocalUpdate(args=args, dataset=train_dataset,
									   idxs=user_groups[idx], logger=logger)
		acc, _ = local_model.inference_test_local(model=global_model2)
		list_acc_test2.append(acc)							
		#test_accuracyL2.append(sum(list_acc_test2)/len(list_acc_test2))
	file_name = expdir+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_vanilla.pkl'.\
		format(args.dataset, args.model, args.epochs, args.frac, args.iid,
			   args.local_ep, args.local_bs)

	with open(file_name, 'wb') as f:
		pickle.dump([train_loss2, train_accuracy2, list_acc_test2], f)

	###########Plotting    
	print('test accuracy for clustered', list_acc_test)
	print('test accuracy for vanilla', list_acc_test2)
	import matplotlib
	import matplotlib.pyplot as plt
	matplotlib.use('Agg')

	# Plot Loss curve
	plt.figure()
	plt.title('Training Loss vs Communication rounds')
	plt.plot(range(len(train_loss)), train_loss, color='r',label = 'cluster based')
	plt.plot(range(len(train_loss2)), train_loss2, color='b', label ='vanilla FL')

	plt.ylabel('Training loss')
	plt.xlabel('Communication Rounds')
	plt.legend()
	plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))
	
	# # Plot Average Accuracy vs Communication rounds
	plt.figure()
	plt.title('Average Accuracy vs Communication rounds')
	plt.plot(range(len(train_accuracy)), train_accuracy, color='r',label='cluster based')
	plt.plot(range(len(train_accuracy2)), train_accuracy2, color='b', label ='vanilla FL')
	plt.ylabel('Average Accuracy')
	plt.xlabel('Communication Rounds')
	plt.legend()
	plt.savefig(dirname+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))
