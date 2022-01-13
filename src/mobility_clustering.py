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
from vehicular_utils import dataRateRB, required_rate, cost_in_RBs ,allocate, capacities, create_wts_dict, create_wt_doubledict, solve_wbm,print_solution,get_selected_edges, canUploadV2V
from VehicularEnv import Freeway,V2IChannels,V2VChannels, Vehicle
from pulp import *
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcl
from scipy.spatial.distance import squareform
import matplotlib


fraction = 0.6
modelname = 'mlp'
datasetname = 'mnist'
resdir = 'diversity_exp/mobility/clustering_shift_new'+ modelname +datasetname
dirname = resdir
os.mkdir(resdir)
test_time = 25
n_clusters_max = 2
m = 6
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
Nmax = 4
fileinfo = resdir+'/exp_info.txt'
info = 'testime'+str(test_time)+'\nm',str(m)+'\n maxcluster'+str(n_clusters_max)+'\n model'+modelname
with open(fileinfo,'w+') as f:
	f.write(str(info))
for i in range (2,tries):
	expdir = resdir+'/exp'+str(i)
	os.mkdir(expdir)
	start_time = time.time()

	# define paths
	path_project = os.path.abspath('..')
	logger = SummaryWriter('logs')

	args = args_parser()
	exp_details(args)
	m2 = int(args.num_users*fraction)
	if args.gpu:
		torch.cuda.set_device(args.gpu)
	device = 'cuda' if args.gpu else 'cpu'

	# load dataset and user groups
	train_dataset1, test_dataset1, user_groups = get_dataset(args)
	train_dataset, test_dataset, clusters = flip_labels_clusters(args,user_groups,train_dataset1,test_dataset1)
	# BUILD MODEL
	Ttrain = [numEpochs*len(user_groups[i])*Ttrain_batch/args.local_bs for i in range(args.num_users)]
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
	train_loss, train_accuracy,cluster_losses = [], [], []
	test_lossL, test_accuracyL = [], []
	val_acc_list, net_list = [], []
	cv_loss, cv_acc = [], []
	print_every = 10
	val_loss_pre, counter = 0, 0
	similarities = []

	E = get_gini(user_groups,train_dataset)
	S = [len(user_groups[idx]) for idx in range(len(user_groups))]
	A = np.zeros(args.num_users)
	Preferences = np.ones((args.num_users,n_clusters_max))	
	for epoch in tqdm(range(args.epochs)):
		flag = True
		env = Freeway(num_vehicles = args.num_users, num_slots = 1, num_channels = NCHANNELS)
		env.build_environment()			
		local_weights1, local_losses1 = [], []					
		global_model.train()
		model = copy.deepcopy(global_model)
		#cost = required_bandwidth(T, Ttrain, args)
		######################################################################################################
		div_indicator = get_importance(E,S,A,args.epochs)
		LLT_ = env.LLT()
		print(LLT_)

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
		print(wts)
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
		local_weights, local_losses = [], []
		local_weights2, local_losses2 = [], []		
		print(f'\n | Global Training Round : {epoch+1} |\n')
		idxs_users = all_participants
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
				if(c in idxs_users):
					A[c] = 0
				else:
					A[c] +=1
				local_model = LocalUpdate(args=args, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
				acc, _ = local_model.inference(model=global_model)
				list_acc.append(acc)
			#list_loss2.append(loss)
			train_accuracy.append(sum(list_acc)/len(list_acc))		
				
					
		elif(epoch==test_time):
			global_model.train()
			ordered = get_order(get_importance(E,S,A,args.epochs)) 
			idxs_users = ordered[:m2]
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
			for U in range(args.num_users):	
				if(U in idxs_users):
					A[U] = 0
				else:
					A[U] +=1
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
			Preferences = np.ones(shape = (args.num_users, len(models)))
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
					Preferences[g][j] = acc


					#list_acc.append(acc)
			print('======================PREFERENCES==================',Preferences)            
			#choose cluster  
			preferred = determine_preferred_model(Preferences)
			print(preferred)
			idx_users_clusters = []
			for j in range(max(clusters_found)):
				idx_users_clusters.append(np.argwhere(preferred == j))
			print(idx_users_clusters)
			idxs_users_flat = [ x.flatten() for x in idx_users_clusters]
			#print(idxs_users)
			
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
			flag = False
			idx_all = all_participants
			print('debug 1', all_participants)
			print('debug 2', idx_all)
			local_losses = []
			new_loss = [ [] for j in range(max(clusters_found))]
			train_loss_clusters = []
			for j in range(max(clusters_found)):
				models[j].train()
				print('debug3', idxs_users_flat[j])
				idxs_users = [value for value in idx_all if value in idxs_users_flat[j]] 
				print('selected', idxs_users)
				sizes = []
				local_weights = []
				if(idxs_users==[]):
					continue
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
				#new_loss[j].append(loss_avg)
				models[j].load_state_dict(global_weights)
				for U in range(args.num_users):
					if(U in idxs_users):
						A[U] = 0
					else:
						A[U] +=1
			loss_avg = sum(local_losses) / len(local_losses)
		
			#train_loss_clusters.append(sum(new_loss[j])/len(new_loss[j]))
			#cluster_losses.append(train_loss_clusters)
			#loss_avg = sum(train_loss_clusters) / len(train_loss_clusters)	
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
		idxs_users_rnd = np.random.choice(range(args.num_users), NCHANNELS, replace=False)
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

	matplotlib.use('Agg')

	# Plot Loss curve
	plt.figure()
	plt.title('Training Loss vs Communication rounds')
	plt.plot(range(len(train_loss)), train_loss, color='r',label = 'cluster based')
	plt.plot(range(len(train_loss2)), train_loss2, color='b', label ='vanilla FL')

	plt.ylabel('Training loss')
	plt.xlabel('Communication Rounds')
	plt.legend()
	plt.savefig(expdir+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
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
	plt.savefig(expdir+'/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
				 format(args.dataset, args.model, args.epochs, args.frac,
						args.iid, args.local_ep, args.local_bs))



# import pickle
# #import matplotlib.pyplot as plt
# #import os 
# #import numpy as np 

# path = resdir
# lossname = path+'/loss_avg.png'
# accname = path+'/acc_avg.png'
# modelname = 'mlp'
# list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]

# def load(t,r,path):
# 	dirname = path+'/exp'+str(t)
# 	if(r==0):
# 		filename = dirname +'/mnist_'+modelname+'_50_C[0.2]_iid[0]_E[1]_B[20]_vanilla.pkl'
# 	else:
# 		filename = dirname +'/mnist_'+modelname+'_50_C[0.2]_iid[0]_E[1]_B[20]_proposed.pkl'
# 	with open(filename,'rb') as f :
# 		x = pickle.load(f)
# 	return x
# sloss_rnd, sacc_rnd, stestrnd = [], [], []
# sloss_opt, sacc_opt, stestopt = [], [], []

# for t in range(len(list_subfolders_with_paths)):
# 	x  = load(t,0,path)
# 	sloss_rnd.append(x[0])
# 	sacc_rnd.append(x[1])
# 	stestrnd.append(x[2])
# 	y = load(t,1,path)
# 	sloss_opt.append(y[0])
# 	sacc_opt.append(y[1])
# 	stestopt.append(y[2])

# moy_rnd_loss = np.array(sloss_rnd).mean(0)
# # max_rnd_loss = np.array(sloss_rnd).max(0)
# # min_rnd_loss = np.array(sloss_rnd).min(0)
# moy_rnd_acc = np.array(sacc_rnd).mean(0)

# moy_opt_loss = np.array(sloss_opt).mean(0)
# # max_opt_loss = np.array(sloss_opt).max(0)
# # min_opt_loss = np.array(sloss_opt).min(0)
# moy_opt_acc = np.array(sacc_opt).mean(0)

# moy_rnd_acc2 = np.array(stestrnd).mean()
# moy_opt_acc2 = np.array(stestopt).mean()

# print(moy_rnd_acc2)
# print(moy_opt_acc2)
# # import matplotlib.pyplot as plt
# matplotlib.use('Agg')

# 	# Plot Loss curve
# plt.figure()
# #plt.title('Training Loss vs Communication rounds')
# plt.plot(range(len(moy_rnd_loss)), moy_rnd_loss, 'b+',  linewidth=1, linestyle='-', label='random')
# plt.plot(range(len(moy_opt_loss)), moy_opt_loss,  'ro',  linewidth=1, linestyle='-',label = 'proposed')
# # plt.plot(range(len(moy_rnd_loss)), min_rnd_loss, 'b+',  linewidth=1, linestyle='--', label='min age-based scheduling')
# # plt.plot(range(len(moy_opt_loss)), min_opt_loss,  'ro',  linewidth=1, linestyle='--',label = 'min diversity-based scheduling')
# # plt.plot(range(len(moy_rnd_loss)), max_rnd_loss, 'b+',  linewidth=1, linestyle=':', label='max age-based scheduling')
# # plt.plot(range(len(moy_opt_loss)), max_opt_loss,  'ro',  linewidth=1, linestyle=':',label = 'max diversity-based scheduling')
# plt.grid()
# plt.ylabel('Training loss')
# plt.xlabel('Communication Rounds')
# plt.legend()
# plt.savefig(lossname)


# plt.figure()
# #plt.title('Accuracy vs Communication rounds')
# plt.ylim(ymin=0.2,ymax=0.85)
# plt.plot(range(int(len(moy_rnd_acc[:50]))), moy_rnd_acc[:50], 'b+',  linewidth=1, linestyle='-', label='Vanilla FL')
# plt.plot(range(int(len(moy_opt_acc[:50]))), moy_opt_acc[:50], 'ro',  linewidth=1, linestyle='-', label = 'CVFL')
# plt.grid()
# plt.ylabel('Train Accuracy')
# plt.xlabel('Communication Rounds')
# plt.legend()
# plt.savefig(accname)