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
from utils import get_dataset, average_weights, exp_details, models_similarity, flip_labels_clusters, newmodel,build_model_by_cluster
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcl
from scipy.spatial.distance import squareform
tries = 7
modelname = 'clustering'
resdir = 'diversity_exp/' + 'baseline_'+ modelname 
#os.mkdir(resdir)
test_time = 1
n_clusters_max = 4
m = 6
for i in range (tries):
	expdir = resdir+'/exp'+str(i)
	#os.mkdir(expdir)
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

	# copy weights
	global_weights = global_model.state_dict()

	# Training
	train_loss, train_accuracy = [], []
	test_lossL, test_accuracyL = [], []
	val_acc_list, net_list = [], []
	cv_loss, cv_acc = [], []
	print_every = 10
	val_loss_pre, counter = 0, 0
	similarities = []



	for epoch in tqdm(range(args.epochs)):
		local_weights, local_losses = [], []
		print(f'\n | Global Training Round : {epoch+1} |\n')


		#m = max(int(args.frac * args.num_users), 1)    
		if (epoch<test_time):
			global_model.train()
			idxs_users = np.random.choice(range(args.num_users), m, replace=False) 
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
		

		elif(epoch==test_time):
			global_model.train()
			idxs_users = np.random.choice(range(args.num_users), m, replace=False) 
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
			preferences = np.zeros(shape= (args.num_users, len(models)))
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
			
		loss_avg = sum(local_losses) / len(local_losses)
		train_loss.append(loss_avg)
		# Calculate avg training accuracy over all users at every epoch



	#     list_acc, list_loss = [], []
	#     global_model.eval()
	#     for c in range(args.num_users):
	#         local_model = LocalUpdate(args=args, dataset=train_dataset,
	#                                   idxs=user_groups[idx], logger=logger)
	#         acc, loss = local_model.inference(model=global_model)
	#         list_acc.append(acc)
	#         list_loss.append(loss)
	#     train_accuracy.append(sum(list_acc)/len(list_acc))
	#     test_acc, test_loss = test_inference(args, global_model, test_dataset)
	#     test_accuracyL.append(test_acc)
	#     # print global training loss after every 'i' rounds
	#     if (epoch+1) % print_every == 0:
	#         print(f' \nAvg Training Stats after {epoch+1} global rounds:')
	#         print(f'Training Loss : {np.mean(np.array(train_loss))}')
	#         print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

	# # Test inference after completion of training
	# test_acc, test_loss = test_inference(args, global_model, test_dataset)

	# print(f' \n Results after {args.epochs} global rounds of training:')
	# print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
	# print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

	# # Saving the objects train_loss and train_accuracy:
	# file_name = expdir+'/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
	#     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
	#            args.local_ep, args.local_bs)

	# with open(file_name, 'wb') as f:
	#     pickle.dump([train_loss, train_accuracy, test_accuracyL], f)

	# print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))









# #    PLOTTING (optional)
#     import matplotlib
#     import matplotlib.pyplot as plt
#     matplotlib.use('Agg')

# #    Plot Loss curve
#     plt.figure()
#     plt.title('Training Loss vs Communication rounds')
#     plt.plot(range(len(train_loss)), train_loss, color='r')
#     plt.ylabel('Training loss')
#     plt.xlabel('Communication Rounds')
#     plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
#                 format(args.dataset, args.model, args.epochs, args.frac,
#                        args.iid, args.local_ep, args.local_bs))
	
#     # Plot Average Accuracy vs Communication rounds
#     plt.figure()
#     plt.title('Average Accuracy vs Communication rounds')
#     plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
#     plt.ylabel('Average Accuracy')
#     plt.xlabel('Communication Rounds')
#     plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
#                 format(args.dataset, args.model, args.epochs, args.frac,
#                        args.iid, args.local_ep, args.local_bs))

#     # Plot Average Accuracy vs Communication rounds
#     plt.figure()
#     plt.title('Average Accuracy vs Communication rounds')
#     plt.plot(range(len(similarities)), train_accuracy, color='k')
#     plt.ylabel('Average Accuracy')
#     plt.xlabel('Communication Rounds')
#     plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
#                 format(args.dataset, args.model, args.epochs, args.frac,
#                        args.iid, args.local_ep, args.local_bs))
