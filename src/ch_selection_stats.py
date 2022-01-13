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
modelsizes = [160000, 320000, 640000, 960000] 
numberRBs = [2,4,6,8,12,16,20]
Power = 0.1
tries = 20
Pmax = 1 #Watt #5 Watt is a high. You can use 1 Watt as the max.
#Ttrain_batch = [0.005,0.0075, 0.01]  #seconds , time to train on a batch of SGD 
Ttrain_batch = [0.005, 0.01, 0.02, 0.03]  #seconds , time to train on a batch of SGD 
num_lanes = 6  
#We can also suppose that the width is negligible 
numEpochs = 1  
D = 2000 # Radius of 2km 
#TOTAL_RBs = 4 #Total number of RBs ( to be adjusted )
BW = 180000   # Bandwidth of a RB Hz
#BW = 1000000
sig2_dBm = -114 # additive white Gaussian noise (dB)
sig2_watts = 10 ** ((sig2_dBm-30)/10) # additive white Gaussian noise (watts)
NCHANNELS = numberRBs[2]
experiment_name = 'number_ch_elected_modelSize_clustering/'+str(NCHANNELS)+'RBs'
path = 'diversity_exp/'+experiment_name
os.mkdir(path)
Nmax = 2
#tries = 1
for t in range(tries):
	file_name = path + '/exp'+str(t)+'.pkl'
	#os.mkdir(dirname)

	path_project = os.path.abspath('..')
	logger = SummaryWriter('logs')

	args = args_parser()
	exp_details(args)
	args.epochs = 15
	#m = args.num_users
	if args.gpu:
		torch.cuda.set_device(args.gpu)
	device = 'cuda' if args.gpu else 'cpu'

	# load dataset and user groups
	train_dataset, test_dataset, user_groups = get_dataset(args)					
	E = get_gini(user_groups,train_dataset)
	S = [len(user_groups[idx]) for idx in range(len(user_groups))]
	total_size = sum(S)
	Ttrain = [ [] for j in modelsizes]
	for j in range(len(modelsizes)):	
		Ttrain[j] = [numEpochs*len(user_groups[i])*Ttrain_batch[j]/args.local_bs for i in range(args.num_users)]
	A1 = [ np.zeros(args.num_users) for x in range(len(modelsizes))]	
	#print(env.channelgains)
	save_all_ch = [ [] for x in modelsizes]
	save_ch_sizes = [ [] for x in modelsizes ]
	save_c_sizes = [ [] for x in modelsizes ]	
	for epoch in tqdm(range(args.epochs)):
		env = Freeway(num_vehicles = args.num_users, num_slots = 1, num_channels = NCHANNELS)
		env.build_environment()
		Tk = env.rate_stay()
		LLT_ = env.LLT()
		Preferences = np.ones((args.num_users,2))				
		for x in range(len(modelsizes)):
			div_indicator = get_importance(E,S,A1[x],args.epochs)			
			r_min = required_rate(Ttrain[x], Tk, modelsizes[x])
			cost, rb = cost_in_RBs(r_min, args, env.channelgains, NCHANNELS)
			print(cost)			
			size = [len(user_groups[i]) for i in range(args.num_users)]
			a,b = allocate(div_indicator, cost, rb, r_min,NCHANNELS)
			print(a)
			print(b)
			save_ch_sizes[x].append(len(a))
			save_all_ch[x].append(a)
			NotSelected = [s for s in range(args.num_users)]
			for y in a:
				NotSelected.remove(y)
			dict_heads, dict_followers = capacities(a, NotSelected, 2)
			print(dict_heads, dict_followers)
			Twait = 2
			wts_time = canUploadV2V(args,a,NotSelected, env.channelgains, Ttrain[x], Twait, LLT_, NCHANNELS, Nmax)
			wts = create_wts_dict(a, NotSelected, Preferences, wts_time)
			print(wts)
			wt = create_wt_doubledict(a, NotSelected,wts)
			p = solve_wbm(a, NotSelected, wt, dict_heads , dict_followers)
			print_solution(p)    
			selected_edges = get_selected_edges(p)
			print(selected_edges)
			all_participants = a
			for alpha in selected_edges:
				_,s = alpha
				all_participants.append(int(s))
			print(all_participants)
			save_c_sizes[x].append(len(all_participants))
			for U in range(args.num_users):
				if(U in a):
					A1[x][U] = 0
				else:
					A1[x][U] +=1





	with open(file_name, 'wb') as f:
		pickle.dump([save_ch_sizes,save_all_ch,save_c_sizes], f)
