import numpy as np 
from scipy.stats import truncnorm
from gain import channel_gain 
import utils 
from options import args_parser 
import math
import VehicularEnv
from pulp import *
Power = 0.2 #Watt 
Pmax = 1 #Watt #5 Watt is a high. You can use 1 Watt as the max.
Ttrain_batch = 0.005 #seconds , time to train on a batch of SGD 
LANES_SPEEDS_Kmh = [[60,80],[80,100],[100,120],[-100,-120],[-80,-100],[-60,-80]] # km/h , to be changed to m/s #Negative speeds?
LANES_SPEEDS = [[17,22],[22,27],[27,33],[-33,-27],[-27,-22],[-22,-17]] # m/s 
num_lanes = 6  
LANES_Y = [3.3 + 3.6*k for k in range(num_lanes)] # what do you mean by this? for k = 0, LANES_Y[0] = 3.3, meaning? # just which lane they are
#We can also suppose that the width is negligible 
xyBSs = np.vstack((1000,25)) # BS is on the side of the road at the 1km mark # You can put it at the side of the road, like (1000, 25)?
numEpochs = 1  
D = 2000 # Radius of 2km 
#TOTAL_RBs = 4 #Total number of RBs ( to be adjusted )
BW = 180000   # Bandwidth of a RB Hz
#BW = 1000000
sig2_dBm = -114 # additive white Gaussian noise (dB)
sig2_watts = 10 ** ((sig2_dBm-30)/10) # additive white Gaussian noise (watts)
S = 160000 #bits 

def initialize(args, xmax = 2000, ymax = max(LANES_Y)):
	''' Generate position and data and calculate gain of each vehicle depending on which lane they are
	'''
	#y = np.random.randint(low=0, high=num_lanes, size=args.num_users)
	#print(y)
	#y_positions = np.array([LANES_Y[y[i]] for i in range(len(y))])
	#print(y_positions)
	#xyUEs = np.vstack((np.random.uniform(low=0.0, high=xmax, size=args.num_users), y_positions)) 
	#Gains = channel_gain(args.num_users,1,TOTAL_RBs, xyUEs, xyBSs,0)
	train_dataset, test_dataset, user_groups = utils.get_dataset(args)
	TrainTime = [numEpochs*len(user_groups[i])*Ttrain_batch/args.local_bs for i in range(args.num_users)]

#	Gains = channel_gain(args.num_users,1,TOTAL_RBs, xyEUs, xyBSs,0)
	return TrainTime, train_dataset, test_dataset, user_groups

def update_location(t, x, speeds):
	''' Calculate new location of each vehicle depending on speed at the moment t
	'''
	new_x = np.zeros(len(x))
	for i in range(len(x)):
		new_x[i] = speeds[i]*t + x[i]
	return new_x


#T_k ia rate of stay 
def required_rate(Ttrain, T_k, S):
	rates = []
	for i in range(len(Ttrain)):
		if (T_k[i]>Ttrain[i]):
			rates.append(S/( T_k[i] - Ttrain[i]))
		else:
			rates.append(math.inf)
	return rates


def dataRateRB(args, Gains,Total_RBs):
	drRB = np.zeros((args.num_users,Total_RBs))
	Gains = Gains.squeeze()
	#print("Shape of gains plzzzzzzzzzzzzzzz",Gains.shape)
	for rb in range(Total_RBs):
		#print('calculating for rb', rb)		
		for eu in range(args.num_users):
			#print('calculating for user', eu)
			#print('this is the list of gains', Gains[eu][args.num_users][rb])
			drRB[eu][rb] = BW * math.log2(1 +np.divide(Power * Gains[eu][args.num_users][rb],sig2_watts))	
	return drRB

def dataRateRB_V2V(args, Gains,cluster_head, Total_RBs):
	drRB = np.zeros((args.num_users,Total_RBs))
	Gains = Gains.squeeze()
	#print("Shape of gains plzzzzzzzzzzzzzzz",Gains.shape)
	for rb in range(Total_RBs):
		#print('calculating for rb', rb)		
		for eu in range(args.num_users):
			#print('calculating for user', eu)
			#print('this is the list of gains', Gains[eu][args.num_users][rb])
			drRB[eu][rb] = BW * math.log2(1 +np.divide(Power * Gains[eu][cluster_head][rb],sig2_watts))	
	return drRB

def upload_time_uniform_V2V():
	return

def cost_in_RBs(r_min, args, gains,Total_RBs):
	nbr_RBs = np.ones(args.num_users)
	cost = np.zeros(args.num_users)
	dataRatesRBs = dataRateRB(args,gains,Total_RBs)
	print(dataRatesRBs)
	reached = []
	for k in range(args.num_users):
		datarates = dataRatesRBs[k][:]
		print('datarates for',k,':',datarates)
		c = 0
		r = 0
		print('required RBs for', k)
		while(r<r_min[k] and c<Total_RBs):
			print(datarates)
			r_ = max(datarates)
			datarates = np.delete(datarates, np.argwhere(datarates == r_))
			r += r_
			c += 1
		cost[k] = c
		#print(cost[k])
		reached.append(r)
		print(reached)			
	return cost, dataRatesRBs


def allocate(div_indicator,cost,dataRatesRBs,r_min,Total_RBs):
	priority = [div_indicator[i]/cost[i] for i in range(len(div_indicator))]
	ordered = utils.get_order(priority)
	b = Total_RBs 
	k = 0
	scheduled = []
	RB_allocation = np.zeros(Total_RBs)
	datarates_schedule = []
	while(b>0):
		datarates = dataRatesRBs[ordered[k]][:]
		r = 0
		print('allocate to', k)
		while(r<r_min[ordered[k]] and b>0 and datarates.size !=0):
			r_ = max(datarates)
			#datarates = np.delete(datarates, np.argwhere(datarates == r_))	
			if (RB_allocation[np.argwhere(datarates == r_)[0]]==0):
				r += r_
				b = b - 1
				RB_allocation[np.argwhere(datarates == r_)[0]] = 1	
			
			datarates = np.delete(datarates, np.argwhere(datarates == r_))				
		datarates_schedule.append(r)
		scheduled.append(ordered[k])
		k = k+1
	#return datarates, indexes of scheduled devices		
	return scheduled, datarates_schedule

####################To be adjusted
# def allocate_random(cost,dataRatesRBs,r_min,Total_RBs):
# 	priority = [ for i in range(len(div_indicator))]
# 	ordered = utils.get_order(priority)
# 	b = Total_RBs 
# 	k = 0
# 	scheduled = []
# 	RB_allocation = np.zeros(Total_RBs)
# 	datarates_schedule = []
# 	while(b>0):
# 		scheduled.append(ordered[k])
# 		datarates = dataRatesRBs[ordered[k]][:]
# 		r = 0
# 		print('required RBs for', k)
# 		while(r<r_min[ordered[k]] and b>0):
# 			r_ = max(datarates)
# 			#get index of r_ in dataRatesRBs
# 			#add condition of it being removed
# 			#datarates = np.delete(datarates, np.argwhere(datarates == r_))	
# 			if (RB_allocation[np.argwhere(datarates == r_)[0]]==0):
# 				r += r_
# 				b = b - 1
# 				RB_allocation[np.argwhere(datarates == r_)[0]] = 1	
			
# 			datarates = np.delete(datarates, np.argwhere(datarates == r_))				
# 		datarates_schedule.append(r)
# 		k = k+1
# 	#return datarates, indexes of scheduled devices		
# 	return scheduled, datarates_schedule	



#######################################################################################
#######################################################################################
###################################Problem 2 functions#################################
#######################################################################################
#######################################################################################

def capacities(Selected, NotSelected, Nmax):
	dict_heads = {}
	dict_followers = {}
	for i in Selected:
		dict_heads[i] = Nmax
	for j in NotSelected:
		dict_followers[j] = 1 
	return dict_heads, dict_followers

def create_wts_dict(Selected, NotSelected, Preferences, LLT, mobility_only = True):
	wts = {}
	for i in Selected:
		if(mobility_only):
			mult = 1
		else:	
			idx_model = np.argmax(Preferences[i,:])[0]
			mult = Preferences[j][idx_model]/3000
			print(idx_model)
		for j in NotSelected:
			wts[(i,j)] = mult*LLT[i][j]
	return wts

def create_wt_doubledict(Selected, NotSelected,wts):
	wt = {}
	for u in Selected:
		wt[u] = {}
		for v in NotSelected:
			wt[u][v] = 0

	for k,val in wts.items():
		u,v = k[0], k[1]
		wt[u][v] = val
	return(wt)

def solve_wbm(Selected, NotSelected, wt, ucap , vcap):

	prob = LpProblem("WBM Problem", LpMaximize)

	# Create The Decision variables
	choices = LpVariable.dicts("e",(Selected, NotSelected), 0, 1, LpInteger)

	# Add the objective function 
	prob += lpSum([wt[u][v] * choices[u][v] 
				   for u in Selected
				   for v in NotSelected]), "Total weights of selected edges"


	# Constraint set ensuring that the total from/to each node 
	# is less than its capacity
	for u in Selected:
		for v in NotSelected:
			prob += lpSum([choices[u][v] for v in NotSelected]) <= ucap[u], ""
			prob += lpSum([choices[u][v] for u in Selected]) <= vcap[v], ""


	# The problem data is written to an .lp file
	prob.writeLP("WBM.lp")

	# The problem is solved using PuLP's choice of Solver
	prob.solve()

	# The status of the solution is printed to the screen
	print( "Status:", LpStatus[prob.status])
	return(prob)


def print_solution(prob):
	# Each of the variables is printed with it's resolved optimum value
	for v in prob.variables():
		if v.varValue > 1e-3:
			print(f'{v.name} = {v.varValue}')
	print(f"Sum of wts of selected edges = {round(value(prob.objective), 4)}")


def get_selected_edges(prob):

	selected_from = [v.name.split("_")[1] for v in prob.variables() if v.value() > 1e-3]
	selected_to   = [v.name.split("_")[2] for v in prob.variables() if v.value() > 1e-3]

	selected_edges = []
	for su, sv in list(zip(selected_from, selected_to)):
		selected_edges.append((su, sv))
	return(selected_edges)          



if __name__ == '__main__':
	args = args_parser()	
	args.num_users = 30
	NCHANNELS = 4
	S = 100000
	Ttrain, train_dataset, test_dataset, user_groups = initialize(args)
	#print(Ttrain)	
	from VehicularEnv import Freeway,V2IChannels,V2VChannels, Vehicle
	env1 = Freeway(num_vehicles = args.num_users, num_slots = 1, num_channels = NCHANNELS)
	env1.build_environment()
	LLT_ = env1.LLT()
	print(LLT_)
	Preferences = np.ones((args.num_users,2))
############################Allocate test##########################			
	Tk = env1.rate_stay()
	r_min = required_rate(Ttrain, Tk, S)
	cost, rb = cost_in_RBs(r_min, args, env1.channelgains, NCHANNELS)
	print(cost)
	size = [len(user_groups[i]) for i in range(args.num_users)]
	age = [0 for i in range(args.num_users)]
	entropy = utils.get_gini(user_groups,train_dataset)
	div_indicator = utils.get_importance(entropy,size,age,50,roE=1/3,roD=1/3,roA =1/3)
	a,b = allocate(div_indicator, cost, rb, r_min,NCHANNELS)
	print(a)
	print(b)
	NotSelected = [i for i in range(args.num_users)]
	for x in a:
		NotSelected.remove(x)
	dict_heads, dict_followers = capacities(a, NotSelected, 2)
	print(dict_heads, dict_followers)
	wts = create_wts_dict(a, NotSelected, Preferences, LLT_)
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

###########################################################################
