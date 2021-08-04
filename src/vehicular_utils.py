import numpy as np 
from scipy.stats import truncnorm
from gain import channel_gain 
import utils 
from options import args_parser 
import math


Power = 0.1 #Watt 
Pmax = 1 #Watt #5 Watt is a high. You can use 1 Watt as the max.
Ttrain_batch = 0.5 #seconds , time to train on a batch of SGD 
LANES_SPEEDS = [[60,80],[80,100],[100,120],[-100,-120],[-80,-100],[-60,-80]] # km/h , to be changed to m/s #Negative speeds?
num_lanes = 6  
LANES_Y = [3.3 + 3.6*k for k in range(num_lanes)] # what do you mean by this? for k = 0, LANES_Y[0] = 3.3, meaning? # just which lane they are
#We can also suppose that the width is negligible 
xyBSs = np.vstack((1000,25)) # BS is on the side of the road at the 1km mark # You can put it at the side of the road, like (1000, 25)?
numEpochs = 1  
D = 2000 # Radius of 2km 
TOTAL_RBs = 10 #Total number of RBs ( to be adjusted )
BW = 180000   # Bandwidth of a RB Hz
sig2_dBm = -114 # additive white Gaussian noise (dB)
sig2_watts = 10 ** ((sig2_dBm-30)/10) # additive white Gaussian noise (watts)

def initialize(args, xmax = 2000, ymax = max(LANES_Y)):
	''' Generate position and data and calculate gain of each vehicle depending on which lane they are
	'''
	y = np.random.randint(low=0, high=num_lanes, size=args.num_users)
	print(y)
	y_positions = np.array([LANES_Y[y[i]] for i in range(len(y))])
	print(y_positions)
	xyUEs = np.vstack((np.random.uniform(low=0.0, high=xmax, size=args.num_users), y_positions)) 
	Gains = channel_gain(args.num_users,1,TOTAL_RBs, xyUEs, xyBSs,0)
	train_dataset, test_dataset, user_groups = utils.get_dataset(args)
	TrainTime = [numEpochs*len(user_groups[i])*Ttrain_batch/args.local_bs for i in range(args.num_users)]

#	Gains = channel_gain(args.num_users,1,TOTAL_RBs, xyEUs, xyBSs,0)
	return TrainTime, Gains, train_dataset, test_dataset, user_groups , y_positions , y , xyUEs


def speeds(args,y):
	''' Generate speed of each vehicle depending on which lane they are
	'''
	speed = np.zeros(args.num_users)
	for i in range(args.num_users):
		speed[i] =  truncnorm.rvs(LANES_SPEEDS[y[i]][0], LANES_SPEEDS[y[i]][1], size=1)
	return speed

def update_location(t, x, speeds):
	''' Calculate new location of each vehicle depending on speed at the moment t
	'''
	new_x = np.zeros(len(x))
	for i in range(len(x)):
		new_x[i] = speeds[i]*t + x[i]
	return new_x
	

def rate_stay(x,speeds):
#Predict T_k  of each vehicle depending on its location and speed
	return [(D-x[i])/speeds[i] for i in range(len(x))]


def LLT(x,y,speeds, cluster_heads_indexes, TR = 300, eps = 1e-08):
		"""Calculate link lifetime
	should be changed to only use cluster_head_indexes instead of everyone
    TR : Transmission Range = 300 m 
    eps is used to avoid division by 0 in case speed_i = speed_j
    We still need to find a solution for the case of v_i = v_j, because even with eps, LLT = 0 does not make sense"""
    LLT_ = np.zeros(len(x),len(x))
	for i in range len(x):
		for j in range(i,len(x)):
			deltaa = speeds[i]-speeds[j]
			LLT_[i][j] = (-deltaa * (x[i]-x[j]) + deltaa * TR)/ max(deltaa**2,eps)
	return LLT_


#T_k ia rate of stay 
def required_rate(Ttrain, T_k, S):
	rates = []
	for i in range(len(Ttrain)):
		if (T_k[i]>Ttrain[i]):
			rates.append(S/( T_k[i] - Ttrain[i]))
		else:
			rates.append(math.inf)
	return rates


def dataRateRB(args, Gains):
    drRB = np.zeros((TOTAL_RBs,args.num_users))
        for rb in range(TOTAL_RBs):
            for eu in range(args.num_users):
                drRB[rb,eu] = BW * np.log2(1 + np.divide(Power * Gains[rb,eu],sig2_watts))
    return drRB

def cost_in_RBs(r_min, args, gains):
	nbr_RBs = np.ones(args.num_users)
	cost = np.zeros(args.num_users)
	dataRatesRBs = dataRateRB(args,gains)
	for k in range(args.num_users):
		datarates = dataRatesRBs[:,k]
		c = TOTAL_RBs
		r = 0
		print('required RBs for', k)
		while(r<rmin and c<TOTAL_RBs):
			r_ = max(datarates)
			datarates.remove(r_)
			r += r_
			c += 1
		cost[k] = c
		print(cost[k])			
	return cost, dataRatesRBs

def allocate(div_indicator,cost,dataRatesRBs):
	priority = [div_indicator[i]/cost[i] for i in range(len(div_indicator))]
	ordered = utils.get_order(priority)
	b = TOTAL_RBs 
	k = 0
	scheduled = []
	RB_allocation = np.zeros(TOTAL_RBs)
	datarates_schedule = []
	while(b>0):
		scheduled.append(ordered[k])
		datarates = dataRatesRBs[:,k]
		c = TOTAL_RBs
		r = 0
		print('required RBs for', k)
		while(r<rmin and c<b):
			r_ = max(datarates)
			#get index of r_ in dataRatesRBs
			#add condition of it being removed
			datarates.remove(r_)
			if (RB_allocation[datarates.index(r_)]==0):
				r += r_
				b -=1
				RB_allocation[datarates.index(r_)] =1	
			else:
				datarates.remove(r_)
		datarates_schedule.append(r)
	#return datarates, indexes of scheduled devices		
	return scheduled, datarates_schedule
		


if __name__ == '__main__':
	args = args_parser()	
	args.num_users = 50
	Ttrain, gains, train_dataset, test_dataset, user_groups , y_positions , y , xyUEs = initialize(args)
	speed = speeds(args,y)
	print(speed)
	#sample of execution
	#x should be inferred from xyUEs
	#T_k = rate_stay(x,speeds)
	#R = required_rate(Ttrain, T_k, S)
	#c , dbrb = cost_in_RBs(R, args, gains)
	#div_indicator = np.random.rand(args.num_users)
	#scheduled, datarates_schedule = allocate(div_indicator,c,dbrb)
