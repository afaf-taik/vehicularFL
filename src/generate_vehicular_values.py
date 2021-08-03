import numpy as np 
from scipy.stats import truncnorm

Power = 1 
Ttrain_batch = 0.5 
LANES_SPEEDS = [[60,80],[80,100],[100,120],[100,120],[80,100],[60,80]]
num_lanes = 6
LANES_Y = [3.3 + 3.6*k for k in range(num_lanes)]
xyBSs = np.vstack((1000,0))

def initialize(args, xmax = 2000, ymax = max(LANES_Y)):
	''' Generate position and data and calculate gain of each vehicle depending on which lane they are
	'''
	y = np.random.randint(low=0, high=num_lanes, size=args.num_users)
	y_positions = np.array([LANES_Y[y[i]]] for i in range(len(y)))
	xyUEs = np.vstack((np.random.uniform(low=0.0, high=xmax, size=args.num_users), y_positions)) 
	Gains = channel_gain(args.num_users,1,1, xyUEs, xyBSs,0)
	train_dataset, test_dataset, user_groups = utils.get_dataset(args)
	TrainTime = [ numEpochs*len(user_groups[i])*Ttrain_batch/args.batch for i in range(args.num_users)]
	gains=[]
	for i in range(args.num_users):
		gains.append(Gains[0,i,0])
	return Ttrain, gains, train_dataset, test_dataset, user_groups , y_positions , y 


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
	for i in range(len(x)):
		new_x = 
	

def rate_stay(x,speeds):
''' Predict T_k  of each vehicle depending on its location and speed
'''
	return [(D-x[i])/speeds[i] for i in range(len(x))]


##############V2X requirments of accuracy : lateral ( 0.1m) longitude (0.5 m)

#TR : Transmission Range = 300 m 
def LLT(x,y,speeds,TR = 300, eps = 1e-08)
	LLT_ = np.zeros(len(x),len(x))
	for i in range len(x):
		for j in range(i,len(x)):
			deltaa = speeds[i]-speeds[j]
			LLT_[i][j] = (-deltaa * (x[i]-x[j]) + deltaa * TR)/ max(deltaa**2,eps)

	return LLT_
def newround(args, y):

	xyUEs = np.vstack()
	Gains = channel_gain(args.num_users,1,1, xyUEs, xyBSs,0)
	gains=[]
	for i in range(args.num_users):
		gains.append(Gains[0,i,0])
	alpha = [1/int(args.num_users) for i in range(int(args.num_users))]
	return alpha, gains