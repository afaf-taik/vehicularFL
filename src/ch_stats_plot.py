import pickle
import matplotlib.pyplot as plt
import numpy as np

def load(t,NCHANNELS):
	experiment_name = 'number_ch_elected_modelSize/'+str(NCHANNELS)+'RBs'
	path = 'diversity_exp/'+experiment_name
	filename = path +'/exp'+str(t)+'.pkl'
	with open(filename,'rb') as f :
		x = pickle.load(f)
	return x[0],x[1]


if __name__=='__main__':
	tries = 50
	NCHANNELS_ = [2,4]
	avg_ch = [[] for x in range(len(NCHANNELS_))]
	how_many_ch = [[] for x in range(len(NCHANNELS_))]
	for t in range(tries):
		for N in range(len(NCHANNELS_)):
			save_ch_sizes,save_all_ch = load(t,NCHANNELS_[N])
			print(save_ch_sizes)
			print('============================')
			print(save_all_ch)
			avg_ch[N].append(np.array(save_ch_sizes).mean(1))			
			all_chs = np.array(save_all_ch)
			for k in range(2):
				how_many_ch[N].append(len(np.unique(all_chs[k].flatten())))
		
#### Lines = Nchannels ########### columns = model size		
	moy_avg_ch = np.array(avg_ch).mean(1)
	print("moy_avg_ch",moy_avg_ch) 
######### ????
	moy_how_many = np.array(how_many_ch).mean(0)
	print("moy_how_many",moy_how_many)




