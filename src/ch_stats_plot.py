import pickle
import matplotlib.pyplot as plt
import numpy as np

def load(t,NCHANNELS):
	experiment_name = 'number_ch_elected_modelSize_clustering/'+str(NCHANNELS)+'RBs'
	path = 'diversity_exp/'+experiment_name
	filename = path +'/exp'+str(t)+'.pkl'
	with open(filename,'rb') as f :
		x = pickle.load(f)
	return x[0],x[1],x[2]


if __name__=='__main__':
	tries = 15
	NCHANNELS_ = [2,4,6]
	avg_ch = [[] for x in range(len(NCHANNELS_))]
	how_many_ch = [[] for x in range(len(NCHANNELS_))]
	how_many_vh_c = [[] for x in range(len(NCHANNELS_))]	
	for t in range(tries):
		for N in range(len(NCHANNELS_)):
			save_ch_sizes,save_all_ch,save_size_c = load(t,NCHANNELS_[N])
			print(save_ch_sizes)
			print('============================')
			print(save_all_ch)
			avg_ch[N].append(np.array(save_ch_sizes).mean(1))			
			all_chs = np.array(save_all_ch)
			for k in range(4):
				how_many_ch[N].append(len(np.unique(all_chs[k].flatten())))
			how_many_vh_c[N].append(np.array(save_size_c).mean(1))
			#avg_ch[N].append(np.array(save_ch_sizes).mean(1))			

#### Lines = Nchannels ########### columns = model size		
	moy_avg_ch = np.array(avg_ch).mean(1)
	print("moy_avg_ch",moy_avg_ch) 
#### Lines = Nchannels ########### columns = model size		
	moy_avg_c = np.array(how_many_vh_c).mean(1)
	print("moy_avg_in_c",moy_avg_c) 

######### ????
	moy_how_many = np.array(how_many_ch).mean(0)
	print("moy_how_many",moy_how_many)




