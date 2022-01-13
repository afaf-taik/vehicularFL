import pickle
import matplotlib.pyplot as plt
import os 
import numpy as np 
import matplotlib
#'clustering_shift_newmlpmnist'
#mobility_normalnoniid_new
modelname = 'mlp'
datasetname = 'mnist'
resdir = 'diversity_exp/mobility/clustering_shift_new'+ modelname +datasetname
resdir = 'diversity_exp/mobility/mobility_normalnoniid_new'

dirname = resdir
path = resdir
lossname = path+'/loss_avg.png'
accname = path+'/acc_avg.png'
modelname = 'mlp'
list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]

def load(t,r,path):
	dirname = path+'/exp'+str(t)
	if(r==0):
		filename = dirname +'/mnist_'+modelname+'_50_C[0.2]_iid[0]_E[1]_B[20]_vanilla.pkl'
	else:
		filename = dirname +'/mnist_'+modelname+'_50_C[0.2]_iid[0]_E[1]_B[20]_proposed.pkl'
	with open(filename,'rb') as f :
		x = pickle.load(f)
	return x
sloss_rnd, sacc_rnd, stestrnd = [], [], []
sloss_opt, sacc_opt, stestopt = [], [], []

for t in range(25):
	x  = load(t,0,path)
	sloss_rnd.append(x[0])
	sacc_rnd.append(x[1])
	stestrnd.append(x[2])
	y = load(t,1,path)
	sloss_opt.append(y[0])
	sacc_opt.append(y[1])
	stestopt.append(y[2])

moy_rnd_loss = np.array(sloss_rnd).mean(0)
# max_rnd_loss = np.array(sloss_rnd).max(0)
# min_rnd_loss = np.array(sloss_rnd).min(0)
moy_rnd_acc = np.array(sacc_rnd).mean(0)

moy_opt_loss = np.array(sloss_opt).mean(0)
# max_opt_loss = np.array(sloss_opt).max(0)
# min_opt_loss = np.array(sloss_opt).min(0)
moy_opt_acc = np.array(sacc_opt).mean(0)

moy_rnd_acc2 = np.array(stestrnd).mean()
moy_opt_acc2 = np.array(stestopt).mean()
moy_rnd_std2 = np.array(stestrnd).std()
moy_opt_std2 = np.array(stestopt).std()

print(moy_rnd_acc2,moy_rnd_std2)
print(moy_opt_acc2,moy_opt_std2 )
# import matplotlib.pyplot as plt
matplotlib.use('Agg')

	# Plot Loss curve
plt.figure()
#plt.title('Training Loss vs Communication rounds')
plt.plot(range(len(moy_rnd_loss)), moy_rnd_loss, 'b+',  linewidth=1, linestyle='-', label='random')
plt.plot(range(len(moy_opt_loss)), moy_opt_loss,  'ro',  linewidth=1, linestyle='-',label = 'proposed')
# plt.plot(range(len(moy_rnd_loss)), min_rnd_loss, 'b+',  linewidth=1, linestyle='--', label='min age-based scheduling')
# plt.plot(range(len(moy_opt_loss)), min_opt_loss,  'ro',  linewidth=1, linestyle='--',label = 'min diversity-based scheduling')
# plt.plot(range(len(moy_rnd_loss)), max_rnd_loss, 'b+',  linewidth=1, linestyle=':', label='max age-based scheduling')
# plt.plot(range(len(moy_opt_loss)), max_opt_loss,  'ro',  linewidth=1, linestyle=':',label = 'max diversity-based scheduling')
plt.grid()
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds')
plt.legend()
plt.savefig(lossname)


plt.figure()
#plt.title('Accuracy vs Communication rounds')
plt.ylim(ymin=0.33,ymax=0.69)
plt.plot(range(int(len(moy_rnd_acc[:50]))), moy_rnd_acc[:50], 'b+',  linewidth=1, linestyle='-', label='Vanilla FL')
plt.plot(range(int(len(moy_opt_acc[:50]))), moy_opt_acc[:50], 'ro',  linewidth=1, linestyle='-', label = 'CVFL')
plt.grid()
plt.ylabel(' Accuracy')
plt.xlabel('Communication Rounds')
plt.legend()
plt.savefig(accname)