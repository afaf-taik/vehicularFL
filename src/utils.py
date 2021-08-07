import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import numpy as np
#from scipy.spatial import distance
import math

def get_dataset(args):
    if args.dataset == 'cifar':
        data_dir = 'data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = 'data/mnist/'
        else:
            data_dir = 'data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                print('i am non iid unequal')
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'psmnist':
        data_dir = 'data/mnist/'
        #permute MNIST
        rng_permute = np.random.RandomState(92916)
        idx_permute = torch.from_numpy(rng_permute.permutation(784)) 
        transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Lambda(lambda x: x.view(-1)[np.array(idx_permute)].view(1, 28, 28) )])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                print('i am non iid unequal')
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    return train_dataset, test_dataset, user_groups



def flip_labels(user_groups,dataset_train, malicious_ids, target_label,malicious_label):
    dataset_train1 = []
    for k, v in user_groups.items():
        counts = np.zeros(10)
        
        if(k in malicious_ids):
            for i in v: 
                img,lbl = dataset_train[int(i)]
                if(lbl==target_label):
                    #ds1 = list(dataset_train[int(i)])
                    ds = img, malicious_label
                    dataset_train1.append(tuple(ds))
                else:
                    dataset_train1.append(dataset_train[int(i)])
        else:
            for i in v: 
                dataset_train1.append(dataset_train[int(i)])
    return dataset_train1

####Used in preliminary 
# def flip_labels_clusters(args,dataset_train, dataset_test, n_clusters = 4 ,original_labels= [1,7,3,5], alternative_labels=[7,1,5,3]):
#     dataset_train1 = []
#     dataset_test1 = []
#     c = 0
#     i = 0
#     while(i<len(dataset_train)):        
#         if(i<int((c+1)*len(dataset_train)/n_clusters)):
#             img,lbl = dataset_train[int(i)]
#             if(lbl==original_labels[c]):
#                     #ds1 = list(dataset_train[int(i)])
#                 ds = img, alternative_labels[c]
#                 dataset_train1.append(tuple(ds))
#             else:
#                 dataset_train1.append(dataset_train[int(i)])
#             i = i+1        
#         else:
#             c = c+1

#         #cluster_ids = [a for a in range(c*len(dataset_test)/n_clusters,(c+1)*len(dataset_test)/n_clusters)]

#     c = 0
#     i = 0
#     while(i<len(dataset_test)):        
#         if(i<int((c+1)*len(dataset_test)/n_clusters)):
#             img,lbl = dataset_test[int(i)]
#             if(lbl==original_labels[c]):
#                     #ds1 = list(dataset_train[int(i)])
#                 ds = img, alternative_labels[c]
#                 dataset_test1.append(tuple(ds))
#             else:
#                 dataset_test1.append(dataset_test[int(i)])
#             i = i+1        
#         else:
#             c = c+1

#     return dataset_train1, dataset_test1

def flip_labels_clusters(args, user_groups,dataset_train, dataset_test, n_clusters = 4 ,labels= [[4,8],[1,7],[3,5],[6,2]]):
    dataset_train1 = []
    dataset_test1 = []
    clusters = []
    for c in range(n_clusters):
        malicious_ids = [a for a in range(int(c*args.num_users/n_clusters), int((c+1)*args.num_users/n_clusters))]
        clusters.append(malicious_ids)
        for k, v in user_groups.items():
            counts = np.zeros(10)
        
            if(k in malicious_ids):
                for i in v: 
                    img,lbl = dataset_train[int(i)]
                    if(lbl==labels[c][0]):
                    #ds1 = list(dataset_train[int(i)])
                        ds = img, labels[c][1]
                        dataset_train1.append(tuple(ds))
                    elif(lbl==labels[c][1]):
                        ds = img, labels[c][0]
                        dataset_train1.append(tuple(ds))

                    else:
                        dataset_train1.append(dataset_train[int(i)])

    c = 0
    i = 0
    while(i<len(dataset_test)):        
        if(i<int((c+1)*len(dataset_test)/n_clusters)):
            img,lbl = dataset_test[int(i)]
            if(lbl==labels[c][0]):
                ds = img, labels[c][1]
                dataset_test1.append(tuple(ds))
            elif(lbl==labels[c][0]):
                ds = img, labels[c][1]
                dataset_test1.append(tuple(ds))

            else:
                dataset_test1.append(dataset_test[int(i)])
            i = i+1        
        else:
            c = c+1
    return dataset_train1, dataset_test1, clusters

def get_dataset_clustered(args):
    if args.dataset == 'cifar':
        data_dir = 'data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = 'data/mnist/'
        else:
            data_dir = 'data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                print('i am non iid unequal')
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'psmnist':
        data_dir = 'data/mnist/'
        #permute MNIST
        rng_permute = np.random.RandomState(92916)
        idx_permute = torch.from_numpy(rng_permute.permutation(784)) 
        transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Lambda(lambda x: x.view(-1)[np.array(idx_permute)].view(1, 28, 28) )])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                print('i am non iid unequal')
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    return train_dataset, test_dataset, user_groups


def get_entropy(user_groups,dataset_train):
    entropy = []
    for k, v in user_groups.items():
        en=0
        counts=np.zeros(10)
        for i in v: 
            _,lbl = dataset_train[int(i)]
            counts[lbl]+=1

        for j in range (len(counts)):
            counts[j]=counts[j]/len(v)
            if(counts[j]!=0):
                en+= -counts[j]*math.log2(counts[j])
        entropy.append(en)
    return entropy

def get_gini(user_groups,dataset_train):
    entropy = []
    for k, v in user_groups.items():
        en=0
        counts=np.zeros(10)
        for i in v: 
            _,lbl = dataset_train[int(i)]
            counts[lbl]+=1

        for j in range (len(counts)):
            counts[j]=counts[j]/len(v)
            en+= counts[j]**2
        en = 1 - en
        entropy.append(en)
    return entropy


def get_importance(entropy,size,age,rnd,roE=1/3,roD=1/3,roA =1/3):
    importance = []
    totalsize = sum(size)
    for i in range(len(entropy)):
        importance.append(roE*entropy[i] + roD * size[i]/totalsize + roA*age[i]/rnd )
    return importance


def get_order(importance):
    e=[]
    for i in range(len(importance)):
        e.append(-1*importance[i])
    return np.argsort(e)
def get_age(age):
    a = []
    for i in range(len(age)):
        a.append(math.log2(1+age[i]))
    return a
'''
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

'''

def average_weights(w,s):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    m = sum(s)
    for key in w_avg.keys():
        w_avg[key] = s[0]*w_avg[key]
        for i in range(1, len(w)):
            w_avg[key] += s[i]*w[i][key]
        w_avg[key] = torch.div(w_avg[key], m)
    return w_avg

###########################################################

def build_model_by_cluster(w,s,clusters, cluster):
    #n_clusters = max(np.unique(clusters))
    c = np.argwhere(clusters == cluster+1).flatten()
    w_temp = [w[i] for i in c]
    s_temp = [s[i] for i in c]    
    return average_weights(w_temp,s_temp)



def determine_preferred_model(preferences):
    return np.argmax(preferences, axis=1)

def aggregate_clusterwise(w,s,clusters, models):
    return        

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def models_similarity(l):
    weights = []
    for i in range(len(l)):
        #for key in l[i].keys():
            #detach().cpu().numpy()
        
        #    weights[i].append(l[i][key].detach().cpu().numpy())
        #weights_[i] = np.array(weights[i])
        weights.append(flatten(l[i]))  

    similarities = np.zeros((len(l),len(l)))
    for i in range(len(l)):
        for j in range(len(l)):
            #+1e-12
            similarities[i][j] = 1 - torch.sum(weights[i]*weights[j])/(torch.norm(weights[i])*torch.norm(weights[j])+1e-15) 
            #similarities[j][i] = similarities[i][j]
    for i in range(len(l)):
        similarities[i][i] = 0 
    return similarities

def newmodel(args, model, global_model):
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    #newmodel = copy.deepcopy(global_model)
    model.to(device)
    model.train()
    #return model.state_dict()



#'''
def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return