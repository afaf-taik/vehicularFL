import matplotlib.pyplot as plt

model_sizes = [160000, 320000, 640000, 960000]
plt.figure()
# plt.plot(model_sizes,[ 7.65333333,7.45333333,7.53333333,7.22666667],linewidth=4, label= '2 RBs', color = '#FCD757') 
# plt.plot(model_sizes,[15.73333333,16.08,16.18666667,15.98666667], linewidth=4,label= '4 RBs', color = '#169873') 
#plt.plot(model_sizes,[22.74666667,22.65333333,22.34666667,23.02666667], linewidth=4,label= '6 RBs', color = '#6B0504') 
#plt.ylabel("Number of Participants")

# plt.plot(model_sizes,[2.57333333,2.48444444,2.51111111,2.41333333],linewidth=4, label= '2 RBs', color = '#FCD757') 
# plt.plot(model_sizes,[5.44,5.56444444,5.67111111,5.47111111], linewidth=4,label= '4 RBs', color = '#169873') 
# plt.plot(model_sizes,[8.52444444,8.41777778,8.23555556,8.52444444], linewidth=4,label= '6 RBs', color = '#6B0504') 
# plt.ylabel("Number of cluster heads in each round")
plt.grid()
plt.legend()

plt.xlabel("Model size")
plt.show()





 
 