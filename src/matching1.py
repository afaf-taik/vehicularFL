import networkx as nx
from pulp import *
import matplotlib.pyplot as plt


#Selected = [1, 2, 3]
#NotSelected = [1, 2, 3, 4]
#ucap = Nmax for all , u are the cluster-heads
#ucap = {1: 1, 2: 2, 3: 2} #u node capacities
# vcap = 1 for all , v are the other vehicles
#vcap = {1: 1, 2: 1, 3: 1, 4: 1} #v node capacities

#weights : R_{v,u} = acc of the model 
#create function to populate dictionnary
# wts = {(1, 1): 0.57, (1, 3): 0.55,
#        (2, 1): 0.48, (2, 4): 0.56,
#        (3, 2): 0.72, (3, 4): 0.6}

def capacities(Selected, NotSelected, Nmax):
    dict_heads = {}
    dict_followers = {}
    for i in Selected:
        dict_heads[i] = Nmax
    for j in NotSelected:
        dict_followers[j] = 1 
    return dict_heads, dict_followers



def create_wts_dict(Selected, NotSelected, Preferences, LLT):
    wts = {}

    for i in Selected:
        idx_model = np.argmax(Preferences[i,:])
        for j in NotSelected:
            wts[(i,j)] = Preferences[j][idx_model]*LLT[i][j]
    return wts

#just a convenience function to generate a dict of dicts

def create_wt_doubledict(Selected, NotSelected):
    wt = {}
    for u in Selected:
        wt[u] = {}
        for v in NotSelected:
            wt[u][v] = 0

    for k,val in wts.items():
        u,v = k[0], k[1]
        wt[u][v] = val
    return(wt)

# #A wrapper function that uses pulp to formulate and solve a WBM
def solve_wbm(Selected, NotSelected, wt):

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

wt = create_wt_doubledict(Selected, NotSelected)
p = solve_wbm(Selected, NotSelected, wt)
print_solution(p)    

selected_edges = get_selected_edges(p)

print(selected_edges)



# if __name__ == '__main__':
#     args = args_parser()    
#     args.num_users = 30
#     NCHANNELS = 4
#     S = 100000
#     Ttrain, train_dataset, test_dataset, user_groups = initialize(args)
#     #print(Ttrain)  
#     from VehicularEnv import Freeway,V2IChannels,V2VChannels, Vehicle
#     env1 = Freeway(num_vehicles = args.num_users, num_slots = 1, num_channels = NCHANNELS)
#     env1.build_environment()
#     #LLT_ = env1.LLT()
#     #print(LLT_)



# ############################Allocate test##########################         
#     Tk = env1.rate_stay()














# #Create a Networkx graph. Use colors from the WBM solution above (selected_edges)
# graph = nx.Graph()
# colors = []
# for u in Selected:
#     for v in NotSelected:
#         edgecolor = 'blue' if (str(u), str(v)) in selected_edges else 'gray'
#         if wt[u][v] > 0:
#             graph.add_edge('u_'+ str(u), 'v_' + str(v))
#             colors.append(edgecolor)


# def get_bipartite_positions(graph):
#     pos = {}
#     for i, n in enumerate(graph.nodes()):
#         x = 0 if 'u' in n else 1 #u:0, v:1
#         pos[n] = (x,i)
#     return(pos)

# pos = get_bipartite_positions(graph)


# nx.draw_networkx(graph, pos, with_labels=True, edge_color=colors,
#        font_size=20, alpha=0.5, width=3)

# plt.axis('off')
# plt.show() 

# print("done")