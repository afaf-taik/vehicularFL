import networkx as nx
from pulp import *
import matplotlib.pyplot as plt

from_nodes = [1, 2]
to_nodes = [1, 2, 3, 4]
ucap = {1: 3, 2: 3} #u node capacities
vcap = {1: 1, 2: 1, 3: 1, 4: 1} #v node capacities

wts = {(1, 1): 0.8, (1, 3): 0.0,(1,2): 0.5, (1,4):0.6,
       (2, 1): 0.4, (2,2): 0.0, (2, 4): 0.9, (2,3):0.8}

#just a convenience function to generate a dict of dicts
def create_wt_doubledict(from_nodes, to_nodes):

    wt = {}
    for u in from_nodes:
        wt[u] = {}
        for v in to_nodes:
            wt[u][v] = 0

    for k,val in wts.items():
        u,v = k[0], k[1]
        wt[u][v] = val
    return(wt)

def solve_wbm(from_nodes, to_nodes, wt):
    ''' A wrapper function that uses pulp to formulate and solve a WBM'''

    prob = LpProblem("WBM Problem", LpMaximize)

    # Create The Decision variables
    choices = LpVariable.dicts("e",(from_nodes, to_nodes), 0, 1, LpInteger)

    # Add the objective function 
    prob += lpSum([wt[u][v] * choices[u][v] 
                   for u in from_nodes
                   for v in to_nodes]), "Total weights of selected edges"


    # Constraint set ensuring that the total from/to each node 
    # is less than its capacity
    for u in from_nodes:
        for v in to_nodes:
            prob += lpSum([choices[u][v] for v in to_nodes]) <= ucap[u], ""
            prob += lpSum([choices[u][v] for u in from_nodes]) <= vcap[v], ""


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
wt = create_wt_doubledict(from_nodes, to_nodes)
p = solve_wbm(from_nodes, to_nodes, wt)
print_solution(p)


selected_edges = get_selected_edges(p)


#Create a Networkx graph. Use colors from the WBM solution above (selected_edges)
graph = nx.Graph()
colors = []
for u in from_nodes:
    for v in to_nodes:
        edgecolor = 'blue' if (str(u), str(v)) in selected_edges else 'gray'
        if wt[u][v] > 0:
            graph.add_edge('u_'+ str(u), 'v_' + str(v))
            colors.append(edgecolor)


def get_bipartite_positions(graph):
    pos = {}
    for i, n in enumerate(graph.nodes()):
        x = 0 if 'u' in n else 1 #u:0, v:1
        pos[n] = (x,i)
    return(pos)

pos = get_bipartite_positions(graph)


nx.draw_networkx(graph, pos, with_labels=True, edge_color=colors,
       font_size=20, alpha=0.5, width=3)

plt.axis('off')
plt.show() 

print("done")