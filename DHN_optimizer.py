import networkx as nx
import pandas as pd
from fairness_diam import fairness_diam
from Filtering_function import Filtering_network

def merge(list1, list2):
    merged_list = tuple(zip(list1, list2))
    return merged_list

def merge3(list1, list2, list3):
    merged_list = tuple(zip(list1, list2, list3))
    return merged_list


def merge4(list1, list2, list3, list4):
    merged_list = tuple(zip(list1, list2, list3, list4))
    return merged_list

df_nodes = pd.read_excel(
    r"C:\Users\marti\OneDrive\Bureaublad\2.xlsx", sheet_name='Sheet1') # Read the nodes
df_edges = pd.read_excel(
    r"C:\Users\marti\OneDrive\Bureaublad\1.xlsx", sheet_name='Sheet1') # Read to the street network
df_edges_in = pd.read_excel(
    r"C:\Users\marti\OneDrive\Bureaublad\3.xlsx", sheet_name='Sheet1') # Read the streets with demand

from_ = df_edges['from']
to_ = df_edges['to']
weight = df_edges['weight']

from_in = df_edges_in['from']
to_in = df_edges_in['to']

node_ = df_nodes['node']
position = df_nodes['position']
demands = df_nodes['demand']

all_nodes = list(node_)

# Define the positions of all nodes
positions = []
x = []
y = []
for n in range(len(position)):
    tuple_ = eval(position[n])
    x.append(tuple_[0])
    y.append(tuple_[1])
    positions.append(tuple_)

# Define the weighted edges of the graph.
weighted_edges = []
for n in range(len(from_)):
    info = (from_[n], to_[n], weight[n])
    weighted_edges.append(info)
we_copy = weighted_edges.copy()

# Remove duplicate edges from the network.
a = Filtering_network(weighted_edges)
weighted_edges = a

# Define the edges with demand
dem_edges = []
for x in range(len(from_in)):
    info = (from_in[x], to_in[x])
    dem_edges.append(info)

for x in dem_edges:
    index = dem_edges.index(x)
    for y in weighted_edges:
        if x[0] == y[0] and x[1] == y[1]:
            edge = (x[0], x[1], y[2])
            dem_edges[index] = edge
        if x[0] == y[1] and x[1] == y[0]:
            edge = (x[0], x[1], y[2])
            dem_edges[index] = edge

# Define the nodes of edges with demand that belong to a street with demand.
nodes_with_demand = []
demand_nodes = []
for n in range(len(node_)):
    if demands[n] > 0 or node_[n] == 's':
        info = (node_[n], demands[n]*50/2)
        if node_[n] != 's':
            abc = node_[n]
            nodes_with_demand.append(abc)
        demand_nodes.append(info)

nodes_amount = demand_nodes

# Manually search where the source will be placed in the network and replace that node by 's'
node_ = list(node_)
for x in node_:
    if x == 7049:
        index = node_.index(x)
        node_[index] = 's'
        
node_positions = merge(node_, positions)
positions = dict(node_positions)
# Add s to the nodes with demand.
demand_nodes.append(('s', 0))

#weighted_edges.append((618, 1493, 10))
#weighted_edges.append((2890, 1493, 50))

# Make a graph of the street network.
test = nx.Graph()
test.add_weighted_edges_from(weighted_edges)

# Define the starting network of the district heating network based on shortest paths.
nodes_with_demand = []
for x in dem_edges:
    option1 = x[0]
    option2 = x[1]
    length1 = nx.shortest_path_length(test, option1, 's', 'weight')
    length2 = nx.shortest_path_length(test, option2, 's', 'weight')
    if length1 < length2:
        nodes_with_demand.append(option1)
    if length2 < length1:
        nodes_with_demand.append(option2)

nodes_with_demand = list(dict.fromkeys(nodes_with_demand))

dist = []
begin_edges = []
for n in range(len(nodes_with_demand)):
    print(nodes_with_demand[n])
    sh_path = nx.shortest_path(test, nodes_with_demand[n], 's')
    if nodes_with_demand[n] == 's':
        pass
    else:
        for o in range(len(sh_path)-1):
            edge = (sh_path[o], sh_path[o+1])
            begin_edges.append(edge)

begin_edges1 = begin_edges.copy()
test1 = nx.Graph()
test1.add_edges_from(begin_edges1)
begin_edges = list(test1.edges)

for i in range(len(begin_edges)):
    index = begin_edges.index(begin_edges[i])
    for j in range(len(weighted_edges)):
        if begin_edges[i][0] == weighted_edges[j][0] and begin_edges[i][1] == weighted_edges[j][1]:
            edge = (weighted_edges[j][0],
                    weighted_edges[j][1], weighted_edges[j][2])
            begin_edges[index] = edge
            print(i)
        elif begin_edges[i][0] == weighted_edges[j][1] and begin_edges[i][1] == weighted_edges[j][0]:
            edge = (weighted_edges[j][0],
                    weighted_edges[j][1], weighted_edges[j][2])
            begin_edges[index] = edge
            print(i)

# network will be the w_dem_edges and begin_network together. Add those first.
begin_edges.extend(dem_edges)
begin_network = begin_edges

network = begin_network
for x in network:
    index = network.index(x)
    edge = (x[0], x[1], x[2])
    network[index] = edge

# Remove options from the streets network that are streets with demand.
print(len(weighted_edges))
for n in network:
    if n in weighted_edges:
        weighted_edges.remove(n)
for we in dem_edges:
    if we in weighted_edges:
        weighted_edges.remove(we)
print(len(weighted_edges))

nodes_amount = nodes_amount.copy()
begin_network = network

# simdem are the streets with demand defined by only the two nodes, no demand.
simdem = []
for x in dem_edges:
    sim = (x[0], x[1])
    simdem.append(sim)
    
start_network = begin_network.copy()

func1 = fairness_diam(start_network, nodes_amount, positions, dem_edges, 1900)
cost_start = func1[3]

checklist = []
for x in dem_edges:
    checklist.append((x[0], x[1]))

count=0 
# The following loop is the Edge-Turn heuristic.
for x in start_network:
    index = start_network.index(x)
    edge = start_network[index]
    node1 = x[0]
    node2 = x[1]
    check = (x[0], x[1])
    checkrev = (x[1], x[0])
    if check in checklist or checkrev in checklist:
        pass
    else:
        #print(index, edge)
        replace = []
        for y in weighted_edges:
            if node1 in y or node2 in y:
                replace.append(y)
        for ed in replace:
            start_network[index] = ed

            test = nx.Graph()
            test.add_weighted_edges_from(start_network)
            test.add_nodes_from(all_nodes)
            path_to_s = []
            
            for no in nodes_amount:
                haspath = nx.has_path(test, no[0], 's')
                path_to_s.append(haspath)
            all_path = all(path_to_s)
            
            print(all_path)
            
            if all_path == False:
                start_network[index] = edge

            if all_path == True: # If all streets with demand are reached, a objective may be used to compare to the previous solution.
                func_in = fairness_diam(start_network, nodes_amount, positions, dem_edges, 1900)
                cost_in = func_in[3]
                print(cost_in)
                if cost_in < cost_start:
                    count = count + 1
                    cost_start = cost_in
                    start_network[index] = ed
                    edge = ed
                    print(cost_in, count)
                
                if cost_in >= cost_start:
                    start_network[index] = edge
    
func_after_edge_turn = fairness_diam(start_network, nodes_amount, positions, dem_edges, 1900)
cost_after_edge_turn = func_after_edge_turn[3]
cost_out = cost_after_edge_turn

# Save the intermediate solution.
with open(r"C:\Users\marti\OneDrive\Bureaublad\notsimple\cost_edge_turn.txt", 'w') as f:
    for line in start_network:
        f.write(f"{line}\n")

after_ET = start_network.copy()


# Check whether removing edges from the network
count_dem = 0
for x in after_ET:
    edge = (x[0], x[1], x[2])
    rev = (x[1], x[0], x[2])
    sim = (x[0], x[1])
    simrev = (x[1], x[0])
    index = after_ET.index(x)
    if sim in simdem or simrev in simdem:
        count_dem = count_dem + 1
        print(index, 'this edge has demand', len(after_ET), count_dem)
        pass
    else:
        after_ET.remove(x)
        test = nx.Graph()
        test.add_weighted_edges_from(after_ET)
        test.add_nodes_from(all_nodes)
        yn = []
        for no in nodes_amount:
            tf = nx.has_path(test, no[0], 's')
            yn.append(tf)
        all_path = all(yn)

        if all_path == True:
            func_in = fairness_diam(after_ET, nodes_amount, positions, dem_edges, 1900)
            cost_in = func_in[3]
            if cost_in <= cost_out:
                cost_out = cost_in
                print('better solution', cost_in)
            if cost_in > cost_out:
                after_ET.insert(index, edge)
                print('worse solution')

        if all_path == False:
            after_ET.insert(index, edge)

func_after_remove1 = fairness_diam(after_ET, nodes_amount, positions, dem_edges, 1900)
cost_after_remove1 = func_after_remove1[3]
cost_out = cost_after_remove1

test = nx.Graph()
test.add_weighted_edges_from(after_ET)
valency_nodes = list(test.degree()) #For the Valency shuffle, the degree of every node is calculated.

nodes_to_remove = []
CC = list(nx.connected_components(test))
for x in CC:
    if len(x) < 10:
        for y in x:
            nodes_to_remove.append(y)

for x in network:
    node1 = 'empty'
    node2 = 'empty'
    if x[0] in nodes_to_remove:
        node1 = x[0]
    if x[1] in nodes_to_remove:
        node2 = x[1]
    if node1 != 'empty' and node2 != 'empty':
        edge = (node1, node2, x[2])
        network.remove(edge)

alledges = []
for x in we_copy:
    if x[0] == 7049:
        edge = ('s', x[1], x[2])
    if x[1] == 7049:
        edge = (x[0], 's', x[2])
    else:
        edge = (x[0], x[1], x[2])
    alledges.append(edge)
    
node = []
important_nodes = []
for x in valency_nodes:
    if x[1] > 2:
        important_nodes.append(x) # If the degree of a node is 3 or higher, it is a high valency node.
        node.append(x[0])

val = nx.Graph()
val.add_nodes_from(node)

net = nx.Graph()
net.add_weighted_edges_from(network)
netnodes = list(net.nodes)

# The following section generates different options for the high valency shuffle.
# Depending on the amount of high valency nodes and the size of the network, 
# It may take quite a while.
###############################################################################################
nodes_with_demand = []
for x in nodes_amount:
    info = x[0]
    nodes_with_demand.append(info)

for x in important_nodes:
    found = False
    for y in network:
        if x[0] in y:
            found = True
    
    if found == False:
        important_nodes.remove(x)
        print('removed', x)
        
ordered = ['empty'] * len(important_nodes)
for x in important_nodes: # This loop searched for the distances to all other nodes from the high valency node.
    index = important_nodes.index(x)
    dists = []
    for y in nodes_with_demand:
        if x[0] == y:
            pass
        else:
            dist = nx.shortest_path_length(net, x[0], y, weight='weight')
            info = ('from', x[0], 'to', y, dist)
            dists.append(info)
    
    ordered[index] = dists

ordered_copy = ordered.copy() # For every node the distance to the other nodes is ordered.
ordered = ordered_copy.copy()

four_closest = ['empty'] * len(ordered)
for x in ordered: # This loop picks the closest four nodes to the high valency node.
    four = []
    index = ordered.index(x)
    for y in range(4):
        aa = min(x, key=lambda x: x[4])
        four.append(aa)
        x.remove(aa)
    four_closest[index] = four

four_paths = ['empty'] * len(ordered)
paths = [] # This loop calculated the paths to the four closest nodes (needed due to steiner nodes.)
for x in four_closest:
    four = []
    index = four_closest.index(x)
    for y in x:
        path = (y[1], y[3])
        paths.append(path)
        four.append(path)
    four_paths[index] = four

possib = [] # This loop calculates the other possibilities if the valancy is shuffled to another node.
for x in four_paths:
    nodes = [] 
    for y in x:
        if y[0] not in nodes:
            nodes.append(y[0])
        if y[1] not in nodes:
            nodes.append(y[1])
    
    for no in nodes:
        from_ = no
        paths = []
        for to in nodes:
            if no == to:
                pass
            else:
                path = (no, to)
                paths.append(path)
    
        possib.append(paths)

test = nx.Graph()
test.add_weighted_edges_from(alledges)

edges =  [] # Here the paths are translated into the right edges.
for x in possib:
    index = possib.index(x)
    print(x)
    e = []
    
    for eds in x:
        p = list(nx.shortest_path(test, eds[0], eds[1], weight='weight'))
        print(eds, p)
        for y in range(len(p)-1):
            node1 = p[y]
            node2 = p[y+1]
            for ed in alledges:
                if node1 in ed and node2 in ed:
                    weight = ed[2]
                    edge = ed
                    e.append(edge)
    e2 = []
    e3 = []
    for li in e:
        if li[2] not in e3:
            e2.append(li)
            e3.append(li[2])
                    
    edges.append(e2)

# Edges contains all possibilities of the high valency shuffle.

paths = nx.Graph()
paths.add_weighted_edges_from(alledges)

nodess = []
for x in edges:
    nodes = []
    for y in x:
        node1 = y[0]
        node2 = y[1]
        if node1 not in nodes:
            nodes.append(node1)
        if node2 not in nodes:
            nodes.append(node2)
    nodess.append(nodes)

# nodess contains all nodes that are active in a certain high valency shuffle.

test = nx.Graph()
test.add_weighted_edges_from(after_ET)

nodelist = []
for x in after_ET:
    if x[0] not in nodelist:
        nodelist.append(x[0])
    if x[1] not in nodelist:
        nodelist.append(x[1])
for y in nodes_amount:
    if y[0] not in nodelist:
        nodelist.append(y[0])


no = nx.Graph()
for nod in nodes_amount:
    no.add_node(nod[0])

# indicate all cycles before a change is made.
cycles_before = list(nx.cycle_basis(test.to_undirected()))

network = after_ET.copy()

func_after_remove1 = fairness_diam(network, nodes_amount, positions, dem_edges, 1900)
cost_after_remove1 = func_after_remove1[3]
cost_out = cost_after_remove1


correct = []
for x in network:
    edge = (x[0], x[1], x[2])
    correct.append(edge)

network = correct

for x in edges:
    index  = edges.index(x)
    deze_nodes = nodess[index]
    
    print(len(nodes_amount))
    
    # This removes the current connection between nodes. may be added later again
    later_toevoegen = []
    for ne in network:
        if ne[0] in deze_nodes and ne[1] in deze_nodes:
            later_toevoegen.append(ne)
            network.remove(ne)
    
    # The new connections are introduced.
    toegevoegd = x
    dezenodes = []
    for wedges in x:
        network.append(wedges)
        if wedges[0] not in dezenodes:
            dezenodes.append(wedges[0])
        if wedges[1] not in dezenodes:
            dezenodes.append(wedges[1])
    
            
    test = nx.Graph()
    test.add_weighted_edges_from(network)
    test.add_nodes_from(nodelist)
    cycles_in = list(nx.cycle_basis(test.to_undirected()))
    new_cycle = False
    
    # check if a new cycle is introduced.
    for cy in cycles_in:
        if cy not in cycles_before:
            new_cycle == True
            cycle = cy
    
    yn = []
    for no in nodes_amount: # Check if all streets with demand may be reached from the source.
        tf = nx.has_path(test, no[0], 's')
        yn.append(tf)
    all_path = all(yn)
    print(all_path)
    
    if all_path == False: # If not, initial edges are returned and the added edges are removed.
        infos = (network, 'not a viable solution')
        for wedges in toegevoegd:
            network.remove(wedges)
        for toe in later_toevoegen:
            network.append(toe)
            
    if all_path == True:
        
        if new_cycle == True: # If new cycles introduced this loop is used.
            print('new cycle')
            edge_to_remove = []
            for x in range(len(cycle)-1):
                node1 = cycle[x]
                node2 = cycle[x+1]
                for wed in network:
                    if node1 in wed and node2 in wed:
                        edge_to_remove.append(wed)
            
            for wed in edge_to_remove:
                network.remove(wed)
            
                test = nx.Graph
                test.add_weighted_edges_from(network)    
                
                yn = []
                for no in nodes_amount:
                    tf = nx.has_path(test, no[0], 's')
                    yn.append(tf)
                all_path = all(yn)
                
                if all_path == False:
                    network.append(wed)
                    
                if all_path == True:
                    func_in = fairness_diam(network, nodes_amount, positions, dem_edges, 1900)
                    cost_in = func_in[3]
                    
                    if cost_in < cost_out:
                        infos = (network, cost_in, 'better solution')
                        network.append(wed)
                        best_network = network
                        cost_out = cost_in
                    if cost_in >= cost_out:
                        network.append(wed)
                            
                print(cost_in, 'in new cycle')
                
        else: # If no new cycle is introduced, this cycle is used.
            func_in = fairness_diam(network, nodes_amount, positions, dem_edges, 1900)
            cost_in = func_in[3]
            if cost_in < cost_out: # check if the new network is better than the previous. 
                cost_out = cost_in
                infos = (network)
                best_network = network.copy()
                print(cost_in, 'better solution')
            if cost_in >= cost_out:
                for wedges in toegevoegd:
                    network.remove(wedges)
                for toe in later_toevoegen:
                    network.append(toe)
                    
    print(index, cost_in, len(network), ((cost_in-cost_out)/(cost_out))*100 )       

# Save the best network fount with valency shuffle.
with open(r"C:\Users\marti\OneDrive\Bureaublad\notsimple\cost_after_valency.txt", 'w') as f:
    for line in best_network:
        f.write(f"{line}\n")

after_VS = best_network.copy()

func_after_valency = fairness_diam(after_VS, nodes_amount, positions, dem_edges, 1900)
cost_after_valency = func_after_valency[3]
cost_out = cost_after_valency

#Check wheter removing edges for the last time will improve the network.

count = 0
for x in after_VS:
    edge = (x[0], x[1], x[2])
    rev = (x[1], x[0], x[2])
    sim = (x[0], x[1])
    simrev = (x[1], x[0])
    index = after_VS.index(x)
    
    if edge in dem_edges or rev in dem_edges:
        print(index, 'this edge has demand', len(after_VS))
        pass
    
    else:
        after_VS.remove(x)
        
        test = nx.Graph()
        test.add_weighted_edges_from(after_VS)
        test.add_nodes_from(nodelist)
        
        yn = []
        for no in nodes_amount:
            tf = nx.has_path(test, no[0], 's')
            yn.append(tf)
        all_path = all(yn)
        
        print(all_path)
        
        if all_path == True:
            func_in = fairness_diam(after_VS, nodes_amount, positions, dem_edges, 1900)
            cost_in = func_in[3]
            
            if cost_in <= cost_out:
                cost_out = cost_in
                print('better solution', cost_in)
                
            if cost_in > cost_out:
                after_VS.insert(index, edge)
                print('worse solution')

        if all_path == False:
            after_VS.insert(index, edge)
            print('not all paths possible')
            
    print(len(after_VS))

#Save the final network

with open(r"C:\Users\marti\OneDrive\Bureaublad\notsimple\verylastbestcost.txt", 'w') as f:
    for line in best_network:
        f.write(f"{line}\n")
    
    
    
    
    
    
    
    
    

