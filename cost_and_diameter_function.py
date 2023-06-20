# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:26:09 2023
@author: Martijn Piket
Contact: martijnpiket@outlook.com

This function may be used in combination with a street network, a set of nodes with demand, a set of
edges with demand and their positions.

Edges should be in the following form: edges = [(a, b, l1), (c, d, l2), ...]
nodes_with_demand = [(a, d1), (b, d2), ...]
edges_with_demand = [(a, b, d), ...]
positions = 

This function is used to calulate pipeline diameters and the cost of the pipelines
in the network.

"""

import numpy as np
import networkx as nx

def merge(list1, list2):
    merged_list = tuple(zip(list1, list2))
    return merged_list

###############################################################################
def Steel(edges, nodes_with_demand, demand_edges, positions):
    
    # First, the lengths of the streets are rounded to integers.
    deze = []
    for xi in edges:
        info = (xi[0], xi[1], round(xi[2]))
        deze.append(info)
    edges=deze
    for x in edges:
        if x[2] == 0:
            index = edges.index(x)
            info = (x[0], x[1], 1)
            edges[index] = info
    edgescopy = edges.copy()
    # edges is the street network in the area, but rounded.
    
    # The demand of all nodes is stored in a list, also a list is created with only the nodes without the demand.
    simple_nodes_with_demand = []
    nodes_with_demand1 = []
    for x in nodes_with_demand:
        index = nodes_with_demand.index(x)
        info = (x[0], x[1])
        nodes_with_demand1.append(info)
        simple_nodes_with_demand.append(x[0])
    nodes_with_demand = nodes_with_demand1
    
    # A graph is defined using networkx. This graph contains the street network
    # and all nodes with demand. This graph is used to define the paths of the
    # pipelines from the source.
    
    testf = nx.Graph()
    testf.add_weighted_edges_from(edgescopy)
    testf.add_nodes_from(simple_nodes_with_demand)
    
    nodes_in_graph = list(testf.nodes)
    
    coordsnd = []
    paths = []
    for x in demand_edges:
        node1 = x[0]
        node2 = x[1]
        for no in nodes_with_demand:
            if node1 == no[0]:
                demand1 = no[1]
            if node2 == no[0]:
                demand2 = no[1]
        sp1 = (nx.shortest_path_length(testf, node1, 's', weight='weight'))
        sp2 = (nx.shortest_path_length(testf, node2, 's', weight='weight'))
        if sp1 < sp2:
            path1 = list(nx.shortest_path(testf, node1, 's', weight='weight'))
            path2 = path1.copy()
            path2.append(node2)
        if sp2 < sp1:
            path2 = list(nx.shortest_path(testf, node2, 's', weight='weight'))
            path1 = path2.copy()
            path1.append(node1)
        if sp1 == sp2:
            print('The shortest path to one end of the street is the same as the shortest path to the other end.')
        
        paths.append((path1, node1, sp1, demand1))
        paths.append((path2, node2, sp2, demand2))
        
        key1 = node1
        coord1 = [sp1, 0]
        key2 = node2
        coord2 = [sp2, 0]
        
        coordsnd.append((key1, coord1))
        coordsnd.append((key2, coord2))
    
    
    # All shortest paths to all clusters are known. All clusters have a certain energy demand.
    # All nodes that appear on a shortest path must carry a certain amount of energy demand.
    # In the following calculation, it is determined how much energy each node carries.
    flows = len(nodes_in_graph) * [0]
    for x in paths:
        E_erbij = x[3]
        for no in x[0]:
            index = nodes_in_graph.index(no)
            E_totaal = flows[index] + E_erbij
            flows[index] = E_totaal
        
    # Here the flow is made insightful. One can print this to make it insightful.
    flow_info = merge(nodes_in_graph, flows)
    sorted_flow = sorted(flow_info, key=lambda x:x[1])
    
    # If a street does not carry flow, it is obsolete and is removed from the graph.
    to_remove = []
    flows_to_remove = []
    for x in sorted_flow:
        if x[1] == 0 and x[0] not in to_remove:
            if x not in flows_to_remove:
                flows_to_remove.append(x)
    for y in flows_to_remove:
        sorted_flow.remove(y)
    edges_to_remove = []
    for ed in edges:
        if ed[0] in to_remove or ed[1] in to_remove:
            if ed not in edges_to_remove:
                edges_to_remove.append(ed)
    
    # Here it is checked whether removing an edge may lead to nodes with demand that
    # cant be reached anymore. Theoretically this should not happen.
    for eds in edges_to_remove:
        edges.remove(ed)
        test = nx.Graph()
        test.add_weighted_edges_from(edges)
        yn = []
        for no in simple_nodes_with_demand:
            tf = nx.has_path(test, no, 's')
            yn.append(tf)
        all_path = all(yn)
        if all_path == False:
            edges.append(ed)
            
        if all_path == True:
            pass
    
    # The flows in the pipelines are increased such that it may lose a certain amount
    # to thermal losses while all clusters still receive their full demand.
    flows = []
    for x in sorted_flow:
        energy_flow = (x[0], x[1]/0.7)
        flows.append(energy_flow)
    
    # Now that all energy flows through the nodes are known, the energy flow through
    # each street is calculated as follows.
    joule_edges = []
    for ed in edges:
        node1 = ed[0]
        node2 = ed[1]
        for x in flows:
            if node1 == x[0]:
                flow1 = x[1]
            if node2 == x[0]:
                flow2 = x[1]
        if flow1 < flow2:
            edge = (node1, node2, flow1)
        if flow2 < flow1:
            edge = (node1, node2, flow2)
        if flow2 == flow1:
            edge = (node1, node2, flow1)         

        joule_edges.append(edge)
    
    
    # In the following calculation, the energy flows are translated into mass flows
    # and the radius of the pipeline.
    
    T_water_source = 50
    T_indoor = 20
    
    kilos = []
    radii = []
    for x in joule_edges:
        kilo = x[2] / ((T_water_source - T_indoor)*4186)
        radius = np.sqrt(((kilo/1000)/(np.pi)))
        info = (x[0], x[1], kilo)
        info2 = (x[0], x[1], radius)
        kilos.append(info)
        radii.append(info2)
    
    # The information of each pipeline is summarized.
    total_edge = []
    for x in radii:
        node1 = x[0]
        node2 = x[1]
        for y in edges:
            if node1 in y and node2 in y:
                edge = (x[0], x[1], x[2], y[2])
        total_edge.append(edge)
    
    # The cost of the network is calculated.
    totalcost = 0
    count = 0
    for x in total_edge:
        count = count + 1
        Area = x[2]*x[2]*3.14
        cost = (x[3])*((Area)**(0.6))
        totalcost = totalcost + cost
    
    # The diameter of each pipeline is calculated.
    diam_edges = []
    for x in total_edge:
        diam = x[2] * 2
        edge = (x[0], x[1], diam)
        diam_edges.append(edge)
    
    return(totalcost, diam_edges, 'obsolete nodes', to_remove, edges, simple_nodes_with_demand, nodes_with_demand, paths, coordsnd, total_edge, flows)
