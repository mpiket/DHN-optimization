# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:26:09 2023
@author: Martijn Piket
Contact: martijnpiket@outlook.com

This function calculates the thermal losses of the pipelines in the district heating
network. Within this function the cost function is incorporated. 

Edges should be in the following form: edges = [(a, b, l1), (c, d, l2), ...]
nodes_with_demand = [(a, d1), (b, d2), ...]
edges_with_demand = [(a, b, d), ...]
positions = 
kilotjes = float that determines the mass flow at the source.
"""

import numpy as np
import math
import networkx as nx
from steelpython import Steel

def fairness_diam(edges, nodes_with_demand, positions, demand_edges, kilotjes):
###############################################################################
    func = Steel(edges, nodes_with_demand, demand_edges, positions)
    cost = func[0] # The cost as calculated in the steel function
    diam = func[1] # The diameters of all pipelines in the network
    clean_network = func[4] # The district heating network with no obsolete pipelines
    just_nodes_with_demand = func[5] 
    nodes_with_demand = func[6] # The paths to all streets with demand
    paths = func[7] 
    coordsnd = func[8] # This list states at which distance from the source a street with demand is encountered.
    kilos = func[10] # The mass flow through every pipeline in the network.

    edges = clean_network.copy()
    
    # First check: All streets with demand may be reached from the source.
    
    test = nx.Graph()
    test.add_weighted_edges_from(edges)
    test.add_nodes_from(just_nodes_with_demand)
    yn = []
    for no in just_nodes_with_demand:
        tf = nx.has_path(test, no, 's')
        yn.append(tf)
    all_path = all(yn)

    # All the nodes that are in the district heating network are defined.
    all_nodes = []    
    for x in edges:
        if x[0] not in all_nodes:
            all_nodes.append(x[0])
        if x[1] not in all_nodes:
            all_nodes.append(x[1])
    
    testf = nx.Graph()
    testf.add_weighted_edges_from(edges)
    testf.add_nodes_from(all_nodes)
    
    stop_info = coordsnd.copy()
    
    #Redefine the paths to all streets with demand.
    check = []
    goede_paths = []
    for x in paths:
        info = (x[1], x[2])
        if info not in check:
            check.append(info)
            goede_paths.append(x)
    paths = goede_paths.copy()
    
    #Remove any small unconnected components. This may affect the checks.
    CC = list(nx.connected_components(testf))
    to_remove = []
    for nodes in CC:
        if len(nodes) < 30:
            for x in nodes:
                to_remove.append(x)
    for y in range(5):
        for x in edges:
            if x[0] in to_remove or x[1] in to_remove:
                edges.remove(x)
    
    testf = nx.Graph()
    testf.add_weighted_edges_from(edges)
    nodes_in_graph = list(testf.nodes)
    
    # The paths to all nodes with demand are already calculated. Now the paths
    # to all nodes that do not have a demand are calculated.
    nodes_that_need_path = []
    for x in nodes_in_graph:
            nodes_that_need_path.append(x)
    
    coordinates = []
    for no in nodes_that_need_path:
        leng = nx.shortest_path_length(testf, no, 's', weight='weight')
        keys_x = no
        coordi = [leng, 0]
        info = (keys_x, coordi)
        coordinates.append(info)
    
    # For all nodes the distance to the source is known, including the source itself.
    coordsnd.extend(coordinates)
    add_s = ('s', [0,0])
    coordsnd.append(add_s)
    
    #For every pipeline the distance at whicht the pipeline begins and the distance
    # at which the pipeline ends are indicated as follows:
    distances = []
    lines = []
    for x in edges:
        node1 = x[0]
        node2 = x[1]
        for y in coordsnd:
            if node1 == y[0]:
                dist1 = y[1][0]
            if node2 == y[0]:
                dist2 = y[1][0]
        info = (node1, node2, (dist1, dist2))
        lines.append(info)
        distances.append(dist1)
        distances.append(dist2)
        
    sort_inzicht = sorted(lines, key=lambda x:x[2][0])
    
    # For every edge the diameters are applied.
    pipes = []
    for e in diam:
        node1 = e[0]
        node2 = e[1]
        for l in lines:
            if node1 == l[0] and node2 == l[1] or node1 == l[1] and node2 == l[0]:
                if l[2][0] < l[2][1]:
                    info = (l[2][0], l[2][1], e[2])
                if l[2][1] < l[2][0]:
                    info = (l[2][1], l[2][0], e[2])
        pipes.append(info)
    
    # The pipelines are made insightful:
    sort_inzicht = sorted(pipes, key=lambda x:x[0])
    # The maximum distance of any node in the network is:
    max_distance = int(max(distances))

    # In the following loop the active diameters are calculated for every meter in the network.
    factors = []
    factor = []
    for c in range((max_distance+1)):
        count=0
        diameters = []
        for pi in pipes:
            if c >= pi[0] and c < pi[1]:
                count=count+1
                info = (c, count, pi)
                diameters.append(pi[2])
        factors.append(diameters)
        info2 = (c, count)
        factor.append(info2)
    factors.remove([])
    toe = len(factors)-1
    factors.append(factors[toe])
    
    clusters = []
    for cl in nodes_with_demand:
        clusters.append(cl[0])
    
    T_ground = 5 # Temperature of the ground
    T_indoor = 20 # indoor temperature
    thick_insul = 0.025 #Thickness of the insulation
    k_insul = 0.043 #0.043 # k of the insulation
    k_steel = 111 # k of the steel
    thick_steel = 0.007 # Thickness of the steel
    cp = 4186 # Heat capacity of the water
    
    stops = []
    info_on_stop = []
    for x in paths:
        stops.append(x[2])
        info = (x[1], x[2], x[3])
        info_on_stop.append(info)
    
    stops.sort()
    #The distances to all nodes are sorted. As well as the paths.
    sorted_demand = sorted(paths, key=lambda elem: elem[2])
    
    
    T_water =  30 # Temperature of the water at the source.
    m_ = kilotjes # Mass flow at the source as defined in the function.
    
    flow_information = []
    massas_information = []
    T_drop = []
    E_pipeline = []
    surfaces = []
    mass_drop = []
    houses_info = []
    nodes_mass = []
    nodes_energy = []
    
    for n in range(max_distance+1): # For every meter until the maximum distance is reached.
        hs = []
        sfs = []
        for xi in range(len(factors[n])): # Every meter contains 1 or more pipelines.
            D_1 = factors[n][xi] # Inner diameter
            D_2 = factors[n][xi] + 2*thick_steel # Inner diameter plus the thickness of the steel
            D_3 = factors[n][xi] + 2*thick_steel + 2*thick_insul # Outer diameter.
            surface_out = np.pi * D_3 #Surface of the pipeline
            h_in = (2*k_steel) / (np.log(D_2/D_1)*D_1)
            h_out = (2*k_insul) / (np.log(D_3/D_2)*D_3)
            over_h = 1/h_in + 1/h_out
            h_both = 1/over_h
            hs.append(h_both)
            sfs.append(surface_out)
        h_total = 0    
        for HS in hs:
            h_total = h_total + HS #Calculating the total heat loss
        total_surface = 0
        for SFS in sfs:
            total_surface = total_surface + SFS #Calculating the sum of surface areas.
        surfaces.append(total_surface) #May be used to compare different district heating networks.
        
        Q_n = total_surface * h_total * (T_water - T_ground) # The heat loss
        
        if math.isnan(m_) == True: #Due to divisions, m_ may become a nan. If this is nan then the mass flow is 0.
            m_ = 0
        
        if m_ == 0: #If the mass flow has become 0 there is now flow left, thus:
            Q = 0
            T_water = 0
            E = 0
            m_ = 0        
            T_drop.append((n, 0))
            E_pipeline.append((n,0))
    
        if m_ > 0: #If there is mass flow, the characteristics are calculated as follows:
            Q = Q_n
            T_water = T_water - (Q/(m_*cp))
            T_drop.append((n, T_water))
            E = (T_water-T_indoor) * m_ * 4186
            E_pipeline.append((n, E))
            mass_drop.append((n, m_))
            
        belangrijke_info = (n, "massa", m_, "temperatuur", T_water, "E", E)
        flow_information.append(belangrijke_info)
        
        if n in stops and m_ > 0 and T_water > 20.1: # If a house or houses is encountered at n meters from the source, loop starts.
            active = []
            for de in sorted_demand: # There may be multiple clusters at n meters.
                if de[2] == n:
                    active.append(de)        
            
            demand_n = []
            for ac in active:
                info = (ac[1], ac[2], ac[3])
                demand_n.append(info) # Contains the clusters and their demand.
            
                for ho in demand_n: # For every cluster that is encountered, the mass flow to it is calculated.
                    oppervlak_woning = ho[2]/50 # The floor area of the cluster
                    m_optimal = (ho[2]/(4186*(T_water-T_indoor))) # The mass flow it needs to satisfy its demand
                    m_max = (oppervlak_woning/600) # The limit of mass flow set by the heat exchanger.
    
                    used = False # Used to check which option is applicable.
                    
                    if m_optimal > m_max and used == False: # The mass needed is larger than the maximum allowed mass flow.
                        print("max mass flow exceeded")
                        nodes_mass.append(ho[0])
                        m_optimal = m_max # the maximum becomes what the cluster takes.
                        m_ = m_ - m_max # the mass flow in the network remains.
                        used = True
                    
                    if m_ - m_optimal <= 0 and used == False: #The mass flow in the network is not enough to satisfy demand
                        print("mass smaller than 0")
                        m_optimal = m_ #The cluster takes all that is left.
                        m_= 0
                        used = True
                    
                    if m_ - m_optimal > 0 and used == False: # There is enough mass flow in the network and the mass 
                    # flow is lower than the maximum set by the heat exchanger.
                        m_ = m_ - m_optimal
                        used = True
                    
                    info_masses = (n, m_optimal, info)
                    massas_information.append(info_masses)
                    
                    energy_derivative = m_optimal * (T_water-T_indoor)* 4186 # The energy that the cluster receives.
                    energy_in = (energy_derivative) 
                    
                    if math.isnan(energy_in) == True: #just a check
                        energy_in = 0
                    
                    if round(energy_in) < round(ho[1]): # If the energy gotten from the network lower than the demand,
                    # the node has a energy deficit.
                        nodes_energy.append(ho[0])
                                     
                    integer_ = energy_in/ho[2] # on a scale of 0 - 1, what is the energy satisfaction.
                    
                    if integer_ < 0:
                        integer_ = 0
                    else:
                        pass
                    
                    info_node = (ac[1], ho[2], energy_in, integer_)
                    houses_info.append(info_node)
                
        if n in stops and m_ <= 0 or n in stops and T_water < 20.1: # If a house is encountered but the mass flow
        # is reduced to 0, or the temperature of the water has become too low, the mass flow becomes unusable.
            active = []
            for de in sorted_demand:
                if de[2] == n:
                    active.append(de)
    
            demand_n = []
            for ac in active:
                info = (ac[1], ac[2], ac[3])
                demand_n.append(info)
                
                for ho in demand_n:
                    oppervlak_woning = ho[2]/30   
                    m_optimal = (ho[2]/(4186*(T_water-T_indoor)))
                    m_max = (oppervlak_woning/125)*0.2
                    
                    m__ = 0
                    
                    optimal_energy = 0
                    integer_ = 0  
                    
                    info5 = (n, m__, info)
                    massas_information.append(info5)
                    
                    info_node = (ac[1], ho[2], optimal_energy, integer_)
                    houses_info.append(info_node)
    
    start_energy = E_pipeline[0][1]
    energy_left = E_pipeline[max_distance-1][1]
    
    cold_house = []
    for x in houses_info:
        if round(x[3], 4) < 1:
            cold_house.append(x)
    
    sort_houses = sorted(houses_info, key=lambda x:x[0])
    
    supply_edges = []
    for x in demand_edges:
        node1 = x[0]
        node2 = x[1]
        found_1 = False
        found_2 = False
        for y in houses_info:
            if node1 == y[0]:
                int1 = y[3]
                found_1 = True
            if node2 == y[0]:
                int2 = y[3]
                found_2 = True
        if found_1 == True and found_2 == True:
            avg_frac = (int1+int2)/2
            edge = (node1, node2, avg_frac)
            supply_edges.append(edge)

    sort_supply = sorted(supply_edges, key=lambda x:x[2])
    
    E_consumed = 0
    for x in houses_info:
        E_consumed = E_consumed + x[2]
    
    perc_consumed = (E_consumed/start_energy)*100
    
    eff = ((E_consumed+energy_left)*100)/(start_energy)
    frac_loss = 100 - eff
    
    len_ = len(T_drop)
    last_ = T_drop[len_-1]
    new_ = (last_[0]+1, last_[1])
    T_drop.append(new_)
    
    temp_edges = []
    for x in lines:
        node1 = x[0]
        node2 = x[1]
        x1 = int(x[2][0])
        x2 = int(x[2][1])
        T1 = T_drop[x1][1]
        T2 = T_drop[x2][1]
        avg_temp = (T1+T2)/2
        edge = (node1, node2, avg_temp)
        temp_edges.append(edge)
    
    return("network diam", diam, 'cost', cost, "clean network", clean_network, "efficiency", eff, "energy left", energy_left, "temperature edges", temp_edges, "supply edges", sort_supply, "Cold houses", cold_house, "frac loss", frac_loss, kilos, E_pipeline, massas_information, flow_information)
