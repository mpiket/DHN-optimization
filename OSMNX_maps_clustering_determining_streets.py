# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:26:09 2023
@author: Martijn Piket
Contact: martijnpiket@outlook.com
"""

import shapely.wkt
import shapely.geometry
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl
import geopandas as gpd
import random
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from matplotlib.patches import Patch

# Functions that ease the merging of lists.
def merge(list1, list2):
    merged_list = tuple(zip(list1, list2))
    return merged_list
def merge3(list1, list2, list3):
    merged_list = tuple(zip(list1, list2, list3))
    return merged_list
def merge4(list1, list2, list3, list4):
    merged_list = tuple(zip(list1, list2, list3, list4))
    return merged_list

ox.config(log_console=True)

# Define the polygon that will be used to create the district heating network.
lat_point_list = [52.007293, 52.004223, 51.994843, 51.987100, 51.984220, 51.997222, 52.000075, 52.002004,52.005782, 52.007293]
lon_point_list = [4.371272, 4.375550, 4.382159, 4.352076, 4.333236,4.331905,4.343922,4.353320,4.360959, 4.371272]
# In the following step, the previous defines polygon is translated into a geopandas dataframe.
polygon_geom = shapely.geometry.Polygon(zip(lon_point_list, lat_point_list))
polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])       
# Saving the polygon in different formats
polygon.to_file(filename='polygon.geojson', driver='GeoJSON')
polygon.to_file(filename='polygon.gpkg', driver="GPKG")
polygon.to_file(filename='polygon.shp', driver="ESRI Shapefile")
# Retrieving the graph of the area, P_G is the projected graph of G.
G = ox.graph.graph_from_polygon(polygon_geom, network_type='all', simplify=False)
P_G = ox.projection.project_graph(G)
# Retrieving the geometries/buildings from the area.
B = ox.geometries.geometries_from_polygon(polygon_geom, tags={'building':True})
S_PG = ox.simplification.consolidate_intersections(P_G, tolerance=8, rebuild_graph=True, dead_ends=False, reconnect_edges=True)
nodes, edges = ox.graph_to_gdfs(G)

Buildings =  pd.DataFrame(B)
Buildings.stack()
Buildings.reset_index(inplace=True)

# The following calculation translates the area of the buildings from deg^2 to m^2. The constants may
# be adjusted to where one is located on the earth.
areas=[]
for n in range(len(Buildings)):
    levels = Buildings['building:levels'][n]
    if pd.isna(levels) == True:
        levels = 1
    levels = int(levels)
    area1 = Buildings['geometry'][n]
    surface = area1.area
    convert = (111000*68000)
    t_area = float(convert*surface)
    a = t_area*levels
    areas.append(a)

# Calculate the centroids of all buildings.
centroids = B.centroid
x_B = list(centroids.x)
y_B = list(centroids.y)

#placing the buildings with their shapes in the street network.
building_coords = [[x_B[i],y_B[i]] for i in range(len(x_B))]
dictionary = dict()
building_areas = merge(building_coords, areas)

# Assigning the locations to all buildings.
for i in range(len(building_coords)):
     node_id = i+1
     dictionary[node_id] = {'x': x_B[i], 'y': y_B[i]}

# Defininf the intersections of the street network.
tmp_list = []
for item_key, item_value in dictionary.items() :
  tmp_list.append({'geometry' : Point(item_value['x'], item_value['y']), 'osmid': item_key, 'y' : item_value['y'], 'x' : item_value['x']})

my_nodes = gpd.GeoDataFrame(tmp_list)
nodes_b = my_nodes
graph2 = ox.graph_from_gdfs(nodes, edges)

##############################################################################
# Example of a plot of the area.
fig, ax = plt.subplots()
legend_elements =  [Patch(facecolor='gainsboro', label='Polygon'), Patch(facecolor='dimgray', label='Roads and paths'), Patch(facecolor='royalblue', label='Intersections')]
polygon.plot(ax=ax, facecolor='gainsboro')
B.plot(ax=ax, facecolor='darkseagreen', edgecolor='black', linewidth=0.8)
edges.plot(ax=ax, linewidth=1, edgecolor='dimgray')
nodes.plot(ax=ax, color='royalblue', alpha=1, markersize=5)
ax.legend(handles=legend_elements)
plt.xlabel('latitude')
plt.ylabel('longitude')

#Building coordinates.
y_h = list(my_nodes['y'])
x_h = list(my_nodes['x']) 
X = merge(y_h, x_h)
X = np.array(X)

#Applying DBSCAN to cluster the buildings of the street network.
db = DBSCAN(eps=0.00013, min_samples=1).fit(X)
labels = db.labels_
clust_df = pd.DataFrame(X)
clusters = (clust_df[db.labels_ != -1])
# Color_clusters contains all clusters.
color_clusters = db.fit_predict(clusters)
no_clusters = len(np.unique(labels) )
no_noise = np.sum(np.array(labels) == -1, axis=0)
print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)
labels = db.labels_
no_clusters = len(set(labels))

# Assigning all buildings to the right cluster.
labels_change = labels.copy()
listtodelete = np.arange(0, no_clusters).tolist()
overwrite = labels_change.copy()
for i in range(no_clusters):
    count = 0
    random_num = random.choice(listtodelete)
    for x in labels_change:
        if x == i:
            index = np.where(labels_change == x)
            count = count + 1
            print(count)
            overwrite[index] = random_num
    listtodelete.remove(random_num)

cluster_and_nodes = no_clusters * ['empty']
area_clusters = no_clusters * ['empty']
building_cluster = merge4(y_h, x_h, labels, areas)

#Caclulate the floor area of every cluster.
for i in range(len(building_cluster)):
    index = building_cluster[i][2]
    node_to_add = (building_cluster[i][0], building_cluster[i][1], building_cluster[i][3])
    a = cluster_and_nodes[index]
    demand = building_cluster[i][3]
    if a == 'empty':
        cluster_and_nodes[index] = [node_to_add]
        area_clusters[index] = demand
    else:
        a = cluster_and_nodes[index]
        a.extend([node_to_add])
        cluster_and_nodes[index] = a
        area_clusters[index] = area_clusters[index] + demand

# Calculating the summed demand and the centroid of every cluster.
xi = []
yi = []
dem2 = []
flooring = []
for x in cluster_and_nodes:
    cord_x = 0
    cord_y = 0
    dem = 0
    for y in x:
        cord_x = y[0] + cord_x
        cord_y = y[1] + cord_y
        dem = dem + y[2]
    xs = cord_x/len(x)
    ys = cord_y/len(x)
    xi.append(xs)
    yi.append(ys)
    dem2.append(dem)
    sizes = dem
    flooring.append(sizes)

# Assiging all clusters a color to plot. Set the normalizer to reasonable values.
cmap = plt.cm.get_cmap('BuPu')
Kolors = []
for x in flooring:
    demands = x * 50
    norm = mpl.colors.Normalize(vmin=5000, vmax=500000)
    Kolor = cmap(norm(demands))
    Kolors.append(Kolor)
    
clustering = []
for n in range(len(area_clusters)):
    info = (cluster_and_nodes[n][0][0], cluster_and_nodes[n][0][1], n, area_clusters[n])
    clustering.append(info)

#Preparation to plot everything.
draw_nodes = np.array(clustering)
draw_nodes1 = draw_nodes.copy()

x = draw_nodes[:,1]
y = draw_nodes[:,0]
s1 = draw_nodes[:,3]/1000
s = draw_nodes[:,3]*50

xs = []
ys = []
for fg in cluster_and_nodes:
    xs.append(fg[0][1])
    ys.append(fg[0][0])

fig1, ax1 = plt.subplots()
plt.title("The demand and locations of clusters")
legend_elements1 =  [Patch(facecolor='gainsboro', label='Polygon'), Patch(facecolor='dimgray', label='Roads and paths')] #Patch(facecolor='olivedrab', label='Centroids of clusters')
#B.plot(ax=ax1, facecolor='silver')
polygon.plot(ax=ax1, facecolor='gainsboro')
edges.plot(ax=ax1, linewidth=1, edgecolor='dimgray')
#nodes.plot(ax=ax1, color='royalblue', alpha=1, markersize=5)
plt.xlabel('latitude')
plt.ylabel('longitude')
#plt.scatter(X[:,1], X[:,0], c=overwrite, cmap='rainbow', marker="o", alpha=1, picker=True) #c=db.labels_.astype(float)
plt.scatter(yi, xi, c=Kolors, marker="o", s=s1)
#lines_gpd.plot(ax=ax1, color='darkgrey')
#plt.scatter(x, y, c=s, cmap='summer', s=s1) # s=s1 for size , cmap='gray',  c=s
#plt.scatter(x, y, c='olivedrab', s=1)
#edges_demand.plot('new_column', cmap='summer', ax=ax1)
ax1.legend(handles=legend_elements1)
sm = plt.cm.ScalarMappable(cmap='BuPu', norm=plt.Normalize(vmin = 5000, vmax=1000000, clip=True))
sm._A = []
plt.colorbar(sm).set_label('Demand of the clusters [J]', size=12) 
plt.show()

# in x are the x coordinates of the clusters, in y the y. Search in graph G to closest edge.
distance = ox.distance.nearest_edges(G, yi, xi, return_dist=False)
distance = merge4(distance, x, y, s)
temp = []
for i in range(len(distance)):
    info = (distance[i][0][0], distance[i][0][1], distance[i][1], distance[i][2], distance[i][3])
    temp.append(info)

edges_pd_df = pd.DataFrame(edges)
edges_arr = edges_pd_df #.unstack()
edges_arr.stack()
edges_arr.reset_index(inplace=True)
df_edges_demand = edges_arr.assign(demand = np.nan)

# Defining the important values of every cluster.
df_clusters = pd.DataFrame(temp, columns=['edge_id1', 'edge_id2','x_cluster', 'y_cluster', 'size_cluster'])

# Some clusters connect to the same closest street, this sums size of the clusters along a street.
skip = []
hit = 0
for i in range(len(df_clusters)):
    thisone = df_clusters['size_cluster'][i]
    if i in skip:
        pass
    else:
        for j in range(len(df_clusters)):
            if i == j:
                pass
            else:
                if df_clusters['edge_id1'][i] == df_clusters['edge_id1'][j] and df_clusters['edge_id2'][i] == df_clusters['edge_id2'][j]:
                    size1 = df_clusters['size_cluster'][j]
                    size2 = df_clusters['size_cluster'][i] 
                    thisone = size1 + size2
                    skip.append(j)
                    hit = hit + 1
                    df_clusters['size_cluster'][i] = thisone

# The following calculation translates the demand of the clusters to the edges found in the street network.
# OSMnx works with 'ids' that have to be coordinated correctly.
abc = [0] * len(edges['osmid'])
hit = 0
for i in range(len(df_edges_demand)):
    for j in range(len(df_clusters)):
        if df_edges_demand['u'][i] == df_clusters['edge_id1'][j] and df_edges_demand['v'][i] == df_clusters['edge_id2'][j]:
            if abc[i] < df_clusters['size_cluster'][j]:
                abc[i] = df_clusters['size_cluster'][j]
            else:
                pass

#Preperation of the plotting
edges = edges.assign(new_column = abc)
colors = list(edges['new_column'])
norm = mpl.colors.Normalize(vmin=0, vmax=max(colors))
all_edges = edges.copy()


zeros = []
for i in range(len(edges)):
    if edges['new_column'][i] == 0:
        zeros.append(i)
zeros.sort(reverse=True)

edges_demand = all_edges.loc[all_edges['new_column'] > 0,:]

ed_df = pd.DataFrame(edges_demand)
ed_df.reset_index(inplace=True)

nd_df = pd.DataFrame(nodes)
nd_df.reset_index(inplace=True)

all_e_df = pd.DataFrame(all_edges)
all_e_df.reset_index(inplace=True)

###############################################################################
# In the following section all the dataframes of the streets with demand, the street network
# and the intersection are translated into lists that can be exported in a excel or txt file.
# The result is that the street network with the edges with demand may be used in other scripts
# to analyze.

# all_e_df contains all information on the streets
# nd_df contains all info on the nodes
# ed_df contains all info on edges with demand.

export_nodes = []
for i in range(len(nd_df)):
    position = (nd_df['x'][i], nd_df['y'][i])
    info = (i, position, 'demand')
    info = list(info)
    export_nodes.append(info)
export_nodes = list(export_nodes)
                

longer_lines = []
export_edges = []
for i in range(len(all_e_df)):
    linestring = all_e_df['geometry'][i]
    len_string = len(linestring.xy[0])-1
    x1 = linestring.xy[0][0]
    y1 = linestring.xy[1][0]
    x2 = linestring.xy[0][len_string]
    y2 = linestring.xy[1][len_string]
    from__ = (x1, y1)
    to__ = (x2, y2)
    print(from__, to__)
    weight__ = all_e_df['length'][i]
    demand__ = all_e_df['new_column'][i]
    info = (from__, to__, weight__, demand__)
    info = list(info)
    export_edges.append(info)
export_edges = list(export_edges)

export_edges_demand = []    
for i in range(len(ed_df)):
    linestring = ed_df['geometry'][i]
    len_string = len(linestring.xy[0])-1
    x1 = linestring.xy[0][0]
    y1 = linestring.xy[1][0]
    x2 = linestring.xy[0][len_string]
    y2 = linestring.xy[1][len_string]
    from__ = (x1, y1)
    to__ = (x2, y2)
    weight__ = ed_df['length'][i]
    demand__ = ed_df['new_column'][i]
    info = (from__, to__, weight__, demand__)
    info = list(info)
    export_edges_demand.append(info)
export_edges_demand = list(export_edges_demand)

for i in range(len(export_edges_demand)):
    punt1 = export_edges_demand[i][0]
    punt2 = export_edges_demand[i][1]
    print(punt1, punt2)
    for j in range(len(export_nodes)):
        if export_nodes[j][1] == punt1:
            export_edges_demand[i][0] = export_nodes[j][0]
            print(export_nodes[j][2])
            if export_nodes[j][2] == 'demand' or export_nodes[j][2] < export_edges_demand[i][3]:
                export_nodes[j][2] = export_edges_demand[i][3]
        if export_nodes[j][1] == punt2:
            export_edges_demand[i][1] = export_nodes[j][0]
            if export_nodes[j][2] == 'demand' or export_nodes[j][2] < export_edges_demand[i][3]:
                export_nodes[j][2] = export_edges_demand[i][3]
    
for i in range(len(export_edges)):
    punt1 = export_edges[i][0]
    punt2 = export_edges[i][1]
    print(punt1, punt2)
    for j in range(len(export_nodes)):
        if export_nodes[j][1] == punt1:
            export_edges[i][0] = export_nodes[j][0]
        if export_nodes[j][1] == punt2:
            export_edges[i][1] = export_nodes[j][0]
        if export_nodes[j][2] == 'demand':
            export_nodes[j][2] = 0
        
export_edges1 = []
for n in range(len(export_edges)):
    correct_edge = (export_edges[n][0], export_edges[n][1], export_edges[n][2])
    export_edges1.append(correct_edge)
export_edges = export_edges1
must_be_in = []
for i in range(len(export_edges_demand)):
    edge__ = [export_edges_demand[i][0], export_edges_demand[i][1]]
    must_be_in.append(edge__)

df_edges_big = pd.DataFrame(export_edges, columns = ['from', 'to', 'weight'])
df_nodes_big = pd.DataFrame(export_nodes, columns = ['node', 'position', 'demand'])
df_edges_demand = pd.DataFrame(must_be_in, columns = ['from', 'to'])

aa = pd.ExcelWriter(r"C:PATH") # Save the street network there
bb = pd.ExcelWriter(r"C:PATH") # Save the nodes here
cc = pd.ExcelWriter(r"C:PATH") # Save the streets with demand here

df_edges_big.to_excel(aa, sheet_name='Sheet1', index=False)
df_nodes_big.to_excel(bb, sheet_name='Sheet1', index=False)
df_edges_demand.to_excel(cc, sheet_name='Sheet1', index=False)

aa.save()
bb.save()
cc.save()

cmap = plt.cm.get_cmap('BuPu')
kleurtjesss = []
for x in range(len(edges_demand)):
    blub = edges_demand['new_column'][x]
    norm = mpl.colors.Normalize(vmin=5000, vmax=500000)
    abcdefghij = cmap(norm(blub))
    print(abcdefghij)
    kleurtjesss.append(abcdefghij)


fig3, ax3 = plt.subplots(figsize=(12,8))
polygon.plot(ax=ax3, facecolor='gainsboro')
edges.plot(ax=ax3, linewidth=1, edgecolor='dimgray')
edges_demand.plot(column = 'new_column', ax=ax3, edgecolor=kleurtjesss, linewidth=2)
sm = plt.cm.ScalarMappable(cmap='BuPu', norm=plt.Normalize(vmin = 5000, vmax=1000000, clip=True))
sm._A = []
plt.colorbar(sm).set_label('Demand of the streets [J]', size=12)
plt.xlabel('latitude')
plt.ylabel('longitude')
ax.legend(handles=legend_elements)
legend_elements =  [Patch(facecolor='gainsboro', label='Polygon'), Patch(facecolor='dimgray', label='Roads and paths'), Patch(facecolor='royalblue', label='Intersections')]
plt.title("Defining the streets with demand")
plt.show()