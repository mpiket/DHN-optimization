import shapely.wkt
import shapely.geometry
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import matplotlib as mpl
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from matplotlib.patches import Patch
from ast import literal_eval
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

# For plotting purposes:
###############################################################################################################################
ox.config(log_console=True)

lat_point_list = [52.007293, 52.004223, 51.994843, 51.987100, 51.984220, 51.997222, 52.000075, 52.002004,52.005782, 52.007293]
lon_point_list = [4.371272, 4.375550, 4.382159, 4.352076, 4.333236,4.331905,4.343922,4.353320,4.360959, 4.371272]

polygon_geom = shapely.geometry.Polygon(zip(lon_point_list, lat_point_list))
polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])       

polygon.to_file(filename='polygon.geojson', driver='GeoJSON')
polygon.to_file(filename='polygon.gpkg', driver="GPKG")
polygon.to_file(filename='polygon.shp', driver="ESRI Shapefile")

G = ox.graph.graph_from_polygon(polygon_geom, network_type='all', simplify=False)
P_G = ox.projection.project_graph(G)
B = ox.geometries.geometries_from_polygon(polygon_geom, tags={'building':True})
F = ox.geometries.geometries_from_polygon(polygon_geom, tags={'landuse'})
S_PG = ox.simplification.consolidate_intersections(P_G, tolerance=8, rebuild_graph=True, dead_ends=False, reconnect_edges=True)
nodes, edges = ox.graph_to_gdfs(G)
nodess, edgess = ox.graph_to_gdfs(S_PG)

Buildings =  pd.DataFrame(B)
Buildings.stack()
Buildings.reset_index(inplace=True)

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
    if a > 170000:
        print(levels, t_area)
    areas.append(a)
    
centroids = B.centroid

x_B = list(centroids.x)
y_B = list(centroids.y)

building_coords = [[x_B[i],y_B[i]] for i in range(len(x_B))]
dictionary = dict()
building_areas = merge(building_coords, areas)

for i in range(len(building_coords)):
     node_id = i+1
     dictionary[node_id] = {
    'x': x_B[i],
    'y': y_B[i]
     }

tmp_list = []
for item_key, item_value in dictionary.items() :
  tmp_list.append({
    'geometry' : Point(item_value['x'], item_value['y']),
    'osmid': item_key,
    'y' : item_value['y'],
      'x' : item_value['x'],
   })
my_nodes = gpd.GeoDataFrame(tmp_list)

nodes_b = my_nodes


cmap = plt.cm.get_cmap('BuPu')
kleurtjesss = []
for x in areas:
    blub = x * 30
    norm = mpl.colors.Normalize(vmin=5000, vmax=500000)
    abcdefghij = cmap(norm(blub))
    print(abcdefghij)
    kleurtjesss.append(abcdefghij)
count1 = 0
count2 = 0
count3  = 0
count4  = 0
count5 = 0
for x in areas:
    if x > 20000:
        count1 = count1 + 1
    if x > 10000 and x < 20000:
        count2 = count2 + 1
    if x > 1000 and x < 10000:
        count3 = count3 + 1
    if x > 100 and x < 1000:
        count4 = count4 + 1
    if x < 100:
        count5 = count5 + 1

print(count1/(len(areas)))
print(count2/(len(areas)))
print(count3/(len(areas)))
print(count4/(len(areas)))
print(count5/(len(areas)))

#####################################################################################################################################
# Defining the street network, the streets with demand and the nodes.
#####################################################################################################################################
df_nodes = pd.read_excel(
    r"C:\Users\marti\OneDrive\Bureaublad\2.xlsx", sheet_name='Sheet1')
df_edges = pd.read_excel(
    r"C:\Users\marti\OneDrive\Bureaublad\1.xlsx", sheet_name='Sheet1')
df_edges_in = pd.read_excel(
    r"C:\Users\marti\OneDrive\Bureaublad\3.xlsx", sheet_name='Sheet1')

from_ = df_edges['from']
to_ = df_edges['to']
weight = df_edges['weight']

from_in = df_edges_in['from']
to_in = df_edges_in['to']

node_ = df_nodes['node']
position = df_nodes['position']
demands = df_nodes['demand']

all_nodes = list(node_)

positions = []
x = []
y = []
for n in range(len(position)):
    tuple_ = eval(position[n])
    x.append(tuple_[0])
    y.append(tuple_[1])
    positions.append(tuple_)

weighted_edges = []
for n in range(len(from_)):
    info = (from_[n], to_[n], weight[n])
    weighted_edges.append(info)
we_copy = weighted_edges.copy()

a = Filtering_network(weighted_edges)
weighted_edges = a

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

nodes_with_demand = []
demand_nodes = []
for n in range(len(node_)):
    if demands[n] > 0 or node_[n] == 's':
        info = (node_[n], demands[n]*30/2)
        if node_[n] != 's':
            abc = node_[n]
            nodes_with_demand.append(abc)
        demand_nodes.append(info)

nodes_amount = demand_nodes

node_ = list(node_)
for x in node_:
    if x == 7049:
        index = node_.index(x)
        node_[index] = 's'
        
node_positions = merge(node_, positions)
positions = dict(node_positions)

demand_nodes.append(('s', 0))
#####################################################################################################################################
# Opening the different optimized district heating networks.

with open(r"C:\Users\marti\OneDrive\Bureaublad\notsimple\verylastbestcost.txt") as f:
    cost_network = [list(literal_eval(line)) for line in f]

with open(r"C:\Users\marti\Downloads\finalnetwork_eff.txt") as f:
    energy_network = [list(literal_eval(line)) for line in f]
    
with open(r"C:\Users\marti\OneDrive\Bureaublad\notsimple\intoesau.txt") as f:
    start_network = [list(literal_eval(line)) for line in f]

with open(r"C:\Users\marti\OneDrive\Bureaublad\notsimple\kruskalafternocycles.txt") as f:
    kruskal_network = [list(literal_eval(line)) for line in f]

# The different networks may be evaluated for their performance at different mass flows, etc.
massas = [1300, 1270, 1240]
cost_info = []
energy_info = []
for m in massas:
    func_cost = fairness_diam(cost_network, nodes_amount, positions, dem_edges, m)
    func_energy = fairness_diam(energy_network, nodes_amount, positions, dem_edges, m)
    
    cold_cost = func_cost[13]
    total_dE = 0
    extent = 0
    cold = 0
    for x in cold_cost:
        extent = extent + (1-x[2])
        if x[2] < 0.99:
            cold = cold + 1
    info = (cold, round(extent, 2))
    cost_info.append(info)
    print(info, m)
    
    cold_energy = func_energy[13]
    total_dE = 0
    extent = 0
    cold = 0
    for x in cold_energy:
        extent = extent + (1-x[2])
        if x[2] < 0.99:
            cold = cold + 1
    info = (cold, round(extent,2))
    energy_info.append(info)
    print(info, m)

temp_edges = func_energy[11]
diam_edges = func_energy[1]

count=0
niet_gevonden = []
temps = []
diams = []
kleuren = []
edgelist = []
for x in temp_edges:
    found = False
    edge = (x[0], x[1])
    for y in diam_edges:
        deze = (y[0], y[1])
        if deze == edge:
            found = True
            count = count + 1
            temps.append(x[2])
            diameter = y[2]
            if diameter > 1:
                w = 6
                c = pltc.to_rgba('darkslategray')
            if diameter < 1 and diameter > 0.6:
                w = 4
                c = pltc.to_rgba('teal')
            if diameter < 0.6 and diameter > 0.08:
                w = 3
                c = pltc.to_rgba('darkturquoise')
            if diameter < 0.1:
                w = 1
                c = pltc.to_rgba('paleturquoise')
            kleuren.append(c)
            diams.append(w)
            edgelist.append((x[0], x[1]))
    if found == False:
        niet_gevonden.append((x, y, count))

G = nx.Graph()
G.add_edges_from(edgelist)

dem_edge = func_energy[13]
cold_nodes = nx.Graph()
cold_nodes.add_weighted_edges_from(dem_edge[0])
# You can use any demand satisfaction here.

source = nx.Graph()
source.add_node('s')

edgelist1 = []
labels = []
for x in dem_edge:
    edgelist1.append((x[0], x[1]))
    procent = round((x[2]*100),1)
    labels.append(procent)
leebels = {edgelist1[i]: labels[i] for i in range(len(edgelist1))}

fig, ax = plt.subplots()
plt.title("The performance of the consumer optimized design at 1200 kg/s")
legend_elements =  [Patch(facecolor='dimgray', label='Polygon'), Patch(facecolor='darkgrey', label='Buildings'), Patch(facecolor='yellow', label='Geothermal well')] 
                    #Patch(facecolor='darkslategray', label='Pipeline diameter > 1m'),
                    #Patch(facecolor='teal', label='Pipeline diameter > 0.5m and < 1m'),
                    #Patch(facecolor='darkturquoise', label='Pipeline diameter > 0.1m and < 0.5m'),
                    #Patch(facecolor='paleturquoise', label='Pipeline diameter < 0.1m')]
polygon.plot(ax=ax, facecolor='dimgray')
B.plot(ax=ax, facecolor='darkgrey', edgecolor='black', linewidth=0.8)
nx.draw(G, pos=positions, edgelist=edgelist, edge_color=temps, width=diams, edge_cmap=plt.cm.coolwarm, node_size=0, alpha=1, edge_vmin=20, edge_vmax=50)
nx.draw_networkx_nodes(source, pos=positions, node_size=150, node_color='yellow', node_shape="2")
nx.draw_networkx_edge_labels(cold_nodes, pos=positions, edge_labels=leebels, font_size=2, font_color='red')
ax.legend(handles=legend_elements)
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin = 20, vmax=50, clip=True))
sm._A = []
plt.colorbar(sm).set_label('Temperature of the water in the pipelines at 1200 [kg]', size=12)
plt.xlabel('latitude')
plt.ylabel('longitude')

y_cost = []
y_energy = []
xs = []
av1 = []
av2 = []
for x in range(9):
    y_cost.append(cost_info[x][1])
    y_energy.append(energy_info[x][1])
    xs.append(massas[x])
    if cost_info[x][0] > 0:
        avg1 = 1 - cost_info[x][1]/cost_info[x][0]
        av1.append(avg1)
    if energy_info[x][0] > 0:
        avg2 = 1- energy_info[x][1]/energy_info[x][0]
        av2.append(avg2)
    










