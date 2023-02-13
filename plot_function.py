import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import cm
import matplotlib.colors 
from ogb.utils import smiles2graph
import pysmiles
from myEdgeDrawer import my_draw_networkx_edge_labels


figure(figsize=(10, 8), dpi=360)

def my_plot_function(saving_path, graph_idx, it_index, smiles, edge_mask):
    graph_from_smile = smiles2graph(smiles)
    mol = pysmiles.read_smiles(smiles, explicit_hydrogen=True)
    elements = nx.get_node_attributes(mol, name = "element")

    src = graph_from_smile['edge_index'][0]
    dst = graph_from_smile['edge_index'][1]
    graph_from_smile = dgl.graph((src, dst))

    # Mean-weighted unidirectional edges 
    unidir_edges = {}
    for x1, x2, m in zip(src, dst, edge_mask):
        x1, x2, x3 = int(x1), int(x2), float(m)
        if (x2, x1) in unidir_edges.keys():
            unidir_edges[(x2, x1)] = (unidir_edges[(x2, x1)] + x3)/2
        else: unidir_edges[(x1, x2)] = x3
    new_edge_weights = unidir_edges.values()  

    src = np.array([cp[0] for cp in unidir_edges.keys()])  
    dst = np.array([cp[1] for cp in unidir_edges.keys()])  
    final_graph = dgl.graph((src, dst))
    print('We have %d nodes.' % final_graph.number_of_nodes())
    print('We have %d edges.' % final_graph.number_of_edges())

    # Mean-weighted undirected edges for plotting 
    nx_G = final_graph.to_networkx().to_undirected()
        
    edge_labels = {k:round(float(v), 2) for k, v in zip(zip(src, dst), new_edge_weights)} 
    edge_labels4color = {k:int(round(float(v), 1)*10) for k, v in zip(zip(src, dst), new_edge_weights)} 
    labels = elements
    
    # CPK COLORING
    node_color = []
    for element in labels.values():
        if element == 'C':
            node_color.append('black')
        elif element == 'H':
            node_color.append('gray')
        elif element == 'O':
            node_color.append('red')
        elif element == 'N':
            node_color.append('blue')
        elif element == 'F':
            node_color.append('green')

    plt.title(smiles)
    pos = nx.kamada_kawai_layout(mol)
    colors = list(edge_labels4color.values())  

    for k, v in mol.edges().items():
        if k not in edge_labels4color.keys():
            edge_labels4color[k] = 0


    colors = edge_labels.values()
 
    cmap = cm.Blues
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    x = list(edge_labels.values())   
    print('Edge weights: ', x)

    m = cm.ScalarMappable(norm=norm, cmap=cmap) 
    colors = m.to_rgba(x)
    edge_color = []
    for color in colors:
        edge_color.append(tuple(color))

    nx.draw(mol, with_labels=True, labels=elements, pos=pos, node_color=node_color, font_color='w', node_size=400)    
    nx.draw_networkx_edges(nx_G, pos=pos, edgelist=edge_labels, edge_color=edge_color, width=5.0)   
    my_draw_networkx_edge_labels(mol, pos=pos, edge_labels=edge_labels, rotate=True)  

    plt.savefig(f'{saving_path}/{graph_idx}_{smiles}_{it_index}.png')
    plt.clf()

    return