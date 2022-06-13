
import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import cm
import matplotlib.colors #as mcolors
# from colorspacious import cspace_converter
# import rasterio
from ogb.utils import smiles2graph
# from pysmiles import read_smiles
import pysmiles
from myEdgeDrawer import my_draw_networkx_edge_labels


# cmap = plt.cm.Blues(np.linspace(0,1,1000))

figure(figsize=(10, 8), dpi=360)

def importingfunction(saving_path, graph_idx, it_index, smiles, edge_mask):
    #'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5'
    # smiles = 'c1c[nH]cc1N'
    hello = smiles2graph(smiles)
    mol = pysmiles.read_smiles(smiles, explicit_hydrogen=True)
    # print(hello)


    src = hello['edge_index'][0]
    dst = hello['edge_index'][1]
    graph_prova = dgl.graph((src, dst))
    # print('We have %d nodes.' % graph_prova.number_of_nodes())
    # print('We have %d edges.' % graph_prova.number_of_edges())

    elements = nx.get_node_attributes(mol, name = "element")

    # ---------------------------------------------------------------------------------
    unidir_edges = {}
    for x1, x2, m in zip(src, dst, edge_mask):
        # print((x1, x2, m))
        x1, x2, x3 = int(x1), int(x2), float(m)
        if (x2, x1) in unidir_edges.keys():
            unidir_edges[(x2, x1)] = (unidir_edges[(x2, x1)] + x3)/2
        else: unidir_edges[(x1, x2)] = x3

    # MEAN-WEIGHTED UNI-DIRECTIONAL EDGES
    new_edge_weights = unidir_edges.values()  

    src = np.array([cp[0] for cp in unidir_edges.keys()])  
    dst = np.array([cp[1] for cp in unidir_edges.keys()])  
    final_graph = dgl.graph((src, dst))
    print('We have %d nodes.' % final_graph.number_of_nodes())
    print('We have %d edges.' % final_graph.number_of_edges())

    nx_G = graph_prova.to_networkx().to_undirected()
    # pos=nx.spring_layout(mol)
        
    edge_labels = {k:round(float(v), 2) for k, v in zip(zip(src, dst), new_edge_weights)} 
    edge_labels4color = {k:int(round(float(v), 1)*10) for k, v in zip(zip(src, dst), new_edge_weights)} 
    # complete_edges = mol.edges().copy()
    # print(edge_labels)
    print(edge_labels4color)

    plt.title(smiles)
    
    
    labels = elements
    # print(labels)
    
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

    # print(node_color)
    # print(edge_labels.values())

    pos = nx.kamada_kawai_layout(mol)
    colors = list(edge_labels4color.values())  #.sort()

    for k, v in mol.edges().items():
        if k not in edge_labels4color.keys():
            edge_labels4color[k] = 0

    
    # print(mol.edges())
    # print(edge_labels4color)
    # colors = list(edge_labels4color.values())
    colors = edge_labels.values()
    # colors = range(len(list(edge_labels.values())))

    # print(colors)
    # print(edge_labels4color)
    # print(mol.edges())
    options = {
        # "node_color": node_color,
        "edge_color": colors,
        "width": 5,
        "edge_cmap": plt.cm.Blues,
        # "with_labels": True,
        # "font_color": 'w',
        "node_size": 400,
        "edge_vmin": 0,
        "edge_vmax": 0.99
        # "labels": elements
    }

    cmap = cm.Blues
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    # x = np.array([0.53, 0.22, 0.85, 0.55, 0.19])
    x = list(edge_labels.values())   #np.array(edge_labels.values())
    print(x)

    m = cm.ScalarMappable(norm =norm, cmap=cmap)  #norm=norm,
    # print(m.to_rgba(x))
    colors = m.to_rgba(x)
    # print(colors)
    nuova = []
    for color in colors:
        nuova.append(tuple(color))

    print(nuova)


    # nx.draw(mol, pos, **options)
    nx.draw(mol, with_labels=True, labels=elements, pos=nx.kamada_kawai_layout(mol), 
            node_color=node_color, font_color='w', node_size=400)
        
    nx.draw_networkx_edges(nx_G, pos=nx.kamada_kawai_layout(mol), edgelist=edge_labels,  #nx_G
                            edge_color=nuova, width=5.0)   #edge_color=edge_labels.values()
                            #edge_vmin=0, edge_vmax=0.99, edge_cmap=plt.cm.Blues) 
    # nx.draw_networkx_edges(mol, pos, **options)

    my_draw_networkx_edge_labels(mol, pos=nx.kamada_kawai_layout(mol), 
                                        edge_labels=edge_labels, rotate=True)  


    plt.savefig(f'{saving_path}/{graph_idx}_{smiles}_{it_index}.png')
    # plt.show()
    plt.clf()

    return



def weight2color(edge_weights):
    cmap = cm.hot
    x = np.array([0.53, 0.22, 0.85, 0.55, 0.19])

    m = cm.ScalarMappable(cmap=cmap)  #norm=norm,
    print(m.to_rgba(x))
    return m.to_rgba(x)


# nx.draw_networkx_nodes(nx_G, pos, node_color='b')

# from rdkit import Chem
# from rdkit.Chem.Draw import IPythonConsole
# from rdkit.Chem import Draw
# IPythonConsole.ipython_useSVG=True


# def mol_with_atom_index(mol):
#     for atom in mol.GetAtoms():
#         atom.SetAtomMapNum(atom.GetIdx())
#     return mol

# mol = Chem.MolFromSmiles(smiles)
# mol
# mol_with_atom_index(mol)