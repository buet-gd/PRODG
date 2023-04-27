
import numpy as np
import pickle as pkl
import networkx as nx
import os
import math

import numpy as np
import scipy.sparse as sp

import click as ck



def load_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'Z' : The community labels in sparse matrix format
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_matrix.data'], loader['adj_matrix.indices'],
                           loader['adj_matrix.indptr']), shape=loader['adj_matrix.shape'])

        if 'attr_matrix.data' in loader.keys():
            X = sp.csr_matrix((loader['attr_matrix.data'], loader['attr_matrix.indices'],
                               loader['attr_matrix.indptr']), shape=loader['attr_matrix.shape'])
        else:
            X = None

        Z = sp.csr_matrix((loader['labels.data'], loader['labels.indices'],
                           loader['labels.indptr']), shape=loader['labels.shape'])

        # Remove self-loops
        A = A.tolil()
        A.setdiag(0)
        A = A.tocsr()

        # Convert label matrix to numpy
        if sp.issparse(Z):
            Z = Z.toarray().astype(np.float32)

        graph = {
            'A': A,
            'X': X,
            'Z': Z
        }

        node_names = loader.get('node_names')
        if node_names is not None:
            node_names = node_names.tolist()
            graph['node_names'] = node_names

        attr_names = loader.get('attr_names')
        if attr_names is not None:
            attr_names = attr_names.tolist()
            graph['attr_names'] = attr_names

        class_names = loader.get('class_names')
        if class_names is not None:
            class_names = class_names.tolist()
            graph['class_names'] = class_names

        return graph



 
def convert_lfr_to_pickle(datadir = '../data/synw/', filename = 'synwcom0.02'):
    G = nx.Graph()
    
    with open(datadir+filename+".nse", "r") as f:

        for line in f:  
            if(line[0]=='#'):
                continue
            line = line.rsplit()
            u = int(line[0]) -1 # zero based index
            v = int(line[1]) -1 # zero based index
            e = float(line[2])        
            G.add_edge(u, v, weight = e)
           

    N = G.number_of_nodes()
    # communities = set()
    maxC = 0
    node_com = [[] for _ in range(N+1)]
    
    with open(datadir+filename+".nmc", "r") as f:
        for line in f:
            line = line.rsplit()
            node = int(line[0]) -1

            for c in line[1:]:
                node_com[node].append(int(c)-1) # zero based index
                maxC = max(maxC, int(c))

    F = [[0]*maxC for _ in range(N)]  # zero based index

    for i in range(N):
        for c in node_com[i]:
            F[i][c] = 1

    
    data = {'G': G, 'F': F}

   

    print("writing pkl")
    with open(datadir+filename+".pkl", 'wb') as outfile:
        pkl.dump(data, outfile, pkl.HIGHEST_PROTOCOL)

    # print("loading pkl")
def load_dataset_pkl(datapath):
    # import pickle

    with open(datapath+".pkl", 'rb') as infile:
        result = pkl.load(infile)

        return result

        

    
@ck.command()
@ck.option(
    '--datadir', '-dd', default='../data/synw/',
    help='data directory')
@ck.option(
    '--filename', '-fn', default='synwcom0.02',
    help='File name')

def main(datadir, filename):
    convert_lfr_to_pickle(datadir = datadir, filename = filename)


if __name__ == '__main__':
    main()