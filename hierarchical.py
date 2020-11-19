import numpy as np
from scipy.spatial.distance import pdist, squareform
from skbio.tree import TreeNode
import networkx as nx
import pandas as pd

import torch 
from tqdm import trange

import copy
import time 

from Cassiopeia.TreeSolver.Node import Node 

def get_lca(x,y):
    '''
    Returns the lowest common ancestor from two irreversibly character arrays.
    All characters inherited were mutated irreversibly from the null state (0).
    
    todo: 
    If either x or y is missing a character at a particular site, then the non-missing character 
    is assumed to be the character present in the lca. If both are missing a particular site, then
    the ancestral site will also be unknown
    
    Args:
        - x (array) of length num_sites where each entry is an integer representing a Cas-9 mutation
        - y (array) of length num_sites where each entry is an integer representing a Cas-9 mutation
    Return:
        (array) of length num_sites where each entry is an integer representing a Cas-9 mutation present in
        the most parsimonious parent of x,y
    '''
    x[x!=y]=0
    return x 

def avg_markov_distance(x,y,probs):
    '''
    Computes the distance between lca->x and lca->y as the site-wise average of log transition probabilities.
    transition probabilities are non-negative for 0->a, 1 for a->a, and 0 for any other b->a. For missing data,
    site is omitted and has no contribution to the distance.
    
    Args:
        - x (array) of length num_sites where each entry is an integer representing a Cas-9 mutation
        - y (array) of length num_sites where each entry is an integer representing a Cas-9 mutation
        - probs (torch.Categorical) is multivariate multinomial distribution (shape num_sites x 1+num_states) 
            to compute the probability of mutating from 0 to state i at a given site. 
        
    '''
    a = get_lca(x,y)
    
    nan_sites = np.isnan(x) + np.isnan(y)
    
    # Wherever lca differs from array, we compute the log-probability of that character being inherited 
    # where x!=0 and x==lca, transition prob is 1
    # where x!=0 and x!=lca, transition prob is prob of mutation going to 0->a
    # where x==0 and x==lca, transition prob is prob of mutation going to 0->0 
    
    # Remove nan values for now to allow us to compute log probs
    x[nan_sites] = 0
    y[nan_sites] = 0
    
    # Compute log-prob of mutations in x,y
    x_prob = -probs.log_prob(torch.DoubleTensor(x)).numpy()
    y_prob = -probs.log_prob(torch.DoubleTensor(y)).numpy()
    
    # First get all the positions where x,y is the same as the lca or is missing  
    # These have no contribution to the distance
    prob_one_x = x==a + nan_sites
    prob_one_y = y==a + nan_sites 
    
    # Sum up the probabilities over contributing sites and average over all non-nan sites 
    dist = (x_prob[~prob_one_x].sum() + y_prob[~prob_one_y].sum())/(len(x)-nan_sites.sum())
    
    return dist  

def avg_markov_distance_matrix(matrix, row, probs):
    '''
    Computes the distance between lca-> and lca->y for every cell y in the matrix as the site-wise average of 
    log transition probabilities.
    transition probabilities are non-negative for 0->a, 1 for a->a, and 0 for any other b->a. For missing data,
    site is omitted and has no contribution to the distance.
    
    Args:
        - matrix (array) of shape n_cells x num_sites where each entry is an integer representing a Cas-9 mutation 
        or NaN representing missing data 
        - row (array) of length num_sites where each entry is an integer representing a Cas-9 mutation 
        or NaN representing missing data 
        - probs (torch.Categorical) is multivariate multinomial distribution (shape num_sites x 1+num_states) 
            to compute the probability of mutating from 0 to state i at a given site. 
        
    '''
    lcas = matrix.copy()
    lcas[~(matrix==row)] = 0

    row_nan = np.isnan(row)
    matrix_nan = np.isnan(matrix)
    matrix_nan[:, row_nan] = True
    
    row[row_nan] = 0
    matrix[matrix_nan] = 0
    
    row_lp = -(probs.log_prob(torch.DoubleTensor(row-lcas))).numpy()
    row_lp[lcas==row] = 0

    matrix_lp = -(probs.log_prob(torch.DoubleTensor(matrix))).numpy()
    matrix_lp[lcas==matrix] = 0
    
    # We have to ignore any sites where the row or matrix contains nans  
    matrix_lp[matrix_nan] = 0
    row_lp[matrix_nan] = 0
    
    # Divide by the number of non-missing sites to get the average
    return (matrix_lp.sum(1)+row_lp.sum(1))/(~matrix_nan).sum(1)

def format_char_vec(a):
    nan_a = np.isnan(a)
    a = a.astype(np.int).astype(str)
    a[nan_a] = '-'
    return list(a)

def cluster_nx(character_matrix, probs=None, names=None):
    '''
    Performs lca-based hierarchical clustering on the character matrix.
    
    Args:
        - character_matrix (array/dataframe) 
        - probs (torch.Categorical) is multivariate multinomial distribution (shape num_sites x 1+num_states) 
            to compute the probability of mutating from 0 to state i at a given site. 
        
    '''
    
    if isinstance(character_matrix, pd.DataFrame):
        names = character_matrix.index.values
        character_matrix = character_matrix.values
                
    num_sites = character_matrix.shape[1]
    
    Ds = []
    joins = []
    
    if names is None:
        names = np.arange(character_matrix.shape[0])
       
    # If no prob features are specified, set a uniform distribution over all.
    if probs is None:
        from torch.distributions.categorical import Categorical
        raise NotImplementedError
        
        probs = Categorical(probs=torch.DoubleTensor(np.ones((num_sites,41))))
    
    print('Computing initial distance matrix')            
    # Compute the distance matrix
    D = squareform(pdist(character_matrix, lambda u,v: avg_markov_distance(u,v,probs)))
    
    tree = nx.DiGraph()
    N_nodes = len(D)
    print('Starting with {0} nodes'.format(N_nodes))
    new_name = N_nodes
    
    tree_nodes = {} 
    for i, name in enumerate(names):
        tree_nodes[name] = Node(name=str(name), character_vec = format_char_vec(character_matrix[i]))
        
    progress = trange(N_nodes-2, desc='Performing agglomerative clustering', leave=True)
    for i in progress:
        s = time.time()
        
        Ds.append(D)
        
        # Convert Q martix to lower triangular form without the diagonal to avoid merging the same site
        D[np.tril_indices(D.shape[0], 0)]  = np.inf
        
        # Now find the argmin (i,j) of Q. These are the sites the be merged
        min_i, min_j = np.unravel_index(np.argmin(D, axis=None), D.shape)
        s = time.time() 
        child1, child2 = names[min_i], names[min_j]
        joins.append((child1, child2))

        # Create a new edge between the merged parent and the children
        new_name += 1
        lca = get_lca(character_matrix[min_i],character_matrix[min_j])
        
        # Create a new node to represent the ancestor of the children
        parent = Node(name=str(new_name), character_vec = format_char_vec(lca))
        tree_nodes[new_name] = parent
        
        child1 = tree_nodes[child1]
        child2 = tree_nodes[child2]
        
        tree.add_edges_from([(parent, child1), (parent, child2)])
               
        names = np.delete(names, [min_i,min_j], axis=0)
        names = np.hstack([names, new_name])
                
        # Now we merge i,j. We need to replace i,j in the feature matrix with lca(i,j).
        character_matrix  = np.delete(character_matrix, [min_i,min_j], axis=0)
        character_matrix  = np.vstack([character_matrix, lca])
        
        # We also need to replace the distance of each site k to i or j with the distance to lca(i,j)

        D = np.delete(np.delete(D, [min_i,min_j], axis=0), [min_i,min_j], axis=1)

        new_D = np.zeros((character_matrix.shape[0], character_matrix.shape[0]))
        new_D[:-1, :-1] = D

        new_D_row = avg_markov_distance_matrix(character_matrix, lca, probs)

        new_D[-1, :] = new_D_row
        new_D[:, -1] = new_D_row
        D = new_D
            
    new_name += 1
    
    # Merge the last two remaining sites to complete the tree
    child1, child2 = names[0], names[1]
    
    child1, child2 = tree_nodes[names[0]], tree_nodes[names[1]]
    parent = Node(name=new_name, character_vec = format_char_vec(lca))
    
    tree.add_edges_from([(parent, child1), (parent, child2)])
    
    return tree, {'Ds':Ds, 'joins':joins}


def cluster(feature_matrix, prob_features, names=None, result_constructor=None):
    fm = copy.deepcopy(feature_matrix)
    fm = fm.values 
    
    Ds = []
    joins = []
    
    if names is None:
        names = np.arange(fm.shape[0])
    
    log_prob_features = np.log(prob_features)
    log_prob_features[-log_prob_features == np.inf] = -10000 #hacky
    
    
    # Compute the distance matrix
    D = squareform(pdist(fm, lambda u,v: (- (u + v - 2*u*v) * log_prob_features).sum()))
    
    tree_nodes = {}
    
    for name in names:
        tree_nodes[name] = TreeNode(name=str(name))
        
    print('Starting with {0} nodes'.format(len(D)))
    new_name = len(D)
    
    new_lcas = {}
    while len(D) > 2:
              
        s = time.time()
        
        Ds.append(D)
        
        # Convert Q martix to lower triangular form without the diagonal to avoid merging the same site
        D[np.tril_indices(D.shape[0], 0)]  = np.inf
        
        # Now find the argmin (i,j) of Q. These are the sites the be merged
        min_i, min_j = np.unravel_index(np.argmin(D, axis=None), D.shape)
        s = time.time() 
        
        joins.append((names[min_i], names[min_j]))

        
        # Create a new TreeNode from the merged children
        
        new_name += 1
        
        
        child_i = tree_nodes[names[min_i]]
        child_j = tree_nodes[names[min_j]]
        new_node = TreeNode(name=str(new_name), length=None, parent=None, children=[child_i, child_j])

            
        child_i.parent = new_node
        child_j.parent = new_node
        
        tree_nodes[new_name] = new_node
        
        
        names = np.delete(names, [min_i,min_j], axis=0)
        names = np.hstack([names, new_name])
                
        # Now we merge i,j. We need to replace i,j in the feature matrix with lca(i,j).
#         lca = lcas[min_i,min_j]
        lca = fm[min_i]*fm[min_j]
        fm  = np.delete(fm, [min_i,min_j], axis=0)
        fm  = np.vstack([fm, lca])
        
        new_lcas[new_name] = lca

        # We also need to replace the distance of each site k to i or j with the distance to lca(i,j)

        D = np.delete(np.delete(D, [min_i,min_j], axis=0), [min_i,min_j], axis=1)

        new_D = np.zeros((fm.shape[0], fm.shape[0]))
        new_D[:-1, :-1] = D

        new_D_row = - ((fm + fm[-1] - 2* fm * fm[-1])*log_prob_features).sum(1)


        new_D[-1, :] = new_D_row
        new_D[:, -1] = new_D_row
        D = new_D
        
        
    new_name += 1
    
    # Merge the last two remaining sites to complete the tree
    child1, child2 = tree_nodes[names[0]], tree_nodes[names[1]]
    root = TreeNode(name = str(new_name), children=[child1, child2])
    child1.parent = root
    child2.parent = root
    
    return root, {'Ds':Ds, 'joins':joins, 'lcas':new_lcas}

