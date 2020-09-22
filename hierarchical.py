import numpy as np
from scipy.spatial.distance import pdist, squareform
from skbio.tree import TreeNode
import copy
import time 

def cluster(feature_matrix, prob_features, names=None, result_constructor=None):
    fm = copy.deepcopy(feature_matrix)
    fm = fm.values 
    
    Ds = []
    joins = []
    
    if names is None:
        names = np.arange(fm.shape[0])
       
    # Determine the lcas for each pair of sites
#     lcas = np.zeros((fm.shape[0], fm.shape[0], fm.shape[1]))
#     for i in range(fm.shape[0]):
#         for j in range(i+1, fm.shape[0]):
#             lcas[i,j] = fm[i]*fm[j]
#             lcas[j,i] = lcas[i,j]
    
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

