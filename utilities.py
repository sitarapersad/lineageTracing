from skbio import TreeNode
import networkx as nx
import numpy as np
import pandas as pd 


def convert_nx_to_tree(G):
    """
    Convert a binary tree, G, represented as a networkx DiGraph to
    an skbio TreeNode
    """
    def tree_nodify(G, node):
        """
        Convert network x 
        """
        # Recursive solution: 
        # If node has children, add the TreeNoded version of (each) child
        # If node has no children, convert to a TreeNode and 

        children = [x[1] for x in G.out_edges(node)] 

        if len(children) == 0:
            return TreeNode(name=str(node.name), 
                            length=None, 
                            parent=None, 
                            children=[])
        else:
            return TreeNode(name=str(node.name), 
                            length=None, 
                            parent=None, 
                            children=[tree_nodify(G, child) for child in children])
    
    root = [x for x in G.nodes() if G.in_degree(x)==0][0]
    return tree_nodify(G, root)

def binarize_character_matrix(f):
    new_f = (f+f.columns.values*1000).astype(int)
    n_values = new_f.max().max() + 1
    xxx = np.zeros((new_f.shape[0], n_values))
    for col in new_f.columns:
        values = new_f[col].astype(int) 
        xxx += np.eye(n_values)[values]

    labels = np.where(xxx.sum(0)>0)[0]

    xx = pd.DataFrame(xxx[:, labels]).astype(int)
    xx.columns = labels
    xx.index = f.index 
    
    xx = xx.loc[:, xx.columns%1000 != 0]
    return xx 

def character_matrix_to_labels(f):
    # Label cells by their character strings 
    str_labels = f.astype(str).values.tolist()
    str_labels = ['|'.join(x) for x in str_labels]

    return str_labels 