from skbio import TreeNode
import networkx as nx
import numpy as np
import pandas as pd 


def format_char_vec(a):
    if isinstance(a, pd.DataFrame):
        a = df_to_array(a)
    missing_vals = a<0
    a = a.astype(str)
    a[missing_vals] = '-'
    return a.tolist()

def char_vec_string(a):
    return '|'.join([str(x) for x in format_char_vec(a)])

def convert_nx_to_tree(G):
    """
    Convert a binary tree, G, represented as a networkx DiGraph to
    an skbio TreeNode. Names of leaf nodes are given by 
    """
    
    MAX_RAND = 50000 #HACKY
    def tree_nodify(G, node, added_tips = {}):
        """
        Convert network x 
        """
        # Recursive solution: 
        # If node has children, add the TreeNoded version of (each) child
        # If node has no children, convert to a TreeNode and 

        children = [x[1] for x in G.out_edges(node)] 
        
        
        if len(children) == 0:
            name = node.char_string
            
            if name in added_tips:
                added_tips[name] += 1
                return None, added_tips
            else:
                added_tips[name] = 1
                return TreeNode(name=name, 
                            length=None, 
                            parent=None, 
                            children=[]), added_tips
        
        else:
            name=str(node.name)                
            if name == 'state-node':
                name = np.random.randint(MAX_RAND)

            # Make sure we do not keep repeated tips, condense 
            clean_children = []
            for child in children:
                node, added_tips = tree_nodify(G, child, added_tips)
                if node is not None:
                    clean_children.append(node)
            # If this node has no clean children, remove it as well
            if len(clean_children) == 0:
                return None, added_tips
            
            return TreeNode(name=str(name), 
                            parent=None, 
                            children=clean_children), added_tips
    
    root = [x for x in G.nodes() if G.in_degree(x)==0][0]
    return tree_nodify(G, root)
    
def convert_tree_to_nx(tree):
    """
    Convert a binary tree, G, represented as an skbio TreeNode to
    a networkx DiGraph.
    """
    
    from cassiopeia.TreeSolver.Node import Node 

    network = nx.DiGraph()
    level_nodes = [tree]
    level_nx = [Node(x.name, is_target=False) for x in level_nodes]
    level = 0

    stop = False
    while not stop:
        successor_nodes = []
        successor_nx = []

        for i, tree_node in enumerate(level_nodes):
            node = level_nx[i]
            # Lookup Cassiopeia node from dictionary that was created when node was probed as a child

            for child_node in tree_node.children:
                # Create CassiopeiaNode for each child and add to the DiGraph
                # If the child is a leaf, then we need to add a character vector

                if child_node.is_tip():
                    child = Node(child_node.name, 
                                 character_vec = child_node.get_character_matrix().replace(-1,'-').values.reshape(-1).tolist(), 
                                 is_target=True)
                else:
                    child = Node(child_node.name, is_target=False)

                network.add_edges_from([(node, child)])

                successor_nodes.append(child_node)
                successor_nx.append(child)

        # Now the successor level is the current level for the next iteration
        level_nodes = successor_nodes
        level_nx = successor_nx

        if len(level_nodes) == 0:
            stop = True 
    return network 

def binarize_character_matrix(f):
    
    if isinstance(f, pd.Series):
        f = f.to_frame()
    
    # Replace nan values with 0s in creating missing data and converting to binary form.
    # In theory this should be missing data in the binary matrix as well?? Not just zero
    f = f.clip(lower=0)
        
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

def get_ordering(tree, n1, n2, n3):
    """
    Return the node that is furthest phylogenetically from the others.
    e.g. If (a,b),c is the true ordering, return c
    
    The lca of (a,c) and (b,c) will the same.
    """
    if tree.lca([n1, n2]) == tree.lca([n1, n3]):
        return n1
    if tree.lca([n1, n2]) == tree.lca([n2, n3]):
        return n2
    if tree.lca([n1, n3]) == tree.lca([n2, n3]):
        return n3

def triplets_correct(true_tree, rtree, n_samples=1000):   
    try:
        n_correct = 0
        failed = 0
        duplicated = 0

        choose_from = [x.name for x in true_tree.tips()]

        for i in range(n_samples):
            # Sample a triplet of cells 
            ix = np.random.choice(choose_from,3,replace=False)
            n1, n2, n3 = ix

            repeat = False
            # ! indicates a node that exists across different branches in the simulated tree
            if n1[-1] == '!':
                n1 = n1.split('-')[0]
                repeat = True

            if n2[-1] == '!':
                n2 = n2.split('-')[0]
                repeat = True

            if n3[-1] == '!':
                n3 = n3.split('-')[0]
                repeat = True
            if repeat:
                duplicated += 1

            true = get_ordering(true_tree, n1,n2,n3)

            try:
                furthest = get_ordering(rtree, n1, n2,n3)
            except Exception as e:
                failed += 1
                continue

            if furthest == true:
                n_correct += 1
        if failed > 0:
            print(f'Failed to find {failed} triplets out of {n_samples} due to nodes missing from reconstructed tree.')
        if duplicated > 0:
            print(f'{duplicated} triplets contained nodes which exist as duplicates across different branches in original tree.')

        n_correct /= (n_samples-failed)

        return n_correct
    except Exception as e:
        print(f'triplets_correct failed with error {e}')

