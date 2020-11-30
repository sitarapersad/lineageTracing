import numpy as np
import networkx as nx

from cassiopeia.TreeSolver.Node import Node 

def format_char_vec(a):
    nan_a = np.isnan(a)
    a = a.astype(np.int).astype(str)
    a[nan_a] = '-'
    return list(a)

from skbio import TreeNode
import networkx as nx

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
    
def convert_tree_to_nx(tree):
    """
    Convert a binary tree, G, represented as an skbio TreeNode to
    a networkx DiGraph.
    """
    
    
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
                                 character_vec = child_node.get_character_matrix().values.reshape(-1).tolist(), 
                                 is_target=False)
                else:
                    child = Node(child_node.name, is_target=True)

                network.add_edges_from([(node, child)])

                successor_nodes.append(child_node)
                successor_nx.append(child)

        # Now the successor level is the current level for the next iteration
        level_nodes = successor_nodes
        level_nx = successor_nx

        if len(level_nodes) == 0:
            stop = True 
    return network 

    
    