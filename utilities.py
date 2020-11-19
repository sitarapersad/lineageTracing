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