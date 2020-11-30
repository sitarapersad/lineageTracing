from skbio import TreeNode
class CellGroupNode(TreeNode):
    """
    An abstract class for all nodes in a tree. 

    Attributes:
    - feature_matrix: dataframe consisting of the feature matrix of all cells contained in this node 
    
    Methods:
    - get_character_string: utility function for getting character string
    - get_name: utility for getting the name of the node
    - get_character_vec: utility for getting the character vector
    - get_edit_distance: calculate the edit distance between two nodes
    - get_modified_hamming_dist: calculate hamming distance between nodes, taking into account mutagenesis process
    - get_mut_length: get a 0 - 1 value indicating the proportion of non-missing characters are different between two nodes

    """
    # Inherit init method from skbio
    def add_character_matrix(self, char_vec):
        """
        A dataframe of character states, of length C for cells in this group. All Nodes in a tree should have the same number of characters.
        """
        self.char_matrix = char_vec 
    
    def get_character_matrix(self):
        return self.char_matrix
        
    def add_feature_matrix(self, feature_matrix):
        """
        A dataframe containing a binarized version of the character states for cells in this group. 
        The length of this depends on the number of unique character/site combinations
        """
        self.feature_matrix = feature_matrix
    
    def get_feature_matrix(self):
        return self.feature_matrix 
    
    def add_defining_muts(self, muts):
        self.muts = muts
        
    def get_all_defining_mutations(self):
        """
        Combine all the mutations which defined this lineage
        """
        return None