from skbio import TreeNode
import numpy as np
import pandas as pd 

from utilities import char_vec_string, binarize_character_matrix
import utilities


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
        if not isinstance(char_vec, pd.DataFrame):
            char_vec = pd.DataFrame(char_vec)
        self.char_matrix = char_vec 
    
    def get_character_matrix(self):
        if not isinstance(self.char_matrix, pd.DataFrame):
            self.char_matrix = pd.DataFrame(self.char_matrix)
        return self.char_matrix
    
    def get_character_array(self):
        return self.get_character_matrix().values.reshape(-1)
    
    def add_feature_matrix(self, feature_matrix):
        """
        A dataframe containing a binarized version of the character states for cells in this group. 
        The length of this depends on the number of unique character/site combinations
        """
        self.feature_matrix = feature_matrix
    
    def get_features_from_characters(self):
        self.feature_matrix = binarize_character_matrix(self.get_character_matrix())
        
    def get_feature_matrix(self):
        return self.feature_matrix 
    
    def add_defining_muts(self, muts):
        self.muts = muts
        
    def get_all_defining_mutations(self):
        """
        Combine all the mutations which defined this lineage
        """
        return None 
    
    def condense_duplicate_leaves(self):
        """
        If all the leaves in a subtree are the same, replace this subtree with a single node.
        """
        if self.is_tip():
            self.name = char_vec_string(self.get_character_array())
            return

        representative = []
        leaves = [x.get_character_array() for x in self.tips()]
        leaves = np.array(leaves)
        residuals = leaves - np.nanmean(leaves, axis=0)
        residuals = np.abs(residuals)
        if np.nansum(residuals) == 0:
            # All tips are the same
            # Pick any leaf as the representative for this node.
            for x in self.tips():
                representative = x 
                break 
            self.children = []
            self.name = char_vec_string(representative.get_character_array())
            self.add_character_matrix(representative.get_character_matrix())
            self.add_feature_matrix(representative.get_feature_matrix())
            return 
        # Otherwise, condense all the children of this tree
        for child in self.children:
            child.condense_duplicate_leaves()
            
        return

    