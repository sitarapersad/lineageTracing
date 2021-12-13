# convert networkx DiGraph G to ete3 tree
def networkx_to_ete(digraph, root_name="ROOT", copy_attr=True):
    import ete3
    import itertools

    subtrees = {node:ete3.Tree(name=node) for node in digraph.nodes()}
    [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), digraph.edges())]
    tree = subtrees[root_name]
    
    if copy_attr:
        # copy attributes of the digraph to the ete3 tree
        for name, node in digraph.nodes.items():
            copy = tree & name  # get first node matching the name
            for feature, data in node.items():
                copy.add_feature(feature, data)

    return tree

# get pandas DataFrame containing character matrix of ete3 tree
def get_ete_cm(tree):
    import pandas as pd
    leaves = tree.get_leaves()
    names = [str(leaf.name) for leaf in leaves]
    cassette_states = [leaf.cassette_state for leaf in leaves]
    df = pd.DataFrame(data=cassette_states)
    df.insert(0, 'name', names)
    df = df.set_index('name')
    return df

# annotate depths in an ete3 tree
def annotate_ete_depths(tree):
    for node in tree.traverse("preorder"):
        if node.is_root():
            node.add_feature('depth', 0)
        else:
            node.add_feature('depth', node.up.depth+1)

# convert character matrix DataFrame to one accepted by Cassiopeia
# (-1 instead of "-" for missing values)
def convert_to_cass_cm(char_matrix):
    char_matrix = char_matrix.replace('-', '-1')
    for column in char_matrix.columns:
        char_matrix[column] = char_matrix[column].astype(int)
    return char_matrix

# compute dict of priors used by Cassiopeia
def compute_cass_priors(char_matrix):
    priors = {}
    for i in range(len(char_matrix.columns)):
        count = char_matrix[i].value_counts()
        if -1 in count:
            del(count[-1])
        freq = count/sum(count)
        priors[i] = freq.to_dict()
    return priors

# generate and add the character_meta attribute to a Cassiopeia tree
def add_char_meta(tree):
    import numpy as np
    import pandas as pd

    char_matrix = tree.character_matrix

    n_leaves = char_matrix.shape[0]
    missing_proportion = (char_matrix == -1).sum(axis=0) / n_leaves
    uncut_proportion = (char_matrix == 0).sum(axis=0) / n_leaves
    n_unique_states = char_matrix.apply(lambda x: len(np.unique(x[(x != 0) & (x != -1)])), axis=0)
    character_meta = pd.DataFrame([missing_proportion, uncut_proportion, n_unique_states],
                                  index=['missing_prop', 'uncut_prop', 'n_unique_states']).T
    tree.character_meta = character_meta


# create cassiopeia tree using character matrix
def create_cass_tree(char_matrix, missing_state_indicator=-1, add_character_meta=True):
    priors = compute_cass_priors(char_matrix)
    import cassiopeia as cas
    tree = cas.data.CassiopeiaTree(character_matrix=char_matrix, priors=priors,
            missing_state_indicator=missing_state_indicator)

    if(add_character_meta):
        add_char_meta(tree)
 
    return tree

# convert ete3 tree to a dendropy tree
def ete_to_dendropy(tree):
    from dendropy import Tree as DTree
    return DTree.get(data=tree.write(format=1), schema='newick')

# get a dendropy StandardCharacterMatrix from an ete3 tree.
# done by mapping integer states to characters
def ete_to_dendropy_cm(tree):
    import dendropy
    leaves = tree.get_leaves()
    names = [str(leaf.name) for leaf in leaves]
    states = [leaf.cassette_state for leaf in leaves]

    new_states = []
    mapping = {"-":chr(0)}
    counter = 1
    for state in states:
        new_state=[]
        for i in range(len(state)):
            if state[i] in mapping:
                newchar = mapping[state[i]]
            else:
                newchar = chr(counter)
                mapping[state[i]] = newchar
                counter += 1
            new_state.append(newchar)
        new_states.append(new_state)

    max_state = ord(max([max(state) for state in new_states]))
    # print(max_state)
    # print(new_states)
    cassette_states = [''.join(state) for state in new_states]
    # print(cassette_states)
    char_dict = dict(zip(names, cassette_states))
    alphabet = dendropy.datamodel.charstatemodel.StateAlphabet(''.join(map(chr,range(0,max_state+1))))
    return dendropy.StandardCharacterMatrix.from_dict(char_dict,default_state_alphabet=alphabet)
