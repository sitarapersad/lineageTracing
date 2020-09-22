def split_tree(x, min_cells = 10):
    """
    Warning! Modifies the object x
    
    @param: x is a dataframe of cells x edit sites containing a record of the crispr edits in the final generation
    """
    # Stop splitting if all the cells are the same or there are fewer than min_cells in the subtree 
    if x.shape[0] <= min_cells:
        x['Label']+='.'
        return x
    
    mode, counts = stats.mode(x.drop('Label',axis=1).values, axis=0, nan_policy = 'omit')
    counts = counts[0]
    mode = mode[0]
    col = np.argsort(counts)[-1]
    edit = mode[col]
    
    # Split tree into cells which contain this edit and those who don't
    ix = (x[col]==edit)
    contains = x.index[ix]
    not_contains = x.index[~ix]

    x.loc[contains,'Label'] += '1'
    x.loc[not_contains,'Label'] += '0'
    
    # Remove this edit (no need to use it again)
    x.loc[ix, col] = np.nan
    
    
    if np.nansum(x.drop('Label', axis=1).values) == 0:
        print('No more mutations to split on, returning.')
        return x 
    
    # Recurse on each individual subtree and label accordingly
    if len(contains) > 0:
        x.loc[contains] = split_tree(x.loc[contains]) 
    
    if len(not_contains) > 0:
        x.loc[not_contains] = split_tree(x.loc[not_contains])
    
    return x

