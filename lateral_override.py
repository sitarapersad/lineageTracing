import tree_utils as l_tree_utils
import numpy as np

def weighbor(est_dists, taxon_namespace, sequence_length):
    est_pdm = l_tree_utils.array_to_pdm(est_dists, taxon_namespace)
    return l_tree_utils.weighbor(est_pdm, sequence_length), None

OVERFLOW_DISTANCE = 10
def counts_to_estimates(counts, num_sites):
    num_same = np.trace(counts)
    diffs_per_site = (num_sites - num_same) / num_sites
    if diffs_per_site >= 3/4:  # when most states are different we can just make the distance large
        dist = OVERFLOW_DISTANCE
    else:
        try:
            dist = (-3 / 4) * np.log(1 - (4 / 3) * diffs_per_site)
        except FloatingPointError:
            # no maximum likelihood estimate
            dist = OVERFLOW_DISTANCE


    variance = np.exp(8 * dist / 3) * diffs_per_site * (1 - diffs_per_site) / num_sites
    return dist, variance

def rate_of_difference(counts):
    N = counts.sum()
    if N == 0: return 0
    same = np.trace(counts)
    diff = N - same
    return diff / N

# compute estimates of distances under JC69 model, as well as variances and rates
def cm_to_estimates(cm):

    from conversion_utils import char_to_int

    dim = len(cm.default_state_alphabet)
    labels = [taxon.label for taxon in cm.taxon_namespace]
    sequences = [cm[label] for label in labels]
    n_species = len(sequences)
    n_sites = len(sequences[0])

    est_dists = np.zeros((n_species, n_species), dtype=np.float64)
    variances = np.zeros((n_species, n_species), dtype=np.float64)

    rates = np.zeros((n_species, n_species), dtype=np.float64)

    # from ctmc import JC69
    for i in range(n_species):
        for j in range(n_species):
            counts = np.zeros((dim, dim), dtype=np.float64)

            # count transitions for non-missing values
            for site in range(n_sites):
                from_idx = char_to_int(sequences[i][site].symbol) - 1
                to_idx = char_to_int(sequences[j][site].symbol) - 1
                if from_idx != -1 and to_idx != -1:  # if neither value missing
                    counts[from_idx, to_idx] += 1

            # take into account missing values
            for site in range(n_sites):
                from_idx = char_to_int(sequences[i][site].symbol) - 1
                to_idx = char_to_int(sequences[j][site].symbol) - 1
                from_missing = from_idx == -1
                to_missing = to_idx == -1
                if from_missing and not to_missing:
                    to_vec = counts[:, to_idx]
                    if sum(to_vec != 0):
                        props = to_vec/sum(to_vec)
                        # print(to_vec)
                        to_vec += props
                        # print(counts[:, to_idx])
                elif to_missing and not from_missing:
                    from_vec = counts[from_idx, :]
                    if sum(from_vec != 0):

                        props = from_vec/sum(from_vec)
                        # print(from_vec)
                        from_vec += props
                        # print(counts[from_idx,:])
            # print(counts.sum())

            dist, variance = counts_to_estimates(counts, n_sites)
            est_dists[i, j] = dist
            variances[i, j] = variance


            rates[i, j] = rate_of_difference(counts)
    return est_dists, variances, rates

def mds(rho, initial_pts, est_dists, variances, hyperboloid, taxon_namespace, sequence_length):
    from cythonised.hyperbolic_mds import MDS
    mds = MDS(rho=rho, lr=0.005, max_step_size=0.1)
    mds.set_points(initial_pts.copy())
    converged, failed, checkpoints = \
        mds.fit(est_dists,
                variances,
                stopping_distance=1e-6,
                max_rounds=100,
                checkpoint_interval=1000)
#         meta = {'converged': converged, 'failed': failed, 'rounds_done': mds.rounds_done}
#         meta['constrained'] = _satisfies_constraints(mds.get_points())
    pw_dists = hyperboloid.pairwise_distances(mds.get_points())
    est_pdm = l_tree_utils.array_to_pdm(pw_dists, taxon_namespace)
    return l_tree_utils.weighbor(est_pdm, sequence_length), None  # meta

def satisfies_constraints(hyperboloid, pts):
    for pt in pts:
        if not hyperboloid.contains(pt):
            return False
    return True

def logalike(rho, initial_pts, rates, hyperboloid, taxon_namespace, sequence_length):
    from cythonised.logalike import LogalikeOptimiser
    logalike = LogalikeOptimiser(rho=rho,
                                 lr=0.005,
                                 max_step_size=0.1)
    logalike.set_points(initial_pts.copy())
    converged, failed, checkpoints = \
        logalike.fit(rates,
                     stopping_distance=1e-6,
                     max_rounds=100,
                     checkpoint_interval=1000)
    meta = {'converged': converged, 'failed': failed, 'rounds_done': logalike.rounds_done}
    meta['constrained'] = satisfies_constraints(hyperboloid, logalike.get_points())
    pw_dists = hyperboloid.pairwise_distances(logalike.get_points())
    est_pdm = l_tree_utils.array_to_pdm(pw_dists, taxon_namespace)
    return l_tree_utils.weighbor(est_pdm, sequence_length), meta
