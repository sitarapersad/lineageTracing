import unittest
import subprocess
import os
import re
import dendropy
from dendropy.model.discrete import Jc69, simulate_discrete_chars
from tree_utils import unrooted_tree_topology_invariant, generate_character_matrix, draw_tree, robinson_foulds
from collections import defaultdict

from tempfile import TemporaryDirectory

CHARS_FN = 'characters.nexus'
INPUT_TREE_FN = 'input_tree.nexus'
OUTPUT_FN = 'best.tree'
BATCH_FILE_FN = 'batch.paup'

CMD = os.path.expanduser('~/paup/paup')


LL_TEMPLATE = """
Begin paup;
[PAUP batch file.  Run with ./paup -n filename]
set autoclose=yes warntree=no warnreset=no storebrlens;
execute %s; [load the characters]
execute %s; [load the tree]
lset nst=1 genfreq=equal; [specify use JC69]
lscores / userbrlen; [calculate the log likelihood using our given branch lengths]
quit;
end;
"""
# NOTE: genfreq instead of basefreq to allow non-nucleotide data

def get_log_likelihood(tree, cm):
    """
    Given a dendropy CharacterMatrix `cm`, and a dendropy Tree `tree`, return
    the log likelihood.
    """
    with TemporaryDirectory() as tmp_dir:

        with open(os.path.join(tmp_dir, CHARS_FN), 'w') as f:
            f.write(cm.as_string(schema='nexus'))

        with open(os.path.join(tmp_dir, INPUT_TREE_FN), 'w') as f:
            tree = dendropy.Tree(tree)
            tree.deroot()  # starting trees must be unrooted
            f.write(tree.as_string(schema='nexus', suppress_taxa_blocks=True))

        with open(os.path.join(tmp_dir, BATCH_FILE_FN), 'w') as f:
            f.write(LL_TEMPLATE % (CHARS_FN, INPUT_TREE_FN))

        proc = subprocess.run([CMD, '-n', BATCH_FILE_FN],
                              stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE, cwd=tmp_dir, check=True)
        _output = proc.stdout.decode()

        # get the value of the LL
        matches = re.findall(r'^-ln L +(\S*)\n', _output, flags=re.MULTILINE)
        if not len(matches) == 1:
            raise ValueError(_output)
        nll = float(matches[0].strip())
        return -1 * nll


BEST_LL_TEMPLATE = """
Begin paup;
[PAUP batch file.  Run with ./paup -n filename]
set autoclose=yes warntree=no warnreset=no;
execute %s; [load the characters]
execute %s; [load the tree]
set criterion=likelihood;
lset nst=1 genfreq=equal; [specify use JC69]
lscores;
savetrees file=%s brlens replace; [write out the tree with the tuned branch lengths]
quit;
end;
"""
# NOTE: genfreq instead of basefreq to allow non-nucleotide data

def get_best_log_likelihood_for_topology(tree, cm):
    """
    Given a dendropy CharacterMatrix `cm`, and a dendropy Tree `tree`, tune the
    branch lengths of the tree, returning the log likelihood of the tuned tree
    and the tuned tree (as a 2-tuple).
    Post: robinson_foulds(tree, returned_tree) == 0
    """
    tmp_dir = ".tmpdir"
    with open(os.path.join(tmp_dir, CHARS_FN), 'w') as f:
        cmstring = cm.as_string(schema='nexus')
        f.write(cm.as_string(schema='nexus'))

    with open(os.path.join(tmp_dir, INPUT_TREE_FN), 'w') as f:
        tree = dendropy.Tree(tree)
        tree.deroot()  # starting trees must be unrooted
        f.write(tree.as_string(schema='nexus', suppress_taxa_blocks=True))

    with open(os.path.join(tmp_dir, BATCH_FILE_FN), 'w') as f:
        f.write(BEST_LL_TEMPLATE % (CHARS_FN, INPUT_TREE_FN, OUTPUT_FN))

    proc = subprocess.run([CMD, '-n', BATCH_FILE_FN],
                          stderr=subprocess.PIPE,
                          stdout=subprocess.PIPE, cwd=tmp_dir, check=True)
    _output = proc.stdout.decode()

    # get the value of the LL
    matches = re.findall(r'^-ln L +(\S*)\n', _output, flags=re.MULTILINE)
    if not len(matches) == 1:
        raise ValueError(_output)
    nll = float(matches[0].strip())

    # load the tuned tree
    with open(os.path.join(tmp_dir, OUTPUT_FN)) as f:
        tuned_tree_nexus = f.read()
    tuned_tree = dendropy.Tree.get_from_string(src=tuned_tree_nexus,
                                               schema='nexus',
                                               taxon_namespace=cm.taxon_namespace)
    return -nll, tuned_tree
