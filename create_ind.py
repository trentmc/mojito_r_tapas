#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import sys

from adts import *
from engine.Ind import *
import pickle
from engine.EngineUtils import loadSynthState
from adts import *
from problems.Problems import ProblemFactory

if __name__== '__main__':            
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.INFO)
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    #set help message
    help = """
Usage: get_ind PROBLEM_CHOICE OUT_FILE

Dump ind for problem number PROBLEM_CHOICE to a pickle file of the ind.
Variables are initialized at random.

Details:
 PROBLEM_CHOICE -- int -- problem number
 OUT_FILE -- string -- the file to write the ind to.  Note that this is
   _not_ a state file, but instead a direct pickle of the ind.
"""

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [3]:
        print help
        sys.exit(0)


    #yank out the args into usable values
    problem_choice = eval(sys.argv[1])
    out_file = sys.argv[2]

    #do the work

    # -load data
    ps = ProblemFactory().build(problem_choice)

    point_meta = ps.embedded_part.part.point_meta
    new_point = point_meta.createRandomUnscaledPoint(with_novelty = False)

    unscaled_optvals = [new_point[var] for var in ps.ordered_optvars]

    ind = NsgaInd(unscaled_optvals, ps)
    ind.genetic_age = 0
    ind.setAncestry([])

    ind.S = None
    ind._ps = None

    fid = open(out_file,'w')
    pickle.dump(ind, fid)
    fid.close()
