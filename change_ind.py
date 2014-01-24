#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import sys

import pickle

if __name__== '__main__':            
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.DEBUG)
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    #set help message
    help = """
Usage: change_ind PROBLEM_NUM IND_FILE OUT_FILE

    Allows to change a pickled ind. The file is loaded from IND_FILE,
    then the python debugger is called, and the resulting ind is saved in
    OUT_FILE

Details:
 PROBLEM_NUM -- int -- see listproblems.py
 IND_FILE -- string -- the file containing the ind  (saved using get_ind.py)
 OUT_FILE -- string -- the file the ind changed ind is to be saved to
"""

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [4]:
        print help
        sys.exit(0)

    #yank out the args into usable values
    problem_choice = eval(sys.argv[1])
    ind_file = sys.argv[2]
    ind_out_file = sys.argv[3]

    #late imports
    from adts import *
    from problems.Problems import ProblemFactory

    #do the work


    # -load data
    ps = ProblemFactory().build(problem_choice)
    if not os.path.exists(ind_file):
        print "Cannot find file with name %s" % ind_file
        sys.exit(0)
    
    fid = open(ind_file,'r')
    ind = pickle.load(fid)
    fid.close()

    ind._ps = ps
    
    print ind.pointSummary()

    unscaled_point = {}
    # print "Please change the unscaled point values through the unscaled_point dict and use the 'c' command to continue..."
    # import pdb; pdb.set_trace()

    unscaled_point['chosen_part_index']      = 0         # load==vdd 'chosen_part_index'
    unscaled_point['input_is_pmos']          = 1         # input_is_pmos
    unscaled_point['Ibias']                  = 2e-3      # Ibias
    unscaled_point['Ibias2']                 = 1e-3      # Ibias2
    unscaled_point['fracVgnd']               = 0.3/0.9       # fracVgnd
    unscaled_point['inputcascode_L']         = 0.4e-6    # inputcascode_L
    unscaled_point['inputcascode_Vgs']       = 0.7       # inputcascode_Vgs
    unscaled_point['inputcascode_is_wire']   = 0         # nputcascode_is_wire
    unscaled_point['inputcascode_recurse']   = 0         # inputcascode_recurse
    unscaled_point['fracAmp']                = 0.5       # fracAmp
    unscaled_point['ampmos_L']               = 0.4e-6    # ampmos_L
    unscaled_point['Vds_internal']           = 0.9       # Vds_internal
    unscaled_point['load_cascode_Vgs']       = 0.7       # load_cascode_Vgs
    unscaled_point['load_fracOut']           = 0.5       # load_fracOut
    unscaled_point['load_cascode_L']         = 0.4e-6    # load_cascode_L
    unscaled_point['load_chosen_part_index'] = 0         # load_chosen_part_index
    unscaled_point['load_fracIn']            = 0.5       # load_fracIn
    unscaled_point['load_L']                 = 0.4e-6    # load_L
    unscaled_point['folder_Vgs']             = 0.7       # folder_Vgs
    unscaled_point['folder_L']               = 0.4e-6    # folder_L
    unscaled_point['inputbias_Vgs']          = 0.7       # inputbias_Vgs
    unscaled_point['inputbias_L']            = 0.4e-6    # inputbias_L
    unscaled_point['degen_choice']           = 0         # degen_choice
    unscaled_point['degen_fracDeg']          = 0.1       # degen_fracDeg

    varnames = ind._ps.ordered_optvars
    for key in unscaled_point.keys():
        idx = varnames.index(key)
        ind.unscaled_optvals[idx] = unscaled_point[key]

    print ind.pointSummary()

    # remove some bogus stuff
    ind.S = None
    ind._ps = None

    # save the modified ind
    fid=open(ind_out_file,'w')
    pickle.dump(ind,fid)
    fid.close()

    
    
