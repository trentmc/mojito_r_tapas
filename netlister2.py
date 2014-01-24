#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import sys

import pickle

from adts import *
from problems.Problems import ProblemFactory

def spiceCommentPrefixBlock(in_str):
    lines = in_str.splitlines()
    out_str = ""
    for line in lines:
        # if no spice prefix, add one
        if line[0:2] != '* ':
            out_str += '* '
            
        # add the original line
        out_str += line + "\n"

    return out_str

def spiceSanitize(in_str):
    lines = in_str.splitlines()
    out_str = ""
    for line in lines:
        # if the line is too long, truncate it
        if len(line) > 1000:
            out_str += line[0:1000] + "\n"
        else:
            # add the line
            out_str += line + "\n"

    return out_str

if __name__== '__main__':            
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.INFO)
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    #set help message
    help = """
Usage: netlister2 PROBLEM_NUM IND_FILE ANNOTATE_POINTS ANNOTATE_BB [ANALYSIS_INDEX ENV_INDEX]

Netlists the ind in IND_FILE.  Can annotate to make it prettier.
Can make it simulatable by also specifying an analysis and env_point.

Note: to netlist an ind in a DB_FILE, use netlister.py.

Details:
 PROBLEM_NUM -- int -- see listproblems.py
 IND_FILE -- string -- the file containing the ind  (saved using get_ind.py)
 ANNOTATE_POINTS -- 0 or 1 -- if 1, add unscaled_point and scaled_point info
 ANNOTATE_BB -- 0 or 1 -- if 1, add building block info
 ANALYSIS_INDEX -- int in {0,1,...,num_analyses-1} (not analysis ID!)
 ENV_INDEX -- int in {0,1,...,num_env_points for analysis} (not env point ID!)
"""

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [5,7]:
        print help
        sys.exit(0)

    #yank out the args into usable values
    problem_choice = eval(sys.argv[1])
    ind_file = sys.argv[2]
    annotate_points = bool(eval(sys.argv[3]))
    annotate_bb_info = bool(eval(sys.argv[4]))

    make_simulatable = (num_args == 7)
    if make_simulatable:
        analysis_index = eval(sys.argv[5])
        env_index = eval(sys.argv[6])

    #do the work

    # -load data
    ps = ProblemFactory().build(problem_choice)
    if not os.path.exists(ind_file):
        print "Cannot find file with name %s" % ind_file
        sys.exit(0)

    # -exit cases
    if make_simulatable:
        if analysis_index >= len(ps.analyses):
            print "Requested analysis_index=%d but only %d analyses available" % (analysis_index, len(ps.analyses))
            sys.exit(0)
        analysis = ps.analyses[analysis_index]
        if env_index >= len(analysis.env_points):
            print "Requested env_index=%d but only %d env_points available" % (env_index, len(analysis.env_points))
            sys.exit(0)

    # -find ind
    fid = open(ind_file,'r')
    ind = pickle.load(fid)
    fid.close()
        
    ind._ps = ps

    #-we'll be building up 'big_s' (a netlist)
    big_s = ''

    # -add design info (and maybe simulation info)
    if not make_simulatable:
        env_point = EnvPoint(is_scaled=True)
        big_s += ind.nominalNetlist(annotate_bb_info = annotate_bb_info, add_infostring=True)
        
    else:
        analysis = ps.analyses[analysis_index]
        env_point = analysis.env_points[env_index]
        variation_data = (RndPoint([]) , env_point, DevicesSetup('UMC180'))
        big_s += analysis.createFullNetlist(ps.embedded_part, ps.scaledPoint(ind), variation_data)

    # -maybe info about unscaled_point, scaled_point
    if annotate_points:
        big_s += "* Netlist=\n%s" % spiceCommentPrefixBlock(ind.pointSummary())

    #successful, so print netlist
    print spiceSanitize(big_s)
        
    
    
    
