#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import sys

from adts import *
from engine.EngineUtils import worstCasePopulationSummaryStr, loadSynthState

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

if __name__== '__main__':            
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.INFO)
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    #set help message
    help = """
Usage: netlister DB_FILE IND_ID ANNOTATE_POINTS ANNOTATE_BB [ANALYSIS_INDEX ENV_INDEX]

Netlists an ind having IND_ID in DB_FILE.  Can annotate to make it prettier.
Can make it simulatable by also specifying an analysis and env_point.

Note: to netlist an ind in an IND_FILE, use netlister2.py.

Details:
 DB_FILE -- string -- e.g. ~/synth_results/state_genXXXX.db or pooled_db.db
 IND_ID -- int -- eg 2212
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
    db_file = sys.argv[1]
    ind_ID = sys.argv[2]
    annotate_points = bool(eval(sys.argv[3]))
    annotate_bb_info = bool(eval(sys.argv[4]))

    make_simulatable = (num_args == 7)
    if make_simulatable:
        analysis_index = eval(sys.argv[5])
        env_index = eval(sys.argv[6])
    else:
        analysis_index = None
        env_index = None

    #do the work

    # -load data
    if not os.path.exists(db_file):
        print "Cannot find file with name %s" % db_file
        sys.exit(0)
    state = loadSynthState(db_file, None)
    ps = state.ps

#     import pdb;pdb.set_trace()
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
    ind = None
    for cand_ind in state.allInds():
        if cand_ind.shortID() == ind_ID:
            ind = cand_ind
            break
    if ind is None:
        print "ind with ID=%s not found in db; use summarize_db to learn IDs" %\
              ind_ID
        sys.exit(0)

    #-we'll be building up 'big_s' (a netlist)
    big_s = ''

    # -add design info (and maybe simulation info)
    if not make_simulatable:
        env_point = EnvPoint(is_scaled=True)
        big_s += ind.nominalNetlist(annotate_bb_info=annotate_bb_info, add_infostring=True)
        
    else:
        analysis = ps.analyses[analysis_index]
        env_point = analysis.env_points[env_index]
        variation_data = (RndPoint([]) , env_point, DevicesSetup('UMC180'))
        big_s += analysis.createFullNetlist(ps.embedded_part, ps.scaledPoint(ind), variation_data)
        
    # -generate whatever is in str(ind)
    big_s += "*-----------------------------------------------------------\n"
    big_s += '* str(ind)=%s:\n' % str(ind)
    
    # -generate summary of performance
    big_s += "*-----------------------------------------------------------\n"
    big_s += '* Summary of worst-case performances:\n'
    big_s += '* '
    big_s += spiceCommentPrefixBlock( worstCasePopulationSummaryStr(ps, [ind]) )

    # -add info about novelty
    big_s += "* -----------------------------------------------------------\n"
    big_s += '* ' + ind.noveltySummary()

    # -maybe info about unscaled_point, scaled_point
    if annotate_points:
        s = spiceCommentPrefixBlock( ind.pointSummary() )
        s += "* Netlist=\n%s" % big_s
        big_s = s


    #
    print big_s
        
    
    
    
      
    
