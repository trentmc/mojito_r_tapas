#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import sys

if __name__== '__main__':            
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.DEBUG)
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    #set help message
    help = """
Usage: summarize_db DB_FILE [SORT_METRIC [BASE_FILE [OUTPUT_ALL]]]

Prints a summary of db contents:
-has one row entry for each _nondominated_ (layer 0) ind
-in each row, give ind ID plus all worst-case metric values

Details:
 DB_FILE -- string -- e.g. ~/synth_results/state_genXXXX.db or pooled_db.db
 SORT_METRIC -- string or None -- if specified, sorts the inds by that SORT_METRIC (Ignored if ps is robust).
 BASE_FILE -- string or None -- if specified, outputs files readable by matlab:
   -Metrics in BASE_FILE_metrics.val (data), BASE_FILE_metrics.hdr (row of metric names)
   -Points in BASE_FILE_points.val (data), BASE_FILE_points.hdr (row of variable names)
   -Topology descriptions in BASE_FILE_topos.val (data), BASE_FILE_topos.hdr (row of choice var names)
 OUTPUT_ALL -- bool or None -- if specified and 'True', then it will output _all_
   the individuals (flat) instead of just the nondominated ones.  Default False.
   (If ps is robust and OUTPUT_ALL is true, then it will give worst-case, NOT nondominated
    population summary).
   
"""

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [2,3,4,5]:
        print help
        sys.exit(0)

    #yank out the args into usable values
    db_file = sys.argv[1]
    output_all = False
    
    sort_metric = None
    if num_args >= 3:
        sort_metric = sys.argv[2]
        if sort_metric == 'None': sort_metric = None
    
    matlab_base_file = None
    if num_args >= 4:
        matlab_base_file = sys.argv[3]
        if matlab_base_file == 'None': matlab_base_file = None
        
    if num_args >= 5:
        output_all = (sys.argv[4] == 'True')

    #do the work
    import engine.EngineUtils as EngineUtils
    from util import mathutil

    # -load data
    if not os.path.exists(db_file):
        print "Cannot find file with name %s" % db_file
        sys.exit(0)
    state = EngineUtils.loadSynthState(db_file, None)
    ps = state.ps

    reference_ind = state.R_per_age_layer[-1][0]
    rnd_ID = reference_ind.rnd_IDs[0]
    metric_names_measured = reference_ind.sim_results[rnd_ID].keys()
    old_ps_metric_names = ps.flattenedMetricNames()
    ps.stripToSpecifiedMetrics(metric_names_measured)
    print "Stripped the following metrics from ps which weren't in inds: %s" % \
          mathutil.listDiff(old_ps_metric_names, metric_names_measured)

    # -validate sort_metric earlier rather than later
    if (sort_metric is not None) and (not ps.doRobust()):
        ok_names = ps.flattenedMetricNames()
        if sort_metric not in ok_names:
            print "Sort_metric '%s' is not in metric names of %s" % (sort_metric, ok_names)
            sys.exit(0)
    
    # -print!
    print state.populationSummaryStr(sort_metric, output_all)
    if matlab_base_file is not None:
        state.populationSummaryToMatlab(matlab_base_file, output_all)

    #done!
    print "Done summarize_db.py"
