#!/usr/bin/env python 
##!/usr/bin/env python2.4

import logging
import os
import sys

INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING

if __name__== '__main__':            
    #set up logging
    logging.basicConfig()
    
    logging.getLogger('master').setLevel(INFO)
    logging.getLogger('engine_utils').setLevel(INFO)
    logging.getLogger('channel').setLevel(INFO)
    
    logging.getLogger('part').setLevel(DEBUG)
    logging.getLogger('analysis').setLevel(DEBUG)
    logging.getLogger('problems').setLevel(INFO)
    logging.getLogger('library').setLevel(DEBUG)
    logging.getLogger('luc').setLevel(INFO)

    #set help message
    help = """
Usage: master DO_ROBUST DO_NOVELTY PROBLEM_NUM POP_SIZE OUTPUT_DIR CLUSTER_ID [RESTART_DB_FILE] [INITIAL_RANDOM_POOL]

 DO_ROBUST -- T (True), F (False) -- do robust optimization? (has yield as another objective in pareto front)
 DO_NOVELTY -- T (True), F (False), A (AlwaysTrue) -- try creating novel topologies too?
 PROBLEM_NUM -- int -- run 'listproblems.py' to see options
 POP_SIZE -- int -- population size
 OUTPUT_DIR -- string -- output directory for state db files.
 CLUSTER_ID -- string -- The ID for a running cluster. The dispatcher, master and all slaves
                         have to run with the same cluster ID in order for clustering to work.
 RESTART_DB_FILE -- string -- (optional) set to a previous state_genXXXX.db to continue a previous run
 INITIAL_RANDOM_POOL -- string -- (optional) set to a previous state_rndXXXX.db to resuse the random inds
                              from a previous run
 """

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [7, 8, 9]:
        print help
        if num_args > 1:
            print "Got %d argument(s), need 6, 7 or 8.\n" % (num_args-1)
        sys.exit(0)

    from adts import *
    from problems.Problems import ProblemFactory
    from engine.Master import Master
    from engine.SynthSolutionStrategy import SynthSolutionStrategy
    from engine.Channel import ChannelStrategy
    from util.constants import DOCS_METRIC_NAME

    #extract inputs
    do_robust = sys.argv[1]
    do_novelty = sys.argv[2]
    problem_num = eval(sys.argv[3])
    pop_size = eval(sys.argv[4])
    output_dir = sys.argv[5]
    cluster_id = sys.argv[6]
    
    if num_args >= 8: restart_file = sys.argv[7]
    else:             restart_file = None
    if restart_file == 'None': restart_file = None
    
    if num_args >= 9: pool_file = sys.argv[8]
    else:             pool_file = None    
    if pool_file == 'None': pool_file = None
    
    print "Arguments: DO_ROBUST=%s, DO_NOVELTY=%s, PROBLEM_NUM=%s, POP_SIZE=%s, OUTPUT_DIR=%s, " \
          "CLUSTER_ID=%s, RESTART_DB_FILE=%s, INITIAL_RANDOM_POOL=%s\n" % \
          (do_robust, do_novelty, problem_num, pop_size, output_dir, cluster_id, restart_file, pool_file)

    #set do_robust
    if do_robust in ['T', 'True']:
        do_robust = True
    elif do_robust in ['F', 'False']:
        do_robust = False
    else:
        print help
        sys.exit(0)

    #set do_novelty, always_with_novelty
    if do_novelty in ['T', 'True']:
        do_novelty = True
        always_with_novelty = False
    elif do_novelty in ['F', 'False']:
        do_novelty = False
        always_with_novelty = False
    elif do_novelty in ['A', 'AlwaysTrue']:
        do_novelty = True
        always_with_novelty = True
    else:
        print help
        sys.exit(0)

    #handle corner cases
    if os.path.exists(output_dir):
        print "\nOutput path '%s' already exists.  Exiting.\n" % output_dir
        sys.exit(0)

    # make directory
    os.mkdir(output_dir)
    # store svn version
    os.system('svn info > ' + output_dir + '/svninfo')
    os.system('svn diff > ' + output_dir + '/svndiff')
        
    #objects for master
    cs = ChannelStrategy('PyroBased', cluster_id)

    ps = ProblemFactory().build(problem_num, None)
    if do_robust:
        ps.devices_setup.makeRobust()
    assert ps.doRobust() == do_robust

    age_gap = 10
    ss = SynthSolutionStrategy(do_novelty, pop_size, age_gap)
    #ss.setTinyRobust() #this is HACK if not commented
    
    ss.max_num_inds = 100e6 #100 million
    ss.metric_weights[DOCS_METRIC_NAME] = 10.0 #have 10x bias for this compared to other metrics
    ss.always_with_novelty = always_with_novelty

    #go!
    master = Master(cs, ps, ss, output_dir, restart_file, pool_file)
    try:
        master.run()
    except:
        #master.cleanup()
        raise
