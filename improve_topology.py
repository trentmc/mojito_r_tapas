#!/usr/bin/env python
import os
import sys

USAGE_STR = """
Usage: improve_topology INPUT_DB CLUSTER_ID IND_SHORTID WEIGHT_INDEX

Improves the ind having shortID IND_SHORTID from INPUT_DB, according to weight vector WEIGHT_INDEX.

 CLUSTER_ID -- string -- The ID for a running cluster. The dispatcher, master and all slaves
                         have to run with the same cluster ID in order for clustering to work.

"""

if __name__== '__main__':

    #grab arguments, exit if needed
    num_args = len(sys.argv)
    if num_args not in [5]:
        print USAGE_STR
        sys.exit(0)

    from engine.Channel import ChannelStrategy
    import engine.EngineUtils
    import engine.Evaluator
    from engine.Master import Master

    input_db_file = sys.argv[1]
    cluster_id =  sys.argv[2]
    ind_shortid = sys.argv[3]
    weight_index = eval(sys.argv[4])

    #corner cases
    if not os.path.exists(input_db_file):
        print "\nInput DB '%s' does not exist.  Exiting.\n" % input_db_file
        sys.exit(0)

    #setup logging
    import logging
    logging.basicConfig()
    logging.getLogger('analysis').setLevel(logging.INFO)
    logging.getLogger('evaluate').setLevel(logging.INFO)
    logging.getLogger('engine_utils').setLevel(logging.INFO)
    logging.getLogger('master').setLevel(logging.INFO)
    logging.getLogger('channel').setLevel(logging.INFO)

    #main work
    print "Begin improve."

    #objects for master
    cs = ChannelStrategy('PyroBased', cluster_id)

    #go!
    master = Master(cs, None, None, None, input_db_file)
    try:
        target = None
        for ind in master.state.allInds():
            if str(ind.shortID()) == str(ind_shortid):
                target = ind
                break
        if target != None:
            print "start with ind %s: %s" % ([target], str(target))
            new_inds = master.improveIndsForWeights([target], [master.state.W[weight_index,:]])
            print "improved ind %s: %s" % ([new_inds[0]], str(new_inds[0]))
        else:
            print "could not find target ind"
    except:
        #master.cleanup()
        raise

    #master.cleanup()
    print "Done improve."
