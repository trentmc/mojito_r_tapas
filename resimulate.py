#!/usr/bin/env python
import os
import sys

USAGE_STR = """
Usage: resimulate INPUT_DB OUTPUT_DB CLUSTER_ID [RANDOMPOOL_DB]
 
Resimulates every individual in INPUT_DB, and saves results to OUTPUT_DB.

 CLUSTER_ID -- string -- The ID for a running cluster. The dispatcher, master and all slaves
                         have to run with the same cluster ID in order for clustering to work.

Helpful for use cases: tweak a testbench's simulation setup; new testbench added; change spec values;
change environmental corners; more
"""

if __name__== '__main__':

    #grab arguments, exit if needed
    num_args = len(sys.argv)
    if num_args not in [4,5]:
        print USAGE_STR
        sys.exit(0)

    from engine.Channel import ChannelStrategy
    import engine.EngineUtils
    import engine.Evaluator
    from engine.Master import Master

    input_db_file = sys.argv[1]
    output_db_file = sys.argv[2]
    cluster_id =  sys.argv[3]

    #corner cases
    if not os.path.exists(input_db_file):
        print "\nInput DB '%s' does not exist.  Exiting.\n" % input_db_file
        sys.exit(0)
    if os.path.exists(output_db_file):
        print "\nOutput DB '%s' already exists.  Exiting.\n" % output_db_file
        sys.exit(0)
    if num_args > 4:
        randompool_db = sys.argv[4]

        if not os.path.exists(randompool_db):
            print "\nRandom pool DB '%s' does not exist.  Exiting.\n" % randompool_db
            sys.exit(0)
    else:
        randompool_db = None

    #setup logging
    import logging
    logging.basicConfig()
    logging.getLogger('analysis').setLevel(logging.INFO)
    logging.getLogger('evaluate').setLevel(logging.INFO)
    logging.getLogger('engine_utils').setLevel(logging.INFO)
    logging.getLogger('master').setLevel(logging.INFO)
    logging.getLogger('channel').setLevel(logging.INFO)

    #main work
    print "Begin resimulate."

    #objects for master
    cs = ChannelStrategy('PyroBased', cluster_id)
    
    #go!
    master = Master(cs, None, None, None, input_db_file, randompool_db)
    if randompool_db:
        try:
            master.resimulate(random_pool_only = True)
            master.random_pool.saveToFile(output_db_file)
        except:
            #master.cleanup()
            raise
    else:
        try:
            master.resimulate()
            master.saveStateToFile(output_db_file)
        except:
            #master.cleanup()
            raise
    #master.cleanup()
    print "Done resimulate."
