#!/usr/bin/env python
import os
import sys

USAGE_STR = """
Usage: mcsimulate INPUT_DB IND_ID OUTPUT_DB CLUSTER_ID [MAX_NUM_RND_PTS]
 
Locally, Monte Carlo simulates INPUT_DB's individual IND_ID (shortID), and saves results to OUTPUT_DB.
If the PS is nominal, then just simulates there.
"""

if __name__== '__main__':

    #grab arguments, exit if needed
    num_args = len(sys.argv)
    if num_args not in [5, 6]:
        print USAGE_STR
        sys.exit(0)
    input_db_file = sys.argv[1]
    ind_ID = sys.argv[2]
    output_db_file = sys.argv[3]
    cluster_id = sys.argv[4]
    
    if num_args >= 6: max_num_rnd_points = eval(sys.argv[5])
    else:             max_num_rnd_points = 100000
    
    #corner cases
    if not os.path.exists(input_db_file):
        print "\nInput DB '%s' does not exist.  Exiting.\n" % input_db_file
        sys.exit(0)
        
    if os.path.exists(output_db_file):
        print "\nOutput DB '%s' already exists.  Exiting.\n" % output_db_file
        sys.exit(0)

    #late imports
    import engine.Channel as Channel
    import engine.EngineUtils as EngineUtils
    from engine.Master import Master

    #setup logging
    import logging
    logging.basicConfig()
    logging.getLogger('analysis').setLevel(logging.INFO)
    logging.getLogger('evaluate').setLevel(logging.INFO)
    logging.getLogger('engine_utils').setLevel(logging.INFO)
    logging.getLogger('master').setLevel(logging.INFO)
    logging.getLogger('channel').setLevel(logging.INFO)

    #
    print "mcsimulate: begin: INPUT_DB=%s, IND_ID=%s, OUTPUT_DB=%s, CLUSTER_ID=%s, MAX_NUM_RND_PTS=%d" % \
          (input_db_file, ind_ID, output_db_file, cluster_id, max_num_rnd_points)
                             
    #objects for master
    cs = Channel.ChannelStrategy('PyroBased', cluster_id)
    
    #go!
    master = Master(cs, None, None, None, input_db_file)
    ind = master.state.getInd(ind_ID)
    if ind is None:
        ind_IDs = [ind.shortID() for ind in master.state.allInds()]
        print "\nCould not find an ind with ID of %s.  IDs are: %s" % (ind_ID, ind_IDs)
        sys.exit(0)
    print "mcsimulate: Found ind with ID %s; it has %d rnd points" % (ind_ID, len(ind.rnd_IDs))
    
    try:
        ind.clearSimulations()
        
        task_data = Channel.TaskData()
        task_data.ind = ind
        task_data.ind.prepareForPickle()
        task_data.num_rnd_points = min(max_num_rnd_points, len(ind.rnd_IDs))

        print "mcsimulate: master-invoked simulations: begin"
        inds = master.generateInds(task_data_list=[task_data], descr="Evaluate ind further")
        print "mcsimulate: master-invoked simulations: done"

        print "mcsimulate: save final output file %s" % output_db_file
        master.saveStateToFile(output_db_file)
    except:
        #master.cleanup()
        raise

    #master.cleanup()
    print "mcsimulate: done"
