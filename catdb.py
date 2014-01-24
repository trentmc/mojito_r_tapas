#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import string
import sys

if __name__== '__main__':            
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.DEBUG)
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    #set help message
    help = """
Usage: catdb OUT_DB IN_DB1 DB1_INDS [IN_DB2 IN_DB3 ...]

Extracts some or all inds from DB1, and can also tack on inds from DB2, DB3, etc.

Details:
 OUT_DB -- string -- the new .db file to create.
 IN_DB1 -- string -- e.g. ~/synth_results/state_genXXXX.db or pooled_db.db
 DB1_INDS -- string -- which inds in DB1 to use?  Can be 'all', or a list of IDs
  such as '83134' or '2983864 38929 322833'
 IN_DB2, IN_DB3, etc -- string -- optional 2nd, 3rd, etc infile.
   Uses all inds.  Merges each age layer
   
"""
    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args <= 3:
        print help
        sys.exit(0)
    
    #yank out the args into usable values
    db_outfile = os.path.abspath(sys.argv[1])
    db1_infile = os.path.abspath(sys.argv[2])
    db1_inds = sys.argv[3]
    other_db_infiles = [os.path.abspath(other_db_infile)
                        for other_db_infile in sys.argv[4:]]

    if other_db_infiles:
        print "WARNING: merging >1 DB from different novelty runs" \
              " will be invalid for doing more synth runs because" \
              " they have different embedded parts!"
        sys.exit(0)

    #some preliminary assertions
    print "Validate inputs: begin"
    
    # -note that for our file checks, we've already made the paths
    # absolute, so that uniqueness checks don't let abs + non-abs path slip thru
    # -unique input files?
    db_infiles = [db1_infile] + other_db_infiles
    assert len(db_infiles) == len(set(db_infiles)), \
           "Need uniquely named input DBs; got %s" % db_infiles
    
    # -do all input db files exist?
    for db_infile in db_infiles:
        if not os.path.exists(db_infile):
            print "Cannot find input file with name %s" % db_infile
            sys.exit(0)

    # -don't accidentally overwrite an input file
    assert not os.path.exists(db_outfile), "Output DB cannot already exist"
    assert db_outfile not in db_infiles, "Output DB cannot be an input DB"

    # -is db1_inds argument valid?
    if db1_inds == 'all':
        pass
    else:
        #this will fail if we cannot convert to int (or other issue)
        dummy = [int(id_str) for id_str in string.split(db1_inds)]
    print "Validate inputs: done"

    #do the work
    print "Will merge inds '%s' from DB %s, and all inds from DBs %s, " \
          " and put result into output DB %s" % \
          (db1_inds, db1_infile, other_db_infiles, db_outfile)
    
    # -load db1 data; it will become the starting point for output state to save
    print "Load DB1: begin"
    import engine.EngineUtils
    state = engine.EngineUtils.loadSynthState(db1_infile, None)
    problem_choice = state.problem_choice
    print "Load DB1: done"
    
    #prune down inds in 'state'
    print "Prune DB1: begin"
    if db1_inds == 'all':
        IDs_used = set([ind.ID
                        for R in state.R_per_age_layer
                        for ind in R])
    else:
        allowed_IDs = [int(id_str) for id_str in string.split(db1_inds)]
        for layer_i, R in enumerate(state.R_per_age_layer):
            state.R_per_age_layer[layer_i] = [ind for ind in R
                                              if ind.ID in allowed_IDs]
        IDs_used = set(allowed_IDs)
    print "Prune DB1: done; tot num inds = %d" % state.numInds()

    #add inds from other dbs' states
    # -we currently need all states to have same num age layers
    for other_db_infile in other_db_infiles:
        print "Add next DB '%s': begin" % other_db_infile

        #-do the load
        other_state = EngineUtils.loadSynthState(other_db_infile, None)
        if other_state.problem_choice != problem_choice:
            print "Cannot add DB because its problem ID is %d, while first db's problem ID is %d" % \
                  (other_state.problem_choice, problem_choice)
        else:
            #-update 'state'
            for layer_i, other_R in enumerate(other_state.R_per_age_layer):
                for ind in other_R:
                    #unique-ify ID if needed
                    if ind.ID in IDs_used:
                        safe_ID_ind = ind.copyWithNewID()
                        print "The ID '%s' of ind from DB '%s' is already taken, so giving it " \
                              " a new ID of '%s'" % (ind.ID, other_db_infile, safe_ID_ind)
                    else:
                        safe_ID_ind = ind
                    IDs_used.add(safe_ID_ind.ID)

                    #add ind
                    state.R_per_age_layer[layer_i].append(safe_ID_ind)
                    
            print "Add next DB: done; tot num inds = %d" % state.numInds()
        
    #now save 'state'
    # -we can't test the asserts because we _are_ violating the R_per_age_layer!
    print "Save final state: begin"
    state.save(db_outfile, test_asserts=False)
    print "Save final state: done"

    #Done!
    print "catdb is complete."
