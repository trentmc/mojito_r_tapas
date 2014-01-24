#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import sys

if __name__== '__main__':            
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.INFO)
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    #set help message
    help = """
Usage: get_ind DB_FILE IND_ID OUT_FILE

Dump ind with id IND_ID from the database DB_FILE to a pickle file of the ind.

Details:
 DB_FILE -- string -- e.g. ~/synth_results/state_genXXXX.db or pooled_db.db
 IND_ID -- int -- eg 2212
 OUT_FILE -- string -- the file to write the ind to.  Note that this is
   _not_ a state file, but instead a direct pickle of the ind.
"""

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [4]:
        print help
        sys.exit(0)

    #late imports
    from adts import *
    import pickle
    from engine.EngineUtils import loadSynthState

    #yank out the args into usable values
    db_file = sys.argv[1]
    ind_ID = sys.argv[2]
    out_file = sys.argv[3]

    #do the work

    # -load data
    if not os.path.exists(db_file):
        print "Cannot find file with name %s" % db_file
        sys.exit(0)
    state = loadSynthState(db_file, None)

    # -find ind
    ind = state.getInd(ind_ID)
    if ind is None:
        print "\nCould not find an ind with ID of %d." % ind_ID
        sys.exit(0)

    ind.S = None
    ind._ps = None
                    
    fid = open(out_file,'w')
    pickle.dump(ind, fid)
    fid.close()
        
    
    
