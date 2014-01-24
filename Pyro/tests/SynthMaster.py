#! /usr/bin/env python
import sys
from engine.Channel import *
import time
import logging
import random
from engine.SynthSolutionStrategy import SynthSolutionStrategy

INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING

log = logging.getLogger('master')
MESSAGE_INTERVAL = 2
nb_tasks = 200

if __name__== '__main__':            
    #set up logging
    logging.basicConfig()

    logging.getLogger('channel').setLevel(INFO)
    logging.getLogger('master').setLevel(INFO)

    # Master workflow
    cs = ChannelStrategy( )
    cs.cluster_id = 0
    cs.channel_type = 'PyroBased'
    cs.channel_file = "testfile"
    
    ch = ChannelFactory( cs ).buildChannel( True )

    ss = SynthSolutionStrategy(do_novelty_gen=False, num_inds_per_age_layer=3)
    ch.registerMaster( ss, 0 )
    master_ID = "my_master_ID"
   
    log.info("Adding tasks...")
    for i in range(0,nb_tasks):
        ch.addTasks([ TaskForSlave(master_ID, "Generate random ind",  TaskData(random.randint(0,1)) ) ])

    log.info( "Waiting for result..." )
    done = False
    all_results = []
    sleeptime = 1
    try:
        while not done:
            results = ch.popResults()
            all_results.extend( results )
            nb_remaining_tasks = len(ch.remainingTasks())
            
            if nb_remaining_tasks > 0:
                if sleeptime % MESSAGE_INTERVAL == 0:
                    log.info( "Still %d tasks in queue, waiting..." % nb_remaining_tasks )
                time.sleep(1)
                sleeptime += 1
                done = False
            else:
                done = True
    except KeyboardInterrupt:
        log.info( "Leaving..." )
        ch.cleanup()
        
    except:
        raise

    log.info("results: %s" % str(all_results))
