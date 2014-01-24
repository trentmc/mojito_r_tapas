#!/usr/bin/env python

import sys
import Pyro.core
from engine.Channel import *

import logging
INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING

log = logging.getLogger('dispatcher')

secs_beteen_status_updates = 10

if __name__== '__main__':            
    #set up logging
    logging.basicConfig()

    logging.getLogger('channel').setLevel(INFO)
    logging.getLogger('dispatcher').setLevel(INFO)

    #set help message
    help = """
Usage: dispatcher CLUSTER_ID [USE_THREADING]

 Starts a dispatcher daemon that forms the center of a Synth cluster. Start this before
 starting the master and/or the slaves.

 CLUSTER_ID -- string -- The ID for a running cluster. The dispatcher, master and all slaves
                         have to run with the same cluster ID in order for clustering to work
 USE_THREADING -- bool -- Use a threaded dispatcher. Defaults to False
 
"""
    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [2,3]:
        print help
        sys.exit(0)

    cluster_id = sys.argv[1]

    if num_args >= 3:
        use_threading = eval(sys.argv[2])
    else:
        use_threading = False


    if use_threading:
        log.info("Creating threaded dispatcher...")
        Pyro.config.PYRO_MULTITHREADED = 1
    else:
        log.info("Creating unthreaded dispatcher...")
        Pyro.config.PYRO_MULTITHREADED = 0
        
    # we need a server daemon
    Pyro.core.initServer()
    daemon = Pyro.core.Daemon()

    # the channel strategy defines what channel to use
    cs = ChannelStrategy('PyroBased', cluster_id)
    
    try:
        dispatcher = PyroChannelDispatcher( cs, daemon )
    except:
        log.info( "Failed to create Dispatcher..." )
        raise
    
    log.info( "Dispatcher is ready..." )

    continueLoop = True
    while continueLoop:
        try:
            daemon.handleRequests( secs_beteen_status_updates )
            # pruning is done when results are retrieved by the master
            # this is only overhead
            #dispatcher.pruneZombieTasks()
            
            log.info("=================================================")
            log.info("Status of Dispatcher for cluster_ID: %s" % cluster_id)
            log.info("%s" % str(dispatcher))
            log.info("==================================================\n")
            
        except KeyboardInterrupt:
            daemon.shutdown( True )
            continueLoop = False

    log.info( "Dispatcher is exiting..." )
