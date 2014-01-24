#! /usr/bin/env python
import sys
import Pyro.core
from engine.Channel import *

import logging
INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING

log = logging.getLogger('dispatcher')
if __name__== '__main__':            
    #set up logging
    logging.basicConfig()

    logging.getLogger('channel').setLevel(DEBUG)
    logging.getLogger('dispatcher').setLevel(DEBUG)

    # we need a server daemon
    Pyro.core.initServer()
    daemon = Pyro.core.Daemon()

    # the channel strategy defines what channel to use
    cs = ChannelStrategy( )
    cs.cluster_id = 0
    cs.channel_type = 'PyroBased'
    cs.channel_file = "testfile"
    
    try:
        dispatcher = PyroChannelDispatcher( cs, daemon )
    except:
        log.info( "Failed to create Dispatcher..." )
        raise
    
    log.info( "Dispatcher is ready..." )
    try:
        daemon.requestLoop()
    except KeyboardInterrupt:
        daemon.shutdown(True)

    log.info( "Dispatcher is exiting..." )
