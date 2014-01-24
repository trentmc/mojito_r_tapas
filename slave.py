#!/usr/bin/env python 
##!/usr/bin/env python2.4

#note: turn on/off aggressive testing via util.constants.AGGR_TEST
import logging
import os
import sys

import signal

INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING
slave = None

def sighandler(signo, stackframe):
    print "signal %s received" % signo
    if slave:
        slave.requestStop()   

if __name__== '__main__':            
    #set up logging
    logging.basicConfig()
    
    logging.getLogger('slave').setLevel(DEBUG)
    logging.getLogger('channel').setLevel(INFO)
    logging.getLogger('engine_utils').setLevel(INFO)
    
    logging.getLogger('gto').setLevel(INFO)
    logging.getLogger('probe').setLevel(INFO)
    logging.getLogger('sgb').setLevel(INFO)
    
    logging.getLogger('cyt').setLevel(WARNING) #useful to turn off outputs in pwl-building
    #logging.getLogger('cyt').setLevel(INFO)   #useful for meeting constraints (at cost of pwl crazy)
    
    logging.getLogger('analysis').setLevel(DEBUG)
    logging.getLogger('part').setLevel(INFO)
    
    logging.getLogger('evaluate').setLevel(INFO)
    logging.getLogger('metric_calc').setLevel(INFO)
    logging.getLogger('pwl').setLevel(DEBUG)
    logging.getLogger('lin').setLevel(WARNING)
    
    logging.getLogger('luc').setLevel(INFO)

    #set help message
    help = """
Usage: slave CLUSTER_ID

 CLUSTER_ID -- string -- The ID for a running cluster. The dispatcher, master and all slaves
                         have to run with the same cluster ID in order for clustering to work   
"""

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [2]:
        print help
        sys.exit(0)

    from adts import *
    from engine.Slave import Slave
    from engine.Channel import ChannelStrategy

    cluster_id = sys.argv[1]
        
    cs = ChannelStrategy('PyroBased', cluster_id)
    #cs = ChannelStrategy('FileBased', cluster_id)
    #cs.channel_file = '/tmp/tmpchannel'
    slave = Slave(cs)
    signal.signal( signal.SIGTERM, sighandler )
    try:
      slave.run()
    except KeyboardInterrupt:
      print "shutdown requested"
      slave.stop()
    except:
      print "Error occurred..."
      raise


def signal_handler( signal_number ):
    """
    A decorator to set the specified function as handler for a signal.
    This function is the 'outer' decorator, called with only the (non-function) 
    arguments
    """
    
    # create the 'real' decorator which takes only a function as an argument
    def __decorator( function ):
        signal.signal( signal_number, function )
        return function
    
    return __decorator
   

    
if __name__ == "__main__":
    """test the decorator"""
    
    sigterm_received = False
    
    @signal_handler(signal.SIGTERM)
    def handle_sigterm(signum, frame):
        """handle sigterm for test"""
        global sigterm_received
        sigterm_received = True
        