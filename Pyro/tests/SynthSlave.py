#! /usr/bin/env python
import sys
from engine.Channel import *
import time
import logging

INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING

log = logging.getLogger('slave')

MESSAGE_INTERVAL = 10

def process(task):
    log.info( "Master wants me to %s for %s" \
              % ( task.descr, str( task.task_data.value ) ) )

    t = task.task_data.value
    string = "%d secs" % t
    log.info( "I think I need %s to fullfil her request" % string )
    
    time.sleep( t )
    return string

if __name__== '__main__':            
    #set up logging
    logging.basicConfig()

    logging.getLogger('channel').setLevel(INFO)
    logging.getLogger('slave').setLevel(INFO)

    # Slave workflow
    cs = ChannelStrategy( )
    cs.cluster_id = 0
    cs.channel_type = 'PyroBased'
    cs.channel_file = "testfile"
    
    ch = ChannelFactory( cs ).buildChannel( False )

    ch.reportForService( random.randint( 0, 10) )
    
    log.info( "Waiting for tasks..." )
    done = False
    all_results = []
    sleeptime = 1
    try:
        while not done:
            task = ch.popTask()
            if task != None:
                log.info( " Got task, processing" )
                result = process( task )
                r = ResultFromSlave(task.descr, ResultData(result, {}), '' )
                ch.addResult( r )
                sleeptime = 1
            else:
                if sleeptime % MESSAGE_INTERVAL == 0:
                    log.info( " No task available..." )
                time.sleep(1)
                sleeptime += 1
            
    except KeyboardInterrupt:
        log.info( "Leaving slave ..." )
        
    except:
        raise
    
