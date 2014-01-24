#!/usr/bin/env python


import commands
import time
import os
import sys
import struct
import copy

import Pyro.core
import Pyro.naming
from Pyro.errors import NamingError

import socket
#socket.setdefaulttimeout(10)
Pyro.core.initClient()
Pyro.config.PYRO_NS_DEFAULTGROUP = ":SynthDrones"

import logging
INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING

log = logging.getLogger('dronemaster')

from RemoteNode import *

if __name__== '__main__':
    #set up logging
    logging.basicConfig()

    logging.getLogger('dronemaster').setLevel(INFO)

    #set help message
    help = """
Usage: dronemaster CLUSTER_ID [TIME_OUT]

 Starts a dronemaster process that monitors a set of drones. drones are daemons that run on
 node that should run slaves. these daemons start and stop the slaves as necessary

 CLUSTER_ID -- string -- The ID for a running cluster. The dispatcher, master and all slaves
                         have to run with the same cluster ID in order for clustering to work
                         make sure there is a dispatcher and a running master before calling the
                         dronemaster
 TIME_OUT -- int/string -- number of seconds to run before the drones (and slaves) are shut-down.
                           is eval()'ed, so can contain a formula. default = 1 year. can be None
                           for default.
"""
    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [2,3]:
        print help
        sys.exit(0)

    cluster_id = sys.argv[1]

    if num_args >= 3 or eval(sys.argv[2]) == None:
        max_runtime_sec = eval(sys.argv[2])
    else:
        max_runtime_sec = 60*60*24*356
     
    # all in seconds
    time_between_checks = 120
    max_hosts = 99

    max_load = 2.0

    files = []
    #files.append('clustertools/pcroom0091.list')
    #files.append('clustertools/pcroom0092.list')
    files.append('clustertools/pcroom0254.list')
    files.append('clustertools/pcroom9156.list')
    #files.append('clustertools/micas.list')
    #files.append('clustertools/test.list')

    hosts = {}
    cnt = 0
    for file in files:
        fid = open(file)
        hostlines = fid.readlines()
        fid.close()

        for l in hostlines:
            if l[0] != '#' and l != "\n":
                try:
                    tmp = l.split()
                    hosts[tmp[0]] = (tmp[0], tmp[1], eval(tmp[2] + ' != False'))
                except:
                    print "bogus line: %s" % l
            cnt += 1
            if cnt > max_hosts:
                break
        if cnt > max_hosts:
            break

    for (mac, h, do_wakeup) in hosts.values():
        print "Loaded host %s at MAC %s (%s)" % (str(h), str(mac), str(do_wakeup))

    stop_time = time.time() + max_runtime_sec
    nodes = {}
    for (mac, h, do_wakeup) in hosts.values():
        nodes[h] = RemoteNode(h, mac, cluster_id, do_wakeup)

    try:
        while time.time() < stop_time:
            start_check_time = time.time()
            for node in nodes.values():

                if node.isAlive():
                    if not node.checkDrone():
                        if not node.connectToDrone():
                            log.info("no drone available, starting one")
                            if not node.startDrone():
                                log.info("the drone couldn't be started, removing node from list")
                            else:
                                if not node.connectToDrone():
                                    log.info("could not connect to drone")
                                    continue
                    #try:
                    if True:
                        # the host is alive
                        node.checkStatus()

                        if node.checkSlaves():
                            print "Slave at %s seems to be OK" % node.host_name
                        else:
                            node.startSlaves()
                    #except:
                    #    log.warning("some communication failed")
                else:
                    node.WakeOnLan()

                print "Checked: %s" % str(node)

            print "===== NODE LIST ====="
            for node in nodes.values():
                print str(node)
                time.sleep(0.1)
            print "====================="

            print "still %s minutes until shutdown" % str((stop_time - time.time()) / 60)
            time_to_sleep = time_between_checks - (time.time() - start_check_time)
            if time_to_sleep > 0 :
                print "waiting %s seconds..." % str(time_to_sleep)
                time.sleep( time_to_sleep )

    except KeyboardInterrupt:
        print "Exit requested, cleaning up..."

    except:
        print "Error occurred, cleaning up..."
        raise

    print "Time over..."
    # logout from the nodes
    for node in nodes.values():
        if node.isAlive():
            os.system('ssh -f %s killall python' % node.host_name)
        print "shutting down %s" % node.host_name
        if not node.checkDrone():
            if not node.connectToDrone():
                log.info("no drone available")
#                if not node.startDrone():
#                    log.info("the drone couldn't be started, removing node from list")
#                else:
#                    if not node.connectToDrone():
#                        log.info("could not connect to drone")
                continue

        
        node.stopSlaves( True )
        # stop the drone
        node.stopDrone()
            
if True: # ungracefull stop
    for node in nodes.values():
        if node.isAlive():
            os.system('ssh -f %s killall python' % node.host_name)
