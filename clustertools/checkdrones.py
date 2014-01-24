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

class RemoteNode():
    """ assumes that the user can login to the host without a password
    """
    def __init__(self, hostname, mac, cluster_id, do_wol):
        tmp = commands.getstatusoutput("whoami")
        if tmp[0] == 0:
            user = tmp[1]
        else:
            print "error getting username"
            sys.exit(-1)

        self.host_name = hostname
        self.mac = mac
        self.load = None
        self.last_check = None

        self.test_port = 22
        self.alive = None
        self.max_load = 10.0

        self.cluster_id = cluster_id
        
        self.base_dir = os.path.realpath('.')

        self.nb_slaves = 2
        self.slave_pids = []
        self.do_WakeOnLan = do_wol

        self.loggedIn = False

        self.drone_location = ":SynthDrones.%s" % hostname
        self.drone = None

    def startDrone(self):
        log_dir = self.base_dir + '/logs/'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_file = log_dir + self.host_name + '.drone.log'
        remote_cmd = '/freeware/bin/python %s/drone.py %s' % (self.base_dir, self.base_dir)
        cmd = 'ssh -f %s "%s 2>&1 > %s"' % (self.host_name, remote_cmd, log_file)
        response = os.system(cmd)
        time.sleep(5)
        if response == 0:
            log.info("remote drone started successfully")
            return True
        else:
            log.info("remote drone startup failed")
            return False

    def stopDrone(self):
        if not self.isAlive():
            return True
        if not self.checkDrone():
            return True
        return self.drone.stopDrone()

    def connectToDrone(self):
        if self.drone != None:
            return False

        # no drone client available, create one
        try:
            # locate the NS
            log.info( 'Searching Naming Service...' )
            ns = Pyro.naming.NameServerLocator().getNS()

            log.info(' Naming Service found at %s, (%s:%d)' \
                     % (ns.URI.address,
                        (Pyro.protocol.getHostname(ns.URI.address) or '??'), \
                        ns.URI.port ) )
        except:
            log.info(" no NS found")
            return False

        # find the dispatcher URI
        try:
            uri = ns.resolve( self.drone_location )
            log.info("Drone found at %s" % self.drone_location )
        except:
            log.warning( 'Could not find drone at %s' % self.host_name )
            return False

        try:
            # get the dispatcher itself
            self.drone = uri.getProxy()
            log.info("Obtained drone proxy...")
        except:
            log.info( 'Could not get drone proxy %s' % uri )
            return False

        # now that we have a drone proxy, we can check whether it fails
        try:
            log.info("Drone at %s" % self.host_name)
            self.cluster_id = self.drone.getClusterId()
            log.info(" drone cluster id: %s" % self.cluster_id)
            self.base_dir = self.drone.getBaseDir()
            log.info(" drone base dir: %s" % self.base_dir)
            log.info(" current drone load: %s" % str(self.drone.getLoad()))
        except:
            log.info( 'Could not contact drone' )
            #raise
            return False
        return True

    def checkDrone(self):
        if self.drone == None:
            return False
        try:
            self.drone.ping()
        except:
            return False
        return True
    
    def isAlive(self):
        """ Is the host responding to requests?
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        try:
            s.connect(( self.host_name, self.test_port ))
            s.close()
            self.alive = True
            #print "Connection accepted, slave is alive"
            return True
        
        except socket.timeout:
            print "Socket timeout, slave is not alive (shut off)"
            self.alive = False
            return False
        
        except:
            print "Connection not accepted, slave is not alive (other OS)"
            self.alive = False
            return False
        
    def checkStatus(self):
        """ What is the host's status?
        """
        if self.isAlive():
            print "Check, %s is alive..." % self.host_name
            load = self.drone.getLoad()
            self.last_check = time.time()
        else:
            print "Skip check, %s not alive..." % self.host_name

    def startSlaves(self):
        if self.isAlive():
            if self.drone.getLoad() < self.max_load:
                log.info("Start slaves, %s is alive..." % self.host_name)
                self.drone.startSlaves(self.nb_slaves)
                pids = self.drone.getSlavePids()
                self.slave_pids = copy.copy(pids)
            else:
                log.info(" skip slave start, load too high")
        else:
            print "Skip start slave, %s not alive..." % self.host_name

    def checkSlaves(self):
        result = True
        if self.isAlive():
            if self.drone.checkSlaves():
                if len(self.drone.getSlavePids()) < self.nb_slaves:
                    print "no slaves running"
                    result = False
            else:
                result = False
        else:
            self.slave_pids = []
            print "%s not alive, slave is dead..." % self.host_name
            result = False
        return result
    
    def stopSlaves(self, be_nice = True):
        if self.isAlive():
            print "Stopping slaves, %s is alive..." % self.host_name
            self.drone.killSlaves()
        else:
            print "Skip stop slave, %s not alive..." % self.host_name
            
    def WakeOnLan( self ):
        if not self.do_WakeOnLan:
            return
    
        print "Sending Wake On Lan packet to %s" % self.host_name
        ethernet_address = self.mac
        # Construct a six-byte hardware address
        addr_byte = ethernet_address.split(':')
        hw_addr = struct.pack('BBBBBB', int(addr_byte[0], 16),
                              int(addr_byte[1], 16),
                              int(addr_byte[2], 16),
                              int(addr_byte[3], 16),
                              int(addr_byte[4], 16),
                              int(addr_byte[5], 16))
        # Build the Wake-On-LAN "Magic Packet"...
        msg = '\xff' * 6 + hw_addr * 16
        # ...and send it to the broadcast address using UDP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(msg, ('<broadcast>', 9))
        s.close()

    def __str__(self):
        try:
            if self.isAlive():
                if self.drone:
                    self.load = self.drone.getLoad()
                    self.slave_pids = self.drone.getSlavePids()
        except:
            log.info("error getting info for host %s" % self.host_name)
            
        s = []
        s += ["RemoteNode={"]
        s += ["Host = %15s, " % (self.host_name)]
        s += ["Alive = %5s, " % (self.alive)]
        s += ["Cluster ID = %15s, " % (self.cluster_id)]
        s += ["last check: %15s (load: %4s), " % \
              (str(self.last_check), str(self.load))]
        s += ["slave pids: %s, " % \
              (str(self.slave_pids))]
        s += ["}"]
        return "".join(s)

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

    if num_args >= 3 and eval(sys.argv[2]) != None:
        max_runtime_sec = eval(sys.argv[2])
    else:
        max_runtime_sec = 60*60*24*356
     
    # all in seconds
    time_between_checks = 120
    max_hosts = 99

    max_load = 2.0

    files = []
    files.append('clustertools/pcroom0091.list')
    files.append('clustertools/pcroom0092.list')
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
                    if node.drone == None:
                        node.connectToDrone()

                    if not node.checkDrone():
                        print "Skip drone..."
		    else:
                        # the host is alive
                        node.checkStatus()
                        node.checkSlaves()
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
#    for node in nodes.values():
#        if node.isAlive():
#            os.system('ssh -f %s killall python' % node.host_name)
#        print "shutting down %s" % node.host_name
#        if not node.checkDrone():
#            if not node.connectToDrone():
#                log.info("no drone available")
#                if not node.startDrone():
#                    log.info("the drone couldn't be started, removing node from list")
#                else:
#                    if not node.connectToDrone():
#                        log.info("could not connect to drone")
                #continue

        
#        node.stopSlaves( True )
        # stop the drone
#        node.stopDrone()
            
#if True: # ungracefull stop
#    for node in nodes.values():
#        if node.isAlive():
#            os.system('ssh -f %s killall python' % node.host_name)
