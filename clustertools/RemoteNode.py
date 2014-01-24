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
    def __init__(self, hostname, mac, cluster_id, nb_slaves=2, do_wol=True):
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

        self.nb_slaves = nb_slaves
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
        remote_cmd = '~ppalmers/bin/python %s/clustertools/drone.py %s' % (self.base_dir, self.base_dir)
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
            log.info("set drone cluster id to: %s" % self.cluster_id)
            self.drone.setClusterId(self.cluster_id)
            log.info("set drone base dir to: %s" % self.base_dir)
            self.drone.setBaseDir(self.base_dir)
            
            log.info("current drone load: %s" % str(self.drone.getLoad()))
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
            if be_nice:
              self.drone.stopSlaves()
            else:
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
            if self.drone:
                self.load = self.drone.getLoad()
                self.slave_pids = self.drone.getSlavePids()
        except:
            log.info("error getting info for host %s" % self.host_name)
            
        s = []
        s += ["RemoteNode={"]
        s += ["Host = %s, " % (self.host_name)]
        s += ["last check: %s (load: %s), " % \
              (str(self.last_check), str(self.load))]
        s += ["slave pids: %s, " % \
              (str(self.slave_pids))]
        s += ["}"]
        return "".join(s)
