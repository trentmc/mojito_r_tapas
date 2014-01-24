#!/usr/bin/env python


from pyssh.sshcontroller import SSHController
import commands
import time
import socket
import os
import struct

socket.setdefaulttimeout(0.2)

class RemoteNode(SSHController):
    """ assumes that the user can login to the host without a password
    """
    def __init__(self, hostname, mac, cluster_id):
        tmp = commands.getstatusoutput("whoami")
        user = tmp[1]
        SSHController.__init__(self, hostname, user, '', ']$ ')

        self.mac = mac
        self.load = None
        self.last_check = None

        self.test_port = 22
        self.alive = None
        self.max_load = 10.0

        self.cluster_id = cluster_id
        
        self.base_dir = os.path.realpath('.')
        self.start_slave_cmd = 'nohup ' + self.base_dir + '/clustertools/startsim.sh ' + \
                               self.base_dir + ' ' + self.cluster_id + ' &'

        self.slave_pid = None
        
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
            uptime = self.run_command('uptime')
            try:
                load = uptime.split()[7]
                self.load = float(load[:-1])
                self.last_check = time.time()
            except:
                print "some error occurred getting the uptime, is %s a trusted host?" % self.host_name
                return False
        else:
            print "Skip check, %s not alive..." % self.host_name

    def startSlave(self):
        if self.isAlive():
            print "Start slave, %s is alive..." % self.host_name
            if tst.load < max_load:
                response = self.run_command(self.start_slave_cmd)
                print "Slave startup responded: " + response
                response = self.run_command('echo $!')
                self.slave_pid = response

                if self.slave_pid=='':
                    self.slave_pid = None
                    
                print " pid = %s" % self.slave_pid
        else:
            print "Skip start slave, %s not alive..." % self.host_name

    def checkSlave(self):
        if self.slave_pid == None:
            print "no slave running according to the node state"
            return False
        if self.isAlive():
            cmd = 'ps -p %s -o comm=' % str(self.slave_pid).strip()
            print cmd
            response = self.run_command(cmd)
            word = response.strip()
            
            if word[0:8] == 'startsim':
                print " slave process still alive"
                return True
            else:
                self.slave_pid = None
                print 'Check returned %s' % response
                return False
        else:
            self.slave_pid = None
            print "%s not alive, slave is dead..." % self.host_name

    def stopSlave(self, be_nice = True):
        if be_nice:
            if self.slave_pid == None:
                print "no slave running according to the node state"
                return False
            if self.isAlive():
                cmd = 'kill %s' % str(self.slave_pid)
                print cmd
                response = self.run_command(cmd)
                self.slave_pid = None
            else:
                self.slave_pid = None
                print "%s not alive, slave is dead..." % self.host_name
        else:
            print "stopping slaves at %s in a not so nice way..." % self.host_name
            self.run_command('killall -9 startsim.sh; killall -9 python; echo "ok"')
       
    def WakeOnLan( self ):
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
        s = []
        s += ["RemoteNode={"]
        s += ["Host = %s, " % (self.host_name)]
        s += ["last check: %s (load: %s), " % \
              (str(self.last_check), str(self.load))]
        s += ["slave pid: %s (load: %s), " % \
              (str(self.slave_pid), str(self.load))]
        s += ["}"]
        return "".join(s)

file = 'clustertools/pcroom0254.list'

fid = open(file)
hostlines = fid.readlines()
fid.close()

max_hosts = 50
cnt = 0
hosts = {}
for l in hostlines:
    if l[0] != '#' and l != "\n":
        try:
            tmp = l.split()
            hosts[tmp[0]] = (tmp[0], tmp[1])
        except:
            print "bogus line: %s" % l
    cnt += 1
    if cnt > max_hosts:
        break

print hosts

max_load = 10.0
cluster_id = 'ppcluster1'

stop_time = time.time() + 60 * 60 * 9
time_between_checks = 10
nodes = {}
for (mac, h) in hosts.values():
    tst = RemoteNode(h, mac, cluster_id)
    nodes[h+'_1'] = RemoteNode(h, mac, cluster_id)
    #nodes[h+'_2'] = RemoteNode(h, mac, cluster_id)
    
## try:
##     while time.time() < stop_time:
##         for node in nodes.values():
##             if node.isAlive():
##                 # the host is alive
##                 node.login()

##                 if node.checkSlave():
##                     print "Slave at %s seems to be OK" % node.host_name
##                 else:
##                     if node.checkSlave():
##                         print "Slave at %s seems to be OK on second sight" % node.host_name
##                     else:
##                         node.checkStatus()
##                         if node.load < max_load:
##                             node.startSlave()

##                 node.logout()
##             else:
##                 node.WakeOnLan()

##             print "Checked: %s" % str(node)

##         time.sleep( time_between_checks )
        
## except KeyboardInterrupt:
##     print "Exit requested, cleaning up..."

## except:
##     print "Error occurred, cleaning up..."

print "Time over..."
# logout from the nodes
for n in nodes.values():
    if n.isAlive():
        n.login()
        n.stopSlave(False)
        n.logout()
