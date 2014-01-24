#!/usr/bin/env python


from pyssh.sshcontroller import SSHController
import commands
import time
import socket
import os
import struct

socket.setdefaulttimeout(1)

class RemoteNode(SSHController):
    """ assumes that the user can login to the host without a password
    """
    def __init__(self, hostname, mac, cluster_id, do_wol):
        tmp = commands.getstatusoutput("whoami")
        user = tmp[1]
        SSHController.__init__(self, hostname, user, '', ']$ ')

        self.keepalive = SSHController(hostname, user, '', ']$ ')
        
        self.mac = mac
        self.load = None
        self.last_check = None

        self.test_port = 22
        self.alive = None
        self.max_load = 10.0

        self.cluster_id = cluster_id
        
        self.base_dir = os.path.realpath('.')
        self.start_slave_cmd = 'nohup ' + self.base_dir + '/clustertools/startsim.sh ' + \
                               self.base_dir + ' ' + self.cluster_id + ''

        self.nb_slaves = 2
        self.slave_pids = []
        self.do_WakeOnLan = do_wol

        self.loggedIn = False
        
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
                uptime = uptime.split()
                idx = 0
                found = False
                for l in uptime:
                    idx += 1
                    if l == 'average:':
                        found = True
                        break
                if found:
                    load = uptime[idx]
                    self.load = float(load[:-1])
                    #print "set load to %s" % str(self.load)
                    
                self.last_check = time.time()
            except:
                print "some error occurred getting the uptime, is %s a trusted host?" % self.host_name
                return False
        else:
            print "Skip check, %s not alive..." % self.host_name

    def startSlaves(self):
        if self.isAlive():
            print "Start slave, %s is alive..." % self.host_name
            if self.load < max_load:
                for i in range(self.nb_slaves):
                    print " starting slave %d" % i
                    response = self.run_command(self.start_slave_cmd)
                    #print "Slave startup responded: " + response
                    #read
                    time.sleep(0.1)
                    fid = open(self.base_dir + '/last_slave_pid')
                    last_slave_pid = fid.read()
                    fid.close()

                    tmp = last_slave_pid.split()
                    
                    if len(tmp) < 2:
                        print "  start failed, last_slave_pid error"
                    else:
                        
                        if response.find('Exit') == -1:
                            self.slave_pids.append(tmp[1])
                            print "  pid = %s" % tmp[1]
                        else:
                            print "  start failed: %s" % response
        else:
            print "Skip start slave, %s not alive..." % self.host_name

    def checkSlaves(self):
        if len(self.slave_pids) == 0:
            print "no slave running according to the node state"
            return False
        result = True
        if self.isAlive():
            for slave_pid in self.slave_pids:
                print " check slave PID=%s" % str(slave_pid)
                cmd = 'ps -p %s -o comm=' % str(slave_pid).strip()
                #print cmd
                response = self.run_command(cmd)
                word = response.strip()
            
                if word[0:6] == 'python':
                    print "  slave process still alive"
                else:
                    self.slave_pids.remove(slave_pid)
                    print '  check returned %s' % response
                    result = False
        else:
            self.slave_pids = []
            print "%s not alive, slave is dead..." % self.host_name
            result = False
        return result
    
    def stopSlaves(self, be_nice = True):
        if be_nice:
            if len(self.slave_pids) == 0:
                print "no slave running according to the node state"
                return False
            if self.isAlive():
                for slave_pid in self.slave_pids:
                    cmd = ' kill -9 %s' % str(slave_pid)
                    print cmd
                    response = self.run_command(cmd)
                self.slave_pids = []
            else:
                self.slave_pids = []
                print "%s not alive, slave is dead..." % self.host_name
        else:
            print "stopping slaves at %s in a not so nice way..." % self.host_name
            self.run_command('killall -9 startsim.sh; killall -9 python; echo "ok"')
            self.slave_pids = []
            
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

    def refreshLogin(self):
        if self.isAlive():
            if self.ssh and self.ssh.isopen:
                pass
            else:
                print " login to %s" % self.host_name
                import pdb;pdb.set_trace()
                self.login()

    def __str__(self):
        s = []
        s += ["RemoteNode={"]
        s += ["Host = %s, " % (self.host_name)]
        s += ["last check: %s (load: %s), " % \
              (str(self.last_check), str(self.load))]
        s += ["slave pids: %s, " % \
              (str(self.slave_pids))]
        s += ["}"]
        return "".join(s)

# all in seconds
max_runtime_sec = 60 * 60 * 11
time_between_checks = 120
max_hosts = 99

max_load = 2.0
cluster_id = 'ppcluster1'

files = []
files.append('clustertools/pcroom0091.list')
files.append('clustertools/pcroom0092.list')
#files.append('clustertools/pcroom0254.list')
#files.append('clustertools/pcroom9156.list')
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
    nodes[h+'_1'] = RemoteNode(h, mac, cluster_id, do_wakeup)
    
try:
    while time.time() < stop_time:
        start_check_time = time.time()
        for node in nodes.values():
            
            if node.isAlive():
                node.login()
                # the host is alive
                node.checkStatus()

                if node.checkSlaves():
                    print "Slave at %s seems to be OK" % node.host_name
                else:
                    if node.checkSlaves():
                        print "Slave at %s seems to be OK on second sight" % node.host_name
                    else:
                        if node.load < max_load:
                            node.startSlaves()
                node.logout()
            else:
                node.WakeOnLan()

            print "Checked: %s" % str(node)

        print "===== NODE LIST ====="
        for node in nodes.values():
            print str(node)
        print "====================="

        print "still %s minutes untill shutdown" % str((stop_time - time.time()) / 60)
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
for n in nodes.values():
    print "shutting down %s" % n.host_name
    if n.isAlive():
        n.login()
        n.stopSlaves( True )
        n.logout()
