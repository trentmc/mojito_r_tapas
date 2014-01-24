#!/usr/bin/env python

import sys
import os
import time
import commands
import Pyro.core
import Pyro.naming
from Pyro.errors import NamingError

import logging
INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING

log = logging.getLogger('drone')

class Drone(Pyro.core.ObjBase):
    """
    """
    # CHECK: how about the byref passing of arguments?
    def __init__(self):
        """
        """
        #preconditions
        Pyro.core.ObjBase.__init__(self)
        
        self.slave_pids = []

        self.base_dir = ''
        self.cluster_id = ''

        self.last_load = None

        self.hostname = commands.getstatusoutput("hostname | gawk -F'.' '{print $1}'")[1]
        self.getLoad()

    def setBaseDir(self, x):
        self.base_dir = x
    def getBaseDir(self):
        return self.base_dir

    def setClusterId(self, id):
        self.cluster_id = id
    def getClusterId(self):
        return self.cluster_id
    def getSlavePids(self):
        return self.slave_pids

    def getLoad(self):
        uptime = commands.getstatusoutput('uptime')
        if uptime[0] == 0:
            uptime = uptime[1].split()
            idx = 0
            found = False
            for l in uptime:
                idx += 1
                if l == 'average:':
                    found = True
                    break
            if found:
                load = uptime[idx]
                self.last_load = float(load[:-1])
                return self.last_load
            else:
                return None
        else:
            return None

    def startSlaves(self, nb_slaves):
        log.info("starting %s slaves" % nb_slaves)
        nb_slaves = int(nb_slaves)
        if self.cluster_id == '':
            log.info("no cluster ID yet")
            return False

        self.reattachSlaves()

        nb_slaves_to_start = nb_slaves - len(self.slave_pids)
        if nb_slaves_to_start <= 0:
            print "already %s slaves running" % str(nb_slaves)
            return
        
        log_dir = self.base_dir + '/logs/'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        for i in range(nb_slaves_to_start):
            log_name = self.hostname + '-' + str(time.time())  + '.log'
            cmd = 'nohup bash ' + self.base_dir + \
            '/clustertools/startsim.sh %s %s' % (self.base_dir, self.cluster_id) \
                  + ' &> /dev/null &' # + log_dir + '/' + log_name + ' &'
            print cmd
            resp = os.system(cmd)
            time.sleep(2) # wait a bit between startup of slaves
            if resp == 0:
                pass
            else:
                print "starting slave %d failed" % i
                
        time.sleep(1)
        self.reattachSlaves()
        pass
        
    def killSlaves(self):
        self.reattachSlaves()
        for pid in self.slave_pids:
            cmd = 'kill -9 ' + str(pid)
            response = commands.getstatusoutput(cmd)
            if response[0] == 0:
                pass
            else:
                print "failed to kill slave PID=%s" % str(pid)

    def stopSlaves(self):
        self.reattachSlaves()
        for pid in self.slave_pids:
            cmd = 'kill -SIGTERM ' + str(pid)
            response = commands.getstatusoutput(cmd)
            if response[0] == 0:
                pass
            else:
                print "failed to SIGTERM slave PID=%s" % str(pid)

    def checkSlaves(self):
        self.reattachSlaves()
        for pid in self.slave_pids:
            cmd = 'ps -p %s -o comm=' % str(pid).strip()
            response = commands.getstatusoutput(cmd)
            if response[0] == 0:
                pass
            else:
                return False
        return True

    def reattachSlaves(self):
        username = commands.getstatusoutput('whoami')
        
        if username[0] == 0:
            if self.cluster_id == '':
                log.info("no cluster ID yet")
                return False

            cmd = 'ps -fC python | grep slave.py | grep ' + self.cluster_id
            response = commands.getstatusoutput(cmd)
            self.slave_pids = []

            if response[0] == 256:
                #no slaves running
                return True
            
            if response[0] == 0:
                
                lines = response[1].split('\n')
                for line in lines:
                    #print line
                    sline = line.split()
                    if username[1] == sline[0]:
                        self.slave_pids.append(sline[1])
                        #print self.slave_pids
                return True
            else:
                print "bad response for ps %s" % response[1]
        else:
            print "bad response for whoami %s" % username[1]
        return False
    
    def stopDrone(self):
        self.daemon.shutdown()
        try:
            self.ns.unregister( ":SynthDrones.%s" % ( self.hostname ) )
        except NamingError:
            log.error("could not unregister drone from nameserver")
            pass
        
    def ping(self):
        return True
    
    def __str__(self):
        s  = ['Drone={\n']
        s += ['hostname: %s\n' % str(self.hostname) ]
        s += ['base dir: %s\n' % str(self.base_dir) ]
        s += ['cluster id: %s, ' % str(self.cluster_id) ]
        s += ['last load: %s\n' % str(self.last_load)]
        s += ['slave pids: %s\n' % str(self.slave_pids)]
        s += ['}']

        return "".join(s)

# The standard I/O file descriptors are redirected to /dev/null by default.
if (hasattr(os, "devnull")):
   REDIRECT_TO = os.devnull
else:
   REDIRECT_TO = "/dev/null"

# Default daemon parameters.
# File mode creation mask of the daemon.
UMASK = 0

# Default working directory for the daemon.
WORKDIR = "/"

def createDaemon():
   """Detach a process from the controlling terminal and run it in the
   background as a daemon.
   """

   try:
      # Fork a child process so the parent can exit.  This returns control to
      # the command-line or shell.  It also guarantees that the child will not
      # be a process group leader, since the child receives a new process ID
      # and inherits the parent's process group ID.  This step is required
      # to insure that the next call to os.setsid is successful.
      pid = os.fork()
   except OSError, e:
      raise Exception, "%s [%d]" % (e.strerror, e.errno)

   if (pid == 0):	# The first child.
      # To become the session leader of this new session and the process group
      # leader of the new process group, we call os.setsid().  The process is
      # also guaranteed not to have a controlling terminal.
      os.setsid()

      # Is ignoring SIGHUP necessary?
      #
      # It's often suggested that the SIGHUP signal should be ignored before
      # the second fork to avoid premature termination of the process.  The
      # reason is that when the first child terminates, all processes, e.g.
      # the second child, in the orphaned group will be sent a SIGHUP.
      #
      # "However, as part of the session management system, there are exactly
      # two cases where SIGHUP is sent on the death of a process:
      #
      #   1) When the process that dies is the session leader of a session that
      #      is attached to a terminal device, SIGHUP is sent to all processes
      #      in the foreground process group of that terminal device.
      #   2) When the death of a process causes a process group to become
      #      orphaned, and one or more processes in the orphaned group are
      #      stopped, then SIGHUP and SIGCONT are sent to all members of the
      #      orphaned group." [2]
      #
      # The first case can be ignored since the child is guaranteed not to have
      # a controlling terminal.  The second case isn't so easy to dismiss.
      # The process group is orphaned when the first child terminates and
      # POSIX.1 requires that every STOPPED process in an orphaned process
      # group be sent a SIGHUP signal followed by a SIGCONT signal.  Since the
      # second child is not STOPPED though, we can safely forego ignoring the
      # SIGHUP signal.  In any case, there are no ill-effects if it is ignored.
      #
      # import signal           # Set handlers for asynchronous events.
      # signal.signal(signal.SIGHUP, signal.SIG_IGN)

      try:
         # Fork a second child and exit immediately to prevent zombies.  This
         # causes the second child process to be orphaned, making the init
         # process responsible for its cleanup.  And, since the first child is
         # a session leader without a controlling terminal, it's possible for
         # it to acquire one by opening a terminal in the future (System V-
         # based systems).  This second fork guarantees that the child is no
         # longer a session leader, preventing the daemon from ever acquiring
         # a controlling terminal.
         pid = os.fork()	# Fork a second child.
      except OSError, e:
         raise Exception, "%s [%d]" % (e.strerror, e.errno)


      if (pid == 0):	# The second child.
         # Since the current working directory may be a mounted filesystem, we
         # avoid the issue of not being able to unmount the filesystem at
         # shutdown time by changing it to the root directory.
         os.chdir(WORKDIR)
         # We probably don't want the file mode creation mask inherited from
         # the parent, so we give the child complete control over permissions.
         os.umask(UMASK)
      else:
         # exit() or _exit()?  See below.
         os._exit(0)	# Exit parent (the first child) of the second child.
   else:
      # exit() or _exit()?
      # _exit is like exit(), but it doesn't call any functions registered
      # with atexit (and on_exit) or any registered signal handlers.  It also
      # closes any open file descriptors.  Using exit() may cause all stdio
      # streams to be flushed twice and any temporary files may be unexpectedly
      # removed.  It's therefore recommended that child branches of a fork()
      # and the parent branch(es) of a daemon use _exit().
      os._exit(0)	# Exit parent of the first child.

   # Close all open file descriptors.  This prevents the child from keeping
   # open any file descriptors inherited from the parent.  There is a variety
   # of methods to accomplish this task.  Three are listed below.
   #
   # Try the system configuration variable, SC_OPEN_MAX, to obtain the maximum
   # number of open file descriptors to close.  If it doesn't exists, use
   # the default value (configurable).
   #
   # try:
   #    maxfd = os.sysconf("SC_OPEN_MAX")
   # except (AttributeError, ValueError):
   #    maxfd = MAXFD
   #
   # OR
   #
   # if (os.sysconf_names.has_key("SC_OPEN_MAX")):
   #    maxfd = os.sysconf("SC_OPEN_MAX")
   # else:
   #    maxfd = MAXFD
   #
   # OR
   #
   # Use the getrlimit method to retrieve the maximum file descriptor number
   # that can be opened by this process.  If there is not limit on the
   # resource, use the default value.
   #
   import resource		# Resource usage information.
   maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
   if (maxfd == resource.RLIM_INFINITY):
      maxfd = MAXFD
  
   # Iterate through and close all file descriptors.
   for fd in range(0, maxfd):
      try:
         os.close(fd)
      except OSError:	# ERROR, fd wasn't open to begin with (ignored)
         pass

   # Redirect the standard I/O file descriptors to the specified file.  Since
   # the daemon has no controlling terminal, most daemons redirect stdin,
   # stdout, and stderr to /dev/null.  This is done to prevent side-effects
   # from reads and writes to the standard I/O file descriptors.

   # This call to open is guaranteed to return the lowest file descriptor,
   # which will be 0 (stdin), since it was closed above.
   os.open(REDIRECT_TO, os.O_RDWR)	# standard input (0)

   # Duplicate standard input to standard output and standard error.
   os.dup2(0, 1)			# standard output (1)
   os.dup2(0, 2)			# standard error (2)

   return(0)

if __name__== '__main__':
    #set help message
    help = """
Usage: drone BASE_DIR
""" 
    num_args = len(sys.argv)
    if num_args not in [2]:
        print help
        sys.exit(0)
        
    #set up logging
    logging.basicConfig()

    logging.getLogger('drone').setLevel(DEBUG)

    cmd = 'ps -fC python | grep drone.py'
    response = commands.getstatusoutput(cmd)

    if response[0] == 0:
        lines = response[1].split('\n')
        if len(lines)>1:
            #we already have a drone
            #print response
            log.info("drone already running")
            sys.exit(0)

    d = Drone()
    d.setBaseDir(WORKDIR)

    WORKDIR = sys.argv[1]
    d.setBaseDir(WORKDIR)
    REDIRECT_TO = WORKDIR+"/logs/" + d.hostname + ".drone.log"
    print REDIRECT_TO
    
    #daemonize
    #retCode = createDaemon()
    
    # we need a server daemon
    Pyro.config.PYRO_MULTITHREADED = 0
    Pyro.core.initServer()
    daemon = Pyro.core.Daemon()
    
    # Pyro dispatcher
    try:
        ns = Pyro.naming.NameServerLocator().getNS()
    except NamingError:
        log.info( "No nameserver found. Please start a nameserver..." )
        sys.exit(-1)

    daemon.useNameServer(ns)

    try:
        ns.createGroup( ":SynthDrones" )
    except NamingError:
        log.info( "SynthDrones group already exists, unregistering..." )

    my_name = '%s' % d.hostname
    my_location = ":SynthDrones.%s" % ( my_name )

    try:
        ns.unregister( my_location )
    except NamingError:
        pass

    uri = daemon.connect( d, my_location )
    log.info( "Drone is listening at %s ..." % my_location )

    try:
        d.daemon = daemon
        d.ns = ns
        daemon.requestLoop()
    except KeyboardInterrupt:
        daemon.shutdown( True )

    log.info( "Drone is exiting..." )
