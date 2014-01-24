"""A SSH Interface class.

An interface to ssh on posix systems, and plink (part of the Putty
suite) on Win32 systems.

By Rasjid Wilcox.
Copyright (c) 2002.

Version: 0.2
Last modified 4 September 2002.

Drawing on ideas from work by Julian Schaefer-Jasinski, Guido's telnetlib and
version 0.1 of pyssh (http://pyssh.sourceforge.net) by Chuck Esterbrook.

Licenced under a Python 2.2 style license.  See License.txt.
"""

DEBUG_LEVEL = 0

import os, getpass
import signal    # should cause all KeyboardInterrupts to go to the main thread
                 # try for Linux, does not seem to be try under Cygwin
import nbpipe
import time

# Constants
SSH_PORT=22
SSH_PATH=''

CTRL_C=chr(3)

READ_LAZY=0
READ_SOME=1
READ_ALL=2

# set the path to ssh / plink, and chose the popen2 funciton to use
if os.name=='posix':
    import fssa    # we can look for ssh-agent on posix
                   # XXX Can we on Win32/others?
    import ptyext  # if my patch gets accepted, change this to check for a
                   # sufficiently high version of python, and assign ptyext=pty
                   # if sufficient.
    sshpopen2=ptyext.popen2
    CLOSE_STR='~.'
    tp=os.popen('/usr/bin/which ssh')
    SSH_PATH=tp.read().strip()
    try:
        tp.close()
    except IOError:
        # probably no child process
        pass
    if SSH_PATH == '':
        tp=os.popen('command -v ssh')  # works in bash, ash etc, not csh etc.
        SSH_PATH=tp.read().strip()
        tp.close()
    if SSH_PATH == '':
        check = ['/usr/bin/ssh', '/usr/local/bin/ssh', '/bin/ssh']
        for item in check:
            if os.path.isfile(item):
                SSH_PATH=item
                break
    PORT_STR='-p '
else:
    sshpopen2=os.popen2
    CLOSE_STR=CTRL_C        # FIX-ME: This does not work.
                            # I think I need to implement a 'kill' component
                            # to the close function using win32api.
    SSH_PATH=''
    PORT_STR='-P '

class mysshError(Exception):
    """Error class for myssh."""
    pass

# Helper functions
def _prompt(prompt):
    """Print the message as the prompt for input.
    Return the text entered."""
    noecho = (prompt.lower().find('password:') >= 0) or \
        (prompt.lower().find('passphrase:') >=0)
    print """User input required for ssh connection.
    (Type Ctrl-C to abort connection.)"""
    abort = 0
    try:
        if noecho:
            response = getpass.getpass(prompt)
        else:
            response = raw_input(prompt)
    except KeyboardInterrupt:
        response = ''
        abort = 1
    return response, abort

class Ssh:
    """A SSH connection class."""
    def __init__(self, username=None, host='localhost', port=None):
        """Constructor.  This does not try to connect."""
        self.debuglevel = DEBUG_LEVEL
        self.sshpath = SSH_PATH
        self.username = username
        self.host = host
        self.port = port
        self.isopen = 0
        self.sshpid = 0  # perhaps merge this with isopen
        self.old_handler = signal.getsignal(signal.SIGCHLD)
        sig_handler = signal.signal(signal.SIGCHLD, self.sig_handler)
        
    def __del__(self):
        """Destructor -- close the connection."""
        if self.isopen:
            self.close()
    
    def sig_handler(self, signum, stack):
        """ Handle SIGCHLD signal """
        if signum == signal.SIGCHLD:
            try:
                os.waitpid(self.sshpid, 0)
            except:
                pass
        if self.old_handler != signal.SIG_DFL:
            self.old_handler(signum, stack)

    def attach_agent(self, key=None):
        if os.name != 'posix':
            # only posix support at this time
            return
        if 'SSH_AUTH_SOCK' not in os.environ.keys():
            fssa.fssa(key)

    def set_debuglevel(self, debuglevel):
        """Set the debug level."""
        self.debuglevel = debuglevel
        
    def set_sshpath(self, sshpath):
        """Set the ssh path."""
        self.sshpath=sshpath
    
    # Low level functions
    def open(self, cmd=None):
        """Opens a ssh connection.
        
        Raises an mysshError if myssh.sshpath is not a file.
        Raises an error if attempting to open an already open connection.
        """
        self.attach_agent()
        if not os.path.isfile(self.sshpath):
            raise mysshError, \
            "Path to ssh or plink is not defined or invalid.\nsshpath='%s'" \
             % self.sshpath
        if self.isopen:
            raise mysshError, "Connection already open."
        sshargs = ''
        if self.sshpath.lower().find('plink') != -1:
            sshargs = '-ssh '
        if self.port and self.port != '':
            sshargs += PORT_STR + self.port + ' '
        if self.username and self.username !='':
            sshargs += self.username + '@'
        sshargs += self.host
        if cmd:
            sshargs += ' ' + cmd
        if self.debuglevel:
            print ">> Running %s %s." % (self.sshpath, sshargs)
        # temporary workaround until I get pid's working under win32
        if os.name == 'posix':
            self.sshin, self.sshoutblocking, self.sshpid = \
                                sshpopen2(self.sshpath + ' ' + sshargs)
        else:
            self.sshin, self.sshoutblocking = \
                                sshpopen2(self.sshpath + ' ' + sshargs)
        self.sshout = nbpipe.nbpipe(self.sshoutblocking)
        self.isopen = 1
        if self.debuglevel:
            print ">> ssh pid is %s." % self.sshpid
        
    def close(self, addnewline=1):
        """Close the ssh connection by closing the input and output pipes.
        Returns the closing messages.
        
        On Posix systems, by default it adds a newline before sending the
        disconnect escape sequence.   Turn this off by setting addnewline=0.
        """
        if os.name == 'posix':
            try:
                if addnewline:
                    self.write('\n')
                self.write(CLOSE_STR)
            except (OSError, IOError, mysshError):
                pass
        output = self.read_lazy()
        try:
            self.sshin.close()
            self.sshoutblocking.close()
        except:
            pass
        if os.name == 'posix':
            try:
                os.kill(self.sshpid, signal.SIGHUP)
            except:
                pass
        self.isopen = 0
        if self.debuglevel:
            print ">> Connection closed."
        return output
        
    def write(self, text):
        """Send text to the ssh process."""
        # May block?? Probably not in practice, as ssh has a large input buffer.
        if self.debuglevel:
            print ">> Sending %s" % text
        if self.isopen:
            while len(text):
                numtaken = os.write(self.sshin.fileno(),text)
                if self.debuglevel:
                    print ">> %s characters taken" % numtaken
                text = text[numtaken:]
        else:
            raise mysshError, "Attempted to write to closed connection."
    
    # There is a question about what to do with connections closed by the other
    # end.  Should write and read check for this, and force proper close?
    def read_very_lazy(self):
        """Very lazy read from sshout. Just reads from text already queued."""
        return self.sshout.read_very_lazy()
    
    def read_lazy(self):
        """Lazy read from sshout.  Waits a little, but does not block."""
        return self.sshout.read_lazy()
    
    def read_some(self):
        """Always read at least one block, unless the connection is closed.
        My block."""
        if self.isopen:
            return self.sshout.read_some()
        else:
            return self.sshout.read_very_lazy()    
        
    def read_all(self):
        """Reads until end of file hit.  May block."""
        if self.isopen:
            return self.sshout.read_all()
        else:
            return self.sshout.read_very_lazy()
        
    # High level funcitons
    def login(self, logintext='Last login:', prompt_callback=_prompt):
        """Logs in to the ssh host.  Checks for standard prompts, and calls
        the function passed as promptcb to process them.
        Returns the login banner, or 'None' if login process aborted.
        """
        self.open()
        banner = self.read_some()
        if self.debuglevel:
            print ">> 1st banner read is: %s" % banner
        while banner.find(logintext) == -1:
            response, abort = prompt_callback(banner)
            if abort:
                return self.close()
            self.write(response + '\n')
            banner = self.read_some()
        return banner
    
    def logout(self):
        """Logs out the session."""
        self.close()
        
    def sendcmd(self, cmd, readtype=READ_SOME):
        """Sends the command 'cmd' over the ssh connection, and returns the
        result.  By default it uses read_some, which may block.
        """
        if cmd[-1] != '\n':
            cmd += '\n'
        self.write(cmd)
        if readtype == READ_ALL:
            return self.read_all()
        elif readtype == READ_LAZY:
            return self.read_lazy()
        else:
            return self.read_some()
    
def test():
    """Test routine for myssh.
    
    Usage: python myssh.py [-d] [-sshp path-to-ssh] [username@host | host] [port]
    
    Default host is localhost, default port is 22.
    """
    import sys
    debug = 0
    if sys.argv[1:] and sys.argv[1] == '-d':
        debug = 1
        del sys.argv[1]
    testsshpath = SSH_PATH
    if sys.argv[1:] and sys.argv[1] == '-sshp':
        testsshpath = sys.argv[2]
        del sys.argv[1]
        del sys.argv[1]
    testusername = None
    testhost = 'localhost'
    testport = '22'
    if sys.argv[1:]:
        testhost = sys.argv[1]
        if testhost.find('@') != -1:
            testusername, testhost = testhost.split('@')
    if sys.argv[2:]:
        testport = sys.argv[2]
        
    testcon = Ssh(testusername, testhost, testport)
    testcon.set_debuglevel(debug)
    testcon.set_sshpath(testsshpath)
    testcon.login()
    
    cmd = None
    while (cmd != 'exit') and testcon.isopen:
        cmd = raw_input("Enter command to send: ")
        print testcon.sendcmd(cmd)
    testcon.close()

if __name__ == '__main__':
    test()
