#############################################################################
#
#	$Id: protocol.py,v 2.94.2.10 2008/05/24 13:30:47 irmen Exp $
#	Pyro Protocol Adapters
#
#	This is part of "Pyro" - Python Remote Objects
#	which is (c) Irmen de Jong - irmen@users.sourceforge.net
#
#############################################################################

import socket, struct, os, time, sys, hmac, types, random, errno, select
import imp, marshal, new, __builtin__
try:
	import hashlib
	md5=hashlib.md5
except ImportError:
	import md5
	md5=md5.md5
import Pyro.constants, Pyro.util

from errors import *
from errors import _InternalNoModuleError
pickle = Pyro.util.getPickle()
Log = Pyro.util.Log

	
if Pyro.util.supports_multithreading():
	from threading import Thread,currentThread
	_has_threading = 1
else:
	_has_threading = 0

if Pyro.util.supports_compression():
	import zlib
	_has_compression = 1
else:
	_has_compression = 0


#------ Get the hostname (possibly of other machines) (returns None on error)
def getHostname(ip=None):
	try:
		if ip:
			(hn,alias,ips) = socket.gethostbyaddr(ip)
			return hn
		else:
			return socket.gethostname()
	except socket.error:
		return None

#------ Get IP address (return None on error)
def getIPAddress(host=None):
	try:
		return socket.gethostbyname(host or getHostname())
	except socket.error:
		return None

	
#------ Socket helper functions for sending and receiving data correctly.


# process optional timeout on socket.
# XXX replace this with python's native socket timeouts. M2Crypto (SSL) needs special care...
def _sock_timeout_send(sock, timeout):
	ssl_select_okay=not hasattr(sock,'pending') or sock.pending()==0
	if timeout and ssl_select_okay:
		r,w,e=safe_select([],[sock],[],timeout)
		if not w:
			raise TimeoutError('connection timeout sending')

def _sock_timeout_recv(sock, timeout):
	ssl_select_okay=not hasattr(sock,'pending') or sock.pending()==0
	if timeout and ssl_select_okay:
		r,w,e=safe_select([sock],[],[],timeout)
		if not r:
			raise TimeoutError('connection timeout receiving')

# Receive a precise number of bytes from a socket. Raises the
# ConnectionClosedError if  that number of bytes was not available.
# (the connection has probably been closed then).
# Never will this function return an empty message (if size>0).
# We need this because 'recv' isn't guaranteed to return all desired
# bytes in one call, for instance, when network load is high.
# Use a list of all chunks and join at the end: faster!
# Handle EINTR states (interrupted system call) by just retrying.
def sock_recvmsg(sock, size, timeout=0):
	while 1:
		if hasattr(sock,'pending'):			# SSL socks have pending...
			# when using SSL, other exceptions occur.
			# need to handle EINTR as well?
			from M2Crypto.SSL import SSLError
			try:
				return _recv_msg(sock,size,timeout)
			except SSLError:
				raise ConnectionClosedError('connection lost')
		else:
			try:
				return _recv_msg(sock,size,timeout)
			except socket.error,x:
				if x.args[0] == errno.EINTR or (hasattr(errno, 'WSAEINTR') and x.args[0] == errno.WSAEINTR):
					# interrupted system call, just retry
					continue
				raise ConnectionClosedError('connection lost: %s' % x)

# select the optimal recv() implementation
if hasattr(socket.socket, "recvall"):   # Irmen's custom socket module extension, see Python patch #1103213
	def _recv_msg(sock,size,timeout):
		_sock_timeout_recv(sock,timeout)
		chunk=sock.recvall(size)
		if len(chunk)!=size:
			err=ConnectionClosedError('connection lost')
			err.partialMsg=chunk    # store the message that was received until now
			raise err
		return chunk
elif hasattr(socket,"MSG_WAITALL") and not Pyro.config.PYRO_BROKEN_MSGWAITALL:
	def _recv_msg(sock,size,timeout):
		_sock_timeout_recv(sock,timeout)
		try:		
			chunk=sock.recv(size, socket.MSG_WAITALL)   # receive all data in one call
		except TypeError:
			# XXX This is caused by a bug in the M2Crypto API, it doesn't like the MSG_WAITALL parameter
			# Once the bug is fixed, the compat function can be rolled back into the _recv_msg below.
			return __recv_msg_compat(sock,size,timeout)
		else:
			if len(chunk)!=size:
				err=ConnectionClosedError('connection lost')
				err.partialMsg=chunk  # store the message that was received until now
				raise err
			return chunk
else:
	def _recv_msg(sock,size,timeout):
		_sock_timeout_recv(sock, timeout)
		return __recv_msg_compat(sock,size,timeout)

def __recv_msg_compat(sock,size,timeout):   # compatibility implementation for non-MSG_WAITALL / M2Crypto
	msglen=0
	msglist=[]
	# Receive chunks of max. 60kb size:
	# (rather arbitrary limit, but it avoids memory/buffer problems on certain OSes -- VAX/VMS, Windows)
	while msglen<size:
		chunk=sock.recv(min(60000,size-msglen))
		if not chunk:
			err = ConnectionClosedError('connection lost')
			err.partialMsg=''.join(msglist)    # store the message that was received until now
			raise err
		msglist.append(chunk)
		msglen+=len(chunk)
	return ''.join(msglist)


# Send a message over a socket. Raises ConnectionClosedError if the msg
# couldn't be sent (the connection has probably been lost then).
# We need this because 'send' isn't guaranteed to send all desired
# bytes in one call, for instance, when network load is high.

def sock_sendmsg(sock,msg,timeout=0):
	try:
		_sock_timeout_send(sock,timeout)
		sock.sendall(msg)
	except socket.error:
		raise ConnectionClosedError('connection lost')


# set socket option to try to re-use a server port if possible
def set_reuse_addr(sock):
	if os.name not in ('nt','dos','ce') and sys.platform!='cygwin':
		# only do this on a non-windows platform. Windows screws things up with REUSEADDR...
		try:
			sock.setsockopt ( socket.SOL_SOCKET, socket.SO_REUSEADDR,
				sock.getsockopt (socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1)
		except:
			pass

# set socket option to enable timeout checking for server sockets.
def set_sock_keepalive(sock):
	if Pyro.config.PYRO_SOCK_KEEPALIVE:
		try:
			sock.setsockopt ( socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1 )
		except:
			Pyro.config.PYRO_SOCK_KEEPALIVE=0    # it didn't work--disable keepalives.


#------ PYRO: adapter (default Pyro wire protocol)
#------ This adapter is for protocol version 4 ONLY
# Future adapters could be downwards compatible and more flexible.

PFLG_COMPRESSED = 0x01		# protocol flag: compressed body
PFLG_CHECKSUM =   0x02		# protocol flag: checksum body
PFLG_XMLPICKLE_GNOSIS =  0x04		# protocol flag: used xml pickling (Gnosis)


_agentImportLock=Pyro.util.getLockObject()
_remoteImportRLock=Pyro.util.getRLockObject()

class PYROAdapter:
	headerFmt = '!4sHHlHl'	# version 4 header (id, ver, hsiz,bsiz,pflags,crc)
	headerID = 'PYRO'
	connectMSG='CONNECT'
	acceptMSG= 'GRANTED'
	denyMSG=   'DENIED'	# must be same length as acceptMSG,
						# note that a 1-character code is appended!

	AUTH_CHALLENGE_SIZE = 16

	headerSize = struct.calcsize(headerFmt)
	version=4				# version 4 protocol
	def __init__(self):
		self.onewayMethods=[]		# methods that should be called one-way
		self.timeout=None			# socket timeout
		self.ident=''				# connection identification
		self.setNewConnectionValidator(DefaultConnValidator())
	def sendAccept(self, conn):		# called by TCPServer
		sock_sendmsg(conn.sock, self.acceptMSG, self.timeout)
	def sendDeny(self, conn, reasonCode=Pyro.constants.DENIED_UNSPECIFIED):	# called by TCPServer
		sock_sendmsg(conn.sock, self.denyMSG+str(reasonCode)[0], self.timeout)
	def __del__(self):
		self.release(nolog=1)
	def recvAuthChallenge(self, conn):
		ver,body,pflags = self.receiveMsg(conn)
		if ver==self.version and len(body)==self.AUTH_CHALLENGE_SIZE:
			return body
		raise ValueError("Received version must be "+`self.version`+" and auth challenge must be exactly "+`self.AUTH_CHALLENGE_SIZE`+" bytes")
	def setNewConnectionValidator(self,validator):
		if not isinstance(validator, DefaultConnValidator):
			raise TypeError("validator must be specialization of DefaultConnValidator")
		self.newConnValidator=validator
	def getNewConnectionValidator(self):
		return self.newConnValidator
	def bindToURI(self,URI):
		# Client-side connection stuff. Use auth code from our own connValidator.
		if URI.protocol not in ('PYRO', 'PYROLOC'):
			Log.error('PYROAdapter','incompatible protocol in URI:',URI.protocol)
			raise ProtocolError('incompatible protocol in URI')
		try:
			self.URI=URI.clone()
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.connect((URI.address, URI.port))
			conn=TCPConnection(sock,sock.getpeername())
			# receive the authentication challenge string, and use that to build the actual identification string.
			try:
				authChallenge=self.recvAuthChallenge(conn)
			except ProtocolError,x:
				# check if we were denied
				if hasattr(x,"partialMsg") and x.partialMsg[:len(self.denyMSG)]==self.denyMSG:
					raise ConnectionDeniedError(Pyro.constants.deniedReasons[int(x.partialMsg[-1])])
				else:
					raise
			# reply with our ident token, generated from the ident passphrase and the challenge
			msg = self._sendConnect(sock,self.newConnValidator.createAuthToken(self.ident, authChallenge, conn.addr, self.URI, None) )
			if msg==self.acceptMSG:
				self.conn=conn
				self.conn.connected=1
				Log.msg('PYROAdapter','connected to',str(URI))
				if URI.protocol=='PYROLOC':
					self.resolvePYROLOC_URI("PYRO") # updates self.URI
			elif msg[:len(self.denyMSG)]==self.denyMSG:
				try:
					raise ConnectionDeniedError(Pyro.constants.deniedReasons[int(msg[-1])])
				except (KeyError,ValueError):
					raise ConnectionDeniedError('invalid response')
		except socket.error:
			Log.msg('PYROAdapter','connection failed to URI',str(URI))
			raise ProtocolError('connection failed')

	def resolvePYROLOC_URI(self, newProtocol):
		# This method looks up the object URI referenced by
		# the PYROLOC string, and updates self.URI in place!
		objectName=self.URI.objectID
		Log.msg('PYROAdapter','resolving PYROLOC name: ',objectName)
		# call the special Resolve method on the daemon itself:
		self.URI.objectID=Pyro.constants.INTERNAL_DAEMON_GUID
		result=self.remoteInvocation('ResolvePYROLOC',0,objectName)
		# found it, switching to regular pyro protocol
		self.URI.objectID=result
		self.URI.protocol=newProtocol

	def _sendConnect(self, sock, ident):
		body=self.connectMSG+ident
		sock_sendmsg(sock, self.createMsg(body), self.timeout)
		return sock_recvmsg(sock, len(self.acceptMSG),self.timeout)

	def release(self,nolog=0):
		if hasattr(self,'conn'):
			if not nolog:
				Log.msg('PYROAdapter','releasing connection')
			self.conn.close()
			del self.conn

	def connected(self):
		if hasattr(self,'conn'):
			return self.conn.connected
		return 0	

	def rebindURI(self, tries=sys.maxint, wait=1):
		t=0
		while t<tries:
			try:
				self.bindToURI(self.URI)
				return
			except ProtocolError:
				t+=1
				if t<tries:
					time.sleep(wait)
		raise TimeoutError('connection lost')
		
	def createMsg(self, body, replyflags=0):
		pflgs=replyflags
		if _has_compression and Pyro.config.PYRO_COMPRESSION:
			before=len(body)
			bz=zlib.compress(body) # default compression level
			if len(bz)<before:
				pflgs|=PFLG_COMPRESSED
				body=bz
		crc=0
		if Pyro.config.PYRO_CHECKSUM and _has_compression:
			crc=zlib.adler32(body)
			pflgs|=PFLG_CHECKSUM
		if Pyro.config.PYRO_XML_PICKLE=='gnosis':
			pflgs|=PFLG_XMLPICKLE_GNOSIS
		return struct.pack(self.headerFmt, self.headerID, self.version, self.headerSize, len(body), pflgs, crc) + body

	def setOneway(self, methods):
		self.onewayMethods.extend(methods)
	def setTimeout(self, timeout):
		if os.name=='java':
			# don't allow the use of the timeout feature in jython because it causes unreliable behavior
			raise RuntimeError("using setTimeout causes unreliable behavior in Jython")
		self.timeout=timeout
	def setIdentification(self, ident, munge=True):
		if ident:
			if munge:
				self.ident=self.newConnValidator.mungeIdent(ident)   # don't store ident itself. 
			else:
				self.ident=ident # per-munged ident string
		else:
			self.ident=''
	def getIdentification(self):
		return self.ident

	# Retrieve code from the remote peer. Works recursively.
	def _retrieveCode(self, mname, level):
		# XXX this is nasty code, and also duplicated in core.py remote_supply_code()
		if mname in sys.modules:
			# module is already loaded, do nothing
			# XXX how can we be sure if the module is "complete"?
			# XXX because things might go wrong during the loading code below?
			return
		try:
			_remoteImportRLock.acquire()	
			# Called by the client-side to obtain code from the server-side.
			# Call the special method on the server to retrieve the code.
			# No need for complex exception stuff like when the server needs
			# code from the client (see handleInvocation): because the server
			# is a Pyro object we can actually *call* it :-)
			module = self.remoteInvocation("remote_retrieve_code",0,mname)
			mname = mname.split('.')
			path = ''
			mod = new.module("pyro-server-context")
			for m in mname:
				path += '.' + m
				# use already loaded modules instead of overwriting them
				real_path = path[1:]
				if sys.modules.has_key(real_path):
					mod = sys.modules[real_path]
				else:
					setattr(mod, m, new.module(real_path))
					mod = getattr(mod, m)
					sys.modules[real_path] = mod
					
			if module[0:4] != imp.get_magic():
				code = compile(module, "<downloaded>", "exec")
			else:
				code = marshal.loads(module[8:])
	 
			importer=None
			try:
				loaded = 0
				# XXX probably want maxtries here...
				while not loaded:
					# install a custom importer to intercept any extra needed modules
					# when executing the module code just obtained from the server
					_agentImportLock.acquire()
					importer = agent_import(__builtin__.__import__)
					__builtin__.__import__ = importer
					_agentImportLock.release()
	 
					try:
						exec code in mod.__dict__
						loaded = 1
					except ImportError:
						mname = importer.name
						if importer is not None:
							__builtin__.__import__ = importer.orig_import
							importer = None
	 
						# XXX probably want maxrecursion here...
						self._retrieveCode(mname, level+1)
	 
			finally:
				if importer is not None:
					__builtin__.__import__ = importer.orig_import
		finally:
			_remoteImportRLock.release()


	def remoteInvocation(self, method, flags, *args):
		if 'conn' not in self.__dict__.keys():
			Log.msg('PYROAdapter','no connection, trying to bind again')
			if 'URI' in self.__dict__.keys():
				self.bindToURI(self.URI)
			else:
				raise ProtocolError('trying to rebind, but was never bound before')
		if method in self.onewayMethods:
			flags |= Pyro.constants.RIF_Oneway
		body=pickle.dumps((self.URI.objectID,method,flags,args),Pyro.config.PYRO_PICKLE_FORMAT)
		sock_sendmsg(self.conn.sock, self.createMsg(body), self.timeout)
		if flags & Pyro.constants.RIF_Oneway:
			return None		# no answer required, return immediately
		ver,answer,pflags = self.receiveMsg(self.conn,1)  # read the server's response, send no further replies
		if answer is None:
			raise ProtocolError('incorrect answer received')

		# Try to get the answer from the server.
		# If there are import problems, try to get those modules from
		# the server too (if mobile code is enabled).
		if not Pyro.config.PYRO_MOBILE_CODE:
			answer = pickle.loads(answer)
		else:
			importer=None
			try:
				_remoteImportRLock.acquire()
				loaded = 0
				# XXX maxtries here...
				while not loaded:
					# install a custom importer to intercept any extra needed modules
					# when unpickling the answer just obtained from the server
					_agentImportLock.acquire()
					importer = agent_import(__builtin__.__import__)
					__builtin__.__import__ = importer
					_agentImportLock.release()
 
					try:
						answer = pickle.loads(answer)
						loaded = 1
					except ImportError:
						mname = importer.name
						if importer is not None:
							__builtin__.__import__ = importer.orig_import
							importer = None
							self._retrieveCode(mname, 0)
 
			finally:
				if importer is not None:
					__builtin__.__import__ = importer.orig_import
				_remoteImportRLock.release()

		if isinstance(answer,PyroExceptionCapsule):
			if isinstance(answer.excObj,_InternalNoModuleError):
				# server couldn't load module, supply it
				# XXX this code is ugly. and duplicated in remote_retrieve_code in core.py
				try:
					importmodule=new.module('-agent-import-')
					mname=answer.excObj.modulename
					# not used: fromlist=answer.excObj.fromlist
					try:
						exec 'import '+mname in importmodule.__dict__
					except ImportError:
						Log.error('PYROAdapter','Server wanted a non-existing module:',mname)
						raise PyroError('Server wanted a non-existing module',mname)
					m=eval('importmodule.'+mname)
					bytecode=None
					if hasattr(m,"_PYRO_bytecode"):
						# use the bytecode that was put there earlier,
						# this avoids recompiles of the source .py if we don't have .pyc bytecode available
						bytecode=m._PYRO_bytecode
					else:
						# try to load the module's compiled source, or the real .py source if that fails.
						# note that the source code (.py) is opened with universal newline mode
						if not hasattr(m,"__file__"):
							raise PyroError("cannot read module source code",mname)
						(filebase,ext)=os.path.splitext(m.__file__)
						if ext.startswith(".PY"):
							exts = ( (".PYO","rb"), (".PYC","rb"), (".PY","rU") )	# uppercase
						else:
							exts = ( (".pyo","rb"), (".pyc","rb"), (".py","rU") )	# lowercase
						for ext,mode in exts:
							try:
								bytecode=open(filebase+ext, mode).read()
								break
							except EnvironmentError:
								pass
					if bytecode:
						self.remoteInvocation("remote_supply_code",0,mname, bytecode, self.conn.sock.getsockname())
						# retry the method invocation
						return self.remoteInvocation(* (method, flags)+args)
					Log.error("PYROAdapter","cannot read module source code for module:", mname)
					raise PyroError("cannot read module source code",mname)
				finally:
					del importmodule
			else:
				# we have an encapsulated exception, raise it again.
				answer.raiseEx()
		return answer

	# (private) receives a socket message, returns: (protocolver, message, protocolflags)
	def receiveMsg(self,conn,noReply=0):
		msg=sock_recvmsg(conn.sock, self.headerSize, self.timeout)
		(hid, ver, hsiz, bsiz, pflags, crc) = struct.unpack(self.headerFmt,msg)
		# store in the connection what pickle method this is
		if pflags&PFLG_XMLPICKLE_GNOSIS:
			conn.pflags|=PFLG_XMLPICKLE_GNOSIS
		if ver!=self.version:
			msg='incompatible protocol version'
			Log.error('PYROAdapter',msg)
			if not noReply:
				# try to report error to client, but most likely the connection will terminate:
				self.returnException(conn, ProtocolError(msg))
			raise ProtocolError(msg)
		if hid!=self.headerID or hsiz!=self.headerSize:
			msg='invalid header'
			Log.error('PYROAdapter',msg)
			Log.error('PYROAdapter','INVALID HEADER DETAILS: ',conn,( hid, ver, hsiz, bsiz,pflags))
			if not noReply:
				# try to report error to client, but most likely the connection will terminate:
				self.returnException(conn, ProtocolError(msg), shutdown=1)
			raise ProtocolError(msg)
		body=sock_recvmsg(conn.sock, bsiz, self.timeout)
		if pflags&PFLG_CHECKSUM:
			if _has_compression:
				if crc!=zlib.adler32(body):
					msg='checksum error'
					Log.error('PYROAdapter',msg)
					if not noReply:
						self.returnException(conn, ProtocolError(msg))
					raise ProtocolError(msg)
			else:
				raise ProtocolError('cannot perform checksum')
		if pflags&PFLG_COMPRESSED:
			if _has_compression:
				body=zlib.decompress(body)
			else:
				# We received a compressed message but cannot decompress.
				# Is this really a server error? We now throw an exception on the server...
				raise ProtocolError('compression not supported')
		return ver,body,pflags

	def _unpickleRequest(self, pflags, body):
		if pflags&PFLG_XMLPICKLE_GNOSIS:
			if Pyro.config.PYRO_XML_PICKLE=='gnosis':
				return pickle.loads(body)
			else:
				return Pyro.util.getXMLPickle('gnosis').loads(body)
		elif Pyro.config.PYRO_XML_PICKLE:
			Log.error('PYROAdapter','xml pickle required, got other pickle')
			raise ProtocolError('xml pickle required, got other pickle')
		else:
			return pickle.loads(body)

	def handleInvocation(self,daemon,conn):
		ver,body,pflags = self.receiveMsg(conn)
		if not body:
			# something went wrong even before receiving the full message body
			return
		if ver!=self.version:
			Log.error('PYROAdapter','incompatible protocol version')
			self.returnException(conn, ProtocolError('incompatible protocol version'))
			return 

		# Unpickle the request, which is a tuple:
		#  (object ID, method name, flags, (arg1,arg2,...))
		importer=fromlist=None
		try:
			try:
				# install a custom importer to intercept any extra needed modules
				# when unpickling the request just obtained from the client
				_agentImportLock.acquire()
				importer=agent_import(__builtin__.__import__)
				__builtin__.__import__=importer
				_agentImportLock.release()
				req=self._unpickleRequest(pflags, body)
				if type(req)!=tuple or len(req)!=4 or type(req[3])!=tuple:
					# sanity check failed
					raise ProtocolError("invalid request data format")
			finally:
				__builtin__.__import__=importer.orig_import

		except ImportError,x:
			if Pyro.config.PYRO_MOBILE_CODE:
				# return a special exception that will be processed by client;
				# it will call the internal 'remote_supply_code' member
				if importer:
					modname=importer.name
					fromlist=importer.fromlist
				else:
					modname = x.args[0][16:]
					fromlist=None
				self.returnException(conn, _InternalNoModuleError(modname,fromlist),0) # don't shutdown!
			else:
				Log.error('PYROAdapter','code problem with incoming object: '+str(x))
				self.returnException(conn, NoModuleError(* x.args))
			return

		try:
			# find the object in the implementation database of our daemon
			o=daemon.implementations[req[0]]
		except (KeyError, TypeError) ,x:
			Log.warn('PYROAdapter','Invocation to unknown object ignored:',x)
			self.returnException(conn, ProtocolError('unknown object ID'))
			return
		else:
			# Do the invocation. We are already running in our own thread.
			if req[2]&Pyro.constants.RIF_Oneway and daemon.threaded:   # flags
				# received a oneway call, run this in its own thread.
				thread=Thread(target=self._handleInvocation2, args=(daemon,req,pflags,conn,o))
				thread.setDaemon(1)   # thread must exit at program termination.
				thread.start()
			else:
				# not oneway or not in threaded mode, just do the invocation synchronously
				self._handleInvocation2(daemon,req,pflags,conn,o)

	def _handleInvocation2(self, daemon, req, pflags, conn, obj):
		try:
			flags=req[2]
			importer=None
			if not Pyro.config.PYRO_MOBILE_CODE:
				res = obj[0].Pyro_dyncall(req[1],flags,req[3])	# (method,flags,args)
			else:
				try:
					# install a custom importer to intercept any extra needed modules
					# when executing the remote method. (using the data passed in by
					# the client may trigger additional imports)
					_agentImportLock.acquire()
					importer=agent_import(__builtin__.__import__)
					__builtin__.__import__=importer
					_agentImportLock.release()
					res = obj[0].Pyro_dyncall(req[1],flags,req[3])	# (method,flags,args)
				finally:
					__builtin__.__import__=importer.orig_import

			if flags&Pyro.constants.RIF_Oneway:
				return		# no result, return immediately
			# reply the result to the caller
			if pflags&PFLG_XMLPICKLE_GNOSIS:
				replyflags=PFLG_XMLPICKLE_GNOSIS
				if Pyro.config.PYRO_XML_PICKLE=='gnosis':
					body=pickle.dumps(res,Pyro.config.PYRO_PICKLE_FORMAT)
				else:
					body=Pyro.util.getXMLPickle('gnosis').dumps(res,Pyro.config.PYRO_PICKLE_FORMAT)
			else:
				replyflags=0
				body=pickle.dumps(res,Pyro.config.PYRO_PICKLE_FORMAT)
			sock_sendmsg(conn.sock, self.createMsg(body,replyflags),self.timeout)
		except ImportError,ix:
			if Pyro.config.PYRO_MOBILE_CODE:
				# Return a special exception that will be processed by client;
				# it will call the internal 'remote_supply_code' member.
				# We have to use this seemingly complex way to signal the client
				# to supply us some code, but it is only a proxy! We can't *call* it!
				if importer:
					# grab the import info from our importer
					name=importer.name
					fromlist=importer.fromlist
				else:
					# XXX the importerror sometimes doesn't contain the package :-(
					name=ix.args[0][16:]
					fromlist=None
				Log.msg('PYROAdapter','failed to import',name)
				self.returnException(conn, _InternalNoModuleError(name,fromlist),0) # don't shutdown!
			else:
				Log.error('PYROAdapter','code problem with incoming object: '+str(ix))
				self.returnException(conn, NoModuleError(* ix.args))
		except Exception:
			# Handle the exception. Pass in if it was a oneway call,
			# those calls don't need any response to be sent.
			daemon.handleError(conn, bool(flags&Pyro.constants.RIF_Oneway))

	def returnException(self, conn, exc, shutdown=1, args=None):
		# return an encapsulated exception to the client
		if conn.pflags&PFLG_XMLPICKLE_GNOSIS:
			pic=Pyro.util.getXMLPickle('gnosis')
		else:
			pic=pickle
		try:
			body=pic.dumps(PyroExceptionCapsule(exc,args),Pyro.config.PYRO_PICKLE_FORMAT)
		except Exception,x:
			# hmm, pickling the exception failed... pickle the string instead
			body=pic.dumps(PyroExceptionCapsule(PyroError(str(x)),args),Pyro.config.PYRO_PICKLE_FORMAT)
		sock_sendmsg(conn.sock, self.createMsg(body),self.timeout)
		if shutdown:
			conn.close()

	def handleConnection(self, conn, tcpserver):
		# Server-side connection stuff. Use auth code from tcpserver's validator.
		try:
			# Validate the connection source (host) immediately,
			# if it's ok, send authentication challenge, and read identification data to validate. 
			(ok,reasonCode) = tcpserver.newConnValidator.acceptHost(tcpserver,conn)
			if ok:
				challenge=tcpserver.newConnValidator.createAuthChallenge(tcpserver,conn)
				if len(challenge)!=self.AUTH_CHALLENGE_SIZE:
					raise ValueError("Auth challenge must be exactly "+`self.AUTH_CHALLENGE_SIZE`+" bytes")
				sock_sendmsg(conn.sock, self.createMsg(challenge),self.timeout)
				ver,body,pflags = self.receiveMsg(conn)
				# only process the message if it makes a bit of sense
				if ver==self.version and body.startswith(self.connectMSG):
					token=body[len(self.connectMSG):]
					(ok,reasonCode) = tcpserver.newConnValidator.acceptIdentification(tcpserver,conn,token,challenge)
					if ok:
						self.sendAccept(conn)
						conn.connected=1
						return 1
					else:
						self.sendDeny(conn,reasonCode)
			else:
				self.sendDeny(conn,reasonCode)
			return 0	
		except ProtocolError:
			# ignore the message if it caused protocol errors
			return 0

# import wrapper class to help with importing remote modules
class agent_import:
	def __init__(self, orig_import):
		self.orig_import=orig_import
	def __call__(self,name, globals={},locals={},fromlist=None):
		# save the import details:
		self.name=name		# note: this must be a str object
		self.fromlist=fromlist
		return self.orig_import(name,globals,locals,fromlist)


#
# The SSL adapter that handles SSL connections instead of regular sockets.
#
class PYROSSLAdapter(PYROAdapter):
	def __init__(self):
		PYROAdapter.__init__(self)
		try:
			from M2Crypto import SSL
		except ImportError:
			raise ProtocolError('SSL not available')

		self.ctx = SSL.Context('sslv23')
		self.ctx.load_cert(os.path.join(Pyro.config.PYROSSL_CERTDIR, Pyro.config.PYROSSL_CLIENT_CERT))
		self.ctx.load_client_ca(os.path.join(Pyro.config.PYROSSL_CERTDIR, Pyro.config.PYROSSL_CA_CERT))
		self.ctx.load_verify_info(os.path.join(Pyro.config.PYROSSL_CERTDIR, Pyro.config.PYROSSL_CA_CERT))
		self.ctx.set_verify(SSL.verify_peer | SSL.verify_fail_if_no_peer_cert,10)
		self.ctx.set_allow_unknown_ca(1)
		Log.msg('PYROSSLAdapter','SSL Context initialized')
	
	def setTimeout(self, timeout):
		PYROAdapter.setTimeout(self, timeout)
		
	def bindToURI(self,URI):
		if URI.protocol not in ('PYROSSL','PYROLOCSSL'):
			Log.error('PYROSSLAdapter','incompatible protocol in URI:',URI.protocol)
			raise ProtocolError('incompatible protocol in URI')
		try:
			from M2Crypto import SSL
			self.URI=URI.clone()
			sock = SSL.Connection(self.ctx,socket.socket(socket.AF_INET, socket.SOCK_STREAM))
			sock.connect((URI.address, URI.port))
			conn=TCPConnection(sock, sock.getpeername())
			# receive the authentication challenge string, and use that to build the actual identification string.
			authChallenge=self.recvAuthChallenge(conn)
			# reply with our ident token, generated from the ident passphrase and the challenge
			msg = self._sendConnect(sock,self.newConnValidator.createAuthToken(self.ident, authChallenge, conn.addr, self.URI, None) )
			if msg==self.acceptMSG:
				self.conn=conn
				self.conn.connected=1
				Log.msg('PYROSSLAdapter','connected to',str(URI))
				if URI.protocol=='PYROLOCSSL':
					self.resolvePYROLOC_URI("PYROSSL") # updates self.URI
			elif msg[:len(self.denyMSG)]==self.denyMSG:
				try:
					raise ConnectionDeniedError(Pyro.constants.deniedReasons[int(msg[-1])])
				except (KeyError,ValueError):
					raise ConnectionDeniedError('invalid response')
		except socket.error:
			Log.msg('PYROSSLAdapter','connection failed to URI',str(URI))
			raise ProtocolError('connection failed')

	def _sendConnect(self, sock, ident):
		return PYROAdapter._sendConnect(self, sock, ident)
	

def getProtocolAdapter(protocol):
	if protocol in ('PYRO', 'PYROLOC'):
		return PYROAdapter()
	elif protocol in ('PYROSSL', 'PYROLOCSSL'):
		return PYROSSLAdapter()
	else:
		Log.error('getProtocolAdapter','unsupported protocol:',protocol)
		raise ProtocolError('unsupported protocol')


#-------- TCPConnection object for TCPServer class
class TCPConnection:
	def __init__(self, sock, addr):
		self.sock = sock
		set_sock_keepalive(self.sock)   # enable tcp/ip keepalive on this socket
		self.addr = addr
		self.connected=0		# connected?	
		self.pflags=0			# protocol flags
	def __del__(self):
		self.close()
	def fileno(self):
		return self.sock.fileno()
	def close(self):
		#self.sock.makefile().flush()
		self.sock.close()
		self.connected=0
	def shutdown(self):
		#self.sock.makefile().flush()
		self.sock.shutdown(2) # no further send/receives
	def __str__(self):
		return 'TCPConnection with '+str(self.addr)+' connected='+str(self.connected)

#-------- The New Connection Validators:
#-------- DefaultConnValidator checks max number of connections & identification
#-------- and ident check is done using hmac-md5 secure hash of passphrase+challenge.
#-------- Contains client- & server-side auth code.
class DefaultConnValidator:
	def __init__(self):
		self.setAllowedIdentifications(None)	# default=accept all (None means all!)
	def acceptHost(self,daemon,connection):
		if len(daemon.connections)>=Pyro.config.PYRO_MAXCONNECTIONS:
			Log.msg('DefaultConnValidator','Too many open connections, closing',connection,'#conns=',len(daemon.connections))
			return (0, Pyro.constants.DENIED_SERVERTOOBUSY)
		return (1,0)
	def acceptIdentification(self, daemon, connection, token, challenge):
		if "all" in self.allowedIDs:
			return (1,0)
		for authid in self.allowedIDs[:]:
			if self.createAuthToken(authid, challenge, connection.addr, None, daemon) == token:
				return (1,0)
		Log.warn('DefaultConnValidator','connect authentication failed on conn ',connection)
		return (0,Pyro.constants.DENIED_SECURITY)
	def createAuthToken(self, authid, challenge, peeraddr, URI, daemon):
		# Called from both client and server, is used to be able to validate the token.
		# client: URI & peeraddr provided, daemon is None
		# server: URI is None, peeraddr and daemon provided.
		# Return hmac-md5 secure hash of our authentication phrase & the challenge.
		return hmac.new(challenge, authid).digest()
	def createAuthChallenge(self, tcpserver, conn):
		# Server-side only, when new connection comes in.
		# Challenge is secure hash of: server IP, process ID, timestamp, random value
		# (NOTE: MUST RETURN EXACTLY AUTH_CHALLENGE_SIZE(=16) BYTES!)
		try:
			pid=os.getpid()
		except:
			pid=id(self)	# at least jython has no getpid()
		string = '%s-%d-%.20f-%.20f' %(str(getIPAddress()), pid, time.time(), random.random())
		return md5(string).digest()
	def mungeIdent(self, ident):
		# munge the identification string into something else that's
		# not easily guessed or recognised, like the md5 hash:
		return md5(ident).digest()
	def setAllowedIdentifications(self, ids):
		if ids is not None:
			if type(ids) in (types.TupleType, types.ListType):
				self.allowedIDs=map(self.mungeIdent, ids)  # don't store ids themselves
			else:
				raise TypeError("ids must be a list")
		else:
			self.allowedIDs=["all"]  # trick: allow all incoming authentications.

	
#-------- basic SSL connection validator, a specialized default validator.
class BasicSSLValidator(DefaultConnValidator):
	def __init__(self):
		DefaultConnValidator.__init__(self)
	def acceptHost(self,daemon,connection):
		(ok,code) = DefaultConnValidator.acceptHost(self, daemon, connection)
		if ok:
			peercert=connection.sock.get_peer_cert()
			return self.checkCertificate(peercert)
		return (ok,code)
	def checkCertificate(self,cert):
		# do something interesting with the cert here, in a subclass :)
		if cert is None:
			return (0,Pyro.constants.DENIED_SECURITY)
		return (1,0)



#-------- Helper class for local storage.
class LocalStorage:
	def __init__(self):
		self.caller=None

#-------- TCPServer base class


class TCPServer:
	def __init__(self, port, host='', threaded=_has_threading,prtcol='PYRO'):
		self._ssl_server = 0
		self.connections = []  # connection threads
		self.initTLS=lambda tls: None  # default do-nothing func
		try:
			if prtcol=='PYROSSL':
				try:
					from M2Crypto import SSL
				except ImportError:
					raise ProtocolError('SSL not available')
				try:
					self.ctx = SSL.Context('sslv23')
					self.ctx.load_cert(os.path.join(Pyro.config.PYROSSL_CERTDIR, Pyro.config.PYROSSL_SERVER_CERT))
					self.ctx.load_client_ca(os.path.join(Pyro.config.PYROSSL_CERTDIR, Pyro.config.PYROSSL_CA_CERT))
					self.ctx.load_verify_info(os.path.join(Pyro.config.PYROSSL_CERTDIR, Pyro.config.PYROSSL_CA_CERT))
					self.ctx.set_verify(SSL.verify_peer | SSL.verify_fail_if_no_peer_cert,10)
					self.ctx.set_allow_unknown_ca(1)
					self._ssl_server = 1
					Log.msg('TCPServer','SSL Context initialized')
				except:
					Log.warn('TCPServer','SSL Context could not be initialized !!!')
				self.setNewConnectionValidator(BasicSSLValidator())
			else:
				self.setNewConnectionValidator(DefaultConnValidator())
				
			# create server socket for new connections
			self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			set_reuse_addr(self.sock)
			self.sock.bind((host,port))
			self.sock.listen(Pyro.config.PYRO_TCP_LISTEN_BACKLOG)
			# rest of members
			self.threaded = threaded
			self.mustShutdown=0  # global shutdown
			self.localStorage=LocalStorage()  # TLS for systems that don't have threads
			return		
		except socket.error,msg:
			raise ProtocolError(msg)
		Log.msg('TCPServer','initialized')
			
	def __del__(self):
		self.closedown(nolog=1)

	def setInitTLS(self, initTLS):
		if not callable(initTLS):
			raise TypeError("initTLS must be callable object")
		self.initTLS=initTLS
		# if in single thread mode, (re-)init the TLS right away.
		if not Pyro.config.PYRO_MULTITHREADED:
			self.initTLS(self.localStorage)
		
	def closedown(self, nolog=0):
		# explicit closedown request
		if len(self.connections)>0:
			if not nolog:
				Log.warn('TCPServer','Shutting down but there are still',len(self.connections),'active connections')
			for c in self.connections[:]:
				if isinstance(c,TCPConnection):
					c.close()
				if isinstance(c,Thread):
					c.join()
			self.connections=[]
		if hasattr(self,'sock'):
			self.sock.close()
			del self.sock

	def setNewConnectionValidator(self,validator):
		if not isinstance(validator, DefaultConnValidator):
			raise TypeError("validator must be specialization of DefaultConnValidator")
		self.newConnValidator=validator
	def getNewConnectionValidator(self):
		return self.newConnValidator

	def connectionHandler(self, conn):
		# Handle the connection and all requests that arrive on it.
		# This is only called in multithreading mode.
		try:
			if self.getAdapter().handleConnection(conn, self):
				Log.msg('TCPServer','new connection ',conn, ' #conns=',len(self.connections))
				while not self.mustShutdown:
					try:
						if not conn.connected:
							# connection has been closed in the meantime!
							raise ConnectionClosedError()
						ins,outs,exs=safe_select([conn],[],[],2)
						if conn in ins or conn in exs:
							self.handleInvocation(conn)
					except ConnectionClosedError:
						# client went away. Exit immediately
						self.removeConnection(conn)
						return 
					except (PyroExceptionCapsule, Exception):
						self.handleError(conn)
			else:
				# log entry has already been written by newConnValidator
				self.removeConnection(conn)
		finally:
			# exiting thread.
			self._removeFromConnectionList(None)

	def _removeFromConnectionList(self, obj):
		if self.threaded and currentThread:
			obj=currentThread()
		try:
			self.connections.remove(obj)
		except ValueError:
			pass


	# this is the preferred way of dealing with the request loop.
	def requestLoop(self, condition=lambda:1, timeout=3, others=[], callback=None):
		while condition() and not self.mustShutdown:
			self.handleRequests(timeout,others,callback)

	def handleRequests(self, timeout=None, others=[], callback=None):
		if others and not callback:
			raise ProtocolError('callback required')
		if self.threaded:
			self._handleRequest_Threaded(timeout,others,callback)
		else:
			self._handleRequest_NoThreads(timeout,others,callback)
	
	def _handleRequest_NoThreads(self,timeout,others,callback):
		# self.connections is used to keep track of TCPConnections
		socklist = self.connections+[self.sock]+others
		ins,outs,exs = safe_select(socklist,[],[],timeout)
		if self.sock in ins:
			# it was the server socket, new incoming connection
			ins.remove(self.sock)
			if self._ssl_server:
				from M2Crypto import SSL
				try:
					csock, addr = self.sock.accept()
					sslsock = SSL.Connection(self.ctx,csock)
					sslsock.setup_addr(addr)
					sslsock.setup_ssl()
					sslsock.set_accept_state()
					sslsock.accept_ssl()
				except SSL.SSLError,error:
					if str(error) in('unexpected eof', 'http request', 'tlsv1 alert unknown ca', 'peer did not return a certificate'):
						return
					else:
						raise
				csock=sslsock

			else:
				csock, addr = self.sock.accept()

			conn=TCPConnection(csock,addr)
			if self.getAdapter().handleConnection(conn, self):
				Log.msg('TCPServer','new connection ',conn, ' #conns=',len(self.connections))
				self.connections.append(conn)
			else:
				# connection denied, log entry has already been written by newConnValidator
				self.removeConnection(conn)

		for c in ins[0:]:
			if isinstance(c,TCPConnection):
				ins.remove(c)
				try:
					self.handleInvocation(c)
					if not c.connected:
						self.removeConnection(c)
				except ConnectionClosedError:
					# client went away.
					self.removeConnection(c)
				except:
					self.handleError(c)
				
		if ins and callback:
			# the 'others' must have fired...
			callback(ins)

	# def handleInvocation(self, conn):	.... abstract method (implemented in subclass)


	def _handleRequest_Threaded(self,timeout,others,callback):
		# self.connections is used to keep track of connection Threads
		socklist = [self.sock]+others
		ins,outs,exs = safe_select(socklist,[],[],timeout)
		if self.sock in ins:
			# it was the server socket, new incoming connection
			if self._ssl_server:
				from M2Crypto import SSL
				try:
					csock, addr = self.sock.accept()
					sslsock = SSL.Connection(self.ctx,csock)
					sslsock.setup_addr(addr)
					sslsock.setup_ssl()
					sslsock.set_accept_state()
					sslsock.accept_ssl()
				except SSL.SSLError,error:
					Log.warn('TCPServer','SSL error: '+str(error))
					# print "SSL Error:",error
					csock.close()
					return
				csock=sslsock

			else:
				csock, addr = self.sock.accept()

			conn=TCPConnection(csock,addr)
			thread=Thread(target=self.connectionHandler, args=(conn,))
			thread.setDaemon(1)   # thread must exit at program termination.
			thread.localStorage=LocalStorage()
			self.initTLS(thread.localStorage)
			self.connections.append(thread)
			thread.start()
		elif callback:
			# the 'others' must have fired...
			callback(ins)

	def getLocalStorage(self):
		# return storage object for this thread.
		if self.threaded:
			return currentThread().localStorage
		else:
			return self.localStorage

	# to be called if a dropped connection is detected:
	def removeConnection(self, conn):
		conn.close()
		self._removeFromConnectionList(conn)
		Log.msg('TCPServer','removed connection ',conn,' #conns=',len(self.connections))

	# to be called to stop all connections and shut down.
	def shutdown(self):
		self.mustShutdown=1

	def getAdapter(self):
		raise NotImplementedError,'must be overridden to return protocol adapter'
	def handleError(self,conn,onewaycall=False):
		raise NotImplementedError,'must be overridden'

	def getServerSockets(self):
		if self.threaded:
			return [self.sock]
		else:
			return map(lambda conn: conn.sock, self.connections)+[self.sock]

# Sometimes safe_select() raises an select.error exception with the EINTR
# errno flag set, which basically tells the caller to try again later.
# This safe_select method works around this case and indeed just tries again.
_selectfunction=select.select
if os.name=="java":
	from select import cpython_compatible_select as _selectfunction
def safe_select(r,w,e,timeout=None):
	while True:
		try:
			if timeout is not None:
				return _selectfunction(r,w,e,timeout)
			else:
				return _selectfunction(r,w,e)
		except select.error,x:
			if x.args[0] == errno.EINTR or (hasattr(errno, 'WSAEINTR') and x.args[0] == errno.WSAEINTR):
				pass
			else:
				raise
