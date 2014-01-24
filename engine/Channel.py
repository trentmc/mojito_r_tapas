"""Channel.py

The aim of Channel is to provide a means for 2-way communication between
 master and slave.

Usage of Channel by Master:
 -initialize once: channel = ChannelFactory(..).buildChannel(..)
 -during a run, call:
    -task = TaskForSlave(task_descr, TaskData object)
    -channel.pushTasks(tasks)
    -tasks = channel.remainingTasks()
    -results = channel.popResults()

Usage of Channel by a Slave:
 -initialize once (though calling >once is safe):
   -channel = ChannelFactory(..).buildChannel(..)
   -synth_ss = channel.solutionStrategy()
   -problem_choice = channel.problemChoice() (use this to build the PS)
   -channel.reportForService(slave.ID)
 -during a run
   -if no task: task = channel.popTask()
   -when have a task: just work on a task
   -when done task:
     -result_data = ResultData(..)
     -task.attachResult(result_data)
     -channel.pushResult(task)

Note that this usage supports 0, 1, or >1 calls by master to update channel
 before calls by slave.

Interface classes:
-Channel -- Both the Master and Slave(s) have a Channel object.
     From their perspective, it is just an object that allows them to send
     tasks and results to each other, but under the hood it manages the
     communications using the file system and pickling or network transfers.
-ChannelFactory -- builds a channel (file-based or pyro-based)
-ChannelStrategy -- sets timeouts, etc
-ChannelData -- Encapsulates the data transport mechanism.
-TaskForSlave -- instructions and/or results; attributes include:
   -descr -- string -- string-based description of task, e.g. 'generate random ind'
   -task_data -- TaskData or None -- task-specific extra data from master, for slave
   -result_data -- ResultData or None -- task-specific result data from slave, for master
   -ID -- unique identifier, set by dispatcher to ease tracking

Implementation classes:
-FileChannel -- File-based channel implementation
-FileChannelData -- what actually gets pickled/unpickled for a FileChannel
-PyroChannel -- class that can be used by the PyroChannelDispatcher to mediate the data
    transport between the master and the slaves
-PyroChannelDispatcher -- see PyroChannel
"""

import copy
import cPickle as pickle
import logging
import os
import random
import time
import types
import re

import threading

import Pyro.core
import Pyro.naming
from Pyro.errors import NamingError

from engine.SynthSolutionStrategy import SynthSolutionStrategy

ALL_TASK_DESCRIPTIONS = [
    "Generate random ind",
    "Generate novel random ind",
    "Generate child",
    "Evaluate ind further",
    "Single simulation",
    "Local optimize",
    "Shut down",
    "Resimulate",
    "ImproveTopologyForWeight"
    ]

#
log = logging.getLogger("channel")

class TaskData:
    #-tack whatever else we want here, depending on the type of task
    def __init__(self):

        # can this type of task data survive zombification
        self.ignore_zombification = False

class ResultData:
    #-to start with, we have a few things already (slave_ID, etc)
    #-then tack whatever else we want here, depending on the type of task.
    
    def __init__(self, slave_ID, num_evaluations_per_analysis):
        #preconditions
        assert isinstance(slave_ID, types.StringType)
        assert isinstance(num_evaluations_per_analysis, types.DictType)

        #set values
        self.slave_ID = slave_ID
        self.num_evaluations_per_analysis = num_evaluations_per_analysis
        
class TaskForSlave:
    def __init__(self, master_ID, task_description, task_data):
        #preconditions
        assert task_description in ALL_TASK_DESCRIPTIONS
        assert isinstance(task_data, TaskData)

        #set data
        self.descr = task_description
        self.task_data = task_data
        
        #attach responses to tasks
        # should be ResultData
        self.result_data = None
        
        #the dispatcher sets this to a certain value.
        # to ease tracking
        self.ID = None

        # the master this task belongs to
        # to avoid processing of tasks that are not for us
        self.master_ID = master_ID
        self.slave_ID = None
       
        # timeout
        self.start_time = None
        self.stop_time = None
        
    def attachResult(self, result_data):
        #preconditions
        assert isinstance(result_data, ResultData)

        #set the data
        self.result_data = result_data
        
    def getRunTime(self):
        if self.start_time != None:
            return time.time() - self.start_time
        else:
            return None
        
    def __str__(self):
        s = []
        s += ["TaskForSlave={"]
        s += ["%s (ID=%s), " % (self.descr, str(self.ID))]
        s += ["start_time: %s (runtime: %s), " % \
              (str(self.start_time), str(self.getRunTime()))]
        s += ["Task Data: %s, " % str(self.task_data)]
        s += ["Result Data:%s, " % str(self.result_data)]
        s += ["/TaskForSlave}"]
        return "".join(s)

class ChannelStrategy:
    def __init__(self, channel_type = 'FileBased', cluster_id = 0):
        # the channel type
        # possibilities:
        #  'FileBased'
        #  'PyroBased'
        self.channel_type = channel_type
        
        # the name for the file for file-based channels
        self.channel_file = ""
        # the time between 2 file open attempts
        self.num_seconds_between_file_access_attempts = 2

        # the maximum time a task can take before it is
        # considered zombified (default: 1hr)
        self.zombification_timeout_secs = 60 * 60
        
        # the ID to be used for a Pyro-based channels
        self.cluster_id = cluster_id
        # the dispatcher name
        self.clusterDispatcherName = "dispatcher"
        # the group prefix
        self.groupPrefix = 'Synth'
        
    def getClusterGroup(self):
        return "%s%s" % ( self.groupPrefix, self.cluster_id )
    
    def setToTinyWaits(self):
        """Call this to have unit tests that won't have to pause for long periods.
        """
        self.num_seconds_between_file_access_attempts = 0.01

    def __str__(self):
        s = []
        s += ["ChannelStrategy={ "]
        s += ["channel_type=%s" % self.channel_type]
        s += ["; channel_file='%s'" % self.channel_file]
        s += ["; num_seconds_between_file_access_attempts=%g" %
              self.num_seconds_between_file_access_attempts]
        s += ["; cluster_id=%s" % self.cluster_id]
        s += ["; dispatcher=%s" % self.clusterDispatcherName]
        s += ["; groupPrefix=%s" % self.groupPrefix]
        s += [" /ChannelStrategy}"]
        return "".join(s)

class ChannelFactory:
    """
    creates channels
    cs = ChannelStrategy (defines channel type and params)
    """
    def __init__(self, cs):
        #preconditions
        assert isinstance(cs, ChannelStrategy)

        self.cs = cs

    def buildChannel(self, is_master = False):
        """
        for_master = build channel for master
        """
        if self.cs.channel_type == 'Local':
            return None
        elif self.cs.channel_type == 'FileBased':
            return FileChannel( self.cs, is_master )
        elif self.cs.channel_type == 'PyroBased':
            return PyroChannel( self.cs, is_master )
        else:
            raise ValueError("Channel type should be 'Local', 'FileBased' or 'PyroBased';"
                             " but got: '%s'" % self.cs.channel_type )

class Channel:
     """Both the Master and Slave(s) have a Channel object.
     From their perspective, it is just an object that allows them to send
     tasks and results to each other, but under the hood it manages the
     communications using the file system and pickling or network transfers.
     """
     def __init__(self, cs):
        #preconditions
        assert isinstance(cs, ChannelStrategy)
        
        self.cs = cs

     #============================================================================
     # enforce the implementation of all methods
     def solutionStrategy(self):
         raise NotImplementedError

     def problemChoice(self):
         raise NotImplementedError
     
     def problemIsRobust(self):
         raise NotImplementedError
     
     def reportForService(self, slave_ID):
         raise NotImplementedError
        
     def leaveService(self, slave_ID):
         raise NotImplementedError
        
     def popTask(self, slave_ID=None):
         raise NotImplementedError
       
     def pushResult(self, task):
         raise NotImplementedError

     def pushTasks(self, tasks):
         raise NotImplementedError

     def tasksWaiting(self):
         raise NotImplementedError
    
     def tasksRunning(self):
         raise NotImplementedError
    
     def popFinishedTasks(self):
         raise NotImplementedError
    
     def registerMaster(self, ss, problem_choice, problem_is_robust):
         raise NotImplementedError

     def slaveIDs(self):
         raise NotImplementedError
    
     def reset(self):
         raise NotImplementedError
        
     def cleanup(self):
         raise NotImplementedError

     def zombieCount(self):
         raise NotImplementedError

     def setZombificationTimeout(self, t):
         raise NotImplementedError
        
class ChannelData:
    """ ChannelData base class. This encapsulates the data transport mechanism.
    
    Data to ease management by master:
    -slave IDs

    Data for two-way communication:
    -tasks from master to slaves
    -results from slaves to master

    Data to ease slave initialization:
    -ss -- SynthSolutionStrategy (so that slaves always have same ss as engine)
    -problem_choice -- so that slaves do not need to have this specified
    -problem_is_robust -- ""
    """
    def __init__(self, cs, support_threading = False):
        #preconditions
        assert isinstance(cs, ChannelStrategy)

        #set data
        self._slave_IDs = []
        
        self._tasks_waiting = []
        self._tasks_running = []
        self._tasks_finished = []
        self._zombified_task_IDs = []
        
        self._last_task_id = 0
        
        self._ss = None
        self._problem_choice = None
        self._problem_is_robust = None

        self._cs = cs

        self._no_threading = not support_threading
        if self._no_threading:
            self._global_lock = None
        else:
            self._global_lock = threading.Lock()

    #============================================================================
    # locking helpers
    def acquireGlobalLock(self, timeout = 600):
        if self._no_threading:
            return True
        lock = self._global_lock
        
        log.debug("acquiring lock %s" % (str(lock)))
        time_to_sleep = 0.1
        
        if False: # debug method
            import inspect
            tmp = inspect.getouterframes(inspect.currentframe())
            log.info("Trace:")
            for t in tmp:
                log.info(" %s:%s" % (str(t[3]), str(t[2])))

        
        waited = 0
        while not lock.acquire( False ) and waited < timeout:
            time.sleep(time_to_sleep)
            waited += time_to_sleep

        if waited < timeout:
            return True
        else:
            log.info("timeout when aquiring global lock")
            import inspect
            tmp = inspect.getouterframes(inspect.currentframe())
            log.info("Trace:")
            for t in tmp:
                log.info(" %s:%s" % (str(t[3]), str(t[2])))
           
            raise ValueError
            return False
        
    def releaseGlobalLock(self):
        if self._no_threading:
            return True
        lock = self._global_lock
        log.debug("releasing lock")
        lock.release()

    #============================================================================
    # Info to ease slave initialization, and make it less error-prone
    def solutionStrategy(self):
        return self._ss

    def problemChoice(self):
        return self._problem_choice

    def problemIsRobust(self):
        return self._problem_is_robust
        
    #=============================================
    # ChannelData routines for Slave
    def reportForService(self, slave_ID):
        if self.acquireGlobalLock():
            if slave_ID not in self._slave_IDs:
                self._slave_IDs.append(slave_ID)
            else:
                self.releaseGlobalLock()
                raise ValueError("Slave already registered")
            log.info("Slave %s reported for service." % slave_ID)
            self.releaseGlobalLock()
            return True
        else:
            log.info("Could not acquire global lock")
            return False

    def leaveService(self, slave_ID):
        log.info("slave %s leaves service" % slave_ID)
        if self.acquireGlobalLock():
            if slave_ID in self._slave_IDs:
                self._slave_IDs.remove(slave_ID)
            else:
                self.releaseGlobalLock()
                raise ValueError("don't know slave")
            
            self.releaseGlobalLock()
            log.debug("Slave %s left service." % slave_ID)
        else:
            log.info("Could not acquire global lock")
            return False

    def popTask(self, slave_ID = None):
        """If self has _tasks, pops and returns 0th task; else returns None
        """

        log.debug("Slave %s wants to pop task", slave_ID)
        
        # if the slave id is not in the list anymore, add
        # it again since it's still alive
        if slave_ID and slave_ID not in self._slave_IDs:
            self._slave_IDs.append(slave_ID)
        
        if self.acquireGlobalLock():
            if len(self._tasks_waiting):
                log.debug(" => start: %s" % str(self) )
                
                # we have the lock, so we can pop
                task = self._tasks_waiting[0]
                self._tasks_waiting.remove(task)
                # release the lock since we won't manipulate the 
                # list anymore
                
                task.slave_ID = slave_ID
                log.debug("Slave %s popped task: %s" % (slave_ID, task))
                task.start_time = time.time()
                self._tasks_running.append(task)
                log.debug(" => result: %s" % str(self) )

                self.releaseGlobalLock()
                return task
            else:
                # release the lock
                log.debug("No task to pop")
            self.releaseGlobalLock()
            return None
        else:
            log.info("Could not acquire lock")
            return None

    def pushResult(self, task):
        #preconditions
        assert isinstance(task, TaskForSlave)
        
        log.debug("Slave wants to add result task %s" % (task))
        log.debug(" => start: %s" % str(self) )

        #find the task
        task_found = False
        if self.acquireGlobalLock():
            for t in self._tasks_running:
                if t.ID == task.ID:
                    task_found = True
                    break

            # was the task zombified?
            if not task_found:
                if task.ID in self._zombified_task_IDs:
                    log.info("slave returning result for zombified task")

                    # check whether the result should be discarded or not
                    if task.task_data.ignore_zombification:
                        # it's already removed from the running queue by
                        # the zombification pruner, so add it only to the
                        # result queue
                        log.info("keeping result for zombie-resistant task %s" % task)
                        task.stop_time = time.time()

                        self._tasks_finished.append(task)
                    else:
                        # the results life ends here...
                        pass
                else:
                    self.releaseGlobalLock()
                    raise ValueError('Task is not in the list of tasks handed out')
            else:
                self._tasks_running.remove(t)

                #main work
                task.stop_time = time.time()
                self._tasks_finished.append(task)
                
            self.releaseGlobalLock()
            log.debug(" => result: %s" % str(self) )
            return True
        else:
            log.debug("Could not aquire lock")
            return False

    #=============================================
    # ChannelData routines for Master
    def pushTasks(self, tasks):
        #preconditions
        log.debug(" Master adding %d tasks..." % len(tasks))
        log.debug(" => start: %s" % str(self) )
        for task in tasks:
            assert isinstance(task, TaskForSlave)
            # assertions without the locks, since that would be
            # too expensive
            assert not task in self._tasks_waiting
            assert not task in self._tasks_running
            assert not task in self._tasks_finished
            task.ID = self._last_task_id
            task.start_time = None
            self._last_task_id += 1
            log.debug(" task: %s" % task)
        
        #main work
        if self.acquireGlobalLock():
            self._tasks_waiting.extend(tasks)
            log.debug(" => result: %s" % str(self) )
            
            self.releaseGlobalLock()
            return True
        else:
            log.debug("Could not aquire lock")
            return False
        
    def tasksWaiting(self):
        # WARNING: BAD because it exposes a lock-protected var
        # do not manipulate the returned list
        log.debug(" Master requesting the waiting tasks (#=%d)" % len(self._tasks_waiting))
        return self._tasks_waiting

    def tasksRunning(self):
        # WARNING: BAD because it exposes a lock-protected var
        # do not manipulate the returned list
        log.debug(" Master requesting the running tasks (#=%d)" % len(self._tasks_running))
        return self._tasks_running
    
    def zombieCount(self):
        log.debug(" Master requesting the zombie count (#=%d)" % len(self._zombified_task_IDs))
        return len(self._zombified_task_IDs)

    def popFinishedTasks(self):
        log.debug("Master popping %d finished tasks" % len(self._tasks_finished) )
        log.debug(" => start: %s" % str(self) )

        if self.acquireGlobalLock():
            tasks_to_return = self._tasks_finished[:]
            self._tasks_finished = []
            self.releaseGlobalLock()
            
            log.debug(" => result: %s" % str(self) )
        
            #remove all tasks that are Zombie
            self.pruneZombieTasks()
            return tasks_to_return
        else:
            log.debug("Could not aquire lock")
            return []

    def registerMaster(self, ss, problem_choice, problem_is_robust):
        """Register a master to the channel"""
        assert isinstance(ss, SynthSolutionStrategy)
        assert isinstance(problem_choice, types.IntType)
        assert isinstance(problem_is_robust, types.BooleanType)

        self._ss = ss
        self._problem_choice = problem_choice
        self._problem_is_robust = problem_is_robust
    
    def slaveIDs(self):
        return self._slave_IDs

    def setZombificationTimeout(self, t):
        self._cs.zombification_timeout_secs = t

    def pruneZombieTasks(self):
        log.debug("Pruning zombie tasks...")
        log.debug(" => start: %s" % str(self) )

        tasks_to_zombie = []
        if self.acquireGlobalLock():
            for task in self._tasks_running:
                runtime = task.getRunTime()
                if runtime != None and \
                   runtime > self._cs.zombification_timeout_secs:
                    log.info("Zombifying task %s" % str(task))
                    # remove from running list
                    self._tasks_running.remove(task)
                    tasks_to_zombie.append(task)
            self.releaseGlobalLock()
        else:
            log.debug("Could not aquire lock")
            return

        # add the tasks to the zombie list
        if self.acquireGlobalLock():
            for task in tasks_to_zombie:
                # keep only the task ID, such that we don't spoil memory
                self._zombified_task_IDs.append(task.ID)
                try:
                  # remove the parent slave from the list
                  if task.slave_ID and (task.slave_ID in self._slave_IDs):
                    self._slave_IDs.remove(task.slave_ID)
                except:
                  log.error("failed to remove slave id")
                  pass # ignore this error
            self.releaseGlobalLock()
            log.debug(" => result: %s" % str(self) )
        else:
            log.debug("Could not aquire lock")
            return
            

    #=====================================================================
    # ChannelData helper Routines
    def reset(self):
        log.debug("Reset ChannelData" )
        
        if self.acquireGlobalLock():
            self._tasks_waiting = []
            self._tasks_running = []
            self._tasks_finished = []
            self._zombified_task_IDs = []
            self._slave_IDs = []
            self.releaseGlobalLock()
        else:
            log.info("could not acquire lock")

    def cleanup(self):
        log.debug("Cleanup ChannelData" )
        self.reset()

    def __str__(self):
        s = []
        s += ["ChannelData={\n"]
        s += [" Last task id   : %s\n" % str(self._last_task_id)]
        s += [" Problem choice : %s (robust=%s)\n" %
              (str(self._problem_choice), self._problem_is_robust)]
        s += [" Zombie timeout : %s\n" % str(self._cs.zombification_timeout_secs)]
        s += [" Zombie count   : %s\n" % str(len(self._zombified_task_IDs))]
        if self._slave_IDs:
            s += ["  Slave_IDs:\n"]
            for slave_ID in self._slave_IDs:
                s += ["   %s\n" % slave_ID]
            s += ["\n"]
            
        if self._tasks_waiting:
            #s += ["  Tasks waiting:\n"]
            #for task in self._tasks_waiting:
            #    s += ["   %s\n" % task]
            s += ["  Tasks waiting: %s" % str(len(self._tasks_waiting))]
            s += ["\n"]
                
        if self._tasks_running:
            #s += ["  Tasks running:\n"]
            #for task in self._tasks_running:
            #    s += ["   %s\n" % task]
            s += ["  Tasks running: %s" % str(len(self._tasks_running))]
            s += ["\n"]
                
        if self._tasks_finished:
            #s += ["  Tasks finished:\n"]
            #for task in self._tasks_finished:
            #    s += ["   %s\n" % task]
            s += ["  Tasks finished: %s" % str(len(self._tasks_finished))]
            s += ["\n"]
                
        s += ["/ChannelData}"]
        return "".join(s)


class FileChannel(Channel):
    """File-based channel implementation
    """
    
    def __init__(self, cs, is_master):
        """Initialize with 'channel_file'.  Master and all Slaves should
        see the _same_ channel file, otherwise they will not be able to
        communicate.
        -'cs' is a ChannelStrategy. Master and all Slaves should
        see the _same_ channel file, otherwise they will not be able to
        communicate.
        -'ss' is a SynthSolutionStrategy or None.
        -'problem_choice' is an int or None
        -'problem_is_robust' is an bool or None
        """
        Channel.__init__(self, cs)

        assert cs.channel_type == 'FileBased'

        self.is_master = is_master
        if is_master:
            log.info( "Creating File channel for master" )
        else:
            log.info( "Creating File channel for slave" )
        
        #set data
        self.channel_file = self.cs.channel_file            
        
        #create channel file if needed
        if not os.path.exists(self.channel_file):
            log.info("Channel file '%s' does not exist yet, so creating it." %
                     self.channel_file)
            data = FileChannelData(cs)
            fid = open(self.channel_file, 'w')
            pickle.dump(data, fid)
            fid.close()
            
            log.info("Successfully created channel file '%s'" %
                     self.channel_file)

        #postconditions
        assert os.path.exists(self.channel_file)

    #============================================================================
    # Info to ease slave initialization, and make it less error-prone
    def solutionStrategy(self):
        """Returns the SynthSolutionStrategy"""
        data = self._loadData()
        ss = data.solutionStrategy()
        return ss

    def problemChoice(self):
        """Returns the problem_choice (an int); use this to create a ProblemSetup."""
        data = self._loadData()
        problem_choice = data.problemChoice()
        return problem_choice
    
    def problemIsRobust(self):
        """Returns whether problem is robust (bool); use this to create a ProblemSetup."""
        data = self._loadData()
        problem_is_robust = data.problemIsRobust()
        return problem_is_robust

    #=====================================================================
    # Channel routines that Slave would call
    def reportForService(self, slave_ID):
        """A Slave should call this during initialization."""
        data = self._loadData()
        data.reportForService(slave_ID)
        self._saveData(data)
        log.info("Slave %s reported for service." % slave_ID)
        
    def leaveService(self, slave_ID):
        """A Slave should call this when exiting."""
        data = self._loadData()
        data.leaveService(slave_ID)
        self._saveData(data)
        log.info("Slave %s left service." % slave_ID)
        
    def popTask(self, slave_ID = None):
        """Pop a task for a slave to use, and return it.
        If there are no tasks, then returns None"""
        data = self._loadData()
        task = data.popTask(slave_ID)
        self._saveData(data)
        log.info("Popped task: %s" % task)
        return task
        
    def pushResult(self, task):
        """Add a result for certain task for a master to see.
        """
        data = self._loadData()
        data.pushResult(task)
        self._saveData(data)
        log.info("Added task result: %s" % task)

    #======================================================================
    # Channel routines that Master would call
    def pushTasks(self, tasks):
        """Add tasks for a slave to see."""
        data = self._loadData()
        data.pushTasks(tasks)
        self._saveData(data)
        log.info("Added %d tasks" % len(tasks))

    def tasksWaiting(self):
        """Returns list of remaining tasks."""
        data = self._loadData()
        return data.tasksWaiting()
    
    def tasksRunning(self):
        """Returns list of running tasks."""
        data = self._loadData()
        return data.tasksRunning()
    
    def popFinishedTasks(self):
        """Pop all results for a master to see, and return them"""
        data = self._loadData()
        results = data.popFinishedTasks()
        self._saveData(data)
        log.info("Popped %d results" % len(results))
        if results:
            for result in results:
                log.info("  Result: %s" % result)
        return results

    def registerMaster(self, ss, problem_choice, problem_is_robust):
        """Register a master to the channel"""
        assert isinstance(ss, SynthSolutionStrategy)
        assert isinstance(problem_choice, types.IntType)
        assert isinstance(problem_is_robust, types.BooleanType)
        
        data = self._loadData()
        data.registerMaster(ss, problem_choice, problem_is_robust)
        self._saveData(data)
        return 

    def slaveIDs(self):
        """Return a list of all slave IDs"""
        data = self._loadData()
        return data.slaveIDs()
    
    def reset(self):
        if self.is_master:
            data = self._loadData()
            data.reset()
            self._saveData(data)
        else:
            raise ValueError("Slave can't request reset")
        
    def cleanup(self):
        """cleans up. the master has nothing more to do."""
        if self.is_master:
            data = self._loadData()
            data.cleanup()
            os.remove(self.channel_file)
        else:
            # check whether the master file still exists
            if os.path.exists(self.channel_file):
                data = self._loadData()
                return data.leaveService(self.slave_ID)
            else:
                log.info("Master has already exited...")

    def zombieCount(self):
        """Returns number of zombified tasks."""
        data = self._loadData()
        return data.zombieCount()
    def setZombificationTimeout(self, t):
        data = self._loadData()
        data.setZombificationTimeout(t)
        self._saveData(data)
      
        

    #=====================================================================
    # Main file manipulation routines
    def _loadData(self):
        fid = open(self.channel_file,'r')
        data = pickle.load(fid)
        fid.close()
        return data

    def _saveData(self, data):
        fid = open(self.channel_file, 'w')
        pickle.dump(data, fid)
        fid.close()
    
    #=====================================================================
    # String utilities
    def __str__(self):
        return self.str2(False)

    def str2(self, include_data_in_file):
        #preconditions
        assert isinstance(include_data_in_file, types.BooleanType)

        #main work
        s = []
        s += ["Channel={"]
        s += ["channel_file=%s" % self.channel_file]
        if include_data_in_file:
            s += [str(self._loadData())]
        s += ["/Channel}"]
        return "".join(s)
        
class FileChannelData(ChannelData):
    """This is what actually gets pickled/unpickled for a FileChannel
    """
    def __init__(self, cs):
        # NOTE: FileChannelData does not support threading
        ChannelData.__init__(self, cs, False)

    #============================================================================
    #for pickling 
    def __getstate__(self):
        """Defines what information is returned for pickling.
        
        Perform some processing so that
        -self can pickle (e.g. cannot have function
         references which are in many ps.analyses objects
        -self doesn't have unneeded references to ind.S and ind._ps 
         (only on Ind for convenience)
        """
        #clear ind.S, ind._ps
        for item in self._tasks_waiting + self._tasks_running + self._tasks_finished:
            data = getattr(item, 'task_data', None) or getattr(item.result, 'result_data', None)
            if data is not None:
                if getattr(data, 'ind', None) is not None:
                    data.ind.S = None
                    data.ind._ps = None
                if getattr(data, 'inds', None) is not None:
                    for ind in data.inds:
                        data.ind.S = None
                        data.ind._ps = None
                    
        #start with all of self's attributes
        d = copy.copy(self.__dict__)

        #delete some attributes
        return d

    def __setstate__(self, d):
        """Defines how to restore self's attributes, given loaded input 'state'
        """
        for key, value in d.items():
            setattr(self, key, value)

class PyroChannel(Channel):
    """The PyroChannel is a class that can be used by the a dispatcher to mediate the data
    transport between the master and the slaves. The system setup would be:
        - one dispatcher
        - one master
        - one or more slaves
        
    The flow is:
        - the dispatcher is created, and creates a PyroChannelServer.
        - the master:
          * creates a PyroChannel(is_master = True)
            this PyroChannel will find the PyroChannelServer and will
            register the master with it (TODO, to make sure there is 
            only one master).
          * the master posts one or more tasks in the PyroChannel by 
            calling the Master channel routines.
          * the master periodically checks for completed tasks, and 
            once it finds some, they are taken from the queue
        - The slaves:
          * create a PyroChannel(is_master = False)
            this PyroChannel will find the PyroChannelServer and will
            register the slave with it.
          * periodically check for tasks that they can perform.
            if a task is present, it is 'taken' from the dispatcher.
            It is then executed and the result is posted back to the 
            dispatcher.
        - the dispatcher's PyroChannelServer maintains a queue for
          tasks and results.
    
    The nice thing about using Pyro is that once the dispatcher has exposed this
    class, both master and slave can call it as if it were a normal Python object.
    """
    def __init__(self, cs, is_master):
        """
        -'cs' is a ChannelStrategy. Master and all Slaves should
        see the _same_ strategy, otherwise they will not be able to
        communicate.
        -'ss' is a SynthSolutionStrategy or None.
        -'problem_choice' is an int or None
        -'problem_is_robust' is a bool or None
        """

        Channel.__init__(self, cs)

        assert cs.channel_type == 'PyroBased'

        self.is_master = is_master
        if is_master:
            log.info( "Creating Pyro channel for master" )
        else:
            # for slaves
            self.slave_ID =  None
            log.info( "Creating Pyro channel for slave" )
        
        log.info( "Using cluster ID: %s" % self.cs.cluster_id )
        log.info( "Using cluster dispatcher name: %s" % self.cs.clusterDispatcherName )

        # both master and slave are dispatcher clients
        Pyro.core.initClient()
        Pyro.config.PYRO_NS_DEFAULTGROUP = ":%s" % \
                                           ( self.cs.getClusterGroup() )

        # locate the NS
        log.info( 'Searching Naming Service...' )
        self.ns = Pyro.naming.NameServerLocator().getNS()

        log.info('Naming Service found at %s, (%s:%d)' \
                 % (self.ns.URI.address,
                    (Pyro.protocol.getHostname(self.ns.URI.address) or '??'), \
                    self.ns.URI.port ) )
        
        # find the dispatcher URI
        dispatcher_location = ":%s.%s" \
                              % ( self.cs.getClusterGroup(), self.cs.clusterDispatcherName )
        try:
            self.uri = self.ns.resolve( dispatcher_location )
        except:
            log.warning( 'Could not find dispatcher' )
            raise

        # get the dispatcher itself
        self.dispatcher = self.uri.getProxy()

    #============================================================================
    # Info to ease slave initialization, and make it less error-prone
    def solutionStrategy(self):
        """Returns the SynthSolutionStrategy"""
        return self.dispatcher.solutionStrategy()

    def problemChoice(self):
        """Returns the problem_choice (an int); use this to create a ProblemSetup."""
        return self.dispatcher.problemChoice()
    
    def problemIsRobust(self):
        """Returns whether the problem is robust (a bool); use this to create a ProblemSetup."""
        return self.dispatcher.problemIsRobust()

    #=====================================================================
    # Channel routines that Slave would call
    def reportForService(self, slave_ID):
        """A Slave should call this during initialization."""
        self.slave_ID = slave_ID
        # slave init
        self.dispatcher.reportForService(slave_ID)
        log.debug("Slave %s reporting for service." % slave_ID)
        
    def leaveService(self, slave_ID):
        """A Slave should call this when bailing out."""
        try:
            # slave init
            self.dispatcher.leaveService(slave_ID)
        except ValueError:
            log.warning( "Master is confused about this slave..." )
            raise
        log.debug("Slave %s leaving service." % slave_ID)
        
    def popTask(self, slave_ID = None):
        """Pop a task for a slave to use, and return it.
        If there are no tasks, then returns None"""
        try:
            task = self.dispatcher.popTask(slave_ID)
            log.debug("Popped task: %s" % task)
        except:
            log.debug("Failed to pop task")
            task = None
        return task
       
    def pushResult(self, task):
        """Add a result for a certain task for a master to see."""
        debugging = False #turn this True or False, depending on if we are debugging vs. running

        #'crashable' code -- but makes it easy to track bugs!
        if debugging:
            try:
                result = self.dispatcher.pushResult(task)
                log.debug("Added task result: %s" % task)
            except:
                import pdb;pdb.set_trace()
            
        #more robust code, but harder to track bugs.  
        else:
            result = None
            numtries = 0
            while numtries < 5:
                try:
                    result = self.dispatcher.pushResult(task)
                    log.debug("Added task result: %s" % task)
                    break
                except:
                    log.debug("Failed to add task result: %s, try %d" % (task, numtries))
                    time.sleep(0.2)
                numtries += 1
            
        return result

    #======================================================================
    # Channel routines that Master would call
    def pushTasks(self, tasks):
        """Add tasks for a slave to see."""
        #preconditions
        for task in tasks:
            assert isinstance(task, TaskForSlave)

            ##can we pickle it? (local test to catch errors more easily)(uncomment for debugging)
            #filename = '/tmp/pickletest_%f' % time.time()
            #try:
            #    pickle.dump(task, open(filename, 'w')) #will die if unsuccessful
            #except:
            #    import pdb; pdb.set_trace()
            #os.remove(filename)

        #main work
        result = None
        numtries = 0
        while numtries < 5:
            try:
                result = self.dispatcher.pushTasks(tasks)
                log.debug("Added %d tasks" % len(tasks))
                break
            except:
                log.debug("Failed to add task: %s, try %d" % (task, numtries))
            numtries += 1
        return result
    
    def tasksWaiting(self):
        """Returns list of remaining tasks."""
        try:
            return self.dispatcher.tasksWaiting()
        except:
            return None
    
    def tasksRunning(self):
        """Returns list of running tasks."""
        try:
            return self.dispatcher.tasksRunning()
        except:
            return None
    
    def popFinishedTasks(self):
        """Pop all results for a master to see, and return them"""
        results = []
        # reduce memory usage
        result = self.dispatcher.popFinishedTask()
        while result != None:
            results.append(result)
            result = self.dispatcher.popFinishedTask()
            
        log.debug("Popped %d results" % len(results))
        if results:
            for result in results:
                log.debug("  Result: %s" % result)
        return results
    
    def registerMaster(self, ss, problem_choice, problem_is_robust):
        """Register a master to the channel"""
        assert isinstance(ss, SynthSolutionStrategy)
        assert isinstance(problem_choice, types.IntType)
        assert isinstance(problem_is_robust, types.BooleanType)
        
        return self.dispatcher.registerMaster(ss, problem_choice, problem_is_robust)

    def slaveIDs(self):
        """Return a list of all slave IDs"""
        slaves = self.dispatcher.slaveIDs()
        return slaves
    
    def reset(self):
        if self.is_master:
            return self.dispatcher.reset()
        else:
            raise ValueError("Slave can't request reset")    
    
    def cleanup(self):
        """cleans up. the master has nothing more to do."""
        if self.is_master:
            return self.dispatcher.cleanup()
        else:
            return self.dispatcher.leaveService(self.slave_ID)
        
    def zombieCount(self):
        """Returns number of zombified tasks."""
        return self.dispatcher.zombieCount()

    def setZombificationTimeout(self, t):
        return self.dispatcher.setZombificationTimeout(t)
   
    #=====================================================================
    # String utilities
    def __str__(self):
        s = []
        s += ["PyroChannel\n"]
        s += ["Dispatcher={" + str( self.dispatcher ) + "}"]
        s += ["/PyroChannelData}\n"]
        return "".join(s)

class PyroChannelDispatcher(Pyro.core.ObjBase, ChannelData):
    """The PyroChannel is a class that can be used by the a dispatcher to mediate the data
    transport between the master and the slaves. The system setup would be:
        - one dispatcher (e.g. SynthDispatcher.py)
        - one master (e.g. SynthMaster.py)
        - one or more slaves (e.g. SynthSlave.py)
        
    The flow is:
        - The master posts one or more tasks in the channel by 
          calling the Master channel routines.
        - The slaves periodically check for tasks that they can perform.
          if a task is present, it is 'taken' from the dispatcher. It is then
          processed and the result is posted back to the dispatcher.
        - the master periodically checks for completed tasks, and once it finds
          some, they are taken from the queue
    
    The nice thing about using Pyro is that once the dispatcher has exposed this
    class, both master and slave can call it as if it were a normal Python object.
    """
    # CHECK: how about the byref passing of arguments?
    def __init__(self, cs, daemon):
        """
        """
        #preconditions
        
        assert isinstance(cs, ChannelStrategy)
        assert cs.channel_type == 'PyroBased'

        # check whether threading is used
        use_threading = (Pyro.config.PYRO_MULTITHREADED != 0)
        if use_threading:
            log.info("PYRO configured for threading, using thread safe objects")

        # initialize parent classes
        Pyro.core.ObjBase.__init__(self)
        ChannelData.__init__(self, cs, use_threading)

        # Pyro dispatcher
        try:
            self.ns = Pyro.naming.NameServerLocator().getNS()
        except NamingError:
            log.info( "No nameserver found. Please start a nameserver..." )
            raise

        self.daemon = daemon
        self.daemon.useNameServer(self.ns)

        log.info( "Using cluster ID: %s" % \
                  self._cs.cluster_id )
        log.info( "Using dispatcher name: %s" % \
                  self._cs.clusterDispatcherName )

        try:
            self.ns.createGroup( ":%s" % self._cs.getClusterGroup() )
        except NamingError:
            log.info( "Synth group already exists, unregistering..." )

        dispatcher_location = ":%s.%s" \
                              % ( self._cs.getClusterGroup(), \
                                  self._cs.clusterDispatcherName )

        try:
                self.ns.unregister( dispatcher_location )
        except NamingError:
                pass

        self.uri = self.daemon.connect( self, dispatcher_location )

    def __str__(self):
        s = []
        s += [" Last task id   : %s\n" % str(self._last_task_id)]
        s += [" Problem choice : %s (robust=%s)\n" %
              (str(self._problem_choice), self._problem_is_robust)]
        s += [" Zombie timeout : %s\n" % str(self._cs.zombification_timeout_secs)]
        s += [" Zombie count   : %s\n" % str(len(self._zombified_task_IDs))]
        if self._slave_IDs:
            s += ["  Slave_IDs:\n"]
            hosts = {}
            sstring = r"SLAVE-(.*)-.*"
            reobj = re.compile(sstring)
            for slave_ID in self._slave_IDs:
                match = reobj.search(slave_ID)
                if match:
                    host = match.group(1)
                    if host in hosts.keys():
                      hosts[host] += 1
                    else:
                      hosts[host] = 1
            for host in hosts.keys():
                s += ["   Host %20s: %s slaves\n" % (host, hosts[host])]
            
        if self._tasks_waiting:
            counts = {}
            for task in self._tasks_waiting:
                if not task.descr in counts.keys():
                    counts[task.descr] = 1
                else:
                    counts[task.descr] += 1

            s += ["  Tasks waiting: %s\n" % str(len(self._tasks_waiting))]
            for d in counts.keys():
              s += ["  %30s: %s\n" % (d, counts[d])]
                
        if self._tasks_running:
            counts = {}
            for task in self._tasks_running:
                if not task.descr in counts.keys():
                    counts[task.descr] = 1
                else:
                    counts[task.descr] += 1
            
            s += ["  Tasks running: %s\n" % str(len(self._tasks_running))]
            for d in counts.keys():
              s += ["  %30s: %s\n" % (d, counts[d])]
                
        if self._tasks_finished:
            counts = {}
            for task in self._tasks_finished:
                if not task.descr in counts.keys():
                    counts[task.descr] = 1
                else:
                    counts[task.descr] += 1
            
            s += ["  Tasks finished: %s\n" % str(len(self._tasks_finished))]
            for d in counts.keys():
              s += ["  %30s: %s\n" % (d, counts[d])]

        return "".join(s)
        
    def popFinishedTask(self):
        log.debug("Master wants to pop task")
        if self._tasks_finished:
            task = self._tasks_finished[0]
            self._tasks_finished.remove(task)
            log.debug("Master popped task: %s" % task)
        else:
            log.debug("No task to pop")
            task = None

        self.pruneZombieTasks()
        return task


