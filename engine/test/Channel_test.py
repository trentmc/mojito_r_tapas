
import logging
import os
import time
import unittest

from util.ascii import stringToAscii
from engine.Channel import *
from engine.SynthSolutionStrategy import SynthSolutionStrategy

log = logging.getLogger("channel")

def getTempChannelFile():
    time_str = str(time.time())
    time_str = time_str.replace(".", "_")
    name = "/tmp/channel_%s.db" % time_str
    #name = '/users/micas/tmcconag/data.pkl'  #HACK
    return name

class ChannelTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

        self.ss = SynthSolutionStrategy(do_novelty_gen=False, num_inds_per_age_layer=3)
        self.problem_choice = 2
       
    def testTaskForSlave(self):
        if self.just1: return
        descr = ALL_TASK_DESCRIPTIONS[0]

        master_ID = 'Master1'
        
        #test successful creation of TaskData (currently no real ways to fail)
        data = TaskData()

        #test failed creations of TaskForSlave
        self.assertRaises(AssertionError, TaskForSlave, master_ID, 'foo', data)
        self.assertRaises(AssertionError, TaskForSlave, master_ID, descr, 'foo')

        #test successful creation of TaskForSlave
        task = TaskForSlave(master_ID, descr, data)
        self.assertEqual(task.descr, descr)
        self.assertEqual(task.task_data, data)
        self.assertEqual(task.result_data, None)
        self.assertEqual(task.ID, None)
        self.assertEqual(task.master_ID, master_ID)
        self.assertTrue('TaskForSlave=' in str(task))

        #test failed creations of ResultData
        self.assertRaises(AssertionError, ResultData, None, {0:1})
        self.assertRaises(AssertionError, ResultData, 'my_slave_ID', 'foo')
        
        #test successful creation of ResultData
        result_data = ResultData('my_slave_ID', {0:3})
        self.assertEqual(result_data.slave_ID, 'my_slave_ID')
        self.assertEqual(result_data.num_evaluations_per_analysis, {0:3})

        #test failed attachResult
        self.assertRaises(AssertionError, task.attachResult, 'foo')

        #test successful attachResult
        task.attachResult(result_data)
        self.assertEqual(task.result_data, result_data)

    def testChannelData(self):
        if self.just1: return
        #setup
        cs = ChannelStrategy()

        #tests specific to ChannelData
        c = ChannelData(cs)
        self.assertEqual(c._slave_IDs, [])
        self.assertEqual(c._tasks_waiting, [])
        self.assertEqual(c._tasks_running, [])
        self.assertEqual(c._tasks_finished, [])
        self.assertEqual(c._cs, cs)
        self.assertTrue('ChannelData=' in str(c))

        #tests that Channels share
        #self._testChannelInterface(c)
        
    def testFileChannel(self):
        if self.just1: return

        log.info("testFileChannel: begin")

        #setup
        cs = ChannelStrategy()
        cs.channel_type = 'FileBased'
        cs.channel_file = getTempChannelFile()

        #tests specific to FileChannel
        # -round 1: test by direct creation of Channel object
        self.assertFalse(os.path.exists(cs.channel_file))
        c = FileChannel(cs, True)
        self.assertTrue(os.path.exists(cs.channel_file))
        c.cleanup()

        # -round 2: test by creation from perspective of master and slave
        c = ChannelFactory( cs ).buildChannel( True )
        self.assertTrue(os.path.exists(cs.channel_file))
        c2 = ChannelFactory( cs ).buildChannel( False ) #points to same channel as 'c'
        c.cleanup()
        c2.cleanup()

        #
        self.assertTrue('Channel=' in str(c))
        
        #tests that Channels share
        c = ChannelFactory( cs ).buildChannel( True )
        self._testChannelInterface(c)
        c.cleanup()
        
        #test zombification
        c = ChannelFactory( cs ).buildChannel( True )
        self._testZombification(c)
        c.cleanup()
        
        log.info("testFileChannel: done")
        
    def testPyroChannel(self):
        if self.just1: return

        log.info("testPyroChannel: begin")
        
        #setup
        cs = ChannelStrategy()
        cs.channel_type = 'PyroBased'
        cs.cluster_id = 'unittest'

        # warning: make sure a dispatcher exists

        #tests specific to PyroChannel
        # -round 1: test by direct creation of Channel object
        pass

        #tests that Channels share
        c = ChannelFactory( cs ).buildChannel( True )
        self._testChannelInterface(c)
        c.cleanup()
        
        #test zombification
        c = ChannelFactory( cs ).buildChannel( True )        
        self._testZombification(c)
        c.cleanup()
        
        log.info("testPyroChannel: done")

    def _testChannelInterface(self, c):
        """Test objects that implement the Channel interface
        """
        master_ID = 'MyMaster'
        self.assertTrue(isinstance(c, FileChannel) or \
                        isinstance(c, PyroChannel))
        
        # start from a clean channel
        c.reset()
        
        #test slaves in service
        c.reportForService(slave_ID = 'slave32')
        self.assertEqual(c.slaveIDs(), ['slave32'])
        # don't allow duplicate registrations
        self.assertRaises(ValueError, c.reportForService, 'slave32')
        self.assertEqual(c.slaveIDs(), ['slave32'])
        c.reportForService(slave_ID = 'slave31')
        self.assertEqual(c.slaveIDs(), ['slave32', 'slave31']) #order = order in

        # test unregistering
        c.leaveService(slave_ID = 'slave32')
        self.assertEqual(c.slaveIDs(), ['slave31'])
        self.assertRaises(ValueError, c.leaveService, 'slave32')
        # now the slave should be able to re-register
        c.reportForService(slave_ID = 'slave32')

        #test tasks...

        #no data
        self.assertEqual(c.popTask(), None)
        self.assertEqual(c.tasksWaiting(), [])
        self.assertTrue(len(str(c)) > 0)

        #no data
        self.assertEqual(c.popFinishedTasks(), [])

        #test result generation
        result_data = ResultData('myID', {0:1})

        descr0 = ALL_TASK_DESCRIPTIONS[0]
        descr1 = ALL_TASK_DESCRIPTIONS[2]

        task0 = TaskForSlave(master_ID, descr0, TaskData())
        task1 = TaskForSlave(master_ID, descr1, TaskData())

        #check whether the nominal channel behavior is correct
        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)

        #master> add task w/o attached result to channel
        c.pushTasks([task1])

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 1)
        self.assertEqual(len(c.tasksRunning()), 0)

        #slave> pop task
        slave_task = c.popTask()
        self.assertNotEqual(slave_task, None)

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 1)

        #slave> add result to task
        slave_task.attachResult(result_data)
        #slave> push the task back into the channel
        c.pushResult(slave_task)

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)

        #master> get results
        result_tasks = c.popFinishedTasks()
        
        self.assertEqual(len(result_tasks), 1)
        self.assertEqual(result_tasks[0].descr, descr1)
        self.assertEqual(c.popFinishedTasks(), [])

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)
        
        #check whether the nominal channel behavior is correct (2)

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)

        #master> add task w/o attached result to channel
        c.pushTasks([TaskForSlave(master_ID, descr1, TaskData())])
        c.pushTasks([TaskForSlave(master_ID, descr0, TaskData())])

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 2)
        self.assertEqual(len(c.tasksRunning()), 0)

        #slave> pop task
        slave_task = c.popTask()

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 1)
        self.assertEqual(len(c.tasksRunning()), 1)

        #slave> add result to task
        slave_task.attachResult(result_data)
        #slave> push the task back into the channel
        c.pushResult(slave_task)

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 1)
        self.assertEqual(len(c.tasksRunning()), 0)

        #slave> pop task
        slave_task = c.popTask()

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 1)

        #slave> add result to task
        slave_task.attachResult(result_data)
        #slave> push the task back into the channel
        c.pushResult(slave_task)

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)

        #master> get results
        result_tasks = c.popFinishedTasks()
        
        self.assertEqual(len(result_tasks), 2)
        self.assertEqual(result_tasks[0].descr, descr1)
        self.assertEqual(result_tasks[1].descr, descr0)
        self.assertEqual(c.popFinishedTasks(), [])

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)

    def _testZombification(self, c):
        """Test whether zombification works
        """
        master_ID = 'MyMaster'
        self.assertTrue(isinstance(c, FileChannel) or \
                        isinstance(c, PyroChannel))

        test_time = 1
        
        # start from a clean channel
        c.reset()
        c.setZombificationTimeout( test_time ) # note: only the dispatches value counts!

        #test result generation
        result_data = ResultData('myID', {0:1})

        descr0 = ALL_TASK_DESCRIPTIONS[0]
        descr1 = ALL_TASK_DESCRIPTIONS[2]

        task0 = TaskForSlave(master_ID, descr0, TaskData())
        task1 = TaskForSlave(master_ID, descr1, TaskData())
        
        #check whether the nominal channel behavior is correct
        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)
        self.assertEqual(c.zombieCount(), 0)

        #master> add task w/o attached result to channel
        c.pushTasks([task1])

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 1)
        self.assertEqual(len(c.tasksRunning()), 0)
        self.assertEqual(c.zombieCount(), 0)

        #slave> pop task
        slave_task = c.popTask()

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 1)
        self.assertEqual(c.zombieCount(), 0)

        log.debug("waiting long enough for zombification...")
        time.sleep(test_time + 0.5)

        #master> pop finished tasks to trigger zombie cleanup
        self.assertEqual(c.popFinishedTasks(), [])
        
        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)
        self.assertEqual(c.zombieCount(), 1)

        #slave> add result to task
        slave_task.attachResult(result_data)
        #slave> push the task back into the channel
        c.pushResult(slave_task) # this should not fail

        # no result should be present for a zombified
        # task
        self.assertEqual(c.popFinishedTasks(), [])

        # test survival
        self._testZombificationSurvival(c)
        
    def _testZombificationSurvival(self, c):
        """Test whether zombification survival works
        """
        master_ID = 'MyMaster'
        self.assertTrue(isinstance(c, FileChannel) or \
                        isinstance(c, PyroChannel))

        test_time = 1
        
        # start from a clean channel
        c.reset()
        c.setZombificationTimeout( test_time ) # note: only the dispatches value counts!

        #test result generation
        result_data = ResultData('myID', {0:1})

        descr0 = ALL_TASK_DESCRIPTIONS[0]

        task0 = TaskForSlave(master_ID, descr0, TaskData())

        #indicate that the task can survive zombification
        task0.task_data.ignore_zombification = True
        
        #check whether the nominal channel behavior is correct
        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)
        self.assertEqual(c.zombieCount(), 0)

        #master> add task w/o attached result to channel
        c.pushTasks([task0])

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 1)
        self.assertEqual(len(c.tasksRunning()), 0)
        self.assertEqual(c.zombieCount(), 0)

        #slave> pop task
        slave_task = c.popTask()

        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 1)
        self.assertEqual(c.zombieCount(), 0)

        log.debug("waiting long enough for zombification...")
        time.sleep(test_time + 0.5)

        #master> pop finished tasks to trigger zombie cleanup
        self.assertEqual(c.popFinishedTasks(), [])
        
        #dispatcher> check task queue status
        self.assertEqual(len(c.tasksWaiting()), 0)
        self.assertEqual(len(c.tasksRunning()), 0)
        self.assertEqual(c.zombieCount(), 1)

        #slave> add result to task
        slave_task.attachResult(result_data)
        #slave> push the task back into the channel
        c.pushResult(slave_task) # this should not fail

        # the result should be present since this is one that
        # survives zombification
        self.assertEqual(len(c.popFinishedTasks()), 1)

    def tearDown(self):
        pass
    
if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger("channel").setLevel(logging.DEBUG)
    
    unittest.main()
