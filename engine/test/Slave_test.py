import unittest

import os
import shutil

from adts import *
from problems.Library import whoami
from problems.Problems import ProblemFactory
from engine.Slave import *
from engine.Channel import ChannelStrategy
from engine.Channel import TaskData, TaskForSlave, ChannelFactory
from engine.SynthSolutionStrategy import SynthSolutionStrategy

def getTempChannelFile():
    time_str = str(time.time())
    time_str = time_str.replace(".", "_")
    name = "/tmp/channel_%s.db" % time_str
    #name = '/users/micas/tmcconag/data.pkl'  #HACK
    return name

class SlaveTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False         #To make True is a HACK
        self.skip_pyro = False     #To make True is a HACK
        self.print_routine = False #To make True is a HACK
        self.do_profiling = False  #To make True is a HACK
        
        self.path = "test_outpath1/"
        self.channel_file = self.path + "temp_channel.db"

        #maybe override just1
        self.just1 = self.just1 or self.do_profiling
               
        #possible cleanup from prev run
        if os.path.exists(self.path): shutil.rmtree(self.path)

    def testBasic(self):
        if self.just1: return
        if self.print_routine: print whoami()
        problem_choice = 2
        
        self.assertRaises(AssertionError, Slave, None)
        self.assertRaises(AssertionError, Slave, "foo")

        cs = ChannelStrategy()
        cs.cluster_id = 0
        cs.channel_type = 'FileBased'
        cs.channel_file = getTempChannelFile()

        ss = SynthSolutionStrategy(do_novelty_gen=False, num_inds_per_age_layer=3)
        ss.max_num_inds = 31
        ss.num_seconds_between_slave_task_requests = 0.1

        os.mkdir(self.path)
        self.assertFalse(os.path.exists(cs.channel_file))
        channel_for_master = ChannelFactory(cs).buildChannel(True)
        channel_for_master.registerMaster(ss, problem_choice, False)
        self.assertTrue(os.path.exists(cs.channel_file))

        slave = Slave(cs)
        
        self.assertEqual(slave.ps.problem_choice, problem_choice)
        self.assertEqual(slave.ss.max_num_inds, ss.max_num_inds)

        self.assertTrue("SLAVE-" in slave.ID)
        self.assertEqual(slave.task, None)
        self.assertEqual(slave.num_inds, 0)
        self.assertEqual(slave.num_evaluations_per_analysis, {})

    def testProblem2_NonNovel_FileBased(self):
        if self.just1: return
        if self.print_routine: print whoami()
        self._testProblem2_Helper(False, 'FileBased')
        
    def testProblem2_Novel_FileBased(self):
        return #HACK fix when novel is working
        if self.just1: return
        if self.print_routine: print whoami()
        self._testProblem2_Helper(True, 'FileBased')
        
    def testProblem2_NonNovel_PyroBased(self):
        if self.just1: return
        if self.skip_pyro: return
        if self.print_routine: print whoami()
        self._testProblem2_Helper(False, 'PyroBased')
        
    def testProblem2_Novel_PyroBased(self):
        return #HACK fix when novel is working
        if self.just1: return
        if self.skip_pyro: return
        if self.print_routine: print whoami()
        self._testProblem2_Helper(True, 'PyroBased')
        
    def _testProblem2_Helper(self, do_novelty_gen, channel_type):
        #
        problem_choice = 2
        
        #set up channel
        os.mkdir(self.path)

        cs = ChannelStrategy( )
        cs.cluster_id = 0
        cs.channel_type = channel_type
        cs.channel_file = getTempChannelFile()

        ss = SynthSolutionStrategy(do_novelty_gen=do_novelty_gen, num_inds_per_age_layer=3)
        ss.max_num_inds = 31
        ss.num_seconds_between_slave_task_requests = 0.1

        channel_for_master = ChannelFactory(cs).buildChannel(True)
        channel_for_master.registerMaster(ss, problem_choice, False)
        channel_for_master.reset()
        
        #set up slave
        slave = Slave(cs)
                        
        #test when no tasks available
        log.info("Running some wait iterations...")
        for i in range(2):
            slave.run__oneIter()
            self.assertEqual(slave.task, None)

        #put a task onto channel
        task_data = TaskData()
        master_ID = "my_master_ID"
        task = TaskForSlave(master_ID, "Generate random ind", task_data)
        channel_for_master.pushTasks([task])

        #let slave grab task and do it
        self.assertEqual(slave.task, None)
        slave.run__oneIter()
        self.assertEqual(slave.task, None) #should be done again

        #grab result from channel, and verify it
        task_with_result = channel_for_master.popFinishedTasks()[0]
        self.assertEqual(task_with_result.descr, "Generate random ind")
        self.assertTrue(hasattr(task_with_result.result_data, "ind"))

        num_evals = task_with_result.result_data.num_evaluations_per_analysis.values()[0]
        self.assertTrue(num_evals > 0)
        self.assertEqual(task_with_result.result_data.slave_ID, slave.ID)        

    def test_ProfileRandomIndGeneration(self):
        """Profile random ind generator; only turn this on for special temporary profiling needs"""
        if not self.do_profiling:
            return
        if self.print_routine: print whoami()
        print ""

        print "Prepare data: begin"
        include_circuit_DOCs = False
        num_topologies = 100
        num_changes_per_topology = 40
        
        if include_circuit_DOCs:
            problem_choice = 13 # == function+circuit analysis DOCs test problem
        else:
            problem_choice = 12 # == function analysis DOCs test problem
        

        os.mkdir(self.path)
        cs = ChannelStrategy()
        cs.cluster_id = 0
        cs.channel_type = "Local"
        cs.channel_file = getTempChannelFile()

        ps = ProblemFactory().build(problem_choice)
        ss = SynthSolutionStrategy(do_novelty_gen=False, num_inds_per_age_layer=3)
        slave = Slave(cs, ps, ss)
        slave.setDoingProfiling()

        print ps
        print "include_circuit_DOCs=%s, num_topologies=%d, num_changes_per_topology=%d" % \
              (include_circuit_DOCs, num_topologies, num_changes_per_topology)
        
        print "Prepare data: done"
            
        #do the profiling...
        import cProfile
        filename = "/tmp/nondom.cprof"
        
        print "Do actual profile run: begin"
        prof = cProfile.runctx(
            "ret_code = self._generateRandomInds(slave, num_topologies, num_changes_per_topology, "
            "include_circuit_DOCs)",
            globals(), locals(), filename)
        print "Do actual profile run: done"

        print "Analyze profile data:"
        import pstats
        p = pstats.Stats(filename)
        p.strip_dirs() #remove extraneous path from all module names

        print ""
        print "======================================================="
        print "Sort by cumulative time in a function (and children)"
        p.sort_stats('cum').print_stats(30)
        print ""

        print ""
        print "======================================================="
        print "Sort by time in a function (no recursion)"
        p.sort_stats('time').print_stats(30)

        print "Analyze profile data: done"

        print "Done overall"

    def _generateRandomInds(self, slave, num_topologies, num_changes_per_topology, include_circuit_DOCs):
        """Helper to test_ProfileRandomGeneration"""
        with_novelty = False
        testdata = ([], [])

        DOC_metric_names = slave.ps.DOCMetricNames()
        eps = 1.0e-10
        print "DOC_metric_names = %s" % DOC_metric_names

        for top_i in range(num_topologies):
            print "Done %d / %d topologies" % (top_i+1, num_topologies)
            
            ind = slave._newRandomInd(with_novelty, testdata)
            slave._evalInd(ind)
            best_ind = ind
            best_cost = ind.DOCsCost()

            for change_i in range(num_changes_per_topology):
                #the next line(s) emulates what was originally:
                best_cost < eps

                slave._randomChangeNonChoiceVars(ind)
                slave._evalInd(ind)
                
                ind_cost = ind.DOCsCost()
                if ind_cost < best_cost:
                    best_ind, best_cost = ind, ind_cost

        
    def tearDown(self):
        if os.path.exists(self.path): shutil.rmtree(self.path)

if __name__ == '__main__':

    import logging
    logging.basicConfig()
    logging.getLogger('slave').setLevel(logging.DEBUG)
    logging.getLogger('channel').setLevel(logging.DEBUG)
    
    unittest.main()
