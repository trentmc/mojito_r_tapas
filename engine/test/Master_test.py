import unittest

import os
import shutil
import time

from adts import *
from engine.Channel import ChannelStrategy
from engine.Master import Master
from regressor.LinearModel import LinearBuildStrategy
from regressor.Probe import ProbeBuildStrategy
from problems.Library import whoami
from problems.Problems import ProblemFactory
from engine.SynthSolutionStrategy import SynthSolutionStrategy
import engine.EngineUtils as EngineUtils

def getTempChannelFile():
    time_str = str(time.time())
    time_str = time_str.replace(".", "_")
    name = "/tmp/channel_%s.db" % time_str
    #name = '/users/micas/tmcconag/data.pkl'  #HACK
    return name

def robustSphereFunc(scaled_point, rnd_point):
    assert scaled_point.is_scaled
    return sum(value**2 for value in scaled_point.itervalues()) + sum(rnd_point.values_list)
                
class MasterTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK
        self.print_routine = False #To set to True is a HACK
        
        #possible cleanup from prev run
        if os.path.exists('test_outpath1'): shutil.rmtree('test_outpath1')
        if os.path.exists('test_outpath2'): shutil.rmtree('test_outpath2')
        if os.path.exists('test_outpath3'): shutil.rmtree('test_outpath3')

    def makeGtoFast(self, ss):
        """Change SS to make GTO optimization fast, via small num opt points, and fast probe"""
        ss.gto_ss.max_num_opt_points = 3 #big testing speed difference! (default is 1500)
        lin_ss = LinearBuildStrategy(y_transforms=["lin"], target_nmse=0.10, regularize=True)
        ss.gto_ss.probe_ss = ProbeBuildStrategy(target_train_nmse = 0.10, max_rank=1, lin_ss=lin_ss)

    def localChannelStrategy(self):
        cs = ChannelStrategy()
        cs.cluster_id = 0
        cs.channel_type = 'Local'
        cs.channel_file = getTempChannelFile()
        return cs

    def testMassivelyMobj(self):
        """testMassivelyMobj: currently this test just makes sure that the run doesn't crash when many objectives"""
        if self.just1: return
        if self.print_routine: print whoami()
        
        ps = ProblemFactory().build(problem_choice=10) #problem 10 is massively mobj function
        self.assertEqual(len(ps.flattenedMetrics()), len(ps.metricsWithObjectives()))
        self.assertTrue(len(ps.flattenedMetrics()) >= 5)
        ss = SynthSolutionStrategy(do_novelty_gen=False, num_inds_per_age_layer=3)
                
        ss.setMaxNumGenerations(1)
        ss.max_num_neutral_vary_tries = 4
        ss.do_plot = False

        engine = Master(self.localChannelStrategy(), ps, ss, 'test_outpath1', None)
        engine.run()
        
    def testNominalSphere2d(self):
        if self.just1: return
        if self.print_routine: print whoami()
        self._testSphere2d(do_robust=False)
        
    def testRobustSphere2d(self):
        if self.just1: return
        if self.print_routine: print whoami()
        self._testSphere2d(do_robust=True)
        
    def _testSphere2d(self, do_robust):
        """This is a nice test for manual inspection of convergence"""        
        ps = ProblemFactory().build(problem_choice=15) #problem 15 is sphere2d
        if do_robust:
            ps.analyses[0].function = robustSphereFunc
            ps.devices_setup.makeRobust()
        num_inds_per_age_layer = 3
        ss = SynthSolutionStrategy(do_novelty_gen=False, num_inds_per_age_layer=num_inds_per_age_layer, age_gap=3)
        self.makeGtoFast(ss)

        if do_robust:
            assert ss.max_num_age_layers == 10
            ss.setMaxNumGenerations(11 * num_inds_per_age_layer + 1) #10 is enough to test all layers, 11 extra
        else:
            ss.setMaxNumGenerations(2 * num_inds_per_age_layer + 1) #enough to generate layers 0,1,2
            
        ss.do_plot = False

        cs = ChannelStrategy()
        cs.cluster_id = 0
        cs.channel_type = 'Local'
        cs.channel_file = getTempChannelFile()

        engine = Master(self.localChannelStrategy(), ps, ss, 'test_outpath1', None)
        engine.run()
        
    def testProblem2_NonNovel_Local(self):
        if self.just1: return
        if self.print_routine: print whoami()
        self._testProblem2_Helper(False, 'Local')
        
#     def testProblem2_Novel_Local(self):
#         if self.just1: return
#         self._testProblem2_Helper(True, 'Local')
        
#     def testProblem2_NonNovel_PyroBased(self):
#         if self.just1: return
#         self._testProblem2_Helper(False, 'PyroBased')
        
#     def testProblem2_Novel_PyroBased(self):
#         if self.just1: return
#         self._testProblem2_Helper(True, 'PyroBased')
        
    def _testProblem2_Helper(self, do_novelty_gen, channel_type):
        #--------------------------------------------------------
        #Set up and invoke round 1 run: start from scratch
        ps = ProblemFactory().build(problem_choice=2)
        num_inds_per_age_layer = 3
        ss = SynthSolutionStrategy(do_novelty_gen=do_novelty_gen, num_inds_per_age_layer=num_inds_per_age_layer, age_gap=3)
        # change the wait times
        ss.num_seconds_between_slave_task_requests = 0.1
        ss.num_seconds_between_master_task_requests = 0.1

        self.assertEqual(ss.do_novelty_gen, do_novelty_gen)
        self.assertEqual(ss.num_inds_per_age_layer, num_inds_per_age_layer)

        ss.setMaxNumGenerations(3+3+3+1)
        ss.max_num_neutral_vary_tries = 4
        ss.do_plot = False

        cs = ChannelStrategy()
        cs.cluster_id = 0
        cs.channel_type = channel_type
        cs.channel_file = getTempChannelFile()

        engine = Master(cs, ps, ss, 'test_outpath1', None)
        engine.run()
        self.assertTrue(os.path.exists('test_outpath1'))
        state = EngineUtils.loadSynthState('test_outpath1/state_gen0001.db', ps)
        self.assertTrue(len(state.allInds()) > 0)
        self.assertTrue(len(state.R_per_age_layer) == 1)
        self.assertTrue(state.generation <= 24)

        #--------------------------------------------------------
        #Set up and invoke round 2 run: continue from run 1 
        ss.setMaxNumGenerations(3)
        
        engine = Master(cs, ps, ss, 'test_outpath2', 'test_outpath1/state_gen0001.db')
        engine.run()

        #--------------------------------------------------------
        #Set up and invoke round 3 run:
        # go from a non-novelty run (round 2) to a novelty run (round 3)
        if False: #HACK - turned off until novelty working again
        #if not do_novelty_gen:
            ss = SynthSolutionStrategy(do_novelty_gen=True, num_inds_per_age_layer=num_inds_per_age_layer)
            ss.setMaxNumGenerations(2)
            ss.max_num_neutral_vary_tries = 4
            ss.do_plot = False 

            engine = Master(cs, ps, ss, 'test_outpath3', 'test_outpath2/state_gen0001.db')
            engine.run()
        
    def tearDown(self):
        #--------------------------------------------------------
        #cleanup all
        if os.path.exists('test_outpath1'): shutil.rmtree('test_outpath1')
        if os.path.exists('test_outpath2'): shutil.rmtree('test_outpath2')
        if os.path.exists('test_outpath3'): shutil.rmtree('test_outpath3')

if __name__ == '__main__':

    import logging
    logging.basicConfig()
    logging.getLogger('master').setLevel(logging.DEBUG)
    logging.getLogger('channel').setLevel(logging.DEBUG)
    
    unittest.main()
