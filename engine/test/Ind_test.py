import copy
import types
import unittest

from adts import *
from engine.Ind import *
from problems.Problems import ProblemFactory
from util.constants import BAD_METRIC_VALUE

def some_function(x):
    return x+2

def function2(x):
    return x-5

def createRandomInd(ps):
    """Creates an non-novel Ind by randomly drawing from ps.embedded_part.part.point_meta."""
    point_meta = ps.embedded_part.part.point_meta
    unscaled_point = point_meta.createRandomUnscaledPoint(with_novelty=False)
    unscaled_optvals = [unscaled_point[var] for var in ps.ordered_optvars]
    ind = NsgaInd(unscaled_optvals, ps)
    ind.genetic_age = 0
    ind.setAncestry([])
    return ind

class IndTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK
        
        an = FunctionAnalysis(some_function, [EnvPoint(True)], 10, 20, False, 10, 20)
        an2 = FunctionAnalysis(function2, [EnvPoint(True)], 10, 20, False, 10, 20)
        dummy_part = AtomicPart('R',['1','2'], PointMeta({}))
        emb_part = EmbeddedPart(dummy_part, dummy_part.unityPortMap(), {})
        self.ps = ProblemSetup(emb_part, [an, an2], None)
        self.ps.devices_setup.makeRobust()

    def testID(self):
        if self.just1: return

        base_ind = Ind([], self.ps)

        #a normal shallow copy will have the same ID
        shallow_copied_ind = copy.copy(base_ind)
        self.assertEqual(base_ind.ID, shallow_copied_ind.ID)

        #to get an ind with a different ID, use copyWithNewId
        # -note that it will retain _everything_ else, including the cached items
        special_copied_ind = base_ind.copyWithNewID()
        self.assertNotEqual(base_ind.ID, special_copied_ind.ID)

        #check rnd_IDs too
        self.assertTrue(base_ind.rnd_IDs == shallow_copied_ind.rnd_IDs == special_copied_ind.rnd_IDs)

    def testNominal(self):
        if self.just1: return
        
        self.ps.devices_setup.makeNominal()
        an = self.ps.analyses[0]
        an2 = self.ps.analyses[1]
        metric_names = [metric.name for metric in self.ps.flattenedMetrics()]
        
        ind = Ind([], self.ps)
        
    def test1(self):
        if self.just1: return
        an = self.ps.analyses[0]
        an2 = self.ps.analyses[1]
        metric_names = [metric.name for metric in self.ps.flattenedMetrics()]
        
        ind = Ind([], self.ps)

        #check out basic attributes
        self.assertEqual(ind.unscaled_optvals, [])
        self.assertEqual(sorted(ind.sim_requests_made.keys()), sorted(ind.rnd_IDs))
        for rnd_ID in ind.rnd_IDs:
            self.assertEqual(sorted(ind.sim_requests_made[rnd_ID].keys()), sorted([an.ID, an2.ID]))
            self.assertEqual(sorted(ind.sim_results[rnd_ID].keys()), sorted(metric_names))
            
        self.assertTrue(len(str(ind)) > 0)

        e = an2.env_points[0]
        rnd_ID = ind.rnd_IDs[3] #arbitrary rnd point

        #try setting a sim result (and see that error)
        self.assertRaises(ValueError, ind.setSimResults, {an2.metric.name:33.2}, rnd_ID, an2, e)

        #ok, set a sim result _properly_
        self.assertFalse(ind.simRequestMade(rnd_ID, an2, e))
        ind.reportSimRequest(rnd_ID, an2, e)
        self.assertTrue(ind.simRequestMade(rnd_ID, an2, e))

        self.assertEqual(ind.sim_results[rnd_ID][an2.metric.name][e.ID], None)
        ind.setSimResults({an2.metric.name:33.2}, rnd_ID, an2, e)
        self.assertEqual(ind.sim_results[rnd_ID][an2.metric.name][e.ID], 33.2)

        #see that we can't report a request twice, or set a sim result twice
        self.assertRaises(ValueError, ind.reportSimRequest, rnd_ID, an2, e)
        self.assertRaises(ValueError, ind.setSimResults, {an2.metric.name:10.5}, rnd_ID, an2, e)

        #fully evaluated yet? (no)
        for num_rnd_points in range(1, len(ind.rnd_IDs)):
            self.assertFalse(ind.fullyEvaluated(num_rnd_points))
        for cur_rnd_ID in ind.rnd_IDs:
            self.assertFalse(ind.fullyEvaluatedAtRndPoint(cur_rnd_ID))

        #retrieve a worst-case value; see that caching didn't happen
        self.assertEqual(ind.worstCaseMetricValueAtRndPoint(rnd_ID, an2.metric.name), 33.2)
        self.assertFalse(ind._cached_wc_metvals[rnd_ID].has_key(an2.metric.name))

        #set enough sim results to be fully evaluated, and re-test
        ind.reportSimRequest(rnd_ID, an, an.env_points[0])
        ind.setSimResults({an.metric.name:10.0}, rnd_ID, an, an.env_points[0])
        self.assertTrue(ind.fullyEvaluatedAtRndPoint(rnd_ID))

        #retrieve a worst-case value; see that caching did happen
        self.assertEqual(ind.worstCaseMetricValueAtRndPoint(rnd_ID, an2.metric.name), 33.2)
        self.assertTrue(ind._cached_wc_metvals[rnd_ID].has_key(an2.metric.name))
        self.assertEqual(ind._cached_wc_metvals[rnd_ID][an2.metric.name], 33.2)

        #not fully evaluated across the board yet, so do that
        self.assertFalse(ind.fullyEvaluated(3))
        self.assertFalse(ind.fullyEvaluated(30))
        num_rnd_points = 0
        for cur_rnd_ID in ind.rnd_IDs:
            num_rnd_points += 1
            if cur_rnd_ID != rnd_ID:
                ind.reportSimRequest(cur_rnd_ID, an, an.env_points[0])
                ind.setSimResults({an.metric.name:10.0}, cur_rnd_ID, an, an.env_points[0])
                ind.reportSimRequest(cur_rnd_ID, an2, an2.env_points[0])
                ind.setSimResults({an2.metric.name:10.0}, cur_rnd_ID, an2, an2.env_points[0])
                
            self.assertTrue(ind.fullyEvaluated(num_rnd_points))
                
            if (num_rnd_points < len(ind.rnd_IDs)) and ((cur_rnd_ID+1) != rnd_ID):
                self.assertFalse(ind.fullyEvaluated(num_rnd_points+1))
                    
    def testIsBad(self):        
        if self.just1: return
        an = self.ps.analyses[0]
        an2 = self.ps.analyses[1]

        #fresh new inds are never bad
        ind = Ind([], self.ps)
        rnd_ID = ind.rnd_IDs[2]
        self.assertFalse(ind.isBadAtRndPoint(rnd_ID))

        #set a non-bad metric value; ind should remain non-bad
        ind.reportSimRequest(rnd_ID, an2, an2.env_points[0])
        ind.setSimResults({an2.metric.name:33.2}, rnd_ID, an2, an2.env_points[0])
        self.assertFalse(ind.isBadAtRndPoint(rnd_ID))

        #set a bad metric value, making ind bad
        ind.reportSimRequest(rnd_ID, an, an.env_points[0])
        ind.setSimResults({an.metric.name:BAD_METRIC_VALUE}, rnd_ID, an, an.env_points[0])
        self.assertTrue(ind.isBadAtRndPoint(rnd_ID))
        self.assertEqual(ind.worstCaseMetricValueAtRndPoint(rnd_ID, an.metric.name), BAD_METRIC_VALUE)
        self.assertEqual(ind.constraintViolationAtRndPoint(rnd_ID, {}), float('Inf'))
        self.assertFalse(ind.isFeasibleAtRndPoint(rnd_ID))

        #can it go into a string?
        self.assertTrue(len(str(ind)) > 0)

    def testForceBad(self):
        if self.just1: return
        ind = Ind([], self.ps)
        rnd_ID = ind.rnd_IDs[2]
        ind.forceFullyBadAtRndPoint(rnd_ID)
        self.assertTrue(ind.isBadAtRndPoint(rnd_ID))
        self.assertTrue(ind.fullyEvaluatedAtRndPoint(rnd_ID))
        
    def testNsgaInd(self):
        if self.just1: return
        
        ps = ProblemFactory().build(1)
        ind = NsgaInd([0.1] * len(ps.ordered_optvars), ps)
        
    def testNominalNetlist(self):
        if self.just1: return

        #not breaking is a sufficient pass here.  (Detailed tests in Analysis_test)
        for loop_i in range(4):
            ps = ProblemFactory().build(41) #problem 41 is single-stage WL dsViAmp

            #nominal devices_setup
            ps.devices_setup.makeNominal()
            ind = createRandomInd(ps)
            netlist1 = ind.nominalNetlist()
            netlist2 = ind.nominalNetlist(annotate_bb_info=True, add_infostring=True)

            #non-nominal devices_setup
            ps.devices_setup.makeRobust()
            ind2 = createRandomInd(ps)
            ind2.unscaled_optvals = ind.unscaled_optvals
            netlist3 = ind2.nominalNetlist()
            netlist4 = ind2.nominalNetlist(annotate_bb_info=True, add_infostring=True)

            #nominalNetlist from nominal vs. non-nominal ps should line up
            self.assertEqual(netlist1, netlist3)
            self.assertEqual(netlist2, netlist4)
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    #if desired, this is where logging would be set up
    
    unittest.main()
