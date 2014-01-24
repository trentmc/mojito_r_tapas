import unittest

import random
import os
import time

#FIXME: unit tests for EngineUtils still need to be written!
from adts import *
from engine.EngineUtils import *
from engine.EngineUtils import \
     _mergeNondominatedSort, _debNondominatedSort, _mergeNondominatedFilter, _simpleNondominatedFilter, \
     _LiteInd, _liteYield
from engine.Ind import Ind
from util.ascii import *
from util.constants import INF

def getTempFile():
    while True:
        time_str = str(time.time())
        time_str = time_str.replace(".", "_")
        name = "/tmp/tst_%s.db" % time_str
        if not os.path.exists(name):
            break
    return name

def f1(x): return x+1
def f2(x): return x+2
def f3(x): return x+3
def f4(x): return x+4
def f5(x): return x+5
def f6(x): return x+6

def dummyPart():
    return AtomicPart('R',['1','2'],PointMeta({}))

def psFromAnalyses(analyses):
    """Easy construction of a PS from a list of analyses"""
    dummy_part = dummyPart()
    emb_part = EmbeddedPart(dummy_part, dummy_part.unityPortMap(), {})
    ps = ProblemSetup(emb_part, analyses, None)
    ps.problem_choice = 0
    return ps

def oneMetricPS_Maximize(thr1):
    """Makes a PS from just 1 metric. (Maximize past thr1).
    """
    an_f1 = FunctionAnalysis(f1, [EnvPoint(True)], thr1, INF, True, 0, 1)
    return psFromAnalyses([an_f1])

def oneMetricPS_Minimize(thr1):
    """Makes a PS from just 1 metric. (Minimize past thr1).
    """
    an_f1 = FunctionAnalysis(f1, [EnvPoint(True)], -INF, thr1, True, 0, 1) 
    return psFromAnalyses([an_f1])

def oneMetricSimulateInd(ind, num_rnd_points):
    """Evaluate an ind, with sim_values of [0.0, 1.0, ..., num_rnd_points-1]"""
    ps = ind._ps
    an = ps.analyses[0]
    e = an.env_points[0]

    for (i, rnd_ID) in enumerate(ind.rnd_IDs[:num_rnd_points]):
        ind.reportSimRequest(rnd_ID, an, e)
        sim_value = float(i)
        sim_results = {an.metric.name : sim_value}
        ind.setSimResults(sim_results, rnd_ID, an, e)

def twoMetricPS_MaxMin(thr1, thr2):
    """Makes a PS from two metrics:
    -first metric is maximize past thr1
    -second metric is minimize past thr2
    """
    #to mos3, add dummy function DOC: constraint of get w > l
    an_f1 = FunctionAnalysis(f1, [EnvPoint(True)], thr1, INF, True, 0, 1) 
    an_f2 = FunctionAnalysis(f2, [EnvPoint(True)], -INF, thr2, True, 0, 1)
    return psFromAnalyses([an_f1, an_f2])

def twoMetricPS_MinMin(thr1, thr2):
    """Like twoMetricPS_MaxMin, except minimize both metrics
    """
    #to mos3, add dummy function DOC: constraint of get w > l
    an_f1 = FunctionAnalysis(f1, [EnvPoint(True)], -INF, thr1, True, 0, 1) 
    an_f2 = FunctionAnalysis(f2, [EnvPoint(True)], -INF, thr2, True, 0, 1) 
    return psFromAnalyses([an_f1, an_f2])

def twoMetricPS_MinLtconstraint(thr1, thr2):
    """Makes a PS from two metrics:
    -first metric is minimize past thr1
    -second metric is just a constraint: get <= thr2 (aim is still minimize)
    """
    #to mos3, add dummy function DOC: constraint of get w > l
    an_f1 = FunctionAnalysis(f1, [EnvPoint(True)], -INF, thr1, True, 0, 1) 
    an_f2 = FunctionAnalysis(f2, [EnvPoint(True)], -INF, thr2, False, 0, 1) 
    return psFromAnalyses([an_f1, an_f2])

def threeMetricPS():
    """Makes a PS from three metrics:
    -all have a threshold of 0
    -all aim to maximize
    """
    an_fs = [FunctionAnalysis(f, [EnvPoint(True)], 0.0, INF, True, 0, 1)
             for f in [f1, f2, f3]]
    return psFromAnalyses(an_fs)

def sixMetricPS():
    """Like threeMetricsPS, but has six metrics"""
    an_fs = [FunctionAnalysis(f, [EnvPoint(True)], 0.0, INF, True, 0, 1)
             for f in [f1, f2, f3, f4, f5, f6]]
    return psFromAnalyses(an_fs)

def indsFromResultsAndPS(results, ps):
    """Construct a list of inds, given results tuples 'results' and problem setup 'ps'."""
    return _indsFromResultsAndPS_sixMetrics(results, ps) #HACK test
    if ps.numMetrics() == 6:
        return _indsFromResultsAndPS_sixMetrics(results, ps)
    else:
        return _indsFromResultsAndPS_oneOrTwoMetrics(results, ps)

def _indsFromResultsAndPS_sixMetrics(results, ps):
    """Construct a list of inds, given results tuples 'results' and problem setup 'ps'."""
    nom_rnd_ID = ps.nominalRndPoint().ID #just nominal for now
    
    inds = []
    for result in results:
        if not isinstance(result, types.TupleType):
            result = (result, )
            
        next_ind = Ind([], ps)

        for (an_index, an) in enumerate(ps.analyses):
            e = an.env_points[0]
            next_ind.reportSimRequest(nom_rnd_ID, an, e)
            sim_results = {an.metric.name : result[an_index]}
            next_ind.setSimResults(sim_results, nom_rnd_ID, an, e)

        inds.append(next_ind)
        
    return inds

def _indsFromResultsAndPS_oneOrTwoMetrics(results, ps):
    """Construct a list of inds, given results tuples 'results' and problem setup 'ps'.
    """
    nom_rnd_ID = ps.nominalRndPoint().ID #just nominal for now
    
    an_f1 = ps.analyses[0]
    e1 = an_f1.env_points[0]
    if ps.numMetrics() > 1:
        an_f2 = ps.analyses[1]
        e2 = an_f2.env_points[0]

    inds = []
    for result in results:
        if ps.numMetrics() > 1:
            (res1, res2) = result
        else:
            res1 = result
            
        next_ind = Ind([], ps)

        next_ind.reportSimRequest(nom_rnd_ID, an_f1, e1)
        next_ind.setSimResults({an_f1.metric.name:res1}, an_f1, e1)

        if ps.numMetrics() > 1:
            next_ind.reportSimRequest(nom_rnd_ID, an_f2, e2)
            next_ind.setSimResults({an_f2.metric.name:res2}, nom_rnd_ID, an_f2, e2)

        inds.append(next_ind)
        
    return inds


def _IDs(inds):
    """Returns a list of IDs, one for each ind."""
    return [ind.ID for ind in inds]

class EngineUtilsTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False         #To make True is a HACK
        self.do_profiling = False  #To make True is a HACK
        
        #maybe override just1
        self.just1 = self.just1 or self.do_profiling
        
    def testNondominatedSort_empty(self):
        if self.just1: return
        F = nondominatedSort([], {'metname':(1,2)})
        self.assertEqual(F,[[]])        
        
    def testNondominatedSort1(self):
        """Test on a synthetic dataset where we have
        -inds on 0th layer: 0,5,6 (feasible)
        -inds on 1st layer: 1,4  (feasible)
        -inds on 2nd layer: 2 (infeasible)
        -inds on 3rd layer: 3 (infeasible)
        """
        if self.just1: return

        res = [(2,1), (2,3), (1,4), (0,5), (3,4), (4,3), (3,2)] #(f1,f2) per ind
        
        ps = twoMetricPS_MaxMin(1.5, 10.0)

        inds = indsFromResultsAndPS(res, ps)

        an_f1 = ps.analyses[0]
        an_f2 = ps.analyses[1]

        #test ind.isFeasibleAtNominal()
        self.assertTrue(inds[0].isFeasibleAtNominal())
        self.assertTrue(inds[1].isFeasibleAtNominal())
        self.assertFalse(inds[2].isFeasibleAtNominal())
        self.assertFalse(inds[3].isFeasibleAtNominal())
        self.assertTrue(inds[4].isFeasibleAtNominal())
        self.assertTrue(inds[5].isFeasibleAtNominal())
        self.assertTrue(inds[6].isFeasibleAtNominal())

        #test ind.nominalConstrainedDominates()
        self.assertTrue(inds[0].nominalConstrainedDominates(inds[1]))
        self.assertTrue(inds[0].nominalConstrainedDominates(inds[2]))
        self.assertTrue(inds[0].nominalConstrainedDominates(inds[3]))
        self.assertFalse(inds[0].nominalConstrainedDominates(inds[4]))
        self.assertFalse(inds[0].nominalConstrainedDominates(inds[5]))
        self.assertFalse(inds[0].nominalConstrainedDominates(inds[6]))
        
        self.assertTrue(inds[2].nominalConstrainedDominates(inds[3]))
        self.assertFalse(inds[3].nominalConstrainedDominates(inds[2]))

        for i in [0,1,4,5,6]:
            self.assertEqual(inds[i].nominalConstraintViolation(), 0.0)
        self.assertTrue(inds[3].nominalConstraintViolation() > \
                        inds[2].nominalConstraintViolation() > 0)
                                 
        #test nondominatedSort, nondominatedFilter, hierNondominatedFilter
        F = nondominatedSort(inds)
        self.assertEqual(len(F), 4) #4 nondominated layers
        F0_IDs = sorted([ind.ID for ind in F[0]])
        F1_IDs = sorted([ind.ID for ind in F[1]])
        F2_IDs = sorted([ind.ID for ind in F[2]])
        F3_IDs = sorted([ind.ID for ind in F[3]])
        
        self.assertEqual(F0_IDs, sorted([inds[0].ID, inds[6].ID, inds[5].ID]))
        self.assertEqual(F1_IDs, sorted([inds[1].ID, inds[4].ID]))
        self.assertEqual(F2_IDs, [inds[2].ID])
        self.assertEqual(F3_IDs, [inds[3].ID])
        
        nondom0_IDs = F0_IDs
        nondom1_IDs = _IDs(nondominatedFilter(inds))
        nondom2_IDs = _IDs(_mergeNondominatedFilter(inds))
        nondom3_IDs = _IDs(_simpleNondominatedFilter(inds))
        nondom4_IDs = _IDs(hierNondominatedFilter(AgeLayeredPop([inds])))
        nondom5_IDs = _IDs(hierNondominatedFilter(AgeLayeredPop(F)))
        self.assertEqual(nondom0_IDs, nondom1_IDs)
        self.assertEqual(nondom0_IDs, nondom2_IDs)
        self.assertEqual(nondom0_IDs, nondom3_IDs)
        self.assertEqual(nondom0_IDs, nondom4_IDs)
        self.assertEqual(nondom0_IDs, nondom5_IDs)
            
    def testNondominatedSort2(self):
        """Tests results of gain and power that caused trouble with Pieter
        -inds on 0th layer: 0,1,4,7,8,9 (feasible)
        -inds on 1st layer: 3,5 (feasible)
        -inds on 2nd layer: 2,6
        """
        if self.just1: return

        #(maximize, minimize)
        gains = [11.293,  32.721,  10.800,  10.655,  11.625,  32.622,  10.501,
                 36.768,  10.075,  10.312]
        powers = [0.00075530,  0.00089390,  0.00164000,  0.00075530,  0.00089030,
                  0.00091880,  0.00075530,  0.00091890,  0.00042520,  0.00075520]
        res = [(gain, power) for gain,power in zip(gains,powers)]

        #gain threshold is 10.0, prnode threshold is 100e-3
        ps = twoMetricPS_MaxMin(10, 100e-3)

        inds = indsFromResultsAndPS(res, ps)

        an_f1 = ps.analyses[0]
        an_f2 = ps.analyses[1]

        #all inds should be feasible
        for ind in inds:
            self.assertTrue(ind.isFeasibleAtNominal())
                                 
        #test nondominatedSort, nondominatedFilter, hierNondominatedFilter
        F = nondominatedSort(inds)
        self.assertEqual(len(F), 3) #3 nondominated layers
        F0_IDs = sorted([ind.ID for ind in F[0]])
        F1_IDs = sorted([ind.ID for ind in F[1]])
        F2_IDs = sorted([ind.ID for ind in F[2]])
        self.assertEqual(F0_IDs, sorted([ inds[0].ID, inds[1].ID, inds[4].ID,
                                          inds[7].ID, inds[8].ID, inds[9].ID]))
        self.assertEqual(F1_IDs, sorted([inds[3].ID, inds[5].ID]))
        self.assertEqual(F2_IDs, sorted([inds[2].ID, inds[6].ID]))
        
        nondom0_IDs = F0_IDs
        nondom1_IDs = _IDs(nondominatedFilter(inds))
        nondom2_IDs = _IDs(_mergeNondominatedFilter(inds))
        nondom3_IDs = _IDs(_simpleNondominatedFilter(inds))
        nondom4_IDs = _IDs(hierNondominatedFilter(AgeLayeredPop([inds])))
        nondom5_IDs = _IDs(hierNondominatedFilter(AgeLayeredPop(F)))
        self.assertEqual(nondom0_IDs, nondom1_IDs)
        self.assertEqual(nondom0_IDs, nondom2_IDs)
        self.assertEqual(nondom0_IDs, nondom3_IDs)
        self.assertEqual(nondom0_IDs, nondom4_IDs)
        self.assertEqual(nondom0_IDs, nondom5_IDs)
        
    def testNondominatedSort3(self):
        if self.just1: return

        N = 50

        gains = [random.random() for i in range(N)]
        powers = [random.random() for i in range(N)]
        res = [(gain, power) for gain,power in zip(gains,powers)]

        ps = twoMetricPS_MaxMin(0.25, 0.75)

        inds = indsFromResultsAndPS(res, ps)

        an_f1 = ps.analyses[0]
        an_f2 = ps.analyses[1]

        #validate that the same results come out of both approaches, and
        # the default (whatever it is)
        Fa = _mergeNondominatedSort(inds)
        Fb = _debNondominatedSort(inds)
        Fc = nondominatedSort(inds)
        for (Fa_layer, Fb_layer, Fc_layer) in zip(Fa, Fb, Fc):
            self.assertEqual(sorted([ind.ID for ind in Fa_layer]),
                             sorted([ind.ID for ind in Fc_layer]))
            self.assertEqual(sorted([ind.ID for ind in Fb_layer]),
                             sorted([ind.ID for ind in Fc_layer]))
            

    def test_ProfileNondominated(self):
        """Only turn this on for special temporary profiling needs"""
        if not self.do_profiling:
            return
        print ""

        #4 test sets: {3d, 6d} x {200 samples, 500 samples}
        print "Load test data: begin"
        from nondom_data_3d_200samples import nondomTestSets_3d_200samples
        from nondom_data_3d_500samples import nondomTestSets_3d_500samples
        from nondom_data_6d_200samples import nondomTestSets_6d_200samples
        from nondom_data_6d_500samples import nondomTestSets_6d_500samples
        
        res_3d_200 = nondomTestSets_3d_200samples()
        res_3d_500 = nondomTestSets_3d_500samples()
        res_6d_200 = nondomTestSets_6d_200samples()
        res_6d_500 = nondomTestSets_6d_500samples()
        print "Load test data: done"

        print "Prepare data: begin"
        ps3 = threeMetricPS()
        ps6 = sixMetricPS()
        
        pops_3d_200 = [indsFromResultsAndPS(res, ps3) for res in res_3d_200]
        pops_3d_500 = [indsFromResultsAndPS(res, ps3) for res in res_3d_500]
        pops_6d_200 = [indsFromResultsAndPS(res, ps6) for res in res_6d_200]
        pops_6d_500 = [indsFromResultsAndPS(res, ps6) for res in res_6d_500]

        tups = [(pops_3d_2003),
                (pops_3d_5003),
                (pops_6d_2006),
                (pops_6d_5006)]
        print "Prepare data: done"
            
        #do the profiling...
        import cProfile
        filename = "/tmp/nondom.cprof"
        
        print "Do actual profile run: begin"
        prof = cProfile.runctx(
            "ret_code = self._applyNondominatedFilters(tups)",
            globals(), locals(), filename)
        print "Do actual profile run: done"

        print "Analyze profile data:"
        import pstats
        p = pstats.Stats(filename)
        p.strip_dirs() #remove extraneous path from all module names

        print ""
        print "======================================================="
        print "Sort by cumulative time in a function (and children)"
        p.sort_stats('cum').print_stats(40)
        print ""

        print ""
        print "======================================================="
        print "Sort by time in a function (no recursion)"
        p.sort_stats('time').print_stats(40)

        print "Analyze profile data: done"

        print "Done overall"

    def _applyNondominatedFilters(self, tups):
        """Helper to test_ProfileNondominated"""
        #choose one of the following
        #nondom_func = _mergeNondominatedFilter
        nondom_func = _simpleNondominatedFilter
        
        for (tup_i, tup) in enumerate(tups):
            (pops) = tup
            for (pop_i, pop) in enumerate(pops):
                print "Running problem #%d / %d; set #%d / %d" % \
                      (tup_i + 1, len(tups), pop_i + 1, len(pops))
                nondom_func(pop)

    def testRandomPool(self):
        if self.just1: return
        
        ps = twoMetricPS_MaxMin(10, 100e-3)
        ind0 = Ind([] ,ps)
        ind1 = Ind([] ,ps)

        # check if nominal behavior works
        rpool = RandomPool(ps)

        rpool.putInds([ind0, ind1])
        inds = rpool.getInds(2)

        seen_0 = False
        seen_1 = False
        for ind in inds:
            if ind == ind0:
                seen_0 = True
            if ind == ind1:
                seen_1 = True
        self.assertEqual(seen_0 and seen_1, True)

        # should be empty now
        self.assertEqual(len(rpool.getInds(99)), 0)

        # we can't add an ind that already has been added
        self.assertRaises(ValueError, rpool.putInds, [ind0, ind1])
        self.assertRaises(ValueError, rpool.putInds, [ind1])
        self.assertRaises(ValueError, rpool.putInds, [ind0])

        # unless we first clear()
        rpool.clear()
        rpool.putInds([ind0])
        rpool.putInds([ind1])        

        # requesting more than the nb of inds in there should
        # be no different
        rpool = RandomPool(ps)

        rpool.putInds([ind0])
        rpool.putInds([ind1])
        
        inds = rpool.getInds(20)
        self.assertEqual(ind0 in inds, True)
        self.assertEqual(ind1 in inds, True)

        # should be empty now
        self.assertEqual(len(rpool.getInds(99)), 0)

        # make all inds selectable again
        rpool.makeAllIndsSelectable()

        # we should get the same inds again
        inds = rpool.getInds(20)
        self.assertEqual(ind0 in inds, True)
        self.assertEqual(ind1 in inds, True)

        
    def testRandomPoolUnselect(self):
        if self.just1: return
        
        ps = twoMetricPS_MaxMin(10, 100e-3)
        ind0 = Ind([] ,ps)
        ind1 = Ind([] ,ps)

        # simple child
        ind2 = Ind([] ,ps)
        ind2.setAncestry([ind0, ind1])

        # an ind with ancestors from
        # multiple generations
        ind3 = Ind([] ,ps)
        ind3.setAncestry([ind2, ind0])

        # a sole ind
        ind4 = Ind([] ,ps)

        rpool = RandomPool(ps)
        rpool.putInds([ind0, ind1, ind2, ind3, ind4])

        # all inds should be selectable
        inds = rpool.getInds(5)
        
        self.assertEqual(ind0 in inds, True)
        self.assertEqual(ind1 in inds, True)
        self.assertEqual(ind2 in inds, True)
        self.assertEqual(ind3 in inds, True)
        self.assertEqual(ind4 in inds, True)

        # make everything selectable again
        rpool.makeAllIndsSelectable()

        # unselect ind2, parents unaffected
        rpool.makeIndsUnselectable([ind2])
        inds = rpool.getInds(5)
        
        self.assertEqual(ind0 in inds, True)
        self.assertEqual(ind1 in inds, True)
        self.assertEqual(ind2 in inds, False)
        self.assertEqual(ind3 in inds, True)
        self.assertEqual(ind4 in inds, True)

        # make everything selectable again
        rpool.makeAllIndsSelectable()

        # unselect ind2, parents too
        rpool.makeIndsUnselectable([ind2], True)
        inds = rpool.getInds(5)
        
        self.assertEqual(ind0 in inds, False)
        self.assertEqual(ind1 in inds, False)
        self.assertEqual(ind2 in inds, False)
        self.assertEqual(ind3 in inds, True)
        self.assertEqual(ind4 in inds, True)

        # make everything selectable again
        rpool.makeAllIndsSelectable()

        # unselect ind3, parents too
        rpool.makeIndsUnselectable([ind3], True)
        inds = rpool.getInds(5)
        
        self.assertEqual(ind0 in inds, False)
        self.assertEqual(ind1 in inds, False)
        self.assertEqual(ind2 in inds, False)
        self.assertEqual(ind3 in inds, False)
        self.assertEqual(ind4 in inds, True)

        # make everything selectable again
        rpool.makeAllIndsSelectable()

        # unselect ind4, parents too
        rpool.makeIndsUnselectable([ind4], True)
        inds = rpool.getInds(5)
        
        self.assertEqual(ind0 in inds, True)
        self.assertEqual(ind1 in inds, True)
        self.assertEqual(ind2 in inds, True)
        self.assertEqual(ind3 in inds, True)
        self.assertEqual(ind4 in inds, False)
        
    def testRandomPoolLoadSave(self):
        if self.just1: return
        
        ps = twoMetricPS_MaxMin(10, 100e-3)
        ps.problem_choice = 1
        ind0 = Ind([] ,ps)
        ind1 = Ind([] ,ps)

        # some children
        ind2 = Ind([] ,ps)
        ind2.setAncestry([ind0, ind1])
        ind3 = Ind([] ,ps)
        ind3.setAncestry([ind2, ind0])
        ind4 = Ind([] ,ps)

        tst_file = getTempFile()

        # create the pool
        rpool = RandomPool(ps)
        rpool.putInds([ind0, ind1, ind2, ind3, ind4])
        rpool.makeIndsUnselectable([ind3], True)

        # save it
        rpool.saveToFile( tst_file )

        # create a new pool and load it
        ps2 = oneMetricPS_Maximize(10)
        ps2.problem_choice = 2
        rpool = RandomPool(ps2)
        # the problem choice doesn't match
        self.assertRaises(ValueError, rpool.loadFromFile, tst_file )

        # create a new pool and load it
        rpool = RandomPool(ps)
        rpool.loadFromFile(tst_file)

        # check the internal structures
        inds_seen = []
        for id in rpool._inds_taken.keys():
            if id == ind0.ID:
                inds_seen.append(ind0)
            if id == ind1.ID:
                inds_seen.append(ind1)
            if id == ind2.ID:
                inds_seen.append(ind2)
            if id == ind3.ID:
                inds_seen.append(ind3)
            if id == ind4.ID:
                inds_seen.append(ind4)
        self.assertEqual(ind0 in inds_seen, True)
        self.assertEqual(ind1 in inds_seen, True)
        self.assertEqual(ind2 in inds_seen, True)
        self.assertEqual(ind3 in inds_seen, True)
        self.assertEqual(ind4 in inds_seen, False)
               
        # should contain only the last ID since the enable state
        # should also be restored
        inds = rpool.getInds(5)

        for ind in inds:
            if ind.ID == ind0.ID:
                self.assertEqual(ind in inds, False)
            if ind.ID == ind1.ID:
                self.assertEqual(ind in inds, False)
            if ind.ID == ind2.ID:
                self.assertEqual(ind in inds, False)
            if ind.ID == ind3.ID:
                self.assertEqual(ind in inds, False)
            if ind.ID == ind4.ID:
                self.assertEqual(ind in inds, True)

        # should be empty now
        self.assertEqual(len(rpool.getInds(99)), 0)

        # adding an ind that is a different ind in memory
        # but has the same ID should fail
        self.assertRaises(ValueError, rpool.putInds ,[ind0, ind4])
        # should still be empty
        self.assertEqual(len(rpool.getInds(99)), 0)
        
        # save to existing file IS allowed.  So by calling this an error should NOT be raised.
        rpool.saveToFile(tst_file)

        #done.  Do cleanup
        os.remove(tst_file)

    def testLiteYield(self):
        if self.just1: return

        #preconditions
        #GOOD is:                         _liteYield, [0], numpy.zeros((5,1)), [True]*5)
        self.assertRaises(AssertionError, _liteYield, [0], numpy.zeros((5)), [True]*5)
        self.assertRaises(AssertionError, _liteYield, [0], numpy.zeros((0, 5)), [True]*5)
        self.assertRaises(AssertionError, _liteYield, [0], numpy.zeros((5, 0)), [True]*5)
        self.assertRaises(AssertionError, _liteYield, [0], numpy.zeros((5, 2)), [True]*5)
        self.assertRaises(AssertionError, _liteYield, [0], numpy.zeros((5, 1)), [True]*4)

        #one objective
        values_X = numpy.zeros((5, 1))
        values_X[:,0] = [10, 20, 30, 40, 50]
        self.assertEqual(_liteYield([0], values_X, [True]*5), 0.0)
        self.assertEqual(_liteYield([10], values_X, [True]*5), 0.2)
        self.assertEqual(_liteYield([11], values_X, [True]*5), 0.2)
        self.assertEqual(_liteYield([41], values_X, [True]*5), 0.8)
        self.assertEqual(_liteYield([50], values_X, [True]*5), 1.0)
        self.assertEqual(_liteYield([51], values_X, [True]*5), 1.0)
        
        self.assertEqual(_liteYield([51], values_X, [False]*5), 0.0)
        self.assertEqual(_liteYield([41], values_X, [True, True, False, True, True]), 0.6)
            
        #two objectives
        values_X = numpy.zeros((5, 2))
        values_X[:,0] = [10, 20, 30, 50, 40]
        values_X[:,1] = [10, 20, 30, 50, 40]
        self.assertEqual(_liteYield([0, 0], values_X, [True]*5), 0.0)
        
        self.assertEqual(_liteYield([10, 10], values_X, [True]*5), 0.2)
        self.assertEqual(_liteYield([9, 10], values_X, [True]*5), 0.0)
        self.assertEqual(_liteYield([10, 9], values_X, [True]*5), 0.0)
        
        self.assertEqual(_liteYield([11, 10], values_X, [True]*5), 0.2)
        
        self.assertEqual(_liteYield([41, 41], values_X, [True]*5), 0.8)
        self.assertEqual(_liteYield([41, 0], values_X, [True]*5), 0.0)
        self.assertEqual(_liteYield([41, 51], values_X, [True]*5), 0.8)
        
        self.assertEqual(_liteYield([41, 31], values_X, [True]*5), 0.6)
        self.assertEqual(_liteYield([50, 50], values_X, [True]*5), 1.0)
        self.assertEqual(_liteYield([51, 51], values_X, [True]*5), 1.0)
        
        self.assertEqual(_liteYield([51, 51], values_X, [False]*5), 0.0)
        self.assertEqual(_liteYield([41, 41], values_X, [True, True, False, True, True]), 0.6)
        
    def testLiteInd(self):
        if self.just1: return
        x, y = None, None

        #one objective
        ind_a = _LiteInd([0], 21)
        self.assertTrue(ind_a.nominalConstrainedDominates(_LiteInd([1], 0), x, y))
        self.assertFalse(ind_a.nominalConstrainedDominates(_LiteInd([0], 0), x, y))
        self.assertFalse(ind_a.nominalConstrainedDominates(_LiteInd([-1], 0), x, y))

        #two objectives
        ind_a = _LiteInd([0, 1], 22)
        self.assertTrue(ind_a.nominalConstrainedDominates(_LiteInd([2, 2], 0), x, y))
        self.assertTrue(ind_a.nominalConstrainedDominates(_LiteInd([1, 1], 0), x, y))
        self.assertTrue(ind_a.nominalConstrainedDominates(_LiteInd([0, 2], 0), x, y))
        self.assertFalse(ind_a.nominalConstrainedDominates(_LiteInd([0, 0], 0), x, y))
        self.assertFalse(ind_a.nominalConstrainedDominates(_LiteInd([-1, 1], 0), x, y))
        self.assertFalse(ind_a.nominalConstrainedDominates(_LiteInd([0, 1], 0), x, y))

    def testExtractYieldSpecsTradeoff_OneObjectiveToMaximize(self):
        if self.just1: return

        #set up robust single-objective problem, with threshold >= 10
        # -note that threshold doesn't matter here, just aim
        ps = oneMetricPS_Maximize(thr1=10) 
        ps.devices_setup.makeRobust()

        ind = Ind([], ps)
        oneMetricSimulateInd(ind, 26) #simulate at rnd points 0,1,...,25; giving values of 0,1,...,25

        lite_inds = extractYieldSpecsTradeoff(ind)

        self.assertEqual(len(lite_inds), 26 - 1) #26 different thresholds, minus one for nom rnd point
        for lite_ind in lite_inds:
            print lite_ind.costs

        #sort lite_inds according to single objective's spec, then test
        specs = [-lite_ind.costs[0] for lite_ind in lite_inds]
        lite_inds = [lite_inds[i] for i in numpy.argsort(specs)]
        for (i, lite_ind) in enumerate(lite_inds):
            self.assertEqual(len(lite_ind.costs), 2)

            #need negative to convert from costs[0] to metric value, because
            # the objective has an aim to minimize, which is the opposite direction of cost.  Same for yield, below.
            found_spec_value = -lite_ind.costs[0] 
            target_spec_value = float(i) + 1.0 #need to add the 1.0 because that was for nominal rnd point
            self.assertEqual(found_spec_value, target_spec_value)

            found_yield = -lite_ind.costs[1]
            target_yield = 1.0 - i / 25.0 
            self.assertAlmostEqual(found_yield, target_yield)

        #test yieldNondominatedFilter: no previous inds, 1 new ind: give same results?
        (lite_inds2, inds2) = yieldNondominatedFilter([], [], [ind])
        self.assertEqual(len(inds2), 1)
        self.assertEqual(inds2[0].ID, ind.ID)
        specs2 = [-lite_ind.costs[0] for lite_ind in lite_inds2]
        self.assertEqual(sorted(specs), sorted(specs2))

        #test yieldNondominatedFilter: the previous inds supplied, 1 new overlapping ind
        # with same ID as before
        (lite_inds3, nondom_inds3) = yieldNondominatedFilter(lite_inds2, inds2, [ind])
        self.assertEqual(len(nondom_inds3), 1)
        specs3 = [-lite_ind.costs[0] for lite_ind in lite_inds3]
        self.assertEqual(sorted(specs), sorted(specs3))
        
        #test yieldNondominatedFilter: the previous inds supplied, 1 new overlapping ind
        # with different ID as before
        (lite_inds4, nondom_inds4) = yieldNondominatedFilter(lite_inds2, inds2, [ind.copyWithNewID()])
        self.assertEqual(len(nondom_inds4), 1)
        specs4 = [-lite_ind.costs[0] for lite_ind in lite_inds4]
        self.assertEqual(sorted(specs), sorted(specs4))

        #------------------------------------------
        #round 2: have duplicate spec values/points...

        #fake-evaluate an ind, with sim_values of [10, 20, 30, 10, 20, 30, 10, 20, 30, 10]
        ind = Ind([], ps)
        an, e = ps.analyses[0], ps.analyses[0].env_points[0]
        for (i, rnd_ID) in enumerate(ind.rnd_IDs[:10]):
            ind.reportSimRequest(rnd_ID, an, e)
            
            if (i % 3) == 0: sim_value = 10.0
            elif (i % 3) == 1: sim_value = 20.0
            elif (i % 3) == 2: sim_value = 30.0
            
            sim_results = {an.metric.name : sim_value}
            ind.setSimResults(sim_results, rnd_ID, an, e)
            
        #extract the tradeoff
        lite_inds = extractYieldSpecsTradeoff(ind)

        #basic test
        self.assertEqual(len(lite_inds), 3) #len(set([10, 20, 30, 10, 20, 30, 10, 20, 30, 10])) == 3

        #sort lite_inds according to single objective's spec, then test
        specs = [-lite_ind.costs[0] for lite_ind in lite_inds]
        lite_inds = [lite_inds[i] for i in numpy.argsort(specs)]
        for (i, lite_ind) in enumerate(lite_inds):
            #need negative to convert from costs[0] to metric value, because
            # the objective has an aim to minimize, which is the opposite direction of cost.  Same for yield, below.
            found_spec_value = -lite_ind.costs[0] 
            target_spec_value = float((i + 1) * 10)
            self.assertEqual(found_spec_value, target_spec_value)

            found_yield = -lite_ind.costs[1]
            target_yield = 1.0 - i / 3.0
            self.assertAlmostEqual(found_yield, target_yield)

        #test yieldNondominatedFilter: no previous inds, 1 new ind: give same results?
        (lite_inds2, inds2) = yieldNondominatedFilter([], [], [ind])
        self.assertEqual(len(inds2), 1)
        specs2 = [-lite_ind.costs[0] for lite_ind in lite_inds2]
        self.assertEqual(specs, specs2)

        #test yieldNondominatedFilter: the previous inds supplied, 1 new overlapping ind
        (lite_inds3, nondom_inds3) = yieldNondominatedFilter(lite_inds2, inds2, [ind])
        self.assertEqual(len(nondom_inds3), 1)
        specs3 = [-lite_ind.costs[0] for lite_ind in lite_inds3]
        self.assertEqual(specs, specs3)

    def testExtractYieldSpecsTradeoff_OneObjectiveToMinimize(self):
        if self.just1: return

        #set up robust single objective problem, with threshold <= 10
        # -note that threshold doesn't matter here, just aim
        ps = oneMetricPS_Minimize(thr1=10) 
        ps.devices_setup.makeRobust()

        ind = Ind([], ps)
        oneMetricSimulateInd(ind, 26) #ind gets values of [0.0, 1.0, ..., 25.0]        

        #extract the tradeoff
        lite_inds = extractYieldSpecsTradeoff(ind)

        #basic test
        self.assertEqual(len(lite_inds), 26 - 1)

        #sort lite_inds according to single objective's spec, then test
        specs = [lite_ind.costs[0] for lite_ind in lite_inds]
        lite_inds = [lite_inds[i] for i in numpy.argsort(specs)]
        for (i, lite_ind) in enumerate(lite_inds):
            self.assertEqual(len(lite_ind.costs), 2)

            #need negative to convert from costs[0] to metric value, because
            # the objective has an aim to minimize, which is the opposite direction of cost.  Same for yield, below.
            found_spec_value = lite_ind.costs[0] 
            target_spec_value = float(i) + 1.0 #need to add the 1.0 because that was for nominal rnd point
            self.assertEqual(found_spec_value, target_spec_value)

            found_yield = -lite_ind.costs[1]
            target_yield = (i + 1.0) / 25.0 
            self.assertAlmostEqual(found_yield, target_yield)

    def testExtractYieldSpecsTradeoff_TwoObjectives_ToMinimizeMinimize(self):
        if self.just1: return

        #set up robust bi-objective problem, with thresholds 0,0
        # -note that threshold doesn't matter here, just aim
        ps = twoMetricPS_MinMin(0, 0) 
        ps.devices_setup.makeRobust()
        an0, an1 = ps.analyses[0], ps.analyses[1]
        e0, e1 = an0.env_points[0], an1.env_points[0]

        ind = Ind([], ps)

        #simulate at a quadrant of points f0={1,2} x f1={10,20}
        for (i, rnd_ID) in enumerate(ind.rnd_IDs[:5]):
            ind.reportSimRequest(rnd_ID, an0, e0)
            ind.reportSimRequest(rnd_ID, an1, e1)
            if i == 0: (v0, v1) = (0, 0) #will be ignored
            elif i == 1: (v0, v1) = (1, 10) #tightest specs, gives worst yield (0.25)
            elif i == 2: (v0, v1) = (1, 20) #med specs, med yield
            elif i == 3: (v0, v1) = (2, 10) #med specs, med yield
            elif i == 4: (v0, v1) = (2, 20) #loosest specs, gives best yield (1.0)
            else: raise
                
            ind.setSimResults({an0.metric.name : v0}, rnd_ID, an0, e0)   
            ind.setSimResults({an1.metric.name : v1}, rnd_ID, an1, e1)     

        #extract the tradeoff
        lite_inds = extractYieldSpecsTradeoff(ind)

        #basic test
        self.assertEqual(len(lite_inds), 4)

        #sort according to yield
        yields = [-lite_ind.costs[2] for lite_ind in lite_inds]
        lite_inds = [lite_inds[i] for i in numpy.argsort(yields)]

        self.assertEqual(lite_inds[0].costs, [1, 10, -0.25]) # (1, 10) for specs and 0.25 for yield
        self.assertTrue((lite_inds[1].costs == [1, 20, -0.50]) or (lite_inds[1].costs == [2, 10, -0.50])) #
        self.assertTrue((lite_inds[2].costs == [1, 20, -0.50]) or (lite_inds[2].costs == [2, 10, -0.50])) #
        self.assertNotEqual(lite_inds[1].costs, lite_inds[2].costs)
        self.assertEqual(lite_inds[3].costs, [2, 20, -1.0]) # (2, 20) for specs and 1.0 for yield
                
        #test yieldNondominatedFilter: no previous inds supplied, 1 new overlapping ind
        (lite_inds2, inds2) = yieldNondominatedFilter([], [], [ind])
        self.assertEqual(len(inds2), 1)
        yields2 = [-lite_ind.costs[2] for lite_ind in lite_inds2]
        self.assertEqual(yields, yields2)

        #test yieldNondominatedFilter: the previous inds supplied, 1 new overlapping ind
        (lite_inds3, inds3) = yieldNondominatedFilter(lite_inds2, inds2, [ind])
        self.assertEqual(len(inds3), 1)
        yields3 = [-lite_ind.costs[2] for lite_ind in lite_inds3]
        self.assertEqual(yields, yields3)
  

        #test yieldNondominatedFilter: the previous inds supplied, 5 new overlapping inds
        (lite_inds3, inds3) = yieldNondominatedFilter(lite_inds2, inds2, [ind]*5)
        self.assertEqual(len(inds3), 1)
        yields3 = [-lite_ind.costs[2] for lite_ind in lite_inds3]
        self.assertEqual(yields, yields3)


    def testyieldNondominatedFilter_TwoObjectives_ToMinimizeMinimize(self):
        if self.just1: return

        #set up robust bi-objective problem, with thresholds 0,0
        # -note that threshold doesn't matter here, just aim
        ps = twoMetricPS_MinMin(0, 0) 
        ps.devices_setup.makeRobust()
        an0, an1 = ps.analyses[0], ps.analyses[1]
        e0, e1 = an0.env_points[0], an1.env_points[0]

        ind_a = Ind([], ps)
        ind_b = Ind([], ps)
        self.assertEqual(ind_a.rnd_IDs, ind_b.rnd_IDs)

        #simulate at a quadrant of points f0={1,2} x f1={10,20}
        for (i, rnd_ID) in enumerate(ind_a.rnd_IDs[:3]):
            ind_a.reportSimRequest(rnd_ID, an0, e0)
            ind_a.reportSimRequest(rnd_ID, an1, e1)
            ind_b.reportSimRequest(rnd_ID, an0, e0)
            ind_b.reportSimRequest(rnd_ID, an1, e1)
            
            if i == 0:
                (va0, va1) = (0, 0) #will be ignored
                (vb0, vb1) = (0, 0) # ""
            elif i == 1:
                (va0, va1) = (1, 10) #a: tightest specs, gives worst yield (0.5)
                (vb0, vb1) = (2, 10) #b: med specs, worst yield (0.5)
            elif i == 2:
                (va0, va1) = (2, 20) #a: loosest specs, gives best yield (1.0)
                (vb0, vb1) = (1, 20) #b: med specs, worst yield (0.5)
                
            else: raise
                
            ind_a.setSimResults({an0.metric.name : va0}, rnd_ID, an0, e0)   
            ind_a.setSimResults({an1.metric.name : va1}, rnd_ID, an1, e1)
            ind_b.setSimResults({an0.metric.name : vb0}, rnd_ID, an0, e0)   
            ind_b.setSimResults({an1.metric.name : vb1}, rnd_ID, an1, e1)       
                
        #test yieldNondominatedFilter
        # -should only get two points in the pareto-optimal front; never the ones with med specs
        (lite_inds2, inds2) = yieldNondominatedFilter([], [], [ind_a, ind_b])
        self.assertEqual(len(lite_inds2), 2)
        self.assertTrue(len(inds2) in [1, 2])
        self.assertEqual(sorted([ind.costs for ind in lite_inds2]), [[1, 10, -0.5], [2, 20, -1.0]])

    def testyieldNondominatedFilter_TwoObjectives_ToMinimizeMinimize_ManyPoints(self):
        if self.just1: return

        #set up robust bi-objective problem, with thresholds 0,0
        # -note that threshold doesn't matter here, just aim
        ps = twoMetricPS_MinMin(0, 0) 
        ps.devices_setup.makeRobust()
        an0, an1 = ps.analyses[0], ps.analyses[1]
        e0, e1 = an0.env_points[0], an1.env_points[0]

        ind = Ind([], ps)

        num_rnd_points = len(ind.rnd_IDs)
        
        for (i, rnd_ID) in enumerate(ind.rnd_IDs):
            ind.reportSimRequest(rnd_ID, an0, e0)
            ind.setSimResults({an0.metric.name : float(num_rnd_points - i)}, rnd_ID, an0, e0)
            ind.reportSimRequest(rnd_ID, an1, e1)
            ind.setSimResults({an1.metric.name : float(i)}, rnd_ID, an1, e1)
                
        #test yieldNondominatedFilter.  Does it handle lotsa points?
        # -the prune_period is set so that the further-pruning will happen at least once too
        # -first round of max_num_combos is not altered
        (lite_inds, inds) = yieldNondominatedFilter([], [], [ind], prune_period=20, max_num_combos=10000)
        
        # -set max_num_combos so that far fewer points
        (lite_inds, inds) = yieldNondominatedFilter([], [], [ind], prune_period=10000, max_num_combos=100)
        self.assertTrue(len(lite_inds) < 100)

        
    def testBAD(self):
        if self.just1: return

        #set up robust bi-objective problem, with thresholds 0,0
        # -note that threshold doesn't matter here, just aim
        ps = twoMetricPS_MaxMin(0, 0) 
        ps.devices_setup.makeRobust()
        an0, an1 = ps.analyses[0], ps.analyses[1]
        e0, e1 = an0.env_points[0], an1.env_points[0]
        metric0, metric1 = an0.metrics[0], an1.metrics[0]

        ind_a, ind_b = Ind([], ps), Ind([], ps)

        num_rnd_points = 10
        
        for (i, rnd_ID) in enumerate(ind_a.rnd_IDs[:num_rnd_points]):
            for ind in [ind_a, ind_b]:
                ind.reportSimRequest(rnd_ID, an0, e0)
                ind.setSimResults({an0.metric.name : float(i)}, rnd_ID, an0, e0)
                ind.reportSimRequest(rnd_ID, an1, e1)
                ind.setSimResults({an1.metric.name : float(i)}, rnd_ID, an1, e1)
                
        #test that we can handle one BAD value
        # -set BAD value, make sure that ind sees it properly
        ind_a.forceFullyBadAtRndPoint(ind_a.rnd_IDs[4]) #good at num_rnd=1..4 (rnd_IDs[0..3]), bad after
        for nr in range(1, num_rnd_points):
            if nr in [1, 2, 3, 4]:
                self.assertFalse(ind_a.isBad(nr))
            else:
                self.assertTrue(ind_a.isBad(nr))
        for (i, rnd_ID) in enumerate(ind_a.rnd_IDs[:num_rnd_points]):
            for metric in [metric0, metric1]:
                expect_bad = (i == 4)
                is_bad = (ind_a.worstCaseMetricValueAtRndPoint(rnd_ID, metric.name) == BAD_METRIC_VALUE)
                self.assertEqual(expect_bad, is_bad)

        # -test yieldNondominatedFilter on one ind with BAD.  It should still be able
        #  to capture a tradeoff, just that the BAD value causes rnd point 4 to be infeasible.
        (lite_inds, inds) = yieldNondominatedFilter([], [], [ind_a])
        self.assertEqual(len(lite_inds), 37)
        self.assertEqual(len(inds), 1)
        
        # -test yieldNondominatedFilter on two inds, one having BAD
        (lite_inds2, inds2) = yieldNondominatedFilter([], [], [ind_a, ind_b])
        self.assertEqual(len(lite_inds2), 46)
        self.assertEqual(len(inds2), 2)
        self.assertEqual(sorted([ind.ID for ind in inds2]), sorted([ind_a.ID, ind_b.ID]))

        #test that we can handle all BAD values
        # -set all BAD, make sure ind sees it properly
        ind_a.forceFullyBad() 
        for num_rnd_points in xrange(1, len(ind_a.rnd_IDs)):
            self.assertTrue(ind_a.isBad(num_rnd_points))
        for (i, rnd_ID) in enumerate(ind_a.rnd_IDs):
            for metric in [metric0, metric1]:
                self.assertEqual(ind_a.worstCaseMetricValueAtRndPoint(rnd_ID, metric.name), BAD_METRIC_VALUE)

        # -test yieldNondominatedFilter on one all-BAD ind .  In this case we shouldn't be
        #  able to find any tradeoffs at all.
        (lite_inds3, inds3) = yieldNondominatedFilter([], [], [ind_a])
        self.assertEqual(lite_inds3, [])
        self.assertEqual(inds3, [])
        
        # -test yieldNondominatedFilter on two inds, one is all-BAD
        (lite_inds4, inds4) = yieldNondominatedFilter([], [], [ind_a, ind_b])
        self.assertEqual(len(inds4), 1)
        self.assertEqual(inds4[0].ID, ind_b.ID)


    def testyieldNondominatedFilter_OneObjectiveToMinimize_OneLessThanConstraint(self):
        if self.just1: return

        #set up robust bi-objective problem, with thresholds 0,0
        # -threshold for f0 doesn't matter here, just aim
        # -threshold for f1 does matter (<=10)
        ps = twoMetricPS_MinLtconstraint(0, 10) 
        ps.devices_setup.makeRobust()
        an0, an1 = ps.analyses[0], ps.analyses[1]
        e0, e1 = an0.env_points[0], an1.env_points[0]

        ind = Ind([], ps)
        
        #simulate at an0, with rnd points 0,1,...,10; giving values of 0.0,1.0,...,10.0
        oneMetricSimulateInd(ind, 11)

        #simulate at an1.  Even-valued rnd points (i=0, 2, ..., 10) get value of 5.0 (feasible),
        # and odd-valued rnd points (i=1, 3, ..., 9) get values of 15.0 (infeasible).
        # Note that rnd point 0 will be ignored later on.
        for (i, rnd_ID) in enumerate(ind.rnd_IDs[:11]):
            ind.reportSimRequest(rnd_ID, an1, e1)
            if (i % 2) == 0: sim_value = 5.0
            else:            sim_value = 15.0
            sim_results = {an1.metric.name : sim_value}
            ind.setSimResults(sim_results, rnd_ID, an1, e1)

        #Simulation values:
        #  rnd_point   f0      f1
        #  ---------  -----  ------ 
        #     1       1.0     15.0
        #     2       2.0      5.0
        #     3       3.0     15.0
        #     4       4.0      5.0
        #     5       5.0     15.0
        #     6       6.0      5.0
        #     7       7.0     15.0
        #     8       8.0      5.0
        #     9       9.0     15.0
        #     10      10.0     5.0
                
        #Should get this pareto-optimal front:
        #  YIELD    f0_thr
        #  -----    -----
        #   0.5      10.0    i.e. 50% of rnd points meet f0 <= 10.0, and all other constraints
        #   0.4      8.0     i.e. 40% of rnd points meet f0 <= 8.0, and all other constraints
        #   0.3      6.0
        #   0.2      4.0
        #   0.1      2.0
        
        (lite_inds, inds) = yieldNondominatedFilter([], [], [ind])
        
        self.assertEqual(len(lite_inds), 5)
        self.assertEqual(len(inds), 1)
        
        yields = [-lite_ind.costs[1] for lite_ind in lite_inds] #sort in ascending order of yield
        lite_inds = [lite_inds[i] for i in numpy.argsort(yields)]

        yields  = [-lite_ind.costs[1] for lite_ind in lite_inds]
        f0_thrs = [ lite_ind.costs[0] for lite_ind in lite_inds]
        self.assertEqual(yields,  [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(f0_thrs, [2.0, 4.0, 6.0, 8.0, 10.0])

        #----------------------------------------------------------------------------------
        #test corner case: no rnd points feasible, so shouldn't get _any_ lite_inds or inds

        ind = Ind([], ps)
        
        #simulate at an0, with rnd points 0,1,...,10; giving values of 0.0,1.0,...,10.0
        oneMetricSimulateInd(ind, 11)

        #simulate at an1, with metric value always infeasible
        for (i, rnd_ID) in enumerate(ind.rnd_IDs[:11]):
            ind.reportSimRequest(rnd_ID, an1, e1)
            sim_results = {an1.metric.name : 15.0} 
            ind.setSimResults(sim_results, rnd_ID, an1, e1)
            
        (lite_inds, inds) = yieldNondominatedFilter([], [], [ind])
        
        self.assertEqual(len(lite_inds), 0)
        self.assertEqual(len(inds), 0)

        
    def tearDown(self):
        pass
    
    
if __name__ == '__main__':

    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.DEBUG)
    
    unittest.main()
