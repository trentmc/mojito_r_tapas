import logging
import unittest

from adts import *
from engine.TloOptimizer import TloSolutionStrategy, TemplateTloOptimizer

class StubPS:
    def __init__(self):
        self.opt_point_meta = PointMeta([ContinuousVarMeta(False, -100.0, +100.0, 'x0'),
                                         ContinuousVarMeta(False,  -20.0,  +40.0, 'x1')])
        

class TloOptimizerTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

    def testSimpleConvergence(self):
        if self.just1: return
        
        ps = StubPS()

        start_opt_point = Point(True, {'x0':-25.0, 'x1':+15.0})
        ss = TloSolutionStrategy(start_opt_point)
        ss.setMaxNumPoints(10000)
        ss.setTargetCost(0.0)

        optimizer = TemplateTloOptimizer(ps, ss)
        optimizer.optimize()

        expected_o = Point(True, {'x0':0.0, 'x1':0.0})
        best_o = optimizer.state.bestOptPoint()
        
        found_it = True
        for name in ps.opt_point_meta.keys():
            tol = 0.05
            found_it = found_it and (abs(expected_o[name] - best_o[name]) < tol)

        self.assertTrue(found_it)

        #test str's
        self.assertTrue(len(str(optimizer)) > 0)
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    import logging

    logging.basicConfig()
    logging.getLogger('tlo').setLevel(logging.INFO)
    
    unittest.main()
