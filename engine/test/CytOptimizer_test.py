import logging
import unittest

from adts import *
from engine.CytOptimizer import CytSolutionStrategy, TemplateCytOptimizer

class StubPS:
    def __init__(self):
        self.opt_point_meta = PointMeta([ContinuousVarMeta(False, -100.0, +100.0, 'x0'),
                                         ContinuousVarMeta(False,  -20.0,  +40.0, 'x1')])
        

class CytOptimizerTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

    def testSimpleConvergence(self):
        if self.just1: return
        for i in range(5):
            self._testSimpleConvergence()
        
    def _testSimpleConvergence(self):
        ps = StubPS()

        ss = CytSolutionStrategy()
        ss.setMaxNumPoints(10000)

        found_it = False
        for try_i in range(5): #give 5 tries to find it 
            optimizer = TemplateCytOptimizer(ps, ss)

            start_opt_point = Point(True, {'x0':-25.0, 'x1':+15.0})
            optimizer.optimize(start_opt_point)

            expected_o = Point(True, {'x0':0.0, 'x1':0.0})
            best_o = optimizer.state.bestInd().opt_point
            try_found_it = True
            for name in ps.opt_point_meta.keys():
                tol = 0.02
                try_found_it = try_found_it and (abs(expected_o[name] - best_o[name]) < tol)

            if try_found_it:
                found_it = True
                break
                
        self.assertTrue(found_it)

        #test str's
        self.assertTrue(len(str(optimizer)) > 0)
        self.assertTrue(len(optimizer.state.detailedStr()) > 0)
        self.assertTrue(len(optimizer.state.detailedIndStr()) > 0)
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    import logging

    logging.basicConfig()
    logging.getLogger('cyt').setLevel(logging.INFO)
    
    unittest.main()
