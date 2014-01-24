import logging
import unittest

from adts import *
from engine.DytOptimizer import *
from util import mathutil

class StubPS:
    def __init__(self):
        self.opt_point_meta = PointMeta([DiscreteVarMeta(numpy.arange(-100.0, +101.0, +1.0), 'x0'), 
                                         DiscreteVarMeta(numpy.arange(0.0, +41.0, +1.0), 'x1')])


class DytOptimizerTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK
        
    def testSimpleConvergence(self):
        if self.just1: return
        for i in range(10):
            self._testSimpleConvergence()
        
    def _testSimpleConvergence(self):
        ps = StubPS()

        ss = DytSolutionStrategy()
        ss.setMaxNumPoints(10000)
        
        optimizer = TemplateDytOptimizer(ps, ss)

        start_opt_point = Point(True, {'x0':-25.0, 'x1':+15.0})
        optimizer.optimize(start_opt_point)

        expected_best_opt_point = Point(True, {'x0':0.0, 'x1':0.0})
        
        self.assertEqual(optimizer.state.bestInd().opt_point,
                         expected_best_opt_point)

        #test str's
        self.assertTrue(len(str(optimizer)) > 0)
        self.assertTrue(len(optimizer.state.detailedStr()) > 0)
        self.assertTrue(len(optimizer.state.detailedIndStr()) > 0)

        
    def tearDown(self):
        pass

if __name__ == '__main__':
    import logging
    import sayo.util.sayolog

    logging.basicConfig()
    logging.getLogger('dyt').setLevel(logging.INFO)
    
    unittest.main()
