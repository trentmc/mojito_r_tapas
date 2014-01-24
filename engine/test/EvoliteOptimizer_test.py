import copy
import unittest
import numpy

from engine.EvoliteOptimizer import EvoliteOptimizer, EvoliteSolutionStrategy

def sphere_function(x):
    """
    N-dimensional sphere, with center at (0,0,...0)
    Want to minimize
    """
    v = sum([xi**2.0 for xi in x])
    return v

class EvoliteOptimizerTest(unittest.TestCase):
    """
    Test different Optimizers for functionality and basic convergence
    abilities.
    """

    def setUp(self):
        self.just1 = False #to make True is a HACK
        
    def testSimpleConvergence(self):
        if self.just1: return

        #set min_x, max_x (don't need all of 'ps' for 'lite' optimizer)
        min_x = numpy.array([-10.0, -10.0])
        max_x = numpy.array([+10.0, +10.0])

        #set ss
        ss = EvoliteSolutionStrategy()
        ss.popsize = 50
        ss.numgen = 200
        ss.init_xs = [numpy.array([9.0,1.0]),
                      numpy.array([3.0,2.0]),
                      numpy.array([2.0,2.1])]

        ss.target_cost = 1.0e-2 #3.0e-4 is doable, but not always
        
        #set function (don't need a full 'evaluator' for 'lite' optimizer)
        function = sphere_function

        #init optimizer and go!
        optimizer = EvoliteOptimizer(min_x, max_x, ss, function)
        optimizer.optimize()

        #retrieve results
        x = optimizer.best_x
        self.assert_(-0.5 < x[0] < 0.5)
        self.assert_(-0.5 < x[1] < 0.5)
        
        self.assertTrue(optimizer.best_cost <= ss.target_cost)
        
    def testScalingUnscaling(self):
        if self.just1: return

        #set min_x, max_x (don't need all of 'ps' for 'lite' optimizer)
        min_x = numpy.array([-10.0, -9.0])
        max_x = numpy.array([-9.0,  -8.0])

        #set ss
        ss = EvoliteSolutionStrategy()
        ss.popsize = 5
        ss.numgen = 2

        #set function (don't need a full 'evaluator' for 'lite' optimizer)
        function = sphere_function

        #init optimizer and go!
        optimizer = EvoliteOptimizer(min_x, max_x, ss, function)
        optimizer.optimize()

        #retrieve results
        x = optimizer.best_x
        self.assert_(-10.0 <= x[0] <= -9.0)
        self.assert_(-9.0 <= x[1] <= -8.0)
        
    def tearDown(self):
        pass

def suite():
    suite = unittest.TestSuite()
    #suite.addTest(unittest.makeSuite(testVarMeta))
    return suite

if __name__ == '__main__':
    import logging

    logging.basicConfig()
    logging.getLogger('evo').setLevel(logging.DEBUG)
    
    unittest.main()
