import unittest

import math
import random
import time

import numpy

from adts import *
from regressor.Pwl import PwlBuildStrategy, PwlModel, PwlFactory, PWL_APPROACHES
from util.octavecall import plotAndPause
from util import mathutil

class PwlTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

    def testTypicalHockey(self):
        """Test fit on a hockey-stick like waveform, with noise"""
        if self.just1: return
        #create training data
        x = numpy.zeros((150), dtype=float)
        y = numpy.zeros(150, dtype=float)
        for i in range(150):
            xi = i * 0.1
            x[i] = xi
            if i < 75:
                y[i] = float(i) + 5.0*random.random()
            else:
                y[i] = 60 + 8.0*random.random()

        #build the model
        ss = PwlBuildStrategy('hockey')
        ss.num_yt_reps = 1
        model = PwlFactory().build(x, y, ss)

        #test the model
        x2 = x 
        yhat = model.simulate(x2)

        #tolerance = 6 sigma
        tol = 6
        min_yhat_lower, min_yhat_upper = (-5.0*tol), (5.0*tol)
        max_yhat_lower, max_yhat_upper = (60 - 8.0*tol), (60 + 8.0*tol)

        # -test yhat
        self.assertTrue(min_yhat_lower <= min(yhat) <= min_yhat_upper)
        self.assertTrue(max_yhat_lower <= max(yhat) <= max_yhat_upper)

        # -test model parameters
        self.assertTrue(len(model.xs) == len(model.ys) == 3)
        self.assertTrue(6.0 <= model.xs[1] <= 6.5)
        self.assertEqual(model.ys[1], model.ys[2])
        self.assertTrue(max_yhat_lower <= model.ys[1] <= max_yhat_upper)

        #plotAndPause(x, yhat, x, y) #to uncomment is a HACK

    def testPerfectFit_Hockey(self):
        """Test fit on a hockey-stick like waveform, with zero noise"""
        if self.just1: return
         
        #create training data
        # -y-offset is y = 5.0
        # -caps off at y = 80.0 (when x = 7.5)
        x = numpy.zeros((150), dtype=float)
        y = numpy.zeros(150, dtype=float)
        for i in range(150):
            xi = i * 0.1
            x[i] = xi
            if i < 75:
                y[i] = float(i) + 5.0
            else:
                y[i] = 75 + 5.0
                
        #build the model
        ss = PwlBuildStrategy('hockey')
        ss.num_yt_reps = 1
        model = PwlFactory().build(x, y, ss)

        #test the model
        x2 = x 
        yhat = model.simulate(x2)

        #test nmse
        self.assertAlmostEqual(mathutil.nmse(yhat, y, min(y), max(y)), 0.0, 2)

        # -test model parameters
        self.assertTrue(len(model.xs) == len(model.ys) == 3)
        self.assertEqual(model.ys[1], model.ys[2])
        tol = 0.1
        self.assertTrue((7.5 - tol) <= model.xs[1] <= (7.5 + tol))
        self.assertTrue((80.0 - tol) <= model.ys[1] <= (80.0 + tol))

        #plotAndPause(x, yhat, x, y) #to uncomment is a HACK

    def testPerfectFit_Bump(self):
        if self.just1: return
        self._testFit_Bump(False)
        
    def testNoisyFit_Bump(self):
        if self.just1: return
        self._testFit_Bump(True)
        
    def _testFit_Bump(self, noisy):

        #construct target data by directly building a pwl model and simulating it
        xs = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
        ys = [-1.0, -1.0, 2.0, 2.0, -1.0, -1.0]
        target_pwl_model = PwlModel(xs, ys)

        x = numpy.arange(-2.0, +8.0, 0.25)
        y = target_pwl_model.simulate(x)
        if noisy:
            for (i, yi) in enumerate(y):
                y[i] = yi + 0.2 * random.random()
        
        #build the model
        ss = PwlBuildStrategy('bump')
        ss.num_yt_reps = 2
        model = PwlFactory().build(x, y, ss)

        #test the model
        x2 = x 
        yhat = model.simulate(x2)
            
        #plotAndPause(x, yhat, x, y) #to uncomment is a HACK

        #test nmse
        if noisy:
            self.assertTrue(mathutil.nmse(yhat, y, min(y), max(y)) < 0.30)
        else:
            self.assertTrue(mathutil.nmse(yhat, y, min(y), max(y)) < 0.02)

            # -test model parameters
            for (xi, yi, target_xi, target_yi) in zip(model.xs, model.ys, xs, ys):
                if target_xi in [1.0, 2.0, 3.0, 4.0]:
                    self.assertAlmostEqual(xi, target_xi, 1)
                self.assertAlmostEqual(yi, target_yi, 1)

    def testFlatY_Hockey(self):
        if self.just1: return
        self._testFlatY('hockey')
        
    def testFlatY_Bump(self):
        if self.just1: return
        self._testFlatY('bump')

    def _testFlatY(self, approach):
        x = numpy.arange(-10.0, 10.0, 0.1)
        y = 3.2 * numpy.ones((len(x)), dtype=float)
        ss = PwlBuildStrategy(approach)
        ss.num_yt_reps = 1
        model = PwlFactory().build(x, y, ss)

    def testRandomData_Hockey(self):
        if self.just1: return
        self._testRandomData('hockey', 5)
        
    def testRandomData_Bump(self):
        if self.just1: return
        self._testRandomData('bump', 3)

    def _testRandomData(self, approach, num_repeats):
        for loop_i in range(num_repeats):
            self._singleRunBuildOnRandomData(approach)
            
    def _singleRunBuildOnRandomData(self, approach):
        x = numpy.array([random.random()**2 for i in range(50)])
        y = numpy.array([random.random()**2 for i in range(50)])
        ss = PwlBuildStrategy(approach)
        ss.num_yt_reps = 1
        model = PwlFactory().build(x, y, ss)
                                  
    def tearDown(self):
        pass

if __name__ == '__main__':

    import logging
    logging.basicConfig()
    logging.getLogger('pwl').setLevel(logging.DEBUG)
    logging.getLogger('cyt').setLevel(logging.INFO)
    #logging.getLogger('lin').setLevel(logging.INFO)
    
    unittest.main()
