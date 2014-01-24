import unittest
import random

import numpy

from regressor.Probe import ProbeFactory, ProbeModel, ProbeBuildStrategy

from util import mathutil

from regressor_data import data2d

class ProbeTest(unittest.TestCase):
    """
    Test Probe
    Also tests Cart, because Probe uses carts.
    """
    
    def setUp(self):
        self.just1 = False #to make True is a HACK

    def testProbe2d(self):
        if self.just1: return

        [xx0, xx1, X, y] = data2d(gridsize=20, funcID=7) #func: 2 + 3*x0 - 4*x1 + 5*x0^2 + 6*x1^2 + 7*x0*x1
    
        ss = ProbeBuildStrategy(max_rank=10)
        self.assertTrue('ProbeBuildStrategy' in str(ss))

        model = ProbeFactory().build(X, y, ss)
        
        self.assertTrue('ProbeModel' in str(model))

        yhat = model.simulate(X)
        e = mathutil.nmse(y, yhat, min(y), max(y))
        self.assertTrue(e < 0.05)

        
def suite():
    suite = unittest.TestSuite()
    #suite.addTest(unittest.makeSuite(testVarMeta))
    return suite

if __name__ == '__main__':

    import logging

    logging.basicConfig()
    logging.getLogger('probe').setLevel(logging.DEBUG)
    logging.getLogger('lin').setLevel(logging.INFO)

    unittest.main()
