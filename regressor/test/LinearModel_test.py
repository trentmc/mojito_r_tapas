import unittest

import numpy

from regressor import LinearModel, RegressorUtils
from util import mathutil
from regressor_data import data2d

class LinearModelTest(unittest.TestCase):
    
    def setUp(self):
        self.just1 = False #to make True is a HACK

    def testLinearModel2d(self):
        if self.just1: return

        [xx0, xx1, X, y] = data2d(gridsize=20, funcID=2) #linear function 
    
        ss = LinearModel.LinearBuildStrategy()
        self.assertTrue('LinearBuildStrategy' in str(ss))

        (minX, maxX) = RegressorUtils.minMaxX(X)
        model = LinearModel.LinearModelFactory().build(X, y, minX, maxX, ss)
        
        self.assertTrue('LinearModel' in str(model))

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
    logging.getLogger('lin').setLevel(logging.DEBUG)

    unittest.main()
