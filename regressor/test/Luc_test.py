import unittest
import logging

import numpy
import time
import math

from adts import *
from regressor.Luc import *

log = logging.getLogger('luc')

# specify the maximum error a regressor can have
regressor_max_error=0.05

class LucTest(unittest.TestCase):

    def setUp(self):
        self.just1 = True #to make True is a HACK
        pass

    def testLowestLevelLuc(self):
        if self.just1: return

        #create training data
        X = numpy.zeros((1,50), dtype=float)
        y = numpy.zeros(50, dtype=float)
        for i in range(50):
            x = i * 0.1
            X[0,i] = x
            y[i] = math.sin(x)

        #build the model
        luc_model = LucModel(None, numpy.transpose(X), y)
        
        #test the model
        X2 = X
        yhat = luc_model.simulate(X2)

        # the borders of the interpolation are not really good
        for yi, yhati in zip(y[1:-1], yhat[1:-1]):
            err = abs(yi - yhati) / ((abs(yi) + abs(yhati))/2 + 1e-20)
            log.info("check: %f, %f, %f"%(yi, yhati, err))
            self.assertTrue( err < regressor_max_error, (yi, yhati, err))
        
    def tearDown(self):
        pass

if __name__ == '__main__':

    import logging
    logging.basicConfig()
    logging.getLogger('luc').setLevel(logging.DEBUG)
    
    unittest.main()
