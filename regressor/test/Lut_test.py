import unittest

import numpy
import time

from adts import *
from regressor.Lut import *

# specify the maximum error a regressor can have
regressor_max_error=0.5

class LutTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK
        pass

    def test1d(self):
        if self.just1: return
        #create training data
        X = numpy.zeros((1,50), dtype=float)
        y = numpy.zeros(50, dtype=float)
        for i in range(50):
            x = i * 0.1
            X[0,i] = x
            y[i] = math.sin(x)

        #build the model
        lut_ss = LutStrategy()
        lut_model = LutFactory().build(X, y, lut_ss)

        #test the model
        X2 = X 
        yhat = lut_model.simulate(X2)

        # the borders of the interpolation are not really good
        for yi, yhati in zip(y[1:-1], yhat[1:-1]):
            self.assertTrue( abs(yi - yhati)/((yi + yhati + 1e-20)/2) < regressor_max_error, (yi,yhati))
            
    def test2d(self):
        if self.just1: return
        #create training data
        gridsize = 10
        X = numpy.zeros((2,gridsize**2), dtype=float)
        y = numpy.zeros(gridsize**2, dtype=float)
        sample_k = 0
        
        for i in range(gridsize):
            x_i = i * 0.1
            for j in range(gridsize):
                x_j = j * 0.1
                X[0,sample_k] = x_i
                X[1,sample_k] = x_j
                y[sample_k] = math.sin(x_i + x_j)
                sample_k=sample_k+1

        #build the model
        lut_ss = LutStrategy()
        lut_model = LutFactory().build(X, y, lut_ss)

        #test the model
        X2 = X
        yhat = lut_model.simulate(X2)
        
        for yi, yhati,x20,x21 in zip(y, yhat,X2[0],X2[1]):
            # ignore the corners
            if not ((x20 <= 0.1) or (x20 >= 4.9) or (x21 <= 0.1) or (x21 >= 4.9)):
                self.assertTrue( abs(yi - yhati)/((yi + yhati + 1e-20)/2) < regressor_max_error, (yi,yhati,x20,x21))
    
    def testSpeed3D(self):
        if self.just1: return
        self._testSpeednD(50, 5)
        
    def _testSpeednD(self,nr_points, dim):
        points = 100*numpy.random.rand(dim, nr_points)
        vals = 100*numpy.random.rand(nr_points)
        
        #build the model
        lut_ss = LutStrategy()
        lut_model = LutFactory().build(points, vals, lut_ss)       
        
        # test the model    
        target_points = points + 0.1
        cnt=0
        
        starttime=time.time()
        
        while cnt < 2:
            yhat = lut_model.simulate(target_points)           
            cnt=cnt+1
        
        elapsed=time.time()-starttime
        
        nb_lookups=nr_points * cnt
        
        lookups_per_sec=nb_lookups / elapsed
        
        #print "%d simulations (%d-D) of %d points took %f seconds (%d lookups/sec)" % ( cnt, dim , nr_points, elapsed, lookups_per_sec)

                              
    def tearDown(self):
        pass

if __name__ == '__main__':

    import logging
    logging.basicConfig()
    logging.getLogger('lut').setLevel(logging.DEBUG)
    
    unittest.main()
