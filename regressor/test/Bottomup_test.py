from random import random, gauss
import unittest

import numpy

from util.mathutil import rail
from regressor.Bottomup import Bottomup, BottomupStrategy
from regressor import RegressorUtils

class BottomupTest(unittest.TestCase):

    def test1d(self):
        x = numpy.arange(20.0, 40.0, 1.0)
        X = numpy.reshape(x, (1, len(x)))
        minX = numpy.array([20.0])
        maxX = numpy.array([40.0])
        
        ss = BottomupStrategy(max_num_centers = 15)
        C = Bottomup().cluster(X, ss, minX, maxX)

        self.assertEqual(C.shape, (1,15))

        for center_i in range(5):
            self.assertTrue(minX[0] <= C[0,center_i] <= maxX[0])
            
        #plot for manual testing
#         from scipy import gplt
#         #all points 'x' are along y=1.0; all centers are along y=2.0
#         gplt.plot(x, numpy.ones(len(x))*1.0, 'with points')
#         gplt.hold('on')
#         gplt.plot(C[0,:], numpy.ones(C.shape[1])*1.2, 'with points')
#         gplt.yaxis((0.8, 1.3))

        I = Bottomup().clusterIndices1d(x, ss)
        self.assertTrue(len(I) <= 15)
        self.assertTrue(len(I) == len(set(I)))
        self.assertTrue(0 <= min(I) < max(I) <= len(x)-1)

        cx = Bottomup().cluster1d(x, ss, False)
        self.assertTrue(len(cx) <= 15)
        self.assertTrue(len(cx) == len(set(cx)))
        
        cx = Bottomup().cluster1d([11.0], ss, False)
        self.assertEqual(cx, [11.0])
        
        cx = Bottomup().cluster1d([11.0, 11.1, 12.0], BottomupStrategy(max_num_centers = 2), False)
        self.assertEqual(len(cx), 2)
        self.assertTrue((cx[0] == 11.0) or (cx[0] == 11.1))
        self.assertTrue(cx[1] == 12.0)
        
        cx = Bottomup().cluster1d([11.0, 11.1, 12.0, 12.1], BottomupStrategy(max_num_centers = 2), True)
        self.assertTrue(len(cx) in [2, 3, 4])
        self.assertTrue(11.0 in cx)
        self.assertTrue(12.1 in cx)
            
        #plot for manual testing
#         from scipy import gplt
#         #all points 'x' are along y=1.0; all centers are along y=2.0
#         gplt.plot(x, numpy.ones(len(x))*1.0, 'with points')
#         gplt.hold('on')
#         gplt.plot([x[i] for i in I], numpy.ones(len(I))*1.2, 'with points')
#         gplt.yaxis((0.8, 1.3))

    def testRemoveDuplicates(self):
        x = numpy.arange(20.0, 40.0, 1.0)
        x[4] = x[0] #insert duplicate
        X = numpy.reshape(x, (1, len(x)))
        minX = numpy.array([20.0])
        maxX = numpy.array([40.0])
        
        ss = BottomupStrategy(max_num_centers = len(x))
        C = Bottomup().cluster(X, ss, minX, maxX)

        self.assertEqual(C.shape, (1,len(x)-1)) #should remove duplicate!
    
        
    def test2d(self):
        for i in range(10):
            self._test2d()
        
    def _test2d(self):
        #create samples
        minX = numpy.array([-5.0, +2.0])
        maxX = numpy.array([+25.0, +4.0])
        diffX = maxX - minX
        
        num_samp_per_group = 15
        num_groups = 3
        num_outliers = 4
        N = num_samp_per_group * num_groups + num_outliers
        X = numpy.zeros((2,N))*0.0
        for group_i in range(num_groups):
            center = [random()*(diffX[0]) + minX[0],
                      random()*(diffX[1]) + minX[1] ]
            std0 = (random()*0.1 + 0.01)*(diffX[0])
            std1 = (random()*0.1 + 0.01)*(diffX[1])
            for samp_i in range(num_samp_per_group):
                x = numpy.array([gauss(center[0], std0),
                                   gauss(center[1], std1)])
                rx = rail(x, minX, maxX)
                
                xi = num_samp_per_group * group_i + samp_i
                X[:,xi] = x

        for outlier_i in range(num_outliers):
            xi = num_samp_per_group * num_groups + outlier_i
            X[:,xi] = numpy.array([random()*(maxX[0]-minX[0]) + minX[0],
                                     random()*(maxX[1]-minX[1]) + minX[1] ])

        #cluster
        ss = BottomupStrategy(max_num_centers=7)
        C = Bottomup().cluster(X, ss, minX, maxX)

        #unit test
        self.assertEqual(C.shape, (2,7))

        #it's ok for returned cluster centers to be outside of the minX, maxX
        # that were passed into cluster(), if a point X[:,i] was outside.
        # But it's not ok for the centers to be outside of the bounds
        # of the union of X and minX/maxX
        for var_i in range(X.shape[0]):
            minX[var_i] = min(minX[var_i], min(X[var_i,:]))
            maxX[var_i] = max(maxX[var_i], max(X[var_i,:]))
            
        for center_i in range(C.shape[1]):
            self.assertTrue(minX[0] <= C[0,center_i] <= maxX[0])
            self.assertTrue(minX[1] <= C[1,center_i] <= maxX[1])
            
        #plot for manual testing
#             from scipy import gplt
#             gplt.hold('on')
#             gplt.plot(X[0,:], X[1,:], 'with points')
#             gplt.plot(C[0,:], C[1,:], 'with points')

    def tearDown(self):
        pass

def suite():
    suite = unittest.TestSuite()
    #suite.addTest(unittest.makeSuite(testVarMeta))
    return suite

if __name__ == '__main__':

    import logging

    logging.basicConfig()
    logging.getLogger('bottomup').setLevel(logging.INFO)

    unittest.main()
