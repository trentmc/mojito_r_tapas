from random import random, gauss
import unittest

import numpy

from regressor.Kmeans import Kmeans, KmeansStrategy
from util.mathutil import rail

class KmeansTest(unittest.TestCase):

    def test1d(self):
        #test the direct-1d call
        x = numpy.arange(20.0, 40.0, 1.0)
        minx = 20.0
        maxx = 40.0
        
        ss = KmeansStrategy(init_num_centers = 5)
        centers = Kmeans().cluster1d(x, ss)
        
        self.assertTrue(5 <= len(centers) <= len(x))
        for center in centers:
            self.assertTrue(minx <= center <= maxx)
            
        #plot for manual testing
        #from scipy import gplt

        #all points 'x' are along y=1.0; all centers are along y=2.0
#        gplt.plot(x, numpy.ones(len(x))*1.0, 'with points')
#        gplt.hold('on')
#        gplt.plot(centers, numpy.ones(len(centers))*1.2, 'with points')
#        gplt.yaxis((0.8, 1.3))

        #test the general n-d call
        X = numpy.reshape(x, (1,len(x)))
        minX = numpy.array([[minx]])
        maxX = numpy.array([[maxx]])
        
        centers = Kmeans().cluster(X, ss, minX, maxX)
        
        self.assertEqual(len(centers.shape), 2)
        self.assertEqual(centers.shape[0], 1)
        self.assertTrue(5 <= centers.shape[1] <= X.shape[1])
        for center in centers[0,:]:
            self.assertTrue(minx <= center <= maxx)
        
        
    def test2d(self):
        return #HACK
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
                x = numpy.array([gauss(center[0], std0), gauss(center[1], std1)])
                rx = rail(x, minX, maxX)
                
                xi = num_samp_per_group * group_i + samp_i
                X[:,xi] = x

        for outlier_i in range(num_outliers):
            xi = num_samp_per_group * num_groups + outlier_i
            X[:,xi] = numpy.array([random()*(maxX[0]-minX[0]) + minX[0],
                             random()*(maxX[1]-minX[1]) + minX[1] ])

        #cluster
        ss = KmeansStrategy()
        centers = Kmeans().cluster(X, ss, minX, maxX)

        #unit test
        self.assertEqual(centers.shape[0], 2)
        self.assertTrue(centers.shape[1] <= N)
        for i in range(centers.shape[1]):
            self.assertTrue(minX[0] <= centers[0,i] <= maxX[0])
            self.assert_(minX[1] <= centers[1,i] <= maxX[1])
            
        #plot for manual testing
#             from scipy import gplt
#             gplt.hold('on')
#             gplt.plot(X[0,:], X[1,:], 'with points')
#             gplt.plot(centers[0,:], centers[1,:], 'with points')

    def tearDown(self):
        pass

def suite():
    suite = unittest.TestSuite()
    #suite.addTest(unittest.makeSuite(testVarMeta))
    return suite

if __name__ == '__main__':

    import logging

    logging.basicConfig()
    logging.getLogger('kmeans').setLevel(logging.INFO)



    unittest.main()
