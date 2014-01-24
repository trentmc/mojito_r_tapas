""" Kmeans.py
 K-means clustering algorithm
"""
import logging
import math
import random

import numpy

from util import mathutil

log = logging.getLogger('kmeans')

class KmeansStrategy:

    def __init__(self, init_num_centers = 10, learn_rate = 0.1,
                 num_reps = 50):
        self.init_num_centers = init_num_centers #(will add outliers to this)
        self.learn_rate = learn_rate
        self.num_reps = num_reps

    def __str__(self):
        s = "KmeansStrategy={ "
        s += 'init_num_centers=%d' % self.init_num_centers
        s += 'learn_rate=%5.2e' % self.learn_rate
        s += 'num_reps=%d' % self.num_reps
        s += '} '
        return s

        
class Kmeans:

    def cluster(self, X, ss, minX, maxX):
        """
        Inputs:  X - 2d array of points [var #][sample #]
        Outputs: C - 2d array of clustered centers [var #][center #]
        """
        n,N = X.shape

        #corner case
        if ss.init_num_centers >= N:
            return X

        #corner case
        if X.shape[0] == 0:
            return X

        #corner case
        if X.shape[0] == 1:
            c = self.cluster1d(X[0,:], ss)
            C = numpy.reshape(c, (1,len(c)))
            return C

        #main case...
        nc = ss.init_num_centers

        #randomly pick init centers
        center_J = random.sample(range(N), nc)
        centers = numpy.take(X, center_J, 1)

        #shift centers iteratively
        dists = numpy.zeros(nc)*0.0
        center_samp_dists = numpy.zeros(N)*0.0
        for rep_i in range(ss.num_reps):
            for sample_i in range(N):
                x = X[:,sample_i]
                for center_i in range(nc):
                    dists[center_i] = _dist01(centers[:,center_i], x, minX, maxX)
                closest_center_i = numpy.argmin(dists)
                delta = centers[:,closest_center_i] - x
                centers[:,closest_center_i] -= ss.learn_rate * delta

                if rep_i == (ss.num_reps-1):
                    center_samp_dists[sample_i] = min(dists)

        #make outliers to be centers as well
        avg = numpy.average(center_samp_dists)
        std = mathutil.stddev(center_samp_dists)
        for sample_i in range(N):
            d = center_samp_dists[sample_i]
            if d < (avg-2.0*std) or d > (avg+2.0*std):
                new_center = X[:,sample_i]
                centers = _addcol(centers, new_center)
                for sample_i2 in range(N):
                    d_old = center_samp_dists[sample_i2]
                    d_new = _dist01(new_center, X[:,sample_i2], minX, maxX)
                    center_samp_dists[sample_i2] = min(d_old, d_new)


        for ci in range(centers.shape[1]):
            centers[:,ci] = mathutil.rail(centers[:,ci], minX, maxX)
        
        return centers

    def cluster1d(self, x, ss, x_already_unique=False):
        """
        Inputs:  x - 1d array of points [sample #]
        Outputs: c - 1d array of clustered centers [center #]

        This algorithm is different than the general n-d case
        because the 1d version allows us to pull tricks to be faster.
        """
        x = numpy.asarray(x)
        assert len(x.shape) == 1

        #corner case
        if len(x)==0:
            return numpy.array([])
        
        minx, maxx = min(x), max(x)
        nc = ss.init_num_centers
        assert nc > 0
        N = len(x)

        #corner case
        if minx == maxx:
            return numpy.array([minx])

        #corner case
        if x_already_unique:
            unique_x = x
        else:
            unique_x = list(set(x))
        if nc >= len(unique_x):
            return numpy.array(unique_x)

        #main case...
        
        #randomly pick init centers: uniformly distributed across [minx, maxx]
        eps = 0.0001
        centers = list(numpy.arange(minx, maxx+eps,
                                      (maxx - minx) / float(nc-1)))

        #so far: center[i-1] < center[i].  Maintain this relationship for speed.

        #shift centers iteratively
        x = sorted(x)
        dists = numpy.zeros(nc)*0.0
        center_samp_dists = numpy.zeros(N)*0.0
        inf = float('Inf')
        for rep_i in range(ss.num_reps):
            start_center_i = 0
            for sample_i, sample_x in enumerate(x):

                #The set of candidate centers is restricted because
                # our centers are ordered in increasing x (therefore
                # we can traverse starting at center 'start_center_i', not '0')
                #We can stop early too (see the 'break') because as
                # soon as we start pulling away from the best so far,
                # we'll never get as close
                best_center_i, best_dist = None, inf
                for center_i in range(start_center_i, nc):
                    dist = abs(centers[center_i] - sample_x)
                    if dist < best_dist:
                        best_center_i = center_i
                        best_dist = dist
                    else:
                        break
                assert best_center_i is not None

                #Adjust closest center's value
                delta = centers[best_center_i] - sample_x
                centers[best_center_i] -= ss.learn_rate * delta

                #Ensure centers stay ordered (FIXME)(or should I bother?)

                last_rep = (rep_i == (ss.num_reps-1))
                if last_rep: 
                    center_samp_dists[sample_i] = best_dist

                #update for next loop
                start_center_i = best_center_i

        #make outliers to be centers as well
        avg = mathutil.average(center_samp_dists)
        std = mathutil.stddev(center_samp_dists)
        for sample_i, sample_x in enumerate(x):
            d = center_samp_dists[sample_i]
            if d < (avg-2.0*std) or d > (avg+2.0*std):
                new_center = sample_x
                centers.append(new_center)
                for sample_i2 in range(N):
                    d_old = center_samp_dists[sample_i2]
                    d_new = abs(new_center - x[sample_i2])
                    center_samp_dists[sample_i2] = min(d_old, d_new)

        #rail centers to [minx, maxx]
        for ci,center in enumerate(centers):
            centers[ci] = min(maxx, max(minx, center) )
        
        return numpy.array(centers)
    
def _addcol(X, x):
    nr,nc = X.shape
    X2 = numpy.zeros((nr,nc+1))*0.0
    X2[:,:nc] = X
    X2[:,nc] = x
    return X2

def _dist01(x1, x2, minX, maxX):
    """Returns the distance between x1 and x2, in range [0,1]
    (i.e. scaled, according to range of each var)"""
    #Corner case
    if len(x1)==1:
        return abs(x1-x2)/(maxX-minX)

    #Main case
    sum01 = sum(((x1i-x2i) / float(mx-mn))**2
                for x1i, x2i, mn, mx in zip(x1,x2,minX,maxX)
                if mn != mx)
    d01 = math.sqrt(sum01) / math.sqrt(len(x1))
    return d01

