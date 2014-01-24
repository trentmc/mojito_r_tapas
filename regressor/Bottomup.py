""" Bottomup.py
Bottom-up clustering algorithm:
 -begin where every point is its own cluster
 -merge the two closest points into a cluster
 -repeat until the target number of clusters is hit

Advantages:
-few parameters
-always catches the outliers

Note: could also have a stopping condition of:
 -stop if min_dist found < target_min_dist
""" 
import logging
import math
import random

import numpy

import util.mathutil as mathutil

log = logging.getLogger('bottomup')

class BottomupStrategy:

    def __init__(self, max_num_centers = 10):
        self.max_num_centers = max_num_centers

    def __str__(self):
        s = "BottomupStrategy={ "
        s += 'max_num_centers=%d' % self.max_num_centers
        s += '} '
        return s

        
class Bottomup:

    def cluster1d(self, x, ss, force_outliers):
        """Returns a list of clustered 1-d points.  (x is a list or 1d array)."""
        I = self.clusterIndices1d(x, ss)
        clustered_x = [x[i] for i in I]
        if force_outliers:
            if min(x) not in clustered_x: clustered_x.append(min(x))
            if max(x) not in clustered_x: clustered_x.append(max(x))
        clustered_x = sorted(clustered_x)
        return clustered_x
    
    def clusterIndices1d(self, x, ss):
        """Like clusterIndices, except x is a 1d array; each entry is a different sample point"""
        X = numpy.reshape(numpy.asarray(x), (1, len(x)))
        minX = numpy.array([min(x)])
        maxX = numpy.array([max(x)])
        I = self.clusterIndices(X, ss, minX, maxX)
        return I

    def clusterIndices(self, X, ss, minX, maxX):
        """Whereas normal cluster() returns new x's which are possibly new points
        in space, this method returns a list of indices to the entries in X which will
        together form the new cluster centers."""
        #cluster
        C = self.cluster(X, ss, minX, maxX)

        #identify an index for each entry in X
        I = []
        for center_i in range(C.shape[1]):
            distances = [_dist01(X[:,i], C[:, center_i], minX, maxX)
                         for i in range(X.shape[1])]
            I.append( numpy.argmin(distances) )

        #uniquify, and put in a nice order
        I = sorted(set(I)) 
        return I

    def cluster(self, X, ss, minX, maxX):
        """
        @description
        
        @arguments.

          X -- 2d array [var #][sample #] -- points
          ss -- BottomupStrategy --
          minX, maxX -- each are a 1d array [var #] - together, they
            specify the range for scaling the distance measures.  They
            are not used in any other way (e.g. if an X[:,i] is outside
            the range of minX, maxX that's ok, and a final cluster
            center may therefore be as well)
        
        @return

          C -- 2d array [var #][center #] -- clustered centers
    
        @exceptions
    
        @notes
        """
        I = mathutil.removeDuplicateRows(numpy.transpose(X), range(X.shape[1]))
        X = numpy.take(X, I, 1)
        
        n,N = X.shape

        #corner case
        if ss.max_num_centers >= N:
            return X

        #corner case
        if X.shape[0] == 0:
            return X

        #corner case
        if ss.max_num_centers == 0:
            return numpy.zeros((n,0), dtype=float)

        #main case...

        #initialize centers = all points in X.  Maintain as a list.
        centers = [X[:,i] for i in range(N)]

        #initial calculate distances between each pair of centers
        distances = [] # distances[i][j] is distance from center i to center j
        for i in range(N):
            distances.append( [ _dist01(centers[i], centers[j], minX, maxX)
                                for j in range(N) ] )

        #keep removing a center until 'centers' is small enough
        min_dist = 0.0
        while len(centers) > ss.max_num_centers:
            
            #choose pair
            (min_i, min_j) = self._minimumDistancePair(distances)

            #create new center
            new_center = self._midpoint(centers[min_i], centers[min_j])

            #replace old two centers with new center
            # -delete 'min_j' entries
            del centers[min_j]
            del distances[min_j]
            for i in range(len(distances)):
                del distances[i][min_j]

            # -replace 'min_i' center with new center
            centers[min_i] = new_center

            # -update distances to new center
            distances[min_i] = [ _dist01(centers[min_i], centers[j], minX, maxX)
                                for j in range(len(distances)) ]
            
            log.info('#centers=%d' % len(centers))
            
        #convert list of centers into 2d array
        C = numpy.transpose(numpy.array(centers, dtype=float))

        return C

    def _minimumDistancePair(self, distances):
        """Returns (min_i, min_j) which incldes
        a pair of indices into 'distances'.  Note that min_i < min_j."""
        min_i, min_j, min_dist  = None, None, float('Inf')
        for i,distances_at_i in enumerate(distances):
            for j, dist in enumerate(distances_at_i):
                if i >= j: continue
                if dist < min_dist:
                    min_i, min_j, min_dist = i, j, dist
        return (min_i,  min_j)

    def _midpoint(self, v1,  v2):
        v = [(v1[i] + v2[i])/2.0 for i in range(len(v1))]
        return v

def _dist01(x1, x2, minX, maxX):
    """Returns the distance between x1 and x2, in range [0,1]
    (i.e. scaled, according to range of each var)"""
    #Corner case: 1d
    if len(x1) == 1:
        mn, mx = minX[0], maxX[0]
        if mn == mx:
            return 0.0
        else:
            return abs(x1[0] - x2[0]) / (maxX[0] - minX[0])

    #Main case
    sum01 = sum(((x1i-x2i) / float(mx-mn))**2
                for x1i, x2i, mn, mx in zip(x1,x2,minX,maxX)
                if mn != mx)
    d01 = math.sqrt(sum01) / math.sqrt(len(x1))
    return d01

