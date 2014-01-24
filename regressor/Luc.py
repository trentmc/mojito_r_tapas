"""Luc.py : Look-Up Cluster

Includes classes:
-LucModel --

"""

import logging

import numpy

from adts import *
from util import mathutil
from util.ascii import *

import copy

log = logging.getLogger('luc')

class LucStrategy:
    """
    @description

      Holds parameters specific to the strategy of building LucModel objects.
      
    @attributes

    """ 
    def __init__(self):
        self.regressor_type = 'Luc' #is this needed?
        
    def __str__(self):
        s = "LucStrategy={"
        s += " /LucStrategy}"  
        return s

class LucFactory:
    """
      Builds a LucModel
    """ 
    def __init__(self):
        pass

    def build(self, X, y, ss):
        """
        @description

          Builds a LucModel, given a target mapping of X=>y and a strategy 'ss'.
        
        @arguments

          X -- 2d array [sample #][input variable #] -- training input data
          y -- 1d array [sample #] -- training output data
          ss -- LucStrategy --
        
        @return

          model -- LucModel object
    
        @exceptions
    
        @notes
        """
        log.info("Build LucModel: begin")
        model = LucModel(None, X, y)
        log.info("Build LucModel: done")
        return model

class LucModel:
    """ Represents a cluster of data points in the LUT point space 
        The idea is that the point space is divided into clusters along
        the 'discrete' axes. This means that a cluster is created for
        every discrete value on that axis. The cluster can contain
        sub-clusters that operate along another axis. The lowest level
        clusters contain a non-uniform set of data points in one dimension.
        
        The interpolation is done by finding the appropriate clusters,
        doing a 1D interpolation in them and then interpolating inbetween
        the clusters.
        
        This approach seems to be yielding good results and has good
        performance, also for large datasets.

        @attributes

          level -- int -- 0 is root level, 1 is child of root, etc.  This
            also specifies input dimension that this object cares about.
            The maximum value is therefore (number of input dimensions) - 1.
          children -- list of Luc -- (is empty if at bottom level)
          xvalues -- sorted, unique list of input training values for
            this object's dimension / level.
    """
    def __init__(self, parent, trX, y):
        """
        @description
            Build the cluster
            
        @arguments
            trX -- 2d array [sample #][input variable #] -- training input data
              Note that this is a transpose of a typical input 'X'.
            y -- 1d array [sample #] array -- training output data
            
        @return

          nothing
    
        @exceptions
    
        @notes                
        """
        # reshape data if necessary
        if len(y.shape) == 2:
            if y.shape[0] != 1 and y.shape[1] != 1:
                raise ValueError("Bad training data shape")
            else:
                if y.shape[0]==1:
                    y_eff = numpy.transpose(y)
                else:
                    y_eff = y
                y = y_eff[:,0]

        #preconditions
        assert len(y.shape) == 1
        assert len(y) == trX.shape[0]
        
        #set base data
        n = trX.shape[1]   #num input vars
        N = trX.shape[0]   #num training points  

        #main work...
        

        #set self.level
        # -case: root
        if parent == None:
            self.level = 0
        # -case: child
        else:
            self.level = parent.level + 1

        self._parent = parent
        
        #figure out the bounds of this model
        self.in_var_min=[]
        self.in_var_max=[]
        
        if self.level==0:
            log.info("Finding model boundaries...")
            
        for i in range(0,n): # only display for the top level
            self.in_var_min.append(min(trX[:, i]))
            self.in_var_max.append(max(trX[:, i]))
            
            if self.level == 0: # only display for the top level
                log.info("Var %d: min = %e, max = %e" % (i, self.in_var_min[i], self.in_var_max[i]))
         
        #set self.children, self.xvalues, self.yvalues; recurse if needed
        self.children = []

        # -base case: at bottom level (i.e. only one variable left)
        if (self.level + 1 == n):
            # yes it is
            
            # we save the values sorted, otherwise we'll have to do it during
            # the interpolation            
            
            # note that we save the values sorted by target value, as we
            # cannot be sure that the I = f(W) is really a function (i.e.
            # invertable).
            # the effect will be that when we interpolate, we'll always choose
            # the smallest W available.            

            perm = numpy.argsort(y)

#             import pdb;pdb.set_trace()
            xvalues_nonunique = numpy.take(trX[:, self.level], perm)
            yvalues_nonunique = numpy.take(y, perm)

            # make sure all xvalues are unique
            xvalues_unique = []
            yvalues_unique = []
            for i in range(0, len(xvalues_nonunique)):
                is_unique = True
                for j in range(i+1, len(xvalues_nonunique)):
                    if xvalues_nonunique[i] == xvalues_nonunique[j]:
                        is_unique = False
                        break
                if is_unique:
                    xvalues_unique.append(xvalues_nonunique[i])
                    yvalues_unique.append(yvalues_nonunique[i])
                        
            self.xvalues = numpy.array(xvalues_unique, dtype=float)
            self.yvalues = numpy.array(yvalues_unique, dtype=float)
                        
            if len(mathutil.uniquifyVector(self.xvalues)) != len(self.xvalues):
                import pdb; pdb.set_trace()

            if len(xvalues_nonunique) != len(xvalues_unique):
                log.info("removed %s nonunique entries" % (len(xvalues_nonunique) - len(xvalues_unique)))
                log.debug(" original X: %s" % (xvalues_nonunique))
                log.debug(" new      X: %s" % (xvalues_unique))
            
            # corner cases
            if len(self.xvalues) < 2:
                log.error("unique dataset too small...")
                self.xvalues = None
                self.yvalues = None
            else:
                # MAGIC: make sure the numbers are significantly larger than 0
                min_y_value = 1e-18
                I = numpy.compress((abs(self.xvalues) > min_y_value), range(N))
    
                self.xvalues = numpy.take(self.xvalues, I)
                self.yvalues = numpy.take(self.yvalues, I)
            
        # -case: not at bottom level, so recurse too
        else:            
            # construct the list of axis-values for this cluster
            # returns a sorted list without duplicates
            self.xvalues = mathutil.uniquifyVector(trX[:,self.level])
            
            # build a cluster for each axis value
            for xvalue in self.xvalues:
                # find the points that are contained in the new child cluster
                I = numpy.compress((trX[:,self.level] == xvalue), range(N))

                #import pdb;pdb.set_trace()

                # build the child
                child = LucModel(self, numpy.take(trX,I,0), numpy.take(y,I))
                self.children.append(child)
                
            # now check whether all children make sense
            # NOTE: we do this after building the child vector
            #       because that allows for extra sanity checks
            new_children = []
            new_xvalues = []
            dropped = 0
            for i in range(0, len(self.xvalues)):
                xvalue = self.xvalues[i]
                child = self.children[i]
                
                if child.xvalues != None and len(child.xvalues) > 1: # keep the child
                    new_children.append(child)
                    new_xvalues.append(xvalue)
                else:
                    dropped += 1

            if dropped:
                log.debug("dropped %d level %d clusters" % \
                         (dropped, self.level+1))
                log.debug(" original  xvals: %s" % str(self.xvalues))
                log.debug(" remaining xvals: %s" % str(numpy.array(new_xvalues)))
                
            self.xvalues = numpy.array(new_xvalues)
            self.children = new_children
                
            
    def getInputUpperBound(self, idx):
        """
        @description
            Returns the validity upper bound for a certain input variable.
            
        @arguments
            idx -- the index of the input variable
            
        @return

          the upper bound for this input variable
    
        @exceptions
    
        @notes                
        """
        return self.in_var_max[idx]
    
    def getInputLowerBound(self, idx):
        """
        @description
            Returns the validity lower bound for a certain input variable.
            
        @arguments
            idx -- the index of the input variable
            
        @return

          the lower bound for this input variable
    
        @exceptions
    
        @notes                
        """
        return self.in_var_min[idx]
                
    def simulate(self, X):
        """
        Simulate for each input point (column) in 2d array X
        """
        N = X.shape[1]
        yhat = numpy.zeros(N, dtype=float)
        for sample_i in range(N):
            #if (sample_i % 1000) == 0:
            #    log.debug("Simulate sample %d / %d" % (sample_i+1, N))
            yhat[sample_i] = self.simulate1(X[:, sample_i])
        return yhat
        
    def simulate1(self, x):
        """ 
        @description
        
            Find the predicted value for the given point based upon the 
            data present in this cluster. This is either interpolation of
            the actual data present, or choosing the best child clusters
            to interpolate between.
            
        @arguments

          x -- 1d array [input var #] -- input value per dimension
            
        @return

          y1 -- float -- the interpolated value
    
        @exceptions
    
        @notes            
            
        """
        #base data
        num_xvalues = len(self.xvalues)
        num_children = len(self.children)

        #corner case: only one x-value
        if num_xvalues == 1:
            log.debug("only one x-value")
            if num_children > 0:
                assert num_children == 1
                return self.children[0].simulate1(x)
            else:
                return self.xvalues[0]
        
        # strip the first entry, as this will be used to select the
        # subclusters
        input_xval = x[self.level]
    
        # find the clusters that should be interpolated
        # calculate distance from input_xval to each xvalue
        xdist = self.xvalues - input_xval

        #log.debug( "xvals: " + str( self.xvalues) )
        
        # find the center element
        # don't do it like this because that will
        # fail when the x=f(y) is not invertible
        # i.e. when f-1(x) is not a function
        #   idxR = numpy.searchsorted(xdist, 0)
        idxR = 0
        while idxR < num_xvalues and xdist[idxR] < 0:
            idxR += 1
    
        # handle extrapolation
        #handle if input_xval >= last x in the dataset
        if idxR == num_xvalues:
            #log.warning('Extrapolation at the right side')
            idxR = num_xvalues - 1
    
        idxL = idxR - 1
    
        #handle if input_xval <= first x in the dataset
        if idxL < 0:
            #if input_xval != self.xvalues[0]:
                #log.warning('Extrapolation at the left side')
            idxL = 0
            idxR = 1
        
        # sanity checks
        assert 0 <= idxL < num_xvalues, (idxL, num_xvalues)
        assert 0 <= idxR < num_xvalues, (idxR, num_xvalues)
        
        # find the left and right y-axis values
        #  -case: not at bottom, so recurse
        if num_children > 0:
            yval_L = self.children[idxL].simulate1(x)
            yval_R = self.children[idxR].simulate1(x)
        
        #  -case: at bottom level; use y-values directly available
        else:
            yval_L = self.yvalues[idxL]
            yval_R = self.yvalues[idxR]
    
        # find the left and right x-axis values
        xval_L = self.xvalues[idxL]
        xval_R = self.xvalues[idxR]

        #log.debug("xL=%e xR=%e x=%e" % (xval_L, xval_R, input_xval))
        #log.debug("yL=%e yR=%e" % (yval_L, yval_R))   

        # compute final interpolated y-value
        if xval_R == xval_L: # this happens if we have multiple identical xvalues
            log.error("trying to prevent divide by zero... (check LUC dataset)")
            if idxR+1 < num_xvalues:
                xval_R = self.xvalues[idxR+1]
                yval_R = self.children[idxR+1].simulate1(x)
            elif idxL-1 > 0:
                xval_L = self.xvalues[idxL-1]
                yval_L = self.children[idxL-1].simulate1(x)
            
            if xval_R == xval_L:
                log.error("bailing out to prevent divide by zero...")
                return 0
            
        try:
            slope = (yval_R - yval_L) / (xval_R - xval_L)
            yval = yval_L + slope * (input_xval - xval_L)
        except:
            import pdb;pdb.set_trace()
        #if not mathutil.isNumber(yval):
        #    import pdb;pdb.set_trace()
        
        #log.debug("slope=%e yval=%e" % (slope, yval))

        # these are checks to see whether the LUC operates
        # correctly.
        if not (xval_L <= input_xval and xval_R >= input_xval):
            #valid = self.isValid1(x)
            #log.warning("input val (%e) not in range [ %e : %e ], isValid==%d, level=%d" \
            #            % (input_xval, xval_L, xval_R, valid, self.level))
            #log.warning("xvals: %s" \
            #            % ( str(self.xvalues) ))
            #assert not (min(self.xvalues) <= input_xval <= max(self.xvalues)), \
            #      ("input xval is not out of range")
            if (min(self.xvalues) <= input_xval <= max(self.xvalues)):
                  log.warning("input xval %s is not out of range [%s %s]" % \
                              (str(input_xval), str(xval_L), str(xval_R)))

        #done
        return yval
    
    def isValid(self, X):
        """
        figure out valid-ness for each input point (column) in 2d array X
        """
        N = X.shape[1]
        yhat = numpy.zeros(N, dtype=float)

        for sample_i in range(N):
            if not self.isValid1(X[:, sample_i]):
                return False
            
        # they are all valid
        return True
        
    def isValid1(self, x):
        """ 
        @description
        
            figure out whether the model is somewhat valid for this
            point
            
        @arguments

          x -- 1d array [input var #] -- input value per dimension
            
        @return

          bool -- is the model valid for this point or not?
    
        @exceptions
    
        @notes            
            
        """
        #base data
        num_xvalues = len(self.xvalues)
        num_children = len(self.children)

        #corner case: only one x-value
        if num_xvalues == 1:
            log.debug("only one x-value")
            
            # NOTE: if there is only one value,
            # we could expect that it is equal
            # to the simulated value
            if num_children > 0:
                assert num_children == 1
                return self.children[0].isValid(x)
            else:
                # there is not much sense in having only one
                # value
                return False
        
        # strip the first entry, as this will be used to select the
        # subclusters
        input_xval = x[self.level]
    
        # find the clusters that should be interpolated
        # calculate distance from input_xval to each xvalue
        xdist = self.xvalues - input_xval

        # find the center element
        # don't do it like this because that will
        # fail when the x=f(y) is not invertible
        # i.e. when f-1(x) is not a function
        #   idxR = numpy.searchsorted(xdist, 0)
        idxR = 0
        
        while idxR < num_xvalues and xdist[idxR] < 0:
            idxR += 1
    
        if idxR == num_xvalues: # extrapolation at right side
            if log.isEnabledFor(logging.DEBUG):
                log.debug("extrapolation at right side: %s > %s" % \
                          ( str(input_xval), str(self.xvalues[idxR-1]) ))
            #return False
            
            # MAGIC: allow right-side extrapolation for lowest level only
            return num_children == 0

        idxL = idxR - 1
    
        if idxL < 0: # extrapolation at left side
            #import pdb; pdb.set_trace()
            if log.isEnabledFor(logging.DEBUG):
                log.debug("extrapolation at left side: %s < %s" % \
                          ( str(input_xval), str(self.xvalues[0]) ))
            return False
        
        # sanity checks
        assert 0 <= idxL < num_xvalues, (idxL, num_xvalues)
        assert 0 <= idxR < num_xvalues, (idxR, num_xvalues)
        
        # find the left and right y-axis values
        #  -case: not at bottom, so recurse
        if num_children > 0:
            is_valid_L = self.children[idxL].isValid1(x)
            if not is_valid_L:
                log.debug("left side is invalid")
                return False
            
            is_valid_R = self.children[idxR].isValid1(x)
            if not is_valid_R:
                log.debug("right side is invalid")
                return False
            
            is_valid = True
        #  -case: at bottom level; use y-values directly available
        else:
            is_valid = True # we are in range

        #try:
        assert input_xval >= self.xvalues[idxL] and input_xval <= self.xvalues[idxR]
        #except:
        #    import pdb;pdb.set_trace()

        return is_valid
