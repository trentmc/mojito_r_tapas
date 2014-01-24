"""PointRegressor.py
"""
import os
import pickle
import logging

import numpy

from util.FileHash import hashfile

log = logging.getLogger('luc')


do_api_check = True

class PointRegressor:
    """
    @description

      Like a regressor, but is aware of the input variable names
      and thus can simulate directly off of an input Point (as opposed to an input array)
      
    @attributes
      
    @notes

      Wraps whatever regressor we specify.
    """
    def __init__(self, regressor=None, input_varnames=[], case_matters=False):
        """
        @description

          Constructor.

        @arguments

          regressor -- a regressor -- has function simulate(X) where
            X is a 2d array [input var #][sample #]
          input_varnames -- list of string --
          case_matters -- bool -- does uppercase vs. lowercase matter for the input vars?

        @return

          new_point_regressor -- PointRegressor object -- 

        @exceptions

        @notes

        """
        # calculate a version id based upon a hash of the source code files
        # this ensures a rebuild of the cached models when their implementation
        # is changed
        
        thisdir = os.path.dirname(os.path.normpath(__file__)) # this is a sort-of HACK
        self._version = hashfile(thisdir + '/' + 'PointRegressor.py') + \
                        hashfile(thisdir + '/' + 'Luc.py')
        log.info("regressor code hash: %s", self._version)
        
        self.regressor = regressor
        if not case_matters:
            input_varnames = [varname.lower() for varname in input_varnames]
        self.input_varnames = input_varnames
        self.case_matters = case_matters

        #We allocate this once, and update with each call to simulate.
        #Therefore less overhead in memory allocation.
        self._X = numpy.zeros((len(input_varnames), 1), dtype=float)

    def loadFromFile(self, filename):
        fn = filename + '.cachedmodel'
        log.info("Trying to load a cached model from %s..." % (fn))

        if os.path.exists(fn):
            #FIXME: we should check the model to see if it corresponds.
            fid = open(fn, 'r')
            model = pickle.load(fid)
            fid.close()

            log.info("File read...")

            if do_api_check:
                try:
                    if self._version != model._version:
                        log.info("PointRegressor api version changed")
                        return False
                except:
                    import pdb;pdb.set_trace()
                    log.info("some API version unavailable...")
                    return False

            self.regressor = model.regressor
            self.input_varnames = model.input_varnames
            self.case_matters = model.case_matters

            self._X = model._X
            self.source_hash = model.source_hash
            
            log.info("Model loaded OK...")
            return True
        
        else:
            log.info("File does not exist")
            return False
        
    def saveToFile(self, filename):
        fn = filename + '.cachedmodel'
        
        fid = open(fn,'w')
        pickle.dump(self, fid)
        fid.close()                
        log.info("Saved model to disk: %s" % (fn))

    def simulatePoint(self, point):
        """
        @description

          Simulates at one input point

        @arguments

          point -- dict mapping input_varname : input_value

        @return

          simulated_output_value -- float          

        @exceptions

        @notes

        """
        #make input point's varnames lowercase if case_matters == False
        if self.case_matters:
            point2 = point
        else:
            point2 = {}
            for varname, varval in point.items():
                point2[varname.lower()] = varval
        
        #set _X
        # -recall that each 'self_varname' is already lowercase, if
        #  case_matters == False
        for var_index, self_varname in enumerate(self.input_varnames):
            self._X[var_index,0] = point2[self_varname]

        #simulate with regressor
        y = self.regressor.simulate(self._X)

        #done
        return y[0]
    
    def isValidForPoint(self, point):
        """
        @description

          Checks if the regressor is valid at one input point

        @arguments

          point -- dict mapping input_varname : input_value

        @return

          bool -- true if the regressor is applicable at this point          

        @exceptions

        @notes

        """
        #make input point's varnames lowercase if case_matters == False
        if self.case_matters:
            point2 = point
        else:
            point2 = {}
            for varname, varval in point.items():
                point2[varname.lower()] = varval
        
        #set _X
        # -recall that each 'self_varname' is already lowercase, if
        #  case_matters == False
        for var_index, self_varname in enumerate(self.input_varnames):
            self._X[var_index,0] = point2[self_varname]

        # check regressor
        return self.regressor.isValid(self._X)

    def getInputUpperBound(self, varname):
        """
        @description
            Returns the validity upper bound for a certain input variable.
            
        @arguments
            varname -- the name of the input variable
            
        @return

          the upper bound for this input variable
    
        @exceptions
    
        @notes                
        """
        if self.case_matters:
            name = varname
        else:
            name = varname.lower()
            
        idx=0
        for var in self.input_varnames:
            if var == name: 
                break
            else:
                idx += 1

        if idx < len(self.input_varnames):
            return self.regressor.getInputUpperBound(idx)
        else:
            raise ValueError("invalid varname")
    
    def getInputLowerBound(self, varname):
        """
        @description
            Returns the validity lower bound for a certain input variable.
            
        @arguments
            varname -- the name of the input variable
            
        @return

        the lower bound for this input variable
        @exceptions
    
        @notes                
        """
        if self.case_matters:
            name = varname
        else:
            name = varname.lower()

        idx=0
        for var in self.input_varnames:
            if var == name: 
                break
            else:
                idx += 1

        if idx < len(self.input_varnames):
            return self.regressor.getInputLowerBound(idx)
        else:
            raise ValueError("invalid varname")
                
