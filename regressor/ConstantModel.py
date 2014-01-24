"""ConstantModel.py
"""

import logging

import numpy

from util import mathutil
from util.constants import BAD_METRIC_VALUE

log = logging.getLogger('lin')

def yIsConstant(y):
    """
    @description

        Determine if the input list is composed of a single value

        #return str('%0.9e' % min(y)) == str('%0.9e' % max(y))

    @arguments
    
        y: list of arithmetic values

    @return

        Boolean: True if all values in y are the same, else false
        
    @exceptions

    @notes
    
        Assumes that y is a list of values.

    """
    return ('%.9e' % min(y)) == ('%.9e' % max(y))

class ConstantModel:
    """
    @description
    
        Simplest model, returns a constant value.

    @arguments

    @return

    @exceptions

    @notes

    """ 
    def __init__(self, constant, numvars):
        """
        @description
        
            Simplest model, returns a constant value.
    
        @arguments
        
            constant: constant value returned by this model
            numvars: number of variables in this model
    
        @return
    
        @exceptions
    
        @notes
            
            The y_transform is fixed as 'linear'
    
        """ 
        if constant == BAD_METRIC_VALUE:
            self.constant = constant
        else:
            self.constant = float(constant) 
        self.numvars = numvars
        self.y_transform = 'lin' # fixed as linear
        self.bases = []

    def simulate(self, X):
        """
        @description
        
        @arguments
        
            X: an array of the form [variable #][sample #]
        
        @return
        
            A vector of length == the number of samples,
            with all data set to the constant value specified
            during construction.
    
        @exceptions
    
        @notes
    
        """
        if self.constant == BAD_METRIC_VALUE:
            return numpy.asarray([BAD_METRIC_VALUE for i in xrange(X.shape[1])])
        elif str(self.constant) == 'nan' or str(self.constant) == 'NaN':
            return numpy.asarray([float('nan') for i in xrange(X.shape[1])])
        else:
            return numpy.ones(X.shape[1], dtype=float) * self.constant
        
    def simulate1(self, x):
        """Simulate a single vector"""
        X = numpy.reshape(x, (len(x), 1))
        return self.simulate(X)[0]

    def influencePerVar(self, X=None):
        """
        @description
        
            Calculate the influence of each variable on the final result.
            Since this is a constant model, every variable has an equal
            influence of 0.0
        
        @arguments
        
            X:  an array of the form [variable #][sample #]
                IGNORED because this is a constant model.
                
        @return
    
            A vector of length == the number of samples,
            with all data set to 0.0

        @exceptions
    
        @notes
    
        """ 
        return numpy.zeros(self.numvars, dtype=float)
    
    def influencePerVarMeanStderrs(self, X=None):
        """
        @description

            Calculate the influence of each variable on the final result.
            Since this is a constant model, every variable has an equal
            influence of mean = 0.0, stderr = 0.0
        
        @arguments

           X -- (ignored) Unlike other models which need input X describing
                    a region, ConstantModels' impacts are not influenced by X,
                    because the constant model is a plane which doesn't
                    change region by region.  
        @return
     
            list(tuple(float, float)) -- [(mean[0], stderr[0]), ...]
    
            where:
    
            mean[i] is influence of variable i. In a constant model, the 
                mean = 0.0 for every var.
            stderr[i] is the stderr of the influence of variable i. In a 
                constant model, the stderr = 0.0 for every var.

        @exceptions
    
        @notes
    
        """ 
        log.warning("Need to fix ConstantModel.influencePerVarMeanStderrs()")
        raise "FIXME - stderrs are NOT 0"
        mean = self.influencePerVar()
        stderr = numpy.zeros(self.numvars, dtype=float)
        return zip(mean, stderr)


    def influencePerVarConfidenceIntervals(self, X=None, confidence_level=0.95):
        """
        @description

          Returns the upper/lower confidence-interval estimate of each
          variable's impact at a specified confidence_level (p-value).

          Since this is a constant model, every variable has an equal
            (mean, lower,upper) influence of (0.0, 0.0,0.0)
          
          WARNING: all the inputs are ignored because this model is constant!
          
        @arguments

          X (optional) -- 2d array -- 
          confidence_level -- float in [0,1] -- 

        @return
        
          intervals -- list of (mean, lower_est, upper_est) tuples
            -- one entry for each model
                       
        @exceptions

        @notes
        """
        intervals = [(0.0, 0.0, 0.0) for var_i in range(self.numvars)]
        return intervals

    
    def influencePer2VarInteractions(self):
        """
        Returns
        -dict of 2var_tuple : relative_influence.

        Note: since this is a ConstantModel, the dict has no 2-var interactions
        and thus the result returned is merely an empty dict.
        """
        return {}
        

    def percModelsDisagreeing(self, X):
        """
        @description
        
            Percentage of submodels that disagree, in range [0.0, 1.0]
            For here, it's always 0.0
        
        @arguments

           X -- 2d array -- each column is an input point, size
                Size is [# input vars][# points]
        
        @return

           p -- 1d array -- percent that each input point disagrees
                Always returns [0.0, 0.0, ....].  Size is [# points]
            
        @exceptions
    
        @notes
        """
        N = X.shape[1]
        return numpy.zeros(N)*0.0

    def __str__(self):
        """
        @description
        
            String format representation of the constant model.
        
        @arguments
        
        @return
    
            String: constant value, number of variables
            
        @exceptions
    
        @notes
    
        """
        if self.constant == BAD_METRIC_VALUE:
            return 'ConstantModel={constant=%s, numvars=%d}' % \
                   (str(self.constant), self.numvars)
        else:
            return 'ConstantModel={constant=%.3e, numvars=%d}' % \
               (self.constant, self.numvars)
        
