""" LinearModel.py
 -Build a LinearModel object using LinearModelFactory.build()
 -Simulate a linear model via LinearModel.simulate()
 
 LinearModelFactory provides two ways to learn the linear model:
 -least-squares (the typical way)
 -'regularized' which uses a lasso derivative called 'threshold gradient
  descent' for potentially better predictive ability and also pruning
  less important variables.  Reference: Friedman and Popescu 2005.  
"""
import copy
import logging
import math
import random
import sys
import traceback

import numpy
import scipy
import scipy.linalg

from util import mathutil
from util.constants import BAD_METRIC_VALUE, INF
import RegressorUtils
from ConstantModel import yIsConstant, ConstantModel
from ChooseModel import chooseModel

log = logging.getLogger('lin')
userlog = log



class RegSS:
    """
    @description

    Solution Strategy for 'regularized' linear learning, which is
    one of the particular sub-approaches to learn a linear model.
    This is used by LinearFactory.regularizedBuild()

    @notes

    """ 
    def __init__(self):
        #===============================================================
        #This parameter governs the degree of pruning and also
        # affects prediction ability.
        # [0.6, 0.0, 1.0] => 1.0 means fewer and more diverse weights
        #Example: set to 0.95 if you want to prune away most bases
        #Example: set to 0.05 if you want to keep most bases
        #If you don't care about pruning / not pruning and you just
        # want maximum prediction ability without testing many different
        # thr values, 0.6 is a safe bet.
        self.thr = 0.95

        #===============================================================
        #Don't change the following parameters unless you really know
        # what you're doing (ie have you read Friedman 2005?)

        # initial learning rate; i.e. delta_mu in Friedman 2005
        self.init_stepsize = 2.0e-3

        # unlike Friedman, we adapt stepsize, so we need a minimum value
        self.min_stepsize = 1.0e-7
        self.max_stepsize = 0.2

        # for increasing and decreasing stepsize
        self.acceleration_rate = 1.15 #1.0=>safe; 1.15=>aggressive
        self.deceleration_rate = 10.0 #[2.0]

        # percent training data (test data used to determine stop)
        self.perc_train = 0.75

        # measure impr_rate over this number of iterations
        self.numiters_for_measure_impr = 20

        #stop if impr rate < min_impr_rate
        self.min_impr_rate = 1.0e-6

        #stop if iters > maxiters
        self.maxiters = 5000

    def __str__(self):
        s = 'RegSS={'
        s += 'thr=%.2f' % self.thr
        s += '; init_stepsize=%.2e' % self.init_stepsize
        s += '; min_stepsize=%.2e' % self.min_stepsize
        s += '; max_stepsize=%.2e' % self.max_stepsize
        s += '; acceleration_rate=%.2e' % self.acceleration_rate
        s += '; deceleration_rate=%.2e' % self.deceleration_rate
        s += '; perc_train=%.3f' % self.perc_train
        s += '; numiters_for_measure_impr=%d' % self.numiters_for_measure_impr
        s += '; min_impr_rate=%.1e' % self.min_impr_rate
        s += '; maxiters=%d' % self.maxiters
        s += ' /RegSS}'
        return s


class LinearBuildStrategy:
    """
    @description
    
        Solution strategy for linear learning, whether least-squares
        or regularized.  Being a solution strategy, it has magic numbers.

        Like most 'SolutionStrategy' or 'BuildStrategy' classes,
        it tries to balance:
        -making it convenient to set commonly-changed attributes, by
         having them in the interface but maybe with default values
        -vs. hiding values which take more thought to set if they
         are used (and those are the non-default values, sometimes
         not accessible via the constructor but must be
         changed by accessing the attribute directly)
         
    @notes

    """
    
    def __init__(self, y_transforms = None, target_nmse = None,
                 regularize = False):
        #list of y_transforms can include: lin,log10,inv,exp10
        if y_transforms is None:
            y_transforms = ['lin']
        self.y_transforms = y_transforms 
        if target_nmse is None:
            target_nmse = 1.0e-5 
        self.target_nmse = target_nmse
        self.regularize = regularize

        self.nancheck_X = False #only enabled if we've pre-transformed X
        self.reg = RegSS()

    def __str__(self):
        s = "LinearBuildStrategy={ "
        s += 'y_transforms=' + str(self.y_transforms)
        s += '; nancheck_X=%s' % self.nancheck_X
        s += '; target_nmse=%5.2e' % self.target_nmse
        s += '; regularize=%s' % self.regularize
        if self.regularize:
            s += '; reg= %s ' % str(self.reg)        
        s += ' /LinearBuildStrategy} '
        return s
        

class LinearModel:
    """
    @description

    A model of y = offset + w0*x0 + w1*x1 + ... 

    @notes

    """ 
    def __init__(self, coefs, y_transform, minX, maxX):
        """
        @description
        
        @arguments
        
            coefs -- 1d array -- coefs[0] is offset, coefs[i+1] is weight
              of variable i.
            y_transform -- string -- identifying the y_transform used
            minX, maxX -- 1d arrays -- minX[i] is min of variable i, maxX[i]..
        
        @return
    
        @exceptions
    
        @notes
    
        """ 
        self.coefs = coefs
        self.y_transform = y_transform
        self.minX = minX
        self.maxX = maxX

        self.sigma_hat_2 = None
        self.Cjj = None
        self.Xshape = None
  
    def zeroInfluenceVars(self):
        """
        @description

            Which vars have no influence?
        
        @arguments
        
        @return
        
            I -- list -- list of indices of the vars with zero-valued
                         coefficients.  Can return an empty list.
    
        @exceptions
    
        @notes
        """ 
        return [index for index, coefficient in enumerate(self.coefs[1:])  \
                if coefficient == 0.0]

    def influencePerVar(self, X=None):
        """
        @description

            Reports the relative importance of each var in this model.

            Note that this is influence across the whole _region_
            defined by [minX,maxX], which is _very_ different from
            sensitivity.  Sensitivity tells the slope, i.e. relative absolute
            coefficient values, whereas this routine multiplies eahc relative
            absolute coefficents for var i by by the width of the region for
            var i.
        
        @arguments

           X -- (ignored) Unlike other models which need input X describing
                    a region, LinearModels' impacts are not influenced by X,
                    because the linear model is a plane which doesn't
                    change region by region.  
        @return
    
            infls -- 1d array -- infls[i] is influence of variable i.  

        @exceptions
    
        @notes
    
        """ 
        if len(self.coefs) == 1:
            return numpy.array([])

        infls = abs(self.coefs[1:])
        for i,(infl,mn,mx) in enumerate(zip(infls, self.minX, self.maxX)):
            if mx == mn:
                infls[i] = 0.0
            else:
                infls[i] = infls[i] * (mx-mn)

        # normalize if possible
        sum_infls = sum(infls)
        if sum_infls > 0.0: 
            infls /= sum_infls
            
        return infls

    def delBases(self, deletion_indices):
        """
        @description

            This method interface exists to show that it should not be
            implemented.  This is because other models (mostly Isle models)
            _can_ safely implement delBases(), but a LinearModel cannot
            because its bases and coefficients are too tied to the inputs.
            
            If you _do_ want a model with linear bases that can
            be removed, then use self.bases with an IsleModel.

            A _differently_ _named_ routine does deletion, but
            with extensive side effects: self.delZeroValuedCoefs()
        
        @arguments
        
        @return
        
            Always returns a NotImplementedError
    
        @exceptions
    
            NotImplementedError
            
        @notes
        """ 
        raise NotImplementedError(
            "to use delBases() with Isle-like behavior, convert to Isle first")

    def delZeroValuedCoefs(self, deletion_indices):
        """
        @description
        
            For each index in deletion_indices, 
                removes all coefs and the corresponding minX, maxX entries
            
        @arguments
        
            deletion_indices: list of indices to be deleted from the data set
        
        @return
        
            Nothing.
            WARNING: this method has extensive side effects!
    
        @exceptions
    
        @notes
    
            WARNING: the X's coming into simulate() after this method is
            called may have a smaller number of input variables than
            the initial data set!
            
            Only the bases of LinearModel exhibit this behavior.
        """ 
        # get the indices to keep (keep all those indices not in
        # deletion_indices)
        keep_indices = mathutil.listDiff(range(len(self.coefs)-1),
                                         deletion_indices)
        
        # keep the constant offset (row 0), plus those rows in keep_indices
        keep_coefs = [self.coefs[0]] + \
                     mathutil.listTake(self.coefs[1:], keep_indices)
        self.coefs = numpy.array(keep_coefs)

        # also remove corr. vars in minX, maxX
        if self.minX is not None:
            self.minX = numpy.take(self.minX, keep_indices)
        if self.maxX is not None:
            self.maxX = numpy.take(self.maxX, keep_indices)        

    def simulate(self, X):
        """
        @description
        
            Simulate this model with inputs X.
        
        @arguments
        
            X -- 2d array -- inputs, one input point per column [var #][sample #]

        @return
    
            y -- 1d array -- output [sample #].  May contain 'inf' values.   

        @exceptions
    
        @notes
         
        """ 
        y = self.coefs[0] + \
            numpy.dot(self.coefs[1:], X) * 1.0 # ensure result is a float

        y_tr = y  # 'transformed' y
        try:
            if   self.y_transform == 'lin':   y_tr =  y
            elif self.y_transform == 'log10': y_tr = 10**y
            elif self.y_transform == 'exp10': y_tr = numpy.log10(y)
            elif self.y_transform == 'inv':   y_tr = 1.0 / y
            else: raise UnknownTransformError(' %s' % self.y_transform)

        except KeyboardInterrupt: raise
        except SystemExit: raise
        except:
            y_tr = y_tr * INF
            
        return y_tr

    def simulateStr(self, expr_str):
        """
        @description
            
            Python-evaluatable string representation which wraps
            self's y_transform around the input 'expr_str'.

            Example: if expr_str is '3.0 + X[2.,:]**2' and
            self.y_transform is 'log10' then
            this routine would return 'numpy.log10(3.0 + X[2.,:]**2)'.
            
        @arguments

            expr_str - any python-evaluatable expression string.  
        
        @return

            simulate_str -- string -- 
    
        @exceptions
    
        @notes
    
        """ 
        if   self.y_transform == 'lin':   return expr_str
        elif self.y_transform == 'log10': return '10**(' + expr_str + ')'
        elif self.y_transform == 'exp10': return 'numpy.log10(' + expr_str+')'
        elif self.y_transform == 'inv':   return '1.0 / (' + expr_str + ')'
        else: raise  UnknownTransformError(' %s' % self.y_transform)
        

    def influencePerVar(self, X=None):
        """
        @description

            Reports the relative importance of each var in this model.

            Note that this is influence across the whole _region_
            defined by [minX,maxX], which is _very_ different from
            sensitivity.  Sensitivity tells the slope, i.e. relative absolute
            coefficient values, whereas this routine multiplies eahc relative
            absolute coefficents for var i by by the width of the region for
            var i.
        
        @arguments

           X -- (ignored) Unlike other models which need input X describing
                    a region, LinearModels' impacts are not influenced by X,
                    because the linear model is a plane which doesn't
                    change region by region.  
        @return
    
            infls -- 1d array -- infls[i] is influence of variable i.  

        @exceptions
    
        @notes
    
        """ 
        if len(self.coefs) == 1:
            return numpy.array([])

        infls = abs(self.coefs[1:])
        for i,(infl,mn,mx) in enumerate(zip(infls, self.minX, self.maxX)):
            if mx == mn:
                infls[i] = 0.0
            else:
                infls[i] = infls[i] * (mx-mn)

        # normalize if possible
        sum_infls = sum(infls)
        if sum_infls > 0.0: 
            infls /= sum_infls
            
        return infls
                    
    def __str__(self):
        s = 'LinearModel={ '
        if self.y_transform:
            s += 'y_transform = %s, ' % self.y_transform
        s += 'y = %5.2e' % self.coefs[0]
        I = numpy.argsort(-1.0 * self.influencePerVar())
        for i in I:
            coef = self.coefs[i+1]
            if coef == 0:
                continue
            if coef < 0: s += ' - '
            else:        s += ' + '
            s += '%g * x%d' % (abs(coef), i)
        s += ' }'
        return s
    
class LinearModelFactory:
    """
    @description

       Builds LinearModel regression objects, invoking
       either least-squares learning or 'regularized' learning to do so.

    @notes
    """ 

    def __init__(self):
        self._covs = None #covariances, used in regularized building

    def quickRegularizedBuild(self, X, y, minX, maxX, ss):
        """Builds at: y_transform=lin; regularized; no test_cols;
        on all input variables (regardless of whether constant or not);
        does not test if X or y are 'good'.
        """
        #ss consistent?
        assert ss is not None
        assert ss.y_transforms == ['lin']
        assert ss.regularize

        #do work
        self.minX, self.maxX = minX, maxX
        (lin_model, norm, e) = self._regularizedBuild(X, y, ss.reg, ss.target_nmse, None)
        lin_model.y_transform = 'lin'

        #done
        return lin_model

    def build(self, X, y, minX, maxX, ss=None, test_cols=None):
        """
        @description
        
            Main routine to build a LinearModel object.
            
        @arguments
        
            X -- 2d array -- training inputs [var #][sample #]
            y -- 1d array -- training outputs [sample #]
            minX, maxX -- 1d arrays -- minX[i] is min of variable i, maxX[i].
            ss -- a LinearBuildStrategy object.  If None, uses defaults.
            test_cols -- if None, use all data for training.  Otherwise,
                         use the non-test_cols columns of X/y for doing
                         the linear regression and the test_cols for
                         selecting the model.
        @return

            linear_model -- LinearModel object

        @exceptions
    
        @notes
    
            n = number of input variables to the model
            N = number of samples
        """
        self.minX, self.maxX = minX, maxX
        n,N = X.shape

        #Validate data, part 1
        if n == 0:
            log.info('no input vars, so return a ConstantModel')
            return ConstantModel(y[0], 0)

        if ss is None:
            ss = LinearBuildStrategy()

        #Validate data, part 2
        assert max(y) < INF
        assert min(y) > -INF
        assert BAD_METRIC_VALUE not in y
        assert X.shape[1] == y.shape[0] #same num samples in X and y?

        s = 'Begin build(); #input dimensions=%d' % X.shape[0]
        s += '; #samples=%d' % X.shape[1]
        s += '; min(y)=%0.2e, max(y)=%0.2e' % (min(y), max(y))
        s += '; regularize? %s' % ss.regularize
        log.info(s)
        log.debug('SS = %s' % ss)

        #Corner case: if y does not vary, just return a ConstantModel
        if yIsConstant(y):
            return ConstantModel(y[0], X.shape[0])

        #Main case...

        # Remove some of X's rows, to just use active_vars 
        (active_X, active_vars) = self._wellBehaved(X)
        if N < len(active_vars) and not ss.regularize:
            s = 'need reasonable # samples (got %d, but %d coefs ' \
                'to learn)' % (N, len(active_vars))
            raise InsufficientDataError(s)
        log.debug('  %d/%d input rows (bases) are non-constant' %
                  (len(active_vars), X.shape[0]))

        #Traverse the set of y-transforms, keep the best model
        # We'll be building the model on active_X 
        log.debug('  y_transforms to try=%s' % ss.y_transforms)
        best_model, best_normality, best_error = None, 0.0, INF
        for y_transform in ss.y_transforms:
            assert y_transform in ['lin','log10','exp10','inv']
            (m, norm, e) = self._buildAtTransform(active_X, y, ss, 
                                                  y_transform, test_cols)
            best_model, best_normality, \
                        best_error = chooseModel(best_model, best_normality,
                                                 best_error, m, norm, e)

        #Corner case: all models are terrible, return a ConstantModel
        if best_model is None:
            log.debug('  Even best model was terrible.  Use median.')
            return ConstantModel(mathutil.median(y), X.shape[0])

        #We've learned the best model on just active_vars, but
        # it has to be valid across _all_ vars.  So fill it in with zeros
        # for non-active vars' coefficients.
        full_coefs = numpy.zeros(n + 1, dtype=float)
        full_coefs[0] = best_model.coefs[0]
        for i,var_index in enumerate(active_vars):
            full_coefs[var_index + 1] = best_model.coefs[i + 1] 
        best_model.coefs = full_coefs

        if best_error == INF: 
            best_model.y_transform = 'lin'

        #Final output message 's'
        s = 'Done. Transform=%s; normality=%.2f' % \
            (best_model.y_transform, best_normality)
        if test_cols is None:
            s += ', train nmse=%5.2e' % best_error
        else:
            s += ', test nmse=%5.2e' % best_error
        s += '; %d/%d nonzero coefs' % \
             (len(numpy.nonzero(best_model.coefs)), len(best_model.coefs))
        log.info(s)

        return best_model

    def _wellBehaved(self, X):
        """
        @description
        
            Reduce the amount of data to process by eliminating all
            rows in X that are constant and all rows that are duplicates
            of another row.
        
        @arguments
        
            X -- 2d array -- training inputs [var #][sample #]

        @return
        
            active_X -- 2d array -- like X, but with possibly fewer rows
            active_vars -- list -- list of rows in X which were kept
    
        @exceptions
    
        @notes
        """
        active_vars = range(X.shape[0]) # all possible rows
        active_vars = mathutil.removeConstantRows(X, active_vars)
        active_X = numpy.take(X, active_vars, 0)
        return (active_X, active_vars)

    def _buildAtTransform(self, X, yIn, ss, y_transform, test_cols):
        """
        @description
        
            Build a model at the specified transform.  Assumes
            that the data has already been firewalled / cleaned by
            its parent routine, self.build().

            This routine first transforms y, then further parcels the work
            into 'unregularizedBuild()' or 'regularizedBuild()', depending
            on what ss says.
            
        @arguments

            y_transform -- string -- one of 'lin','log10','exp10','inv'
            X, yIn, ss, test_cols -- like build(), except that
            X has had some rows pruned, so don't worry about that anymore.
            yIn is still NOT transformed.
            
        @return
        
            model -- LinearModel object -- what this routine builds
            norm -- float -- measure of normality.  0.0=worst, 1.0=best
            e -- error -- training nmse if test_cols is None, else test nmse
    
        @exceptions
    
        @notes
    
        """ 
        log.debug('  Try y_transform=%s' % y_transform)
        y = None
        try:
            if   y_transform == 'lin':   y = yIn
            elif y_transform == 'log10': y = numpy.log10(yIn)
            elif y_transform == 'exp10': y = 10**yIn
            elif y_transform == 'inv':   y = 1.0 / yIn
            else: raise AssertionError("unknown y_transform")
            if y_transform != 'lin':
                assert max(y) < INF
                assert min(y) > -INF
                assert not mathutil.hasNan(y)
        except KeyboardInterrupt: raise
        except SystemExit: raise
        except:
            log.debug('    Applying transform not successful')
            return (None, 0.0, INF)
        if y_transform != 'lin':
            log.debug('    Applying transform successful')

        if (len(y) > 3) and ss.regularize:
            model, norm, e = self._regularizedBuild(X, y, ss.reg, ss.target_nmse, test_cols)
            if model is not None:
                model.y_transform = y_transform
        else:
            model, norm, e = self._unregularizedBuild(X, y, y_transform,  test_cols)

        if test_cols is None: s = 'train'
        else:                 s = 'test'
        log.debug('    Done y_transform=%s: normality=%.2f, %s nmse=%5.2e' % (y_transform, norm, s, e))
        return model, norm, e

    def _unregularizedBuild(self, X, y, y_transform, test_cols):
        """
        @description
        
            Apply the typical (unregularized) least-squares regression.
            Returns a LinearModel and nmse.  'y' is already transformed.

            Its 'twin' which has identical inputs and outputs but
            builds a model differently is 'regularizedBuild()'.
            y_transform is not used, except for to set it into linear_model.
        
        @arguments

            X, yIn, ss, test_cols -- like buildAtTransform(),
            except that yIn IS now transformed according to y_transform.
        
        @return

            model, norm, e -- like buildAtTransform()            
    
        @exceptions
    
        @notes    
        """ 
        if test_cols is None:
            #Build a model using _all_ the training data.
            # Its error 'e' is the training error.
            (model, norm, e) = self._leastSquares(X, y)
            if model is not None: 
                model.y_transform = y_transform
            
        else:
            #Build a model using a subset of the input data (trn_cols),
            # then simulate it on the test_cols.  Its error 'e'
            # is the test error.
            trn_cols = mathutil.listDiff(range(X.shape[1]), test_cols)
            (model, norm, e) = self._leastSquares(numpy.take(X, trn_cols,1), \
                                                  numpy.take(y, trn_cols))
            if model is not None:
                model.y_transform = y_transform
            if e != INF: #(successful build)
                temp_model = LinearModel(
                                 model.coefs, 
                                 y_transform = 'lin',
                                 minX = model.minX, 
                                 maxX = model.maxX
                             )
                test_yhat = temp_model.simulate(numpy.take(X, test_cols, 1))
                test_y = numpy.take(y, test_cols)
                e = mathutil.nmse(test_yhat, test_y, min(y), max(y))
                norm = mathutil.normality(test_yhat - test_y)

        return (model, norm, e)
            
    def _leastSquares(self, X, y):
        """
        @description
        
            Apply the typical (unregularized) least-squares regression.
        
        @arguments

            X, yIn, ss -- like unregularizedBuild()
        
        @return

            model, norm, e -- like unregularizedBuild()   
    
        @exceptions

           If least squares call fails, then it returns:
           model, norm, e = None, 0.0, INF
           
        @notes
        
        If b is a matrix then x is also a matrix with corresponding columns.
        If the rank of A is less than the number of columns of A or greater than
        the numer of rows, then residuals will be returned as an empty array
        otherwise resids = sum((b-dot(A,x)**2).
        Singular values less than s[0]*rcond are treated as zero.
        """ 
        [n,N] = X.shape 
        
        # A must have an extra row of ones to hold the constant term
        A = numpy.zeros((n+1,N), dtype=float)
        A[0,:] = numpy.ones(N, dtype=float)
        A[1:n+1,:] = X

        # Invoke the worker routine!
        log.debug('About to invoke scipy.linalg.lstsq()')
        try:
            (coefs, resids, rank, sing_vals) = \
                scipy.linalg.lstsq( scipy.asarray(numpy.transpose(A)),
                                    scipy.asarray(numpy.transpose(y)) )

            if len(coefs.shape) == 2 :
                [m,M] = coefs.shape # need to make 1-d array
                if m > 1:
                    nrows = m
                else:
                    nrows = M
                coefs = numpy.reshape(coefs,(nrows,))

            #sometimes coefs comes back as Float64.  We want Float32 ('float')
            coefs = numpy.array(coefs, dtype=float)
            
            temp_model = LinearModel(coefs, y_transform = 'lin',
                                     minX = self.minX, maxX = self.maxX)
            yhat = temp_model.simulate(X)
            train_e = mathutil.nmse(yhat, y, min(y), max(y))
            norm = mathutil.normality(yhat - y)
            
            model = LinearModel(coefs, y_transform=None,
                                minX = self.minX, maxX = self.maxX)
            return (model, norm, train_e)
        except KeyboardInterrupt: raise
        except SystemExit: raise
        except Exception, e:
            log.warning('scipy.linalg.lstsq() failed.')
            log.warning('Details:\n%s', traceback.format_exc())

            model, norm, train_e = None, 0.0, INF
            return (model, norm, train_e)
        log.warning('Should never get here')


    def _regularizedBuild(self, Xin, yin, ss, target_test_nmse, test_cols):
        """
        @description

            Builds a linear model using 'regularized' linear learning,
            as described in Apply Friedman and Popescu, 'Gradient
            Directed Regularization', 2004.
        
            Its 'twin' which has identical inputs and outputs but
            builds a model differently is 'unregularizedBuild()'.
        
        @arguments

            X, yIn, ss, y_transform, test_cols -- like buildAtTransform(),
            except that yIn IS now transformed according to y_transform.
        
        @return

            model, norm, e -- like buildAtTransform()            
    
        @exceptions
    
        @notes        
        """

        #Condition X and y: subtract each row's mean, then divide by stddev
        n = Xin.shape[0]
        y_avg, y_std = numpy.average(yin), mathutil.stddev(yin)
        X_avgs, X_stds = mathutil.averagePerRow(Xin), mathutil.stddevPerRow(Xin)
        (X, y, test_X, test_y, yX) = self._trainTestData( \
            Xin, X_avgs, X_stds, yin, y_avg, y_std, ss, test_cols)
        data = (X, y, test_X, test_y, yX)

        #initialize coefs (learn these)
        a = numpy.zeros(n + 1, dtype=float)

        #reset the cached covariances
        self._covs = {}

        #main iterative loop
        inf = INF
        failed, best_e, best_a = False, inf, a
        round_i = 0
        init_stepsize = ss.init_stepsize
        while True:
            log.debug("Do regularized lin learning round #%d" % (round_i+1))
            next_failed, next_e, next_a = self._regularizedBuild_OneRound( \
                data, ss, target_test_nmse, test_cols, a, init_stepsize)
            
            if round_i == 0 and next_failed:
                failed = True
            
            if next_failed or next_e >= best_e: #unsuccessful, so stop
                break
            
            best_e, best_a = next_e, next_a
            if next_e < target_test_nmse: #successful, and updated, so stop
                break
            
            init_stepsize /= ss.deceleration_rate
            round_i += 1
        log.debug("Done regularized lin learning rounds")

        #We'd learned the coefficients on a scaled X and y.
        # The following lines put the coefficients back into proper space.
        if not failed:
            coefs = numpy.zeros(n+1, dtype=float)
            coefs[0] = best_a[0] * y_std + y_avg
            for j in range(1,n+1):
                coefs[j] = best_a[j] * y_std / X_stds[j-1]
                coefs[0] -= coefs[j] * X_avgs[j-1]
            temp_model = LinearModel(coefs=coefs, 
                                     y_transform='lin', minX = self.minX,
                                     maxX = self.maxX)
            norm = mathutil.normality(temp_model.simulate(test_X) - test_y) 
            model = LinearModel(coefs=coefs, y_transform=None,
                                minX = self.minX, maxX = self.maxX)
        else:
            log.debug('Reg. build failed, so return a constant')
            model = ConstantModel(y[0], X.shape[0])
            norm, best_e = 0.0, INF
                                
        self._covs = None
            
        return (model, norm, best_e)
        
    def _regularizedBuild_OneRound(self, data, ss, target_test_nmse,
                                   test_cols, a, init_stepsize):
        (X, y, test_X, test_y, yX) = data
        n = X.shape[0]
        miny, maxy = min(min(y),min(test_y)), max(max(y),max(test_y))
            
        stepsize = init_stepsize

        #initialize g
        g = numpy.zeros(n+1, dtype=float) #gradient
        for j in range(1,n+1):
            g[j] = self._cov(yX, 0, j)

        best_e = INF
        iters, recent_es = 0, [best_e]
        failed = False
        while True:
            iters += 1

            #==================================================================
            #The bottom of this while loop to compute gradient 'g', plus
            # the next four lines, is where the real work gets done:
            # 'g' tells the rate of change of each coefficient
            # then 'f' flags the variables which still have non-small gradients
            # then 'h' is like 'g' except small-valued gradients are set to 0
            # then 'a', the coef values, are updated in the diretion of 'h'.
            # 
            f = numpy.where(abs(g) >= ss.thr*max(abs(g)), 1.0, 0.0); f[0] = 0.0
            h = f * g #'generalized' gradient (like g, but small gi's = 0)
            a += stepsize * h

            #==================================================================
            #Report progress; maybe stop.
            test_yhat = numpy.dot(a[1:], test_X) + a[0]
            e = mathutil.nmse(test_yhat, test_y, miny, maxy)
            impr_rate, recent_es = self._updateRecentErrors(e, recent_es, ss)
            
            if e < best_e:
                best_a, best_e = copy.copy(a), e
                stepsize *= ss.acceleration_rate
                stepsize = min(ss.max_stepsize, stepsize)
            else:
                log.debug('Stop: test nmse stopped improving'); break
                    
            log.debug('Iter=%d:test nmse=%.4e, step=%.1e, impr_rate=%.2e'%
                      (iters, best_e, stepsize, impr_rate))
                               
            if e < target_test_nmse:
                log.debug('Stop: target test nmse hit'); break
            elif iters > ss.maxiters:
                log.debug('Stop: num iters > max'); break
            elif stepsize <= ss.min_stepsize:
                log.debug('Stop: stepsize < min'); break
            elif len(recent_es)==ss.numiters_for_measure_impr and \
                     impr_rate < ss.min_impr_rate:
                log.debug('Stop: improvement stagnated'); break
            elif best_e == INF:
                log.debug("Failure stop: best e still inf")
                failed = True;  break

            #==================================================================
            #This is the other part of the work:
            # Update our estimation of the gradient.  Note the covariance calc.
            G = list(numpy.nonzero(f))
            for j in range(1, n+1):
                g[j] -= stepsize * sum(h[k] * self._cov(yX, j, k)  for k in G)

        return failed, best_e, best_a

    def _trainTestData(self, Xin, X_avgs, X_stds, yin, y_avg, y_std, ss,
                      test_cols):
        """
        @description

          Helper routine for regularizedBuild that preps its
          data in a way that it finds useful.
    
        """ 
        [n,Nall] = Xin.shape
        allI = range(Nall)

        if test_cols is None:
            perc_test = 1.0 - ss.perc_train
            test_cols = RegressorUtils.generateTestColumns(yin, perc_test)
        else:
            assert min(test_cols) >= 0 and max(test_cols) < Nall
        I = mathutil.listDiff(allI, test_cols)    

        #train info
        X, y = numpy.take(Xin, I, 1), numpy.take(yin, I)
        X = self._unbias(X, X_avgs, X_stds)
        y = (y - y_avg) / y_std

        #test info, for early stopping
        test_X = numpy.take(Xin, test_cols, 1)
        test_y = numpy.take(yin, test_cols)
        
        test_X = self._unbias(test_X, X_avgs, X_stds)
        test_y = (test_y - y_avg) / y_std

        #for cleaner calcs
        yX = numpy.zeros(((n+1), len(I)), dtype=float)
        yX[0, :] = y
        yX[1:,:] = X
        
        return X, y, test_X, test_y, yX

    def _updateRecentErrors(self, e, recent_es, ss):
        """
        @description
        
          Helper routine for regularizedBuild that lets it track its 'smoothed'
          error vs. time.
        """ 
        recent_es.append(e)
        if len(recent_es) > ss.numiters_for_measure_impr:
            recent_es = recent_es[-ss.numiters_for_measure_impr:]
        impr_rate = (max(recent_es) - min(recent_es))
        return (impr_rate, recent_es)        

    def _cov(self, yX, i1, i2):
        """
        @description
        
          Return covariance between yX[i1,:] amd yX[i2,:]
          Uses caching: if self._covs if key exists, else calcs it and saves it 

          Helper routine for regularizedBuild.
        """ 
        iu,iv = i1,i2
        if iv < iu:
            iu,iv = i2,i1

        key = str((iu,iv))
        if not self._covs.has_key(key): #self.cov_calc[iu,iv]:
            self._covs[key] = numpy.dot(yX[iv,:], yX[iu,:]) / yX.shape[1]

        return self._covs[key]
        
    def _unbias(self, Xin, avgs, stds):
        """
        @description
        
            Remove bias from the values in the input matrix by
            first subtracting each row by that row's mean, then
            dividing each row by that row's stddev.  Thus each
            row no offset, and equal variance.
        
        @arguments
        
            X -- 2d array -- training inputs [var #][sample #]
            avgs: a vector of average values for each row in Xin
            stds: a vector of std_dev for each row in Xin
            
        @return

            unbiased_X -- 2d array -- like X, except unbiased
            
        @exceptions
    
            If the stds vector contains a zero, the divide by zero 
            exception will be thrown.
            
        @notes
    
        """ 
        X = numpy.zeros(Xin.shape, dtype=float)
        
        try:
            for i in range(Xin.shape[0]):
                X[i,:] = (Xin[i,:] - avgs[i]) / stds[i]
        except KeyboardInterrupt: raise
        except SystemExit: raise
        except:
            log.debug('Standard deviation of 0 when removing bias in unbias')
            
        return X
        
