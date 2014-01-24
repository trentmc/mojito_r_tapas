"""Isle.ps. ISLE = Iterative Sampled Learning Ensemble.  Ref. Friedman 2004.

Isles leave the 'base discovery' to other modelers, but they are good at
pulling it all together cleanly.
"""

import logging

import copy
import math

import numpy

from util import mathutil
from IsleBase import IsleBase, LinearBase, CartBase, PolyBase, \
     QuadraticBase, HSBase, ProductBase, IsleBasesFactory, pruneRedundantBases
from LinearModel import LinearModelFactory, LinearBuildStrategy, LinearModel

log = logging.getLogger('isle')

def isleXterms(X11, bases11):
    Xterms11 = numpy.ones((len(bases11), X11.shape[1]))*1.0
    for base_i,base11 in enumerate(bases11):
        Xterms11[base_i,:] = base11.simulate(X11)
    return Xterms11        
        
class IsleModel:
    """Generalized model"""
    def __init__(self, lin_model, bases11, minX, maxX, X11):
        assert isinstance(lin_model, LinearModel), lin_model.__class__
        self.lin_model = lin_model

        for i,base11 in enumerate(bases11):
            setmean = isinstance(base11, PolyBase) or \
                      isinstance(base11, QuadraticBase) or \
                      (isinstance(base11, LinearBase) and base11.mean is None)
            
            if setmean:
                base11.setMeanAndStddev(X11)
                
        self.bases11 = bases11
        
        self.y_transform = lin_model.y_transform
        self.numvars = len(minX)
        self.minX = minX
        self.maxX = maxX

    def __len__(self):
        return len(self.bases11)
        
    def simulate(self, X):
        """
        Note that children do not have to provide this!
        
        Inputs:  X - model inputs [var #][sample #]
        Outputs: y - model output [sample #]

        See that each base simulates in X11 (scaled to [-1,+1]) space,
        NOT in X (unscaled) space.
        """
        X11 = mathutil.scaleTo01(X, self.minX, self.maxX)*2.0 - 1.0
        return self.lin_model.simulate( isleXterms(X11, self.bases11) )

    def uncertainty(self, X):
        """
        Inputs:  X - model inputs [var #][sample #]
        Outputs: u - uncertainty of output [sample #], each entry in [0,1]
        """
        return numpy.array([self.dist01toClosest(X[:,sample_i]) \
                              for sample_i in range(X.shape[1])])
        
    def dist01toClosest(self, x):
        """Distance from x to closest point in self.trainX,
        scaled to [0,1] based on self.minX and self.maxX
        Note that an IsleModel does not normally store self.trainX (for memory)
        so the caller has to tack it on in order to use this routine."""
        range_x = self.maxX - self.minX
        active_vars = [i for i,range_x_i in enumerate(range_x) if range_x_i > 0]
        n = float(len(active_vars))
        x2 = numpy.take(x, active_vars)
        range_x2 = numpy.take(range_x, active_vars)
        X2 = numpy.take(self.trainX, active_vars, 0)
        return min([math.sqrt( sum( ((x2 - X2[:,samp_i])/range_x2)**2 )/n ) \
                    for samp_i in range(X2.shape[1])])
    

    def influencePerBase(self, X11=None):
        return numpy.array([self.influenceOfBase(base_i, X11)
                              for base_i in range(len(self.bases11))])

    def influenceOfBase(self, base_i, X11=None):
        #Note that using coefs is different than lin_model.influencePerVar()
        # because lin.influencePerVar takes into account variable scaling
        # which we account for elswehere in IsleModel.
        w = abs(self.lin_model.coefs[base_i+1])
        
        base11 =  self.bases11[base_i]

        #global infl
        if X11 is None: 
            if isinstance(base11, LinearBase):
                infl = w * base11.stddev
            elif isinstance(base11, PolyBase):
                infl = w * base11.stddev
            elif isinstance(base11, QuadraticBase):
                infl = w * base11.stddev
            elif isinstance(base11, ProductBase):
                infl = w * base11.stddev
            else:
                raise AssertionError('unsupported type of base')

        #local infl
        else:
            if isinstance(base11, LinearBase):
                infl = w * sum( abs(X11[base11.var_index] - base11.mean) )
            elif isinstance(base11, PolyBase):
                infl = w * sum( abs(base11.simulate(X11) - base11.mean) )
            elif isinstance(base11, QuadraticBase):
                infl = w * sum( abs(base11.simulate(X11) - base11.mean) )
            elif isinstance(base11, ProductBase):
                infl = w * sum( abs(base11.simulate(X11) - base11.mean) )
            else:
                raise AssertionError('unsupported type of base')
                
        return infl
                                     
    def influencePerVar(self, X=None):
        """        
        Inputs:  X - (optional) prediction points [desvar #][sample #]
        Outputs: I - influence estimates [desvar #]
        """
        if X is None:
            X11 = None
        else:
            X11 = mathutil.scaleEachRowToPlusMinus1(X, self.minX, self.maxX)
        infl_per_base = self.influencePerBase(X11)
            
        infl_per_var = numpy.zeros(self.numvars)*0.0
        for base_i, (base11, base_infl) in enumerate(zip(self.bases11,
                                                         infl_per_base)):
            if isinstance(base11, LinearBase):
                infl_per_var[base11.var_index] += base_infl
            elif isinstance(base11, PolyBase):
                dvars = [var_i
                         for var_i,exp in enumerate(base11.explist)
                         if exp != 0]
                for var in dvars:
                    infl_per_var[var] += base_infl / float(len(dvars))
            elif isinstance(base11, QuadraticBase):
                infl_per_var[base11.var_index] += base_infl
            elif isinstance(base11, ProductBase):
                n = len(self.minX)
                dvars = [var_i for var_i in range(n) if base11.hasVar(var_i)]
                for var in dvars:
                    infl_per_var[var] += base_infl / float(len(dvars))
                
            else:
                raise AssertionError('unsupported type of IsleBase')
        
        if sum(infl_per_var) > 0.0:
            infl_per_var = infl_per_var / sum(infl_per_var)
            
        return infl_per_var
        
    def zeroInfluenceVars(self):
        """Returns list of vars with an influence of 0"""
        return [var for var,infl in enumerate(self.influencePerVar()) if infl==0]
    
    def delZeroInfluenceBases(self):
        """Removes bases with a coefficient of 0.  Returns deleted bases."""
        del_I = self.lin_model.zeroInfluenceVars()
        return self.delBases(del_I)

    def importantBases(self, cumul_infl_cutoff=0.999):
        """Returns the bases which contribute to the input cumulative influence
        value of overall influence, but not the remaining bases"""
        infl_per_base = self.influencePerBase()
        important_bases_I = mathutil.mostImportantVars(infl_per_base,
                                                       cumul_infl_cutoff)
        return list(numpy.take(self.bases11, important_bases_I))

    def delLowInfluenceBases(self, cumul_infl_cutoff, max_num_bases=10000):
        """Measures impact per base, then 
        -removes bases that whose impact is not part of the cumul_infl_cutoff-impact target;
        -if # bases is still > max, removes the lesser-impact bases too
        Returns removed bases"""
        infls = self.influencePerBase()
        keep_I = mathutil.mostImportantVars(infls, cumul_infl_cutoff)
        if len(keep_I) > max_num_bases:
            keep_I = list(numpy.argsort(-infls))[:max_num_bases]
        del_I = mathutil.listDiff(range(len(self.bases11)), keep_I)
        return self.delBases(del_I)

    def delBases(self, del_I):
        """Removes all bases in indices 'del_I'. Returns the deleted bases"""
        all_I = range(len(self.bases11))
        keep_I = mathutil.listDiff(all_I, del_I)
        self.lin_model.delZeroValuedCoefs(del_I)
        del_bases11 = mathutil.listTake(self.bases11, del_I)
        self.bases11 = mathutil.listTake(self.bases11, keep_I)
        assert len(self.lin_model.coefs)-1 == len(self.bases11)
        return del_bases11

    def __str__(self):        
        s = 'Isle model={'
        s += 'y_transform=%s' % self.y_transform
        s += '; tot # bases=%d' % len(self.bases11)
        s += '; %s' % self.basisFunctionInfoStr()
        s += '; actual model f(x) = '
        s += '%g' % self.lin_model.coefs[0]

        max_num_bases = 30 #magic number alert. max num bases to show
        if self.lin_model.coefs[0] is None:
            keep_I = range(max_num_bases)
        else:
            keep_I = [i for i in range(len(self.bases11))
                      if self.lin_model.coefs[i+1] != 0]
            
            if len(keep_I) > max_num_bases:
                infls = self.influencePerBase()
                sum_infls = sum(infls)
                if sum_infls > 0:
                    infls /= sum_infls #normalize

                #magic number alert
                keep_I = mathutil.mostImportantVars(infls, 0.95) 
                if len(keep_I) > max_num_bases:
                    keep_I = list(numpy.argsort(-infls))[:max_num_bases]

        for i in keep_I:
            coef = self.lin_model.coefs[i+1]
            if coef is None:
                coef_str = 'None'
                s += ' + '
            else:
                coef_str = '%g' % abs(coef)
                if coef_str == '0':
                    coef_str = '%.3e' % abs(coef)
                if coef < 0: s += ' - '
                else:        s += ' + '
            
            base11 = self.bases11[i]

            s += '%s * %s' % (coef_str, base11)

        num_bases_left = len(self.bases11) - len(keep_I)
        if num_bases_left > 0:
            s += ' + (%d other bases...)' % num_bases_left

        s += ' }'
        return s

    def basisFunctionInfoStr(self):
        #return a string that reports relative influence of
        # different classes of bases

        #fill in dicts of class:count, class:infl
        count_per_class, infl_per_class = {}, {}
        for base_i, base11 in enumerate(self.bases11):
            coef = self.lin_model.coefs[base_i+1]
            c = str(base11.__class__)
            
            rloc = c.rfind('.')
            if rloc != -1:
                c = c[rloc+1:]
                
            if not count_per_class.has_key(c):
                count_per_class[c] = 0
                infl_per_class[c] = 0.0
                
            count_per_class[c] += 1
            if coef is None:   infl_per_class[c] = None
            else:              infl_per_class[c] += abs(coef)
        num_classes = len(infl_per_class)

        #output s
        s = ''
        
        # -case when no coefs available
        if self.lin_model.coefs[0] is None:
            s += 'num_bases_per_basis_function_class={'
            for i,c in enumerate(sorted(infl_per_class.keys())):
                s += '%s:%d' % (c, count_per_class[c])
                if i < num_classes-1: s += ', '
            s += '}'

        # -case when coefs available
        else:       
            ordered_class_names = mathutil.mostImportantVars(infl_per_class, 1.0)
            denom = sum(infl_per_class.values())
            if denom == 0.0: denom = 1.0
            
            s += 'rel_influence_per_basis_function_class={'     
            for i,c in enumerate(ordered_class_names):
                s += '%s:%g' % (c, infl_per_class[c]/denom)
                if i < num_classes-1: s += ', '
            s += '}'
                
            s += '; num_bases_per_basis_function_class={'    
            for i,c in enumerate(ordered_class_names):
                s += '%s:%d' % (c, count_per_class[c])
                if i < num_classes-1: s += ', '
            s += '}'
            
        return s



class ConstructiveIsleBuildStrategy:

    def __init__(self, *args):
        max_order, max_num_pregrow_bases, y_transforms, lin_ss = args[0]
        self.max_order = max_order
        self.max_num_pregrow_bases = max_num_pregrow_bases 
        self.y_transforms = y_transforms
        self.lin_ss = lin_ss

        self.perc = 1.0-1.0/1000.0

    def __str__(self):
        s = 'ConstructiveIsleBuildStrategy={'
        s += ' max_order=%d' % self.max_order
        s += '; max_num_pregrow_bases=%d' % self.max_num_pregrow_bases
        s += '; y_transforms=%s' % self.y_transforms
        s += '; perc=%.2e' % self.perc
        s += '; lin_ss=%s' % self.lin_ss
        s += ' /ConstructiveIsleBuildStrategy}'
        return s
    
    
class IsleModelFactory:
    
    def buildFromBases(self, bases11, X, X11, y, minX, maxX, lin_ss, test_cols):
        """
        @description
        
          Build an IsleModel from bases.  Each base can be
          any sort of an IsleBase, e.g. a LinearBase, or other

        @arguments

          bases11 -- list of IsleBase objects -- each of these
            has been trained on X11, not (unnormalized) X
          X -- 2d array -- inputs, with each row i ranging in [minX[i], maxX[i]]
          X11 -- 2d array -- like X, except each row is ranging in [-1,+1]
          y -- 1d array -- target outputs, one entry per column of X
          minX, maxX -- 1d array, 1d array -- (unnormalized) min and max; this
            is the info that tells how to scale from X to X11
          lin_ss -- LinearBuildStrategy object -- parameters on how
            to learn the weighting coefficients
          test_cols -- indices of training points to use as test (vs. train) data
            Can be None.
            
        @return

          isle_model -- an IsleModel object -- what gets built
        
        @exceptions
    
        @notes

          We pass in X and X11 because we've already had to compute it
          in order to learn bases11
        """
        assert X.shape == X11.shape
        assert len(y) == X.shape[1]
        assert minX.shape[0] == maxX.shape[0] == X.shape[0]

        bases11 = pruneRedundantBases(bases11)
        
        #compute the values of the bases at X11
        XT = isleXterms(X11, bases11)
        
        # -corner case: if num samples < num_lin_coefficients needed
        #               then use regularize (which doesn't care; LS cares)
        active_base_I = range(XT.shape[0]) 
        active_base_I = mathutil.removeConstantRows(XT, active_base_I)
        lin_ss = copy.deepcopy(lin_ss)
        if XT.shape[1] < len(active_base_I) and not lin_ss.regularize:
            s = 'Had to change from LS learning to regularized, b/c'
            s += ' had %d bases to learn linear coefs for, but just %d samples'%\
                 (len(active_base_I), XT.shape[1])
            log.warning(s)
            lin_ss.regularize = True

        if (lin_ss is not None and lin_ss.regularize) and test_cols is None:
            all_cols = range(len(y))
            random.shuffle(all_cols)
            perc_test = 0.25 #magic number
            num_test = len(y) * perc_test
            test_cols = all_cols[:num_test]

        #learn the linear weighting coefficients
        linear_model = LinearModelFactory().build(XT, y, None, None,
                                                  lin_ss, test_cols)
            
        #create an isle model and return it
        assert len(linear_model.coefs) == ( len(bases11)+1 )
        isle_model = IsleModel(linear_model, bases11, minX, maxX, X11)
        return isle_model        

    def buildFromConstantValue(self, constant_value, minX, maxX):
        coefs = numpy.array([constant_value])
        linear_model = LinearModel(coefs, 'lin', minX, maxX)
        isle_model = IsleModel(linear_model, [], minX, maxX, None)
        return isle_model
    
    def buildConstructively(self, init_bases11, X, X11, y, minX, maxX,
                            ss, test_cols):
        """
        @description
        
          Build an IsleModel from bases.  Each base can be
          any sort of an IsleBase, e.g. a LinearBase, or other

          Strategy:
          -start with input bases (could be ultra-simple)
          -iteratively expand the highest-impact bases with 'higherOrderBases'
          routine specific to each Base

        @arguments

          initial_bases11 -- list of IsleBase objects -- each of these
            has been trained on X11, not (unnormalized) X.  
          X -- 2d array -- inputs, with each row i ranging in [minX[i], maxX[i]]
          X11 -- 2d array -- like X, except each row is ranging in [-1,+1]
          y -- 1d array -- target outputs, one entry per column of X
          minX, maxX -- 1d array, 1d array -- (unnormalized) min and max; this
            is the info that tells how to scale from X to X11
          ss -- a ConstructiveIsleBuildStrategy object 
          test_cols -- indices of training points to use as test (vs. train) data
            
        @return

          isle_model -- an IsleModel object -- what gets built
        
        @exceptions
    
        @notes

          We pass in X and X11 because we've already had to compute it
          in order to learn init_bases11
        """
        full_n,N = X.shape
        if N < 10:
            raise InsufficientDataError('need reasonable # samples (got %d)'%N)
        log.info('Begin build(); #vars=%d, #samples=%d (train=%d, test=%d)' %
                 (full_n, N, N-len(test_cols), len(test_cols)))
        log.debug('ss=%s' % ss)

        if full_n == 0:
            return self.buildFromConstantValue(y[0], minX, maxX)
        if yIsConstant(y):
            return self.buildFromConstantValue(y[0], minX, maxX)

        minX11 = -1.0 * numpy.ones(X.shape[0], dtype=float)
        maxX11 = +1.0 * numpy.ones(X.shape[0], dtype=float)
        
        #build model, iterating through y-transforms
        best_model, best_e = None, float('Inf')
        for y_transform in ss.y_transforms:
            log.info('Try at y_transform=%s' % y_transform)
            (model,e) = self.buildHelper(init_bases11[:], X, X11, y, minX, maxX,
                                         test_cols, ss, y_transform)
            log.info('Done y_transform=%s; test nmse = %.3e' % (y_transform,e))
            if e < best_e:
                best_model, best_e = model, e
            if best_e < ss.lin_ss.target_nmse and len(ss.y_transforms)>1:
                log.debug('Stop iterating on transforms b/c target nmse hit')
                break

        log.info('Done. Final test e=%.2e' % best_e)
        log.debug('Final model=%s' % str(best_model))
        return best_model
    
    def buildHelper(self, bases11, X, X11, y, minX, maxX,
                    test_cols, ss, y_transform):
        """Build constructively, but more aggressively:
        -init lin learning on all init bases
        -then iteratively add all the interaction terms of high-impact vars
        """
        ok_vars = mathutil.removeConstantRows(X11, range(X11.shape[0]))
        tabu_vars = mathutil.listDiff(range(X11.shape[0]), ok_vars)
        bases11 = [base11 for base11 in bases11
                   if not base11.influencedOnlyBy(tabu_vars, X11.shape[0])]
        
        ss.lin_ss.y_transforms = [y_transform]

        tabu_bases11 = []

        test_e_per_iter, models_per_iter = [],[]
        more_to_add = True
        while True:
            s = 'Call buildFromBases() with these bases:'
            for base11 in bases11: s += '%s\n' % base11
            log.debug(s)

            model = self.buildFromBases(bases11, X, X11, y, minX, maxX,
                                        ss.lin_ss, test_cols)
            tabu_vars += model.zeroInfluenceVars()
            tabu_bases11 += model.delZeroInfluenceBases()
            if len(model)==0: more_to_add = False

            trn_e, test_e = self.errors(model, X, y, test_cols)
            s = 'Constructive iter #%d' % len(test_e_per_iter)
            s += '; train nmse=%.2e, test nmse=%.2e' % (trn_e, test_e)
            s += '; #bases=%d; #tabu bases=%d, #tabu vars=%d' % \
                 (len(bases11), len(tabu_bases11), len(tabu_vars))
            log.info(s)
            test_e_per_iter.append(test_e)
            models_per_iter.append(model)
            
            if not more_to_add:
                log.info('Stop: no new bases'); break
            elif test_e == float('Inf'):
                log.info('Stop: test nmse is Inf'); break
            elif test_e > 1.0:
                log.info('Stop: test nmse horrible'); break
            elif test_e < ss.lin_ss.target_nmse:
                log.info('Stop: test nmse hit'); break
            elif self._stalled(test_e_per_iter):
                log.info('Stop: stalled (test nmse not improving)'); break

            tabu_bases11 += model.delLowInfluenceBases(ss.perc,
                                                       ss.max_num_pregrow_bases)
            
            #add new bases.  Try highest-infl base first, then 2nd-highest, etc
            infls = model.influencePerBase()
            more_to_add = False
            for base_i in numpy.argsort(-infls):
                all_tabu_bases11 = bases11 + tabu_bases11
                new_bases11 = bases11[base_i].higherOrderBases(all_tabu_bases11,
                                                               tabu_vars, ss)
                if len(new_bases11) > 0:
                    more_to_add = True
                    bases11.extend(new_bases11)
                    break
            
        model = models_per_iter[numpy.argmin(test_e_per_iter)]
        test_e = min(test_e_per_iter)
        return model, test_e

    def errors(self, model, X, y, test_I):
        trn_I = mathutil.listDiff(range(X.shape[1]), test_I)
        yhat = model.simulate(X)
        trn_e = mathutil.transformedNmse(numpy.take(yhat,trn_I),
                                         numpy.take(y,trn_I),
                                         min(y), max(y), model.y_transform)
        test_e = mathutil.transformedNmse(numpy.take(yhat, test_I),
                                          numpy.take(y,test_I),
                                          min(y), max(y), model.y_transform)
        return trn_e, test_e
    
    def _stalled(self, e_per_iter): #magic number alert
        if len(e_per_iter) < 6:  return False

        recent_best = min(e_per_iter[-3:])
        nonrecent_best = min(e_per_iter[-6:-3])
        if recent_best*1.0001 >= nonrecent_best: return True

        return False
