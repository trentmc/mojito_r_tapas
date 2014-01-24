"""Mars.py

Multivariate adaptive regression splines
(== piecewise polynomials)
"""

import logging
import math

import numpy

from LinearAlgebra import linear_least_squares        #can be unstable
#from scipy.linalg import lstsq as linear_least_squares #more stable(has LAPACK)

from util import mathutil
from Isle import IsleModel, IsleModelFactory
from IsleBase import IsleBase, LinearBase, HSBase, ProductBase
from Kmeans import KmeansStrategy, Kmeans

log = logging.getLogger('mars')

class MarsBuildStrategy:
    def __init__(self):

        #q==1 means first-order continuous (PWL)
        #q==2 means second-order continuous (ie derivatives are PWL)
        #(recommend q=1 unless you really need second-order continuous)
        self.q = 1

        #
        self.max_num_bases = 50 #[15, 5..50]
        self.thr_num_bases_where_stop_consider_HS = 2000 #20 #[5, 0..max_num_bases]

        #This is another limit on the maximum number of var interactions
        # (We usually leave it to 'big' so that the formula above is used;
        #  but it does have the option of being invoked)
        self.max_num_var_interactions = 5000

        #This controls variable interactions.
        #If set to 1, it makes Mars allow no variable interactions
        #In problems with lots of variables and a small relative amount
        # of data, we don't want to allow too many interactions, so a value
        # of 2 to 3 is recommended.
        self.max_num_prod_terms_per_base = 3 #[3, 1..8]

        #convergence will stop if best_lof < near_ideal_lof
        #this should be <= 1.0e-6 or so
        self.near_ideal_lof = 1.0e-8

        #forward-stepwise will stop if this gets achieved (no need to
        # spend computational effort on what's likely overfitting anyway)
        self.target_train_nmse = 0.001 #[0.001, 0.0001...0.05]
        
        self.preset_influence_summary = None

    def __str__(self):
        s = "MarsBuildStrategy={"
        s += ' q = %d' % self.q
        s += '; max_num_bases = %d' % self.max_num_bases
        s += '; thr_num_bases_where_stop_consider_HS = %d' % \
             self.thr_num_bases_where_stop_consider_HS
        s += '; max_num_prod_terms_per_base = %d' % \
             self.max_num_prod_terms_per_base
        s += '; near_ideal_lof = %.2e' % self.near_ideal_lof
        s += '; target_train_nmse = %.2e' % self.target_train_nmse
        s += ' /MarsBuildStrategy}'
        return s
                  
    def setFast(self):
        """
        For some unit tests, calling this will speed the run of the test
          by making some model build parameters very high-speed, low accuracy.
        """
        self.max_num_bases = 7

           
class MarsFactory:
    """
    Builds a MarsModel.
    
    Note that for speed, this class invokes a few 'coding horrors' here
    and there.
    """
    
    def __init__(self):
        self._resetAttributes()
        
    def _resetAttributes(self):
        """Reset the attributes that are cached in 'self'.
        These attributes exist in order to speed up computations.
        This method only needs to be called right before building a new model.
        """
        
        #Basic input data
        self.X11 = None      #inputs 'X' but scaled to [-1,+1] using minX, maxX
        self.minX = None     #min per var
        self.maxX = None     #max per var
        self.y = None        #target output
        self.y_range = None  #=(max(y) - min(y))
        self.ss = None       #a MarsBuildStrategy object

        #For restricting vars and interactions.
        #Details in _initializeCandSplitvars().
        self.important_vars = None
        self.important_interactions = None

        #For restricting splitvals at a given var
        #Details in _initializeCandSplitvals()
        self.clustered_splitvals11 = None
        self.cand_splitvals11 = None
        
        #Data related to prod_bases
        #Details in _initializeProdBases()
        self.prod_bases11 = None
        self.vars_at_prod_base11 = None
        self.B = None
        self.num_knots = None

        #Cached _candidate_ prod base info
        self.y_of_hs_term = None
        self.y_of_cand_prod_base_with_new_hs_term = None
        self.y_of_cand_prod_base_with_new_lin_term = None

        #For test data calcs
        self.test_B = None
        
    def build(self, X, y, minX, maxX, ss, test_cols):
        """Build a MarsModel object"""
        self._resetAttributes()
        
        #preconditions
        assert len(y) == X.shape[1]
        assert minX.shape[0] == maxX.shape[0] == X.shape[0]
        assert max(y) > min(y), 'need  non-constant y'
        assert X.shape[0] > 0, 'need >0 input vars'

        #output about problem
        n, N = X.shape
        s = 'MARS Build start; # vars=%d, #samples=%d (%d train, %d test)' % \
            (n, N, N-len(test_cols), len(test_cols))
        s += ', min(y)=%.3e, max(y)=%.3e' % (min(y), max(y)) 
        log.info(s)
        log.info('Strategy=%s',ss)
        
        #set internal variables
        #Note that they use _training_ data here
        X11 = mathutil.scaleTo01(X, minX, maxX)*2.0 - 1.0
        trn_cols = mathutil.listDiff(range(N), test_cols)
        self.X11 = numpy.take(X11, trn_cols, 1)
        self.minX, self.maxX = minX, maxX
        self.y = numpy.take(y, trn_cols)
        self.y_range = max(y) - min(y)
        self.ss = ss
        
        self.test_X11 = numpy.take(X11, test_cols, 1)
        self.test_y = numpy.take(y,test_cols)

        #Data used elsewhere
        self._initializeCandSplitvals()
        self._initializeCandSplitvars()
        
        #Forward...
        self._forwardStepwise()
        assert len(self.prod_bases11) > 0 

        #Backward...
        bases11 = self._backwardStepwise()

        #Make mean and stddev attributes non-None so that influence can
        # be calculated
        for base11 in bases11:
            base11.setMeanAndStddev(self.X11)
            
        #done
        if len(bases11[0].bases11) == 0:
            bases11 = bases11[1:] #remove the '1' base
        isle_fact = IsleModelFactory()
        mars_model = isle_fact.buildFromBases(bases11, X, X11, y,
                                              minX, maxX, None, None)
        log.info('Final mars model: %s' % mars_model)
        return mars_model
            
    def _forwardStepwise(self):
        """
        @description

          1. Stepwise-Forward-build an IsleModel from cand_bases one-at-a-time:
             -Add the next cand_base that reduces training nmse the most.
             -Stop if test nmse hasn't improved  for awhile

          Returns best_trn_bases

        @arguments

          X -- 2d array -- inputs, with each row i ranging in [minX[i], maxX[i]]
          y -- 1d array -- target outputs, one entry per column of X
          minX, maxX -- 1d array, 1d array -- (unnormalized) min and max;
          ss -- MarsBuildStrategy object
            
        @return

          best_trn_bases - best bases from training (+ a few more, good
            to  pass into backwards regression)
          best_test_bases - the bases that gave best test nmse
        
        @exceptions
    
        @notes
        """
        log.info('Begin forward stepwise addition of lin and HS bases')
        self._initializeProdBases()
        inf = float('Inf')
        best_overall_lof = inf
        while True:
            if best_overall_lof < self.ss.near_ideal_lof:
                log.info('Stop: lof < near_ideal_lof'); break
            if self._M()+2 > self.ss.max_num_bases: 
                log.info('Stop: max num bases hit'); break
            best_lof, best_m, best_v, best_t, best_e = inf, None, None, None,None
            for m in range(self._M()):
                num_prod_terms = len(self.prod_bases11[m].bases11)
                if num_prod_terms >= self.ss.max_num_prod_terms_per_base:
                    continue
                for v in self._candSplitvars(m):
                    lof, t, e = self._splitValWithBestLof(m, v)
                    if lof < best_lof:
                        best_lof, best_m, best_v, best_t, best_e = lof, m, v, t,e
            if best_lof > best_overall_lof:
                log.info('Stop: not improving lof anymore'); break
            if best_lof == inf:
                log.info('Stop: best_lof was inf'); break
            log.debug('Add base: build off m=%s, splitvar/val=%d/%s' %
                      (best_m, best_v, best_t))
            best_overall_lof = best_lof
            self._updateProdBases(best_m, best_v, best_t) 
            log.info('M=%d, best_lof=%.3e, best (trn) nmse=%.3e' %
                     (self._M(), best_lof, best_e))
            if best_e < self.ss.target_train_nmse:
                log.info('Stop: nmse < target_train_nmse'); break
            #self._plotCurrent()
        log.info('Done forward stepwise addition of lin and HS bases')

    def _backwardStepwise(self):
        """
        Each iteration of the following algorithm causes one
        basis function to be deleted -- the one whose removal
        either improvesthe fit the most or degrades it the least.

        Note that this algorithm does _not_ modify any internal
        data structures, unlike forward regression.  However,
        it does depend on the internal data generated by forward
        regression to still be there.  Yes it's sloppy -- but it
        has speed advantages.
        """
        log.info('Begin backward stepwise removal of lin and HS bases')
        M = self._M()
        best_J = range(M) #indices of best overall bases 
        best_K = range(M) #indices of best-in-iter bases (may be smaller than J)
        best_overall_nmse = self._testNmseAtSelectedBases(best_J)
        log.info('best overall nmse with M=%d (no bases removed) = %.7e' %
                 (M, best_overall_nmse))
        for M_i in range(M, 2, -1):
            best_nmse_M_i = float('Inf')
            prev_best_K = best_K
            L = best_K
            for m in prev_best_K:
                if m==0: continue #never remove the 1's base
                K = mathutil.listDiff(L, [m])
                nmse = self._testNmseAtSelectedBases(K)
                if nmse < best_nmse_M_i:
                    best_nmse_M_i = nmse; best_K = K
                eps = 0.00001
                if nmse <= best_overall_nmse*(1.0 + eps):
                    best_overall_nmse = nmse; best_J = K
            s = 'M_i=%d, best_test_nmse_M_i=%.7e (%d non-"1" bases)' % \
                (M_i, best_nmse_M_i, len(best_K)-1)
            s += ', best_overall_test_nmse=%.7e (%d non-"1" bases)' % \
                 (best_overall_nmse, len(best_J)-1)
            log.info(s)
            log.debug('Best-in-iter bases remaining: \n%s' %
                      strBases(numpy.take(self.prod_bases11, best_K), True))

        log.info('Done backward stepwise removal of bases; final test nmse=%.3e'%
                 best_overall_nmse)
        best_bases11 = list(numpy.take(self.prod_bases11, best_J))
        log.debug('Final chosen bases: \n%s' % strBases(best_bases11, True))
        return best_bases11

    def _testNmseAtSelectedBases(self, K):
        """Each entry of 'K' is an index into self.prod_bases11.
        Use_test_data = calc nmse using test_X/test_y or just usual (trn) X/y ?
        """
        assert max(K) <= self._M()

        #calc coefs
        tran_sub_B = numpy.transpose( numpy.take(self.B, K, 0) )
        tran_y = numpy.transpose(self.y)
        try:
            (coefs, resids, rank, sing_vals) = \
                    linear_least_squares( tran_sub_B, tran_y )
            
            #sometimes coefs comes back as Float64.  We want Float32 ('float')
            coefs = numpy.array(coefs, dtype=float)
        except:
            return float('inf')

        #fill test_B if needed
        if self.test_B is None:
            self.test_B = numpy.zeros((self._M(), len(self.test_y)),
                                        dtype=float)
            for m, base11 in enumerate(self.prod_bases11):
                self.test_B[m,:] = base11.simulate(self.test_X11)
        test_tran_sub_B = numpy.transpose( numpy.take(self.test_B, K, 0) )

        #sim and sse calc
        test_tran_yhat = numpy.dot(coefs, test_tran_sub_B)
        test_yhat = numpy.transpose(test_tran_yhat)
        sse = self._sse(test_yhat, self.test_y)
        nmse = math.sqrt(sse)
        return nmse

        #if we were to calc lof...
        #num_knots = sum(self.prod_bases11[k_m].numKnots() for k_m in K)
        #lof = self._lof(sse, len(K), num_knots)
        #return lof        

    def _initializeCandSplitvals(self):
        """Initializes:
        -self.clustered_splitvals11[v] - all available splitvals for var 'v'
        -self.cand_splitvals11[m][v] - subset of clustered_splitvals11[v],
                                       for base m
        """
        #
        self.clustered_splitvals11 = _calcClusteredSplitvals(self.X11)

        #
        self.cand_splitvals11 = [] 
        n = self.X11.shape[0]
        for m in range(self.ss.max_num_bases):
            splitvals_at_base = []
            for ni in range(n):
                #'None' means that not computed yet
                splitvals_at_base.append(None) 
            self.cand_splitvals11.append(splitvals_at_base)

    def _candSplitvals(self, m, v):
        """ Returns a set from X11[v,i], i=1,2,..N, for all i that 
        self.prod_bases11[m].simulate(X11[:,i]) > 0
        """
        #on-the-fly caching
        if self.cand_splitvals11[m][v] is None:
            #case 1: building off a '1' base, or a linear base; in
            # both cases all points are still firing.  So use pre-clustered vals.
            prod_base = self.prod_bases11[m]
            if isinstance(prod_base, LinearBase) or prod_base.isTrivial():
                 self.cand_splitvals11[m][v] = self.clustered_splitvals11[v]

            #case 2: we've already zeroed at some i's, so recluster rest
            else:
                y_m = self.B[m,:]
                n, N = self.X11.shape
                firing_t = [t for t,yi in zip(self.X11[v,:], y_m) if yi!=0.0]
                if len(firing_t) == 0:
                    #no options!
                    cand_t = numpy.array([])
                elif len(firing_t) > 0.90*N:
                    #not that many sliced away yet, so just use pre-clustered
                    cand_t = self.clustered_splitvals11[v]
                else:
                    #enough sliced away to justify clustering more
                    cand_t = _clusterVals(firing_t, n)
                    
                log.debug('m=%d, v=%d, num_firing_t=%d, num_cand_t=%d' %
                          (m, v, len(firing_t), len(cand_t)))
                self.cand_splitvals11[m][v] = cand_t
            
        return self.cand_splitvals11[m][v]

    def _initializeCandSplitvars(self):
        """
        self.important_vars is a list of important vars
        self.important_interactions[var] holds a set of all the vars
          that 'var' interacts significantly with
        """
        important_vars, important_interaction_tups = \
                        _calcImportantVarsAndInteractions(self.X11, self.y,
                                                          self.ss)
        self.important_vars = important_vars
        
        n = self.X11.shape[0]
        self.important_interactions = [set([]) for v in range(n)]
        for (var_i, var_j) in important_interaction_tups:
            self.important_interactions[var_i] |= set([var_j])
            self.important_interactions[var_j] |= set([var_i])
        
    def _candSplitvars(self, m):
        """What variables are eligible to interact with base m?"""
        if m==0: #this is the '1' b ase
            return self.important_vars
        else:            
            cand_vars = set()
            for var in self.vars_at_prod_base11[m]:
                cand_vars |= self.important_interactions[var]
            return list(cand_vars)

    def _M(self):
        """Returns number of currently chosen bases"""
        return len(self.prod_bases11)
                
    def _initializeProdBases(self):
        """Initialize data related to 'bases learned so far':
        -self.prod_bases11 - data structure, with one base ('1'),
        -self.B - whole matrix 'B' [maxM][N]; set 0th row to 1's
        -self.vars_per_prod_base11 -- list of vars used at each prod base
        -self.num_knots -- sum of (# knots per base)
        """
        #prod_bases11[m] = ProductBase object
        self.prod_bases11 = [ProductBase([])] 

        #vars_at_prod_base11[m] = vars that prod_base11[m] uses
        self.vars_at_prod_base11 = [[]]

        #self.B[m][i] is results of simulating X11[:,i] on self.prod_bases11[m]
        #-start big so that we don't need to keep reallocating memory
        #-while the 0th row will always be 1's (reflecting self.prod_bases11[0]),
        # other rows will change
        N = self.X11.shape[1]
        self.B = numpy.ones((self.ss.max_num_bases, N), dtype=float)

        #cached count of knots for chosen bases so far
        self.num_knots = 0

        #cached sim information
        #these use a mix of lists and dicts in order to minimize
        # the key-searching effort during retrieval

        #-y_of_hs_term[splitvar][splitval] = sim_outputs
        n = self.X11.shape[0]
        self.y_of_hs_term = [{} for var in range(n)] 

        #-y_of_base_with_new_hs_term[parent_m][splitvar][splitval] = y
        self.y_of_base_with_new_hs_term = [] 
        for m in range(self.ss.max_num_bases):
            list_at_m = [{} for var in range(n)]
            self.y_of_base_with_new_hs_term.append(list_at_m)
            
        #-y_of_base_with_new_lin_term[parent_m][var] = y
        self.y_of_base_with_new_lin_term = []
        for m in range(self.ss.max_num_bases):
            list_at_m = [None for var in range(n)]
            self.y_of_base_with_new_lin_term.append(list_at_m)

        
    def _updateProdBases(self, m, v, t):
        """If t is not None:
        -Add two more bases (HSBase) to self.prod_bases11, etc.
        -Else add one linear base to self.prod_bases11
        Update other prod_base related attributes."""
        if t is not None:
            #prod_bases11
            #prod_bases11[m] holds the choice of ProductBase for basis #m
            pos_base = ProductBase( self.prod_bases11[m].bases11 + 
                                    [HSBase(-1.0, v, t, self.ss.q)])
            neg_base = ProductBase( self.prod_bases11[m].bases11 + 
                                    [HSBase(+1.0, v, t, self.ss.q)])
            self.prod_bases11.extend([pos_base, neg_base])

            #vars_at_prod_base11
            #vars_at_prod_base11[m] holds the vars that basis #m uses
            # (Q: Why cached rather than querying prod_bases11[m]?
            #  A: for speed )
            parent_vars = self.vars_at_prod_base11[m]
            self.vars_at_prod_base11.extend([ parent_vars + [v],
                                              parent_vars + [v] ])

            #B
            #B[m,:] is the simulation of the input test data for basis #m
            M = self._M()
            self.B[M-2,:] = pos_base.simulate(self.X11)
            self.B[M-1,:] = neg_base.simulate(self.X11)

            #num_knots
            #num_knots, an int, is the sum of knots from basis #0, #1, ..., #M-1
            self.num_knots += pos_base.numKnots()
            self.num_knots += neg_base.numKnots()
        else:
            #prod_bases11
            avg = numpy.average(self.X11[v,:])
            stddev = mathutil.stddev(self.X11[v,:])
            prod_base = ProductBase( self.prod_bases11[m].bases11 + 
                                     [LinearBase( v, avg, stddev )] )
            self.prod_bases11.append(prod_base)

            #vars_at_prod_base11
            parent_vars = self.vars_at_prod_base11[m]
            self.vars_at_prod_base11.append( parent_vars + [v] )

            #B
            M = self._M()
            self.B[M-1,:] = prod_base.simulate(self.X11)

            #num_knots
            #If lin entries is the only entry in the product base, then
            # 0 new knots are added.  But there may be HSBase entries.
            self.num_knots += prod_base.numKnots() 

        #consistency checks
        assert len(self.prod_bases11) == self._M()
        for m, base11 in enumerate(self.prod_bases11):
            for var in range(self.X11.shape[0]):
                has_var = base11.hasVar(var)
                if var in self.vars_at_prod_base11[m]: assert has_var
                else: assert not has_var
        assert self.num_knots == \
                   sum(base11.numKnots() for base11 in self.prod_bases11)

        #return
        log.debug('Updated prod bases: \n%s' % strBases(self.prod_bases11, True))

    def _plotCurrent(self):
        """Plot the current prod_bases11's x vs y (single variable input only)"""
        if self.X11.shape[0] > 1: return
        y = self.y
        M = self._M()
        tran_B = numpy.transpose(self.B[:M,:])    
        yhat = self._linLearnAndCalcYhat(tran_B, numpy.transpose(y))
        #from scipy import gplt;
        #gplt.figure() #this will create a new figure for each plot
        #gplt.plot(self.X11[0,:], y, self.X11[0,:], yhat)
        
        
    def _splitValWithBestLof(self, m, v):
        """Find the splitval 't' that gives the best lof at bases[m].
        Returns best_lof, best_t, best_nmse.
        
        If best_t comes back as a 'None' but best_lof is not inf,
        then it says that a _linear_ basis function is best"""
        #general data setup
        M = self._M()
        tran_y = numpy.transpose(self.y)

        #specific data setup 
        cand_splitvals11 = self._candSplitvals(m, v)

        tran_aug_B = numpy.transpose(self.B[:M+2,:])

        inf = float('Inf')
        best_lof,  best_sse, best_t = inf, inf, None
        if M < self.ss.thr_num_bases_where_stop_consider_HS:
            num_fail = 0
            cand_t = self._candSplitvals(m, v)
            for t in cand_t:

                tran_aug_B[:,-2] = self._simProdBaseWithNewHSTerm(m, -1.0, v, t)
                tran_aug_B[:,-1] = self._simProdBaseWithNewHSTerm(m, +1.0, v, t)

                try:
                    yhat = self._linLearnAndCalcYhat(tran_aug_B, tran_y)
                    sse = self._sse(yhat, tran_y)
                    lof = self._lof(sse, M+2, self.num_knots+2)
                    if lof < best_lof:
                        best_lof, best_sse, best_t = lof, sse, t
                except KeyError:
                    #nothing to do because we merely haven't improved sse
                    num_fail += 1

            if num_fail == len(cand_t):
                log.warning('All the cand splits failed at m=%d, v=%d' % (m,v))

        #try linear too
        tran_aug_B[:,-2] = self._simProdBaseWithNewLinTerm(m, v)
        try:
            yhat = self._linLearnAndCalcYhat(tran_aug_B[:,:-1], tran_y)
            sse = self._sse(yhat, tran_y)
            lof = self._lof(sse, M+1, self.num_knots+0)
        except KeyError:
            sse = inf
            lof = inf
            
        
        log.debug('Parent base (m=%d): %s; splitvar=%d, best lof from non-lin=%.3e (sse=%.3e,M=%d,#knots=%d), from lin=%.3e (sse=%.3e,M=%d,#knots=%d)' % (m, self.prod_bases11[m], v, best_lof, best_sse, M+2, self.num_knots+2, lof, sse, M+1, self.num_knots+0))
        if lof < best_lof:
            #note that t = 'None' for linear
            best_lof, best_sse, best_t = lof, sse, None 

        if best_sse == inf: best_nmse = inf
        else:               best_nmse = math.sqrt(best_sse)

        return best_lof, best_t, best_nmse

    def _linLearnAndCalcYhat(self, tran_B, tran_y):
        (coefs, resids, rank, sing_vals) = \
                linear_least_squares( tran_B, tran_y )    
        #sometimes coefs comes back as Float64.  We want Float32 ('float')
        coefs = numpy.array(coefs, dtype=float)
        
        tran_yhat = numpy.dot(coefs, tran_B)
        yhat = numpy.transpose(tran_yhat)
        return yhat

    def _sse(self, yhat, y):
        """Returns pseudo_sse = sse + w * max_scaled_e"""
        scaled_diff = ((yhat - y) / self.y_range)
        sse = numpy.average(scaled_diff ** 2)
        #max_scaled_e = max(abs(scaled_diff))
        #sse += 0.01 * max_scaled_e
        return sse

    def _lof(self, sse, M, num_knots):
        if sse == float('Inf'):
            return sse
        #lof = self._GCV(sse, M, num_knots) #HACK
        lof = sse
        return lof

    def _GCV(self, sse, M, num_knots):
        """Friedman's recommended loss function.
        Eqn (9.20), pp. 287, The Elements of Statistical Learning
        """
        complexity = M + 3.0*num_knots #see p. 287 book, p.18-20 mars paper
    
        N = float(self.X11.shape[1])
        if (complexity / N) >= 1.0:
            #log.warning('gcv is returning inf because complexity overtaking N')
            return float('Inf')
        gcv = sse / (1.0 - complexity / N)**2
        return gcv

    def _simProdBaseWithNewHSTerm(self, parent_m, sign, var, splitval):
        
        #cache if not there
        if not self.y_of_base_with_new_hs_term[parent_m][var].has_key(splitval):
            y = self.B[parent_m,:] * self._simHsBase(sign, var, splitval)
            self.y_of_base_with_new_hs_term[parent_m][var][splitval] = y

        #return cached value
        return self.y_of_base_with_new_hs_term[parent_m][var][splitval]

    def _simProdBaseWithNewLinTerm(self, parent_m, var):

        #cache if not there
        if self.y_of_base_with_new_lin_term[parent_m][var] is None:
            y = self.B[parent_m,:] * self.X11[var]
            self.y_of_base_with_new_lin_term[parent_m][var] = y

        #return cached value
        return self.y_of_base_with_new_lin_term[parent_m][var]
    
    def _simHsBase(self, sign, splitvar, splitval):

        #cache if not there
        if not self.y_of_hs_term[splitvar].has_key(splitval):
            inf = float('Inf')
            y = numpy.clip(self.X11[splitvar,:] - splitval, 0.0, inf)
            if self.ss.q != 1:
                y = y**self.ss.q
            self.y_of_hs_term[splitvar][splitval] = y

        #return cached value (times 'sign')
        return sign * self.y_of_hs_term[splitvar][splitval]


def _concatArrays(X1, X2):
    """Add the rows of X2 to X1, and return the result"""   
    new_X = numpy.zeros((X1.shape[0]+X2.shape[0], X1.shape[1]), dtype=float)
    new_X[:X1.shape[0],:] = X1
    new_X[X1.shape[0]:,:] = X2
    return new_X
                    

def _calcClusteredSplitvals(X11):
    """Choose unique, clustered-down splitvals"""
    
    log.info('Calculate splitvals per var...')
    clustered_splitvals11 = {}
    num_clustered_splitvals11 = {}

    n = X11.shape[0]
    for var in range(n):
        vals11 = _clusterVals(X11[var, :], n)
        clustered_splitvals11[var] = vals11
        num_clustered_splitvals11[var] = len(vals11)

    log.debug('Num splitvals per var: %s' % num_clustered_splitvals11)

    return clustered_splitvals11

def _clusterVals(x, numvars):
    """Returns a clustered-down version of input 1d-array or list 'x'."""
    assert len(x) > 0
    unique_x = list(set(x))

    #magic number alert (next several lines)
    N = len(unique_x)

    #why N/7: simple version of Friedman's Mars formulas (43)(45)
    #why /(numvars/40): when lotsa vars, bias away from splits and to lin
    #eg when numbers=378, N=756, then sugg_nc = about 11
    sugg_nc = int( N/7.0 / max(1.0, (numvars/40.0)) )
    upper_bound = min(30, N)

    if numvars == 1:   lower_bound = 20
    elif numvars == 2: lower_bound = 15
    elif numvars == 3: lower_bound = 12
    else:              lower_bound = 10
    lower_bound = min(lower_bound, N)
    
    nc = min(upper_bound, max(lower_bound, sugg_nc))
    ss = KmeansStrategy(init_num_centers = nc, learn_rate = 0.1,
                        num_reps = 10)
    clustered_x = Kmeans().cluster1d(unique_x, ss, True)
    return clustered_x
        
    
def _calcImportantVarsAndInteractions(X, y, mars_ss):
    n = X.shape[0]
    active_vars = mathutil.removeConstantRows(X, range(n))
    n_active = len(active_vars)
        
    important_vars = active_vars
    important_interactions = [set([var_i, var_j])
                              for var_i in range(n)
                              for var_j in range(var_i)
                              if var_i in active_vars
                              and var_j in active_vars]
    return important_vars, important_interactions

        
def strBases(bases, newline):
    s = ''
    if newline: s += '\n'
    for i,base in enumerate(bases):
        s += str(base)
        if newline: s += '\n'
        elif i < (len(bases)-1):
            s += ",   "
    return s


