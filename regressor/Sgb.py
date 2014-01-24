""" Sgb.py: Builds and simulates SGB models 

Ref: J. Friedman, Stochastic gradient boosted trees, 1999

SgbModel holds a bunch of boosted CartModels.
    
"""

import logging
import random
import copy
import math

import numpy

from util import mathutil
from Cart import CartFactory, CartBuildStrategy
from ConstantModel import yIsConstant, ConstantModel
import RegressorUtils


log = logging.getLogger('sgb')        

class SgbBuildStrategy:
            
    def __init__(self, max_carts=100, learning_rate=0.01, target_trn_nmse=0.05):
        """
        max_carts     - max # iterations (though it may stop
                        sooner if cond'ns met)
        lrn_rate      - learning rate, non-aggressive settings are:
                       [.005 for N=500 samples, .05 for N=5000]
        """
        self.max_carts = max_carts         
        self.learning_rate = learning_rate  
        self.target_trn_nmse = target_trn_nmse
        self.target_test_nmse = 0.0

        self.delta_trn_e_thr=0.001 #[.001,] stop if lpf'd change in nmse is <this
        self.min_N_trn_per_cart=1  #[5,3..7] min num allowed trn samples per cart
        
        #Will govern cart_ss.max_depth, PER CART (influences #var interactions)
        self.cand_depths  = [3,     4,    5,    6,    7  ]
        self.depth_biases = [20.0, 20.0, 10.0, 5.0,  3.0]
        
        self.cart_ss = CartBuildStrategy()
        self.cart_ss.max_depth = -1  #force setting it later!

    def __str__(self):
        s = "SgbBuildStrategy={"
        s += '  max_carts=  %g' % self.max_carts
        s += '; learning rate= %g' % self.learning_rate
        s += '; target_trn_nmse= %.2e' % self.target_trn_nmse
        s += '; delta_trn_e_thr= %.2e' % self.delta_trn_e_thr
        s += '; min_N_trn_per_cart= %d' % self.min_N_trn_per_cart
        s += '; cand_depths/biases=%s' % \
             str(zip(self.cand_depths, self.depth_biases))
        s += '; cart_ss=%s' % str(self.cart_ss)
        s += ' /SgbBuildStrategy}'
        return s
                  
    def setFast(self):
        self.max_carts = 10
        self.learning_rate = 0.40
        self.target_nmse = 0.20
        
class SgbModel:

    def __init__(self, carts, weights, miny, maxy, numvars):
        assert len(carts) == len(weights)-1
        self.carts = carts
        self.weights = weights   #offset, plus one weight per submodel
        self.miny, self.maxy = miny, maxy
        self.numvars = numvars

    def simulate(self, X):
        """
        Inputs:  X - model inputs [var #][sample #]
        Outputs: yhat - model output [sample #]
        """
        
        # Simulate one cart at a time.
        [n, N] = X.shape        
        offset = float(self.weights[0])
        yhat = numpy.zeros(N, dtype=float) + offset

        for cart_i, cart in enumerate(self.carts):
            w = self.weights[cart_i+1]
            if w > 0.0:
                yhat += w * cart.simulate(X)
        
        #rail
        miny, maxy = self.miny, self.maxy
        for i in xrange(N):
            yhat[i] = max(miny, min(maxy, yhat[i]))
            
        #done
        return yhat

    def isConstant(self):
        return len(self.carts)==0

    def __str__(self):
        s = "SgbModel={"
        s += "num carts=%s" % str(len(self.carts))
        s += "}"
        return s
    
class SgbFactory:
        
    def build(self, X, y, ss, test_cols=None):
        """
        High-level interface of SGB model builder.  

         X          - training inputs; [1..n vars][1..N samples]
         y          - training outputs [1..N samples]
         ss         - build strategy
         test_cols  - columns of X,y to use for test data (use rest for trn) 
        """
        [n, N] = X.shape
        assert len(y) == N
        if N == 0:
            raise InsufficientDataError('need >0 samples')
        if n == 0:
            log.info('no input vars, so return a ConstantModel')
            return ConstantModel(y[0], 0)
            
        if test_cols is None: test_cols = []

        s = 'Build start; # vars=%d, #samples=%d ' % (n, N)
        s += '(#train=%d, #test=%d), target_test_nmse=%s' % \
             (N-len(test_cols), len(test_cols), str(ss.target_test_nmse))
        s += ', miny=%.3e, maxy=%.3e' % (min(y), max(y)) 
        log.info(s)
        log.debug('Strategy=%s',ss)

        const_model = ConstantModel(numpy.average(y), X.shape[0])
        if yIsConstant(y):
            log.info('max(y) == min(y); will just provide a constant')
            return const_model

        carts, w = self.findCarts(X, y, ss, test_cols)
        model = SgbModel(carts, w, min(y), max(y), X.shape[0])
        return model

    def findCarts(self, full_X, full_y, ssIn, test_cols):
        """Find carts.  Note that y is pre-transformed"""
        ss = self._biasConstantVarsToZero(ssIn, full_X)
        trn_cols = mathutil.listDiff(range(full_X.shape[1]), test_cols)
        trn_X = numpy.take(full_X, trn_cols, 1)
        trn_y = numpy.take(full_y, trn_cols)
        test_X = numpy.take(full_X, test_cols, 1)
        test_y = numpy.take(full_y, test_cols)

        n, trn_N = trn_X.shape
        n, test_N = test_X.shape
        range_trn_N = range(trn_N)
        range_test_N = range(test_N)

        check_period = 10

        # root node has initial (constant) estimate = mean of targets
        offset = numpy.average(trn_y)
        carts = {} # cart_i : CartModel
        trn_y = trn_y - offset
        test_y = test_y - offset
        
        miny, maxy = min(full_y) - offset, max(full_y) - offset

        #N_cart = num training samples per cart
        min_N_trn_per_cart = min(ss.min_N_trn_per_cart, trn_N)
        N_cart = int(trn_N**0.75) #heuristic: like 0.2*trn_N for trn_N < 1000,
                                  # but is 1/2 that when trn_N = 5000
        N_cart = max(min_N_trn_per_cart, N_cart)

        #build tree portion
        trn_yhat = numpy.zeros(trn_N, dtype=float)
        test_yhat = numpy.zeros(test_N, dtype=float)

        trn_e, test_e, trn_norm = float('Inf'), float('Inf'), 0.0
        trn_es, test_es = {},{} #num_carts : error (not on all vals of num_carts)
        learning_rate = ss.learning_rate
        cart_i = 0
        while cart_i < ss.max_carts:
            mask = numpy.zeros(trn_N)
            for i in random.sample(range_trn_N, N_cart):
                mask[i] = 1
            sub_X = numpy.compress(mask, trn_X, 1)
            sub_y = numpy.array([trn_y[i] - trn_yhat[i]
                                 for i in range_trn_N if mask[i]])

            choice_i = mathutil.randIndex(ss.depth_biases)
            ss.cart_ss.max_depth = ss.cand_depths[choice_i]
            cart = CartFactory().build(sub_X, sub_y, ss.cart_ss)

            carts[cart_i] = cart

            trn_yhat += learning_rate * cart.simulate(trn_X)
                
            if test_cols:
                test_yhat += learning_rate * cart.simulate(test_X)

            if cart_i < 4 or (cart_i % check_period)==0:
                trn_e = mathutil.nmse(trn_y, trn_yhat, miny, maxy)
                trn_es[cart_i] = trn_e
                delta_trn_e = self._recentDeltaNmse(trn_es)
                trn_norm = mathutil.normality(trn_y - trn_yhat)
                s = 'Iter=%d, trn nmse=%.4f, change=%.3e, trn normality=%.3f' % \
                    (cart_i, trn_e, delta_trn_e, trn_norm)
                if test_cols:
                    test_e = mathutil.nmse(test_y, test_yhat, miny, maxy)
                    test_es[cart_i] = test_e
                    delta_test_e = self._recentDeltaNmse(test_es)
                    s += '; test nmse=%.4f, change=%.3e' % (test_e, delta_test_e)
                log.debug(s)

                if test_cols and test_e < ss.target_test_nmse:
                    log.info('Stop: test_nmse < target_test_nmse');  break
                
                elif trn_e < ss.target_trn_nmse:
                    log.info('Stop: trn_nmse < target_trn_nmse'); break

                elif test_cols and delta_test_e+0.0001<0 and cart_i>10:
                    log.info('Stop: starting to overfit (delta_test_e<0)'); break
                
                elif delta_trn_e < 0:
                    learning_rate /= 2.0
                    log.debug('learning worsened,so shrink lrnrate to %.1e'%
                              learning_rate)

                elif (delta_trn_e < ss.delta_trn_e_thr) and (cart_i > 40):
                    log.info('Stop: delta nmse < thr=%.2e' % ss.delta_trn_e_thr)
                    break
                
            cart_i += 1

        #convert 'carts' from dict to list
        num_carts = len(carts)
        carts = [carts[i] for i in range(num_carts)]
                
        #report
        s = 'Done findCarts(); #carts=%g; trn nmse=%.2e' % (len(carts), trn_e )
        s += '; trn norm=%.2f' % trn_norm
        if test_cols: s += '; test nmse=%.2e' % test_e
        log.info(s)

        #maybe prune down
        if test_cols:
            best_numcarts = _keyWithLowestValue(test_es) + 1
            carts[best_numcarts:] = []
            log.info('Pruned down #carts to %d, to recover test nmse=%.2e' %
                     (len(carts), min(test_es)))

        #create 'weights'
        weights = [offset] + [learning_rate]*len(carts)
            
        return carts, weights

    def _biasConstantVarsToZero(self, ssIn, full_X):
        ss = copy.deepcopy(ssIn)
        n = full_X.shape[0]
        I = range(n)
        I = mathutil.removeConstantRows(full_X, I)
        const_I = mathutil.listDiff(range(n), I)
        if ss.cart_ss.var_biases is None:
            ss.cart_ss.var_biases = [1.0 for i in range(n)]
            for i in const_I:  ss.cart_ss.var_biases[i] = 0.0
        else:
            assert len(ss.cart_ss.var_biases) == n, "need one var bias per var"
            for i in const_I:
                ss.cart_ss.var_biases[i] = 0.0
                    
        return ss
    
    def _recentDeltaNmse(self, nmses_dict):
        ks = sorted(nmses_dict.keys())
        nmses = [nmses_dict[k] for k in ks]
        
        if len(nmses)==0 or len(nmses)==1:
            return 1.0
        else:
            trunc_nmses = nmses[-8:]
            n = len(trunc_nmses)
            mask = numpy.zeros(n)
            for i in range((n+1)/2):
                mask[i] = 1
            return numpy.average(numpy.compress(mask,   trunc_nmses)) -   \
                   numpy.average(numpy.compress(1-mask, trunc_nmses))
        

def _keyWithLowestValue(my_dict):
    assert len(my_dict) > 0
    best_key, best_value = None, float('Inf')
    for key, value in my_dict.items():
        if value < best_value:
            best_key, best_value = key, value
    return best_key

