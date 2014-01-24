"""Probe.py.  Builds order-2 polynomials using the PROBE algorithm.

Reference: Li, Le, & Pileggi, Statistical Performance Modeling and Optimization, FTEDA, section 3.2.2.2
 (Algorithms 3.1 and 3.2)
"""
import logging
import math
import random
import types

import numpy

from util import mathutil
from engine.EngineUtils import coststr
from regressor.ConstantModel import yIsConstant, ConstantModel
from regressor.LinearModel import LinearModelFactory, LinearBuildStrategy

log = logging.getLogger('probe')

tr = numpy.transpose
dot = numpy.dot

class ProbeBuildStrategy:
            
    def __init__(self, target_train_nmse=None, max_rank=None, lin_ss=None):
        #preconditions
        assert (target_train_nmse is None) or isinstance(target_train_nmse, types.FloatType)
        assert (max_rank is None) or isinstance(max_rank, types.IntType)
        assert (lin_ss is None) or isinstance(lin_ss, LinearBuildStrategy)

        #set values...
        if target_train_nmse is None:
            self.target_train_nmse = 0.01
        else:
            self.target_train_nmse = target_train_nmse

        if max_rank is None:
            self.max_rank = 2
        else:
            self.max_rank = max_rank
        
        self.min_improvement_nmse = 1.0e-4
        
        self.max_k = 30 #max num iterations in building a rank-1 model
        
        self.lin_ss = lin_ss

    def __str__(self):
        s = "ProbeBuildStrategy={"
        s += ' target_train_nmse=%.2e' % self.target_train_nmse
        s += '; max_rank=%d' % self.max_rank
        s += '; min_improvement_nmse=%.2e' % self.min_improvement_nmse
        s += '; max_k=%d' % self.max_k
        s += '; lin_ss=%s' % self.lin_ss
        s += ' /ProbeBuildStrategy}'
        return s

class ProbeModel:
    
    def __init__(self, rank1_models):
        """
        @description
        
        @arguments

          rank1_models -- lst of Rank1ProbeModel
        
        @return
    
        @exceptions
    
        @notes
    
        """ 
        self.rank1_models = rank1_models

    def simulate(self, X):
        N = X.shape[1]
        y = numpy.zeros(N, dtype=float)
        for rank1_model in self.rank1_models:
            y += rank1_model.simulate(X)

        return y

    def simulate1(self, x):
        """Simulate a single vector"""
        X = numpy.reshape(x, (len(x), 1))
        return self.simulate(X)[0]

    def rank(self):
        return len(self.rank1_models)

    def __str__(self):
        s = []
        s += ["ProbeModel={\n"]
        s += [" rank=%d" % self.rank()]
        s += ["/ProbeModel}\n"]
        return "".join(s)

class Rank1ProbeModel:
    """This is a quadratic model, parameterized by A, B, and C
    """
    
    def __init__(self, Q_k, Q_km1, B, C):
        """
        @description
        
        @arguments

            Q_k -- 2d array [numvars] -- quadratic coefficients
            Q_km1 -- 2d array [numvars] -- quadratic coefficients
            B -- 1d array [numvars] -- linear coefficients
            C -- float -- offset
        
        @return
    
        @exceptions
    
        @notes
    
        """ 
        self.Q_k = Q_k
        self.Q_km1 = Q_km1
        self.B = B
        self.C = C

    def simulate(self, X):
        """
        @description
        
            Simulate this model with inputs X.
        
        @arguments
        
            X -- 2d array [numvars][num samples] -- inputs

        @return
    
            y -- 1d array [num samples] -- outputs

        @exceptions
    
        @notes
         
        """ 
        Q_k, Q_km1, B, C = self.Q_k, self.Q_km1, self.B, self.C
        N = X.shape[1]
        
        y = numpy.zeros(N, dtype=float)
        for i in range(N):
            x = X[:, i]

            v = dot( x , Q_k ) #don't need to tr(x) because it's a 1d array
            v = dot( v , Q_km1 )
            v = dot( v , x )
            
            y[i] = v + dot( tr(B), x ) + C

        return y

    def __str__(self):
        s = []
        s += ["Rank1ProbeModel={\n"]
        s += ["\nQ_k:\n%s\n" % self.Q_k]
        s += ["\nQ_km1:\n%s\n" % self.Q_km1]
        s += ["\nB:\n%s\n" % self.B]
        s += ["\nC:\n%s\n" % self.C]
        s += ["/Rank1ProbeModel}\n"]
        return "".join(s)
    
class ProbeFactory:
        
    def build(self, X, y, ss):
        """
         X -- 2d array -- training inputs; [1..num vars][1..num samples]
         y -- 1d array -- training outputs [1..num samples]
         ss -- ProbeBuildStrategy -- build strategy
        """
        #preconditions
        assert X.shape[1] == y.shape[0]
                          
        #initialize
        s = 'ProbeFactory.build: begin.  # vars=%d, #samples=%d' % X.shape
        log.info(s)
        log.debug('Strategy=%s' % ss)

        #corner case
        const_model = ConstantModel(numpy.average(y), X.shape[0])
        if yIsConstant(y):
            log.info('max(y) == min(y); will just provide a constant')
            return const_model

        #main work...
        rank = 0
        residual_y = y
        model = ProbeModel([])
        e = float('Inf')
        while True:
            rank += 1
            if rank > ss.max_rank:
                log.debug("Stop because hit max rank of %d" % ss.max_rank)
                break

            log.debug("")
            log.debug("Build another rank-1 model (overall rank will be %d)" % rank)
            model.rank1_models.append(self.buildRank1ProbeModel(X, residual_y, ss))
            yhat = model.simulate(X)
            next_e = mathutil.nmse(yhat, y, min(y), max(y))
            log.info("For model of overall rank %d, overall nmse = %.3e" % (rank, next_e))

            if next_e < ss.target_train_nmse:
                log.debug("Stop because nmse < thr")
                e = next_e
                break

            if next_e > e:
                log.debug("Stop because nmse worsened.  Go back to overall nmse = %.3e" % e)
                model.rank1_models[-1:] = []
                break

            e = next_e
            residual_y = residual_y - yhat

        log.info('ProbeFactory.build: done.  Final nmse = %.3e' % e)
                
        return model
            
    def buildRank1ProbeModel(self, X, y, ss):

        n = X.shape[0] #num vars
        N = X.shape[1] #num samples

        #step 1
        #(start from a set of samples X=>y)

        #step 2
        k = 1
        Q_km1 = numpy.array([random.random() for i in xrange(n)])
        nmse_km1 = float('Inf')
        best_nmse, best_Q_k, best_Q_km1, best_lin_model = None, None, None, None

        while True:
            norm = self.norm1d(Q_km1)
            if norm == 0.0:
                break
            
            #step 3
            Q_km1 = Q_km1 / norm

            #step 4
            bases_per_sample = []
            for sample_i in range(N):
                x = X[:, sample_i]

                r = numpy.dot(Q_km1, x)

                #build up all the basis entries at this sample
                bases = []
                
                # -quadratic coefficients
                bases += list(r * x)

                # -linear coefficients
                bases += list(x)

                # -don't need to add offset (linear model factory handles that)

                #
                bases_per_sample.append(bases)

            XB = tr(numpy.array(bases_per_sample))
            lin_model = LinearModelFactory().build(XB, y, None, None, ss.lin_ss)
            Q_k = numpy.array(lin_model.coefs[1:1+n])

            #step 5
            yhat = lin_model.simulate(XB)
            nmse_k = mathutil.nmse(y, yhat, min(y), max(y))

            if (best_nmse is None) or (nmse_k < best_nmse):
                best_nmse, best_Q_k, best_Q_km1, best_lin_model = nmse_k, Q_k, Q_km1, lin_model

            delta_nmse = abs(nmse_k - nmse_km1)
            log.debug("  Iteration #%d: nmse = %.3e, delta_nmse = %.3e; best nmse = %.3e" %
                      (k, nmse_k, delta_nmse, best_nmse))
            
            if delta_nmse < ss.min_improvement_nmse:
                log.debug('  Stop: rate of improvement < thr')
                break

            if k > ss.max_k:
                log.debug('  Stop: hit max num iterations')
                break
            
            Q_km1 = Q_k
            nmse_km1 = nmse_k
            k += 1

        #step 6
        Q_k, Q_km1, lin_model = best_Q_k, best_Q_km1, best_lin_model
        model = self._buildRank1ProbeModel(Q_k, Q_km1, lin_model)

        #done
        return model

    def _buildRank1ProbeModel(self, Q_k, Q_km1, lin_model):
        n = Q_k.shape[0]
        
        C = lin_model.coefs[0]
        B = numpy.asarray(lin_model.coefs[-n:])
        model = Rank1ProbeModel(Q_k, Q_km1, B, C)
        return model

    def norm1d(self, X):
        """Returns the 2-norm of 1d array"""
        m = X.shape[0]

        sum = 0.0
        for m_i in range(m):
            sum += X[m_i]**2

        return math.sqrt(sum)
    
    def norm2d(self, X):
        """Returns the 2-norm of 2d matrix"""
        m,n = X.shape

        sum = 0.0
        for m_i in range(m):
            for n_i in range(n):
                sum += X[m_i, n_i]**2

        return math.sqrt(sum)

