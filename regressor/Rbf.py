""" Rbf.py
 Build a radial basis function (RBF) model;
  equivalent to a support vector machine.
 Simulate an rbf model.
""" 
import logging
import sayo.util.sayolog
import copy
import math

import numpy 

from sayo.Exceptions import *
from sayo.util import mathutil
from sayo.engine.adts import *
from sayo.engine.evaluator import *
from LinearModel import LinearModelFactory, LinearModel, LinearBuildStrategy

log = sayo.util.sayolog.getLogger('rbf')
userlog = sayo.util.sayolog.getLogger('user')

class RbfBuildStrategy:
    
    def __init__(self, basis_type='cs20'):
        self.target_nmse = 5.0e-2
        self.lin_ss = None
        
        self.basis_type = basis_type
        if basis_type == 'gaussian':
            self.min_sigma_exp = -2.5     #min_sigma = 10**min_sigma_exp. 
            self.max_sigma_exp = -0.5     #..                             
            self.num_sigma_steps = 5      #
        elif basis_type == 'cs20':
            self.min_sigma_exp = 0.0   
            self.max_sigma_exp = 0.0     
            self.num_sigma_steps = 1
        else:
            raise AssertionError("unknown basis_type: %s" % basis_type)

    def __str__(self):
        s = "RbfBuildStrategy={ "
        s += 'basis_type=%s, ' % self.basis_type
        s += 'target_nmse=%5.2e, ' % self.target_nmse
        s += 'min_sigma_exp=%5.2e, ' % self.min_sigma_exp
        s += 'max_sigma_exp=%5.2e, ' % self.max_sigma_exp
        s += 'lin_ss=%s, ' % str(self.lin_ss)
        s += 'num_sigma_steps=%d, ' % self.num_sigma_steps
        s += '} '
        return s
        

class RbfModel:

    def __init__(self, rbf_bases, bases_lin_model, lin_lin_model,
                 trainX=None):
        self.rbf_bases = rbf_bases
        
        self.bases_lin_model = bases_lin_model
        self.lin_lin_model = lin_lin_model

        self.trainX = trainX

        self.range_x, self.nonzero_range_vars = None, None
        if trainX is not None:
            self.range_x = numpy.array([max(trainX[i,:]) - min(trainX[i,:]) \
                                          for i in range(trainX.shape[0])])
            self.nonzero_range_vars = []
            for i in range(trainX.shape[0]):
                if self.range_x[i] > 0.0:
                    self.nonzero_range_vars.append(i)

    def simulate(self, X):
        """
        Inputs:  X - inputs [var #][sample #]
        Outputs: f - output [sample #]
        """
        At = numpy.transpose(self.rbf_bases.basisOuts(X))
        Gt = X
        y = self.bases_lin_model.simulate(At) + \
            self.lin_lin_model.simulate(Gt)
        return y

    def uncertainty(self, X):
        """ Returns u, where u[i] is uncertainty for X[:,i]"""
        return numpy.array([self.dist01toClosest(X[:,i]) \
                              for i in range(X.shape[1])])
        
        
    def dist01toClosest(self, x):
        """ Returns uncertainty in an estimate at x; range is [0,1]"""
        assert self.trainX is not None, "need to set trainX to use this"
        close = self.closestTrainPoint(x)

        d01 = 0.0
        for i in self.nonzero_range_vars:
            d01 += ((close[i] - x[i])/self.range_x[i])**2
        d01 = math.sqrt(d01 / float(len(x)))
        return d01

    def closestTrainPoint(self, x):
        """ Returns the training point that is closest to x.
        x must be 1-dimensional; returns a point as a 1-d vec.
        """
        d = [math.sqrt( sum( (x - self.trainX[:,j])**2 ) ) \
             for j in range(self.trainX.shape[1])]
        return self.trainX[:,numpy.argmin(d)]
            

class RbfBases:
    def __init__(self, basis_type, sigmas, centers, minX, maxX):
        self.basis_type = basis_type
        self.sigmas = sigmas
        self.centers = centers
        self.minX, self.maxX = minX, maxX

    def basisOuts(self, X):
        nb = len(self.sigmas)
        N = X.shape[1]
        At = numpy.zeros((nb,N))*0.0 #each row is all outputs for one basis
        for sample_i in range(N):
            x = X[:,sample_i]
            for bi, sigma in enumerate(self.sigmas):
                center = self.centers[:,bi]
                d01 = self._dist01(x, center)
                z = d01 / sigma
                if self.basis_type == 'gaussian': 
                    At[bi, sample_i] += math.exp(-z**2)
                else:
                    t = z
                    At[bi, sample_i] += \
                              (1.0-t)**5*(1.0+5.0*t+9.0*t**2+5.0*t**3+1.0*t**4)
        y = numpy.transpose(At)
        return y

    def linearModels(self, X, y, lin_ss):
        p,n = X.shape
        A = self.basisOuts(X)
        G = numpy.transpose(X)
        cat_AG = self._catAG(A, G)
        
        z = numpy.zeros(p)*0.0
        ytarg = numpy.concatenate((y, z))

        lambd_c = LinearModelFactory().build(cat_AG, ytarg, lin_ss).coefs
        lambd = lambd_c[:n+1]
        c = numpy.concatenate((numpy.zeros(1)*0.0, lambd_c[n+1:]))
        return (LinearModel(lambd, None, 'lin'),
                LinearModel(c,     None, 'lin')  )

    def _catAG(self, A, G):
        """Concatenate G and A in a special way"""
        cat = numpy.concatenate
        tr = numpy.transpose
        p = G.shape[1]
        z = numpy.zeros((p,p))*0.0
        return cat( (cat( (  A,    G), 1),
                     cat( ( tr(G), z), 1) ), 0)
        
    def _dist01(self, x1, x2):
        """Returns the distance between x1 and x2, in range [0,1]
        (i.e. scaled, according to range of each var)"""
        sum01 = 0.0
        for i, (x1i, x2i, mn, mx) in enumerate(zip(x1, x2, self.minX,self.maxX)):
            if mn != mx:
                sum01 += ((x1i-x2i) / float(mx-mn))**2
        d01 = math.sqrt(sum01) / float(len(x1))
        return d01

class RbfFactory:
        
    def build(self, X, y, ss):
        """
        Inputs:  X - prediction points [var #][sample #]
                 y - training outputs  [sample #]
        Outputs: RbfModel
        """
        assert max(y) < float('Inf')
        assert min(y) > float('-Inf')
        n,N = X.shape
        if N < 10:
            raise InsufficientDataError('need reasonable # samples (got %d)'%N)
        if n == 0:
            raise InsufficientDataError('need >0 input variables')
        assert N == y.shape[0]

        minX, maxX = numpy.zeros(X.shape[0])*0.0, numpy.zeros(X.shape[0])*0.0
        for var in range(X.shape[0]):
            minX[var], maxX[var] = min(X[var,:]), max(X[var,:])

        userlog.info('Begin.  Min(y)=%0.2e, max(y)=%0.2e' % (min(y), max(y)))
        userlog.info('#input dims=%d, # trn samples=%d; basis_type=%s' %
                 (X.shape[0], X.shape[1], ss.basis_type))

        m, e = self.buildAcrossSigmas(X, y, ss, minX, maxX)
        model = RbfModel(m.rbf_bases, m.bases_lin_model, m.lin_lin_model, X)
        
        userlog.info('Done.  Final nmse=%5.2e' % e)
        return model

    def buildAcrossSigmas(self, X, y, ss, minX, maxX):
        centers = copy.deepcopy(X)
        mn, mx = ss.min_sigma_exp, ss.max_sigma_exp
        sigma_exps = [ss.min_sigma_exp]
        assert mx >= mn, "max sigma_exp must be >= min"
        if mn == mx:
            ss.num_sigma_steps = 1
        if ss.num_sigma_steps > 1:
            step = float(mx - mn) / float(ss.num_sigma_steps-1)
            sigma_exps = numpy.arange(mx, mn-step, -step)
        best_e, best_sigma_exp, best_model = float('Inf'), None, None
        for step_i, sigma_exp in enumerate(sigma_exps):
            sigmas = numpy.array([10**sigma_exp]*centers.shape[1])
            bases = RbfBases(ss.basis_type, sigmas, centers, minX, maxX)
            blin, llin = bases.linearModels(X, y, ss.lin_ss)
            model = RbfModel(bases, blin, llin)
            e = mathutil.nmse(model.simulate(X), y, min(y), max(y))
            userlog.info('Sigma=%5.2e, nmse=%5.2e' % (10**sigma_exp, e))
            if e < best_e:
                best_e, best_sigma_exp, best_model = e, sigma_exp, model
            if best_e < ss.target_nmse:
                break
            
        return best_model, best_e
