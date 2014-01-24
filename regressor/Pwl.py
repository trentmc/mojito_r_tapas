"""Pwl.py

1-d piecewise linear regressors
-hockey sticks
-...
"""
from itertools import izip
import logging

import numpy

from adts import *
from engine.CytOptimizer import CytSolutionStrategy, CytState
from engine.YtAdts import YTPoint
from regressor.LinearModel import LinearBuildStrategy, LinearModelFactory
from util import mathutil
from util.octavecall import plotAndPause

log = logging.getLogger('pwl')

PWL_APPROACHES = ['hockey', 'bump']

class PwlBuildStrategy:
    """
    @description
    
        Solution strategy for building pwl models
        
    @notes

    """

    def __init__(self, approach):
        #preconditions
        assert approach in PWL_APPROACHES

        #set values
        self.approach = approach

        target_correlation = 0.9999
        self.target_cost = 1.0 - target_correlation
        
        self.lin_ss = LinearBuildStrategy(
            y_transforms=['lin'], target_nmse=0.00, regularize=True)
        self.lin_ss.reg.thr = 0.5
        
        self.cyt_ss = CytSolutionStrategy()
        self.cyt_ss.setMaxNumPoints(5000)

        self.num_yt_reps = 4

    def __str__(self):
        s = ''
        s += 'PwlbuildStrategy={'
        s += ' lin_ss = %s' % self.lin_ss
        s += '; cyt_ss=%s' % self.cyt_ss
        s += '; num_yt_reps=%d' % self.num_yt_reps
        s += ' /PwlbuildStrategy}'
        return s

class PwlModel:
    """
    @description

      A 1-d piecewise linear model.  Hockey stick models are a subset of this.

    @attributes

      xs, ys -- list of float -- each line segment is characterized by
        going from (xs[i-1], ys[i-1]) to (xs[i], ys[i]).
        If the input x is < xs[0] or > xs[-1], it extrapolates.

    @notes

    """ 
    def __init__(self, xs, ys):
        """
        @description

        @arguments
        
            xs, ys -- list of float 
        
        @return
    
        @exceptions
        
          xs and ys need to be sorted in ascending order of x
    
        @notes
        """
        #preconditions
        assert sorted(xs) == xs, 'xs must be sorted in ascending order'
        assert len(xs) == len(ys), 'need a y[i] for every x[i]'
        assert len(xs) >= 1, 'need at least one entry'

        #set data
        self.xs = xs
        self.ys = ys
        self.num_points = len(xs)

    def simulate(self, x_samples):
        """
        @description
        
            Simulate this model for each input sample
        
        @arguments
        
            x_samples -- 1d array -- x_samples[i] is x value for sample i

        @return
    
            y -- 1d array -- output [sample #].

        @exceptions
    
        @notes
         
        """
        y = numpy.array([self.simulate1(x_sample)
                           for x_sample in x_samples])
        return y

    def simulate1(self, x):
        #corner case
        if len(self.xs) == 1:
            return self.ys[0]

        #main case...
        xs = self.xs
        ys = self.ys

        #loop to find the target i & i+1
        for i in range(self.num_points - 1):
            if (xs[i+1] > x):
                break
            
        #simply apply equation of "line through two points"
        x1, x2, y1, y2 = xs[i], xs[i+1], ys[i], ys[i+1]
        return y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
            

    def __str__(self):
        s = 'PwlModel={(x,y)='
        for (i, (xi, yi)) in enumerate(zip(self.xs, self.ys)):
            s += '(%g,%g)' % (xi, yi)
            if i < (len(self.xs) - 1):
                s += ','
        s += ' /PwlModel}'
        return s

class PwlData:
    """Holds input 'x', and output 'y', plus simple functions of them: min, max, range."""
    
    def __init__(self, x, y):
        #condition data
        x = numpy.asarray(x)
        y = numpy.asarray(y)

        #preconditions
        assert len(x.shape) == 1
        assert len(y.shape) == 1

        #set data
        self.x = x
        self.min_x, self.max_x = min(x), max(x)
        self.xrange = max(x) - min(x)
        self.y = y
        self.min_y, self.max_y = min(y), max(y)

    def setResampledData(self, resampled_x, resampled_y):
        #condition data
        resampled_x = numpy.asarray(resampled_x)
        resampled_y = numpy.asarray(resampled_y)

        #preconditions
        assert len(resampled_x.shape) == 1
        assert len(resampled_y.shape) == 1
        
        self.resampled_x = resampled_x
        self.resampled_y = resampled_y

class PwlFactory:
    """Builds pwl models"""

    def __init__(self):
        self.ss = None
        self.data = None

    def build(self, x, y, ss):
            
        #preconditions
        assert ss.approach in PWL_APPROACHES
        assert isinstance(ss, PwlBuildStrategy)

        #corner case
        min_N = 5
        if (len(x) < min_N) or (len(y) <= min_N):
            log.warning('Not enough data to work with, so returning a model of 0')
            return PwlModel([0.0],[0.0])

        #main work...
        
        log.info('Build pwl with approach %s: begin' % ss.approach)
        
        #set data
        self.ss = ss
        self.data = PwlData(x, y)

        #set pm, init_opt_point, base_func, data.resampled_{x,y}
        if ss.approach == 'hockey':
            #One nonlinear parameter: inflection_x.
            # Lin-learning determines offset & bias afterwards; use r^2 during
            vm = ContinuousVarMeta(False, self.data.min_x, self.data.max_x,
                                   'inflection_x')
            pm = PointMeta([vm])
            init_opt_point = Point(
                True, {'inflection_x' : (self.data.min_x + self.data.max_x) / 2.0 })
            base_func = self._pwlHockeyBase

            self.data.setResampledData(self.data.x, self.data.y)
            
        elif ss.approach == 'bump':
            #Four nonlinear parameters: x1, x2, x3, x4
            # -Lin-learning determines offset & bias.
            # -use heuristics to set initial values and ranges.  This is ok
            #  because if the heuristics don't give a good fit, then 
            #  the waveform is bad!
            
            dy = [y[i+1] - y[i] for i in range(len(y) - 1)]
            init_x1, init_x2, init_x3, init_x4 = None, None, None, None
            max_dy, min_dy = max(dy), min(dy)
            for (xi, dyi) in izip(x[:-1], dy):
                if init_x1 is None:
                    if dyi > 0.2 * max_dy:
                        init_x1 = xi
                elif init_x2 is None:
                    if dyi < 0.8 * max_dy:
                        init_x2 = xi
                elif init_x3 is None:
                    if dyi < 0.2 * min_dy:
                        init_x3 = xi
                elif init_x4 is None:
                    if dyi > 0.8 * min_dy:
                        init_x4 = xi
            log.info("init_x1=%s, init_x2=%s, init_x3=%s, init_x4=%s" %
                     (init_x1, init_x2, init_x3, init_x4))

            mn, mx, rng = self.data.min_x, self.data.max_x, self.data.xrange
            if init_x1 is None: init_x1 = mn + 0.00 * rng
            if init_x2 is None: init_x2 = mn + 0.25 * rng
            if init_x3 is None: init_x3 = mn + 0.50 * rng
            if init_x4 is None: init_x4 = mn + 0.75 * rng

            init_x1, init_x2, init_x3, init_x4 = sorted([
                init_x1, init_x2, init_x3, init_x4])
            
            x1_min, x1_max = sorted([mn + (init_x1-mn)*0.75, init_x2 + 0.02*rng])
            x2_min, x2_max = sorted([init_x1 - 0.02*rng, (init_x2 + init_x3) / 2.0])
            x3_min, x3_max = sorted([(init_x2 + init_x3) / 2.0, init_x4 + 0.02*rng])
            x4_min, x4_max = sorted([init_x3 - 0.02*rng, mx - 0.25*(mx - init_x4)])

            if x1_max == x1_min: x1_min, x1_max = mn, mx
            if x2_max == x2_min: x2_min, x2_max = mn, mx
            if x3_max == x3_min: x3_min, x3_max = mn, mx
            if x4_max == x4_min: x4_min, x4_max = mn, mx
                
            pm = PointMeta([
                ContinuousVarMeta(False, x1_min, x1_max, 'x1'),
                ContinuousVarMeta(False, x2_min, x2_max, 'x2'),
                ContinuousVarMeta(False, x3_min, x3_max, 'x3'),
                ContinuousVarMeta(False, x4_min, x4_max, 'x4'),
                ])

            init_opt_point = Point(True, {'x1':init_x1, 'x2':init_x2,
                                          'x3':init_x3, 'x4':init_x4})
            log.info('Waveform fitting variable ranges: %s' % pm)
            log.info('Waveform fitting initial point: %s'% init_opt_point)
            base_func = self._pwlBumpBase

            #resample:
            # Three regions: 1 - flat region before, 2 - middle, 3 - flat after
            x1min = pm['x1'].min_unscaled_value
            x4max = pm['x4'].max_unscaled_value
            num1 = sum((xi < x1min) for xi in self.data.x)
            num3 = sum((xi > x4max) for xi in self.data.x)
            num2 = len(self.data.x) - num1 - num3
            target_num1 = 20 #magic number alert
            target_num2 = 100 #magic number alert
            target_num3 = 20 #magic number alert
            rate1 = max(1, int(num1 / float(target_num1)))
            rate2 = max(2, int(num2 / float(target_num2)))
            rate3 = max(3, int(num3 / float(target_num3)))
            I1 = [i for (i, xi) in enumerate(self.data.x)
                  if (xi < x1min) and ((i % rate1) == 0)]
            I3 = [i for (i, xi) in enumerate(self.data.x)
                  if (xi > x4max) and ((i % rate3) == 0)]
            I13 = I1 + I3
            I2 = [i for (i, xi) in enumerate(self.data.x)
                  if (i not in I13) and ((i % rate2) == 0)]
            I = I1 + I2 + I3
                
            self.data.setResampledData([self.data.x[i] for i in I],
                                       [self.data.y[i] for i in I])
            log.info('Resampled data from size %d to %d' %
                     (len(self.data.x), len(self.data.resampled_x)))

        else:
            raise NotImplementedError('ss.approach of %s not implemented yet' %
                                      ss.approach)


        #find 'best_opt_point'.  Maybe Repeat YT         
        best_opt_point, best_cost = None, float('Inf')
        for yt_rep in range(ss.num_yt_reps):

            #initialize cyt state
            state = CytState(pm, ss.cyt_ss, init_opt_point)

            #main cyt optimization loop
            log.debug('Begin run %d / %d of cyt' % (yt_rep+1, ss.num_yt_reps))
            loop_i = 0
            while True:
                loop_i += 1
                
                #[at this point: simulate 'inds' and set each ind's 'cost']
                inds = state.indsWithoutCosts()
                for ind in inds:
                    cost = self._cost(ind.opt_point, base_func)
                    ind.cost_tup = (cost, )

                if (loop_i % 10) == 0:
                    log.debug(str(state))
                #else:
                #    log.debug2(str(state))
                    
                state.update()

                if ind.cost_tup[0] <= self.ss.target_cost:
                    break
                if state.doStop():
                    break

            if best_opt_point is None or (state.bestInd().cost() < best_cost):
                best_opt_point = state.bestInd().opt_point
                best_cost = state.bestInd().cost()
                log.debug('Run %d / %d done.  Best cost so far = %.5e' %
                         (yt_rep + 1, ss.num_yt_reps, best_cost))

            if ind.cost_tup[0] <= self.ss.target_cost:
                log.debug('Hit target cost, so can stop cyt runs')
                break

        log.debug('Done all runs of cyt.  Best cost=%.5e' % best_cost)

        #build final model
        # -create the X, y for linear learning
        best_opt_point = best_opt_point
        pwl_base = base_func(best_opt_point)
        X_base = numpy.zeros((1 ,len(self.data.x)), dtype=float)
        X_base[0,:] = pwl_base.simulate(self.data.x)

        # -do linear learning
        minX, maxX = [min(X_base[0,:])], [max(X_base[0,:])]
        log.info('About to do lin learning; len(self.y) = %d' %
                 len(self.data.y))
        lin_model = LinearModelFactory().build(X_base, self.data.y, minX, maxX,
                                               self.ss.lin_ss)
        
        # -compute coefficients to characterize the final PWL model
        xs = pwl_base.xs
        X_base = numpy.zeros((1,len(xs)), dtype=float)
        X_base[0,:] = pwl_base.simulate(xs)
        ys = list(lin_model.simulate(X_base))
        
        # -construct the final PWL model
        pwl_model = PwlModel(xs, ys)

        #plotAndPause(x, y, x, pwl_model.simulate(x)) #to uncomment is a HACK

        #done
        log.debug('Build hockey stick: done. Result: %s' % pwl_model)
        return pwl_model

    def _cost(self, opt_point, base_function):
        
        #this pwl model does not have target scale and bias (and that's ok)
        pwl_base = base_function(opt_point)

        yhat = pwl_base.simulate(self.data.resampled_x)

        #correlation is cheaper to compute than nmse because
        # it is independent of scale and bias (nmse needs scale
        # and bias, which therefore needs linear learning, which takes time!)
        #-make sure that a correlation of -1.0 is worst (i.e. don't square r)
        r = mathutil.correlation(yhat, self.data.resampled_y)
        cost = 1.0 - r

        return cost

    def _pwlHockeyBase(self, opt_point):
        """Return a pwl model for a hockey stick that does not have the
        target scale and bias incorporated yet.
        """
        xs = [self.data.min_x - self.data.xrange, opt_point['inflection_x'],
              self.data.max_x + self.data.xrange]
        ys=  [self.data.min_y, self.data.max_y, self.data.max_y]
        return PwlModel(xs, ys)
    
    def _pwlBumpBase(self, opt_point):
        """Return a pwl model for a bump that does not have the target scale and bias
        incorporated yet.
        """
        eps = self.data.xrange / 1000.0
        
        #note how we maximize flexibility of the search and avoid
        # ordinality constraints by merely sorting the opt point values.
        # (Ref. Rothlauf 2005)
        sub_xs = sorted(opt_point.values())

        xs = [self.data.min_x - self.data.xrange] + sub_xs + \
             [self.data.max_x + self.data.xrange]

        ys = [self.data.min_y, self.data.min_y, self.data.max_y,
              self.data.max_y, self.data.min_y, self.data.min_y]
        return PwlModel(xs, ys)
        
