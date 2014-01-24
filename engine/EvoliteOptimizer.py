""" EvoliteOptimizer

Lightweight evolutionary programming optimizer.
With Cauchy Mutation (ref. X. Yao IEEE TEC 1999)

**Minimizes.**

EvoliteOptimizer is _not_ a general optimizer; it does _not_
  use ADT objects, and instead stays at the numpy 'array' level.  Thus
  it can run significantly faster for appropriate problems.  It only
  works on nominal problems (ie no random awareness).
"""

import copy
import random
import logging
import math

import numpy

from engine.EngineUtils import coststr
import util.mathutil as mathutil

log = logging.getLogger('evo')

class EvoliteSolutionStrategy:

    def __init__(self, popsize=50, numgen=50, num_mc=5):
        self.popsize = popsize #population size
        self.numgen = numgen   #number of generation
        self.num_mc = num_mc   #number of MC samples
        
        self.mstd = 0.005; #std dev of stepsize in a gauss mut as if_ [0,1]
        self.init_xs = [] # list of points 'x'

        self.prob_crossover = 0.2 #always mutate, but cross over too how often?

        self.numgen_stagnated_to_stop=10 #how many gen of no improve before stop?

        self.target_cost = None #if non-None, it will stop if it's hit this target cost

    def __str__(self):
        s = 'EvoliteSolutionStrategy={'
        s += ' popsize=%d' % self.popsize
        s += '; numgen=%d' % self.numgen
        s += '; num_mc=%d' % self.num_mc
        s += '; mstd=%.3e' % self.mstd
        s += '; num init xs=%d' % len(self.init_xs)
        s += '; prob_crossover=%.2f' % self.prob_crossover
        s += '; numgen_stagnated_to_stop=%d' % self.numgen_stagnated_to_stop
        s += '; target_cost=%s' % self.target_cost
        s += ' /EvoliteSolutionStrategy}'
        return s

    def setFast(self):
        self.popsize = 5
        self.numgen = 5

class EvoliteOptimizer: 
    
    def __init__(self, min_x, max_x, ss, function):
        """
        @description

          Set up the opt problem based on inputs.
          
        @arguments

          min_x -- 1d array -- [opt variable #] -- defines the minimum bounds
            for each variable allowed during search
          max_x -- 1d array -- [opt variable #] -- defines the maximum bounds
          ss -- EvoliteSolutionStrategy object --
          function -- callable object such that a call to = function(X) returns
            'costs',  which is a 1d array of outputs [sample #].
            'X' is a 2d array of [opt variable #][sample #].

        @return

          EvoliteOptimizer object

        @exceptions

        @notes
        """
        self.min_x = min_x
        self.max_x = max_x
        self.ss = ss
        self.function = function
        
        self.best_x = copy.copy(min_x) #arbitrary initial setting
        self.best_cost = float('Inf')
        
    def optimize(self):   
        """
        @description

          Optimizes in the space defined by (self.min_x, self.max_x) in
          order to find a self.best_x that minimizes cost, where
          costs = self.function(X).
          
        @arguments

          <<none>>

        @return

          <<none>>, except that self.best_x has been updated with the
            lowest-cost 'x' encountered during optimization.

        @exceptions

        @notes
        """  
        log.info('Begin opt.')

        #time-saving local variables
        scaled_stds = self.ss.mstd * (self.max_x - self.min_x)
        n = len(self.min_x)

        #initialize population
        pop_X = numpy.zeros((n,self.ss.popsize), dtype=float)
        for ind_i in range(self.ss.popsize):
            pop_X[:,ind_i] = self._random_x()
        
        num_init = min(len(self.ss.init_xs), self.ss.popsize)
        for i in range(num_init):
            pop_X[:,i] = self.ss.init_xs[i]

        #generational loop:
        # -calc cost per ind in pop
        # -choose parents
        # -remember best; report
        # -new pop = vary parents
        last_improve_gen = -1
        for gen in range(self.ss.numgen):
            costs = self.function(pop_X)
            for cost in costs:
                assert mathutil.isNumber(cost)
            
            parent_X = numpy.take(pop_X, self._select_i(costs), 1)
            parent_X2 = numpy.take(pop_X, self._select_i(costs), 1) #for xover

            if min(costs) < self.best_cost:
                last_improve_gen = gen
                I = numpy.argmin(costs)
                self.best_x = copy.copy(pop_X[:, I])
                self.best_cost = costs[I]

            if (gen % 5 == 0) or (gen == self.ss.numgen-1):
                log.info('Gen=%d, Best:cost=%5.7e, x=%s...' % (gen, self.best_cost, self.best_x[:2]))
                log.debug('(Sorted) pp costs=%s' % coststr(sorted(costs)))

            if (self.ss.target_cost is not None) and (self.best_cost <= self.ss.target_cost):
                log.info('Stop because hit target cost of %.3e' % self.ss.target_cost)
                break

            if (gen - last_improve_gen) > self.ss.numgen_stagnated_to_stop:
                log.info('Stop because no improvement in last %d generations' % self.ss.numgen_stagnated_to_stop)
                break

            #elitism
            parent_X[:,0] = self.best_x
            
            #vary
            pop_X = parent_X
            for ind_i in range(self.ss.popsize):
                do_crossover = (random.random() < self.ss.prob_crossover)
                for j, (mn, mx) in enumerate(zip(self.min_x,self.max_x)):
                    #maybe xover (uniform xover)
                    if do_crossover and random.random() < 0.5:
                        base_v = parent_X2[j, ind_i]
                    else:
                        base_v = pop_X[j, ind_i]
                        
                    #new value 'v'
                    v = base_v + _cauchy(scaled_stds[j])
                    
                    #rail 'v'
                    v  = min(self.max_x[j], max(self.min_x[j], v))

                    #set 'v'
                    pop_X[j, ind_i] = v

        #done
        log.info('Done opt. Best:cost=%5.7e, best_x=%s...\n' % (self.best_cost, self.best_x[:2]))
        
    def _random_x(self):
        """Random individual 'x' generated from uniform distribution"""
        return numpy.array([random.uniform(mn,mx)
                              for (mn,mx) in zip(self.min_x, self.max_x)])
        
    def _select_i(self, costs):
        """ Apply tournament selection to N inds, each w/ a corr. cost """
        N = len(costs)
        winner_Is = []
        for tournament_i in range(N):
            ind1_I = random.randint(0,N-1)
            ind2_I = random.randint(0,N-1)
            winner_I = ind1_I
            if costs[ind2_I] < costs[ind1_I]:
                winner_I = ind2_I
            winner_Is.append(winner_I)
        return winner_Is

def _cauchy(sigma):
    """Draws a random scalar number from a Cauchy distribution,
    with mean = 0.0 and sigma = 'sigma'.  A Cauchy distribution is
    similar to Gaussian, except the tails are way fatter.
    """
    z = random.uniform(0.0, 1.0)
    return sigma * math.tan(math.pi * (z-0.5) )
