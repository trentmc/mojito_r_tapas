"""TloOptimizer

TloOptimizer = Trust-region LOcal optimizer

Uses a trust-region method with modeling to do optimization.

Its interface is like Dyt.

Minimizes cost
"""

import copy
import logging
import math
import random
import types

import numpy

from adts import *
from engine.EngineUtils import coststr, LHS
from engine.YtAdts import YTPoint, buildCytPointMeta
from engine.EvoliteOptimizer import EvoliteSolutionStrategy, EvoliteOptimizer
from regressor.LinearModel import LinearBuildStrategy
from regressor.Probe import ProbeBuildStrategy, ProbeFactory
from util import mathutil 

log = logging.getLogger("tlo")

#===============================================================
#start magic numbers

class TloSolutionStrategy:

    def __init__(self, init_opt_point):
        self.init_opt_point = init_opt_point
        self.max_num_opt_points = 1500
        self.num_samples_per_active_var = 2
        self.target_cost = None
        
        self.init_radius = 0.50
        self.max_radius = 0.50
        self.stop_radius = 1.0e-4 #how precise should the final answer be?

        #improvement-ratio thresholds 
        self.loose_thr = 0.01 #eta1
        self.tight_thr = 0.2  #eta2

        #growth rates
        self.gamma1 = 0.667 #growth when we do poorly (want <1, i.e. to actually shrink)
        self.gamma2 = 1.5 #growth when we do well   (want >1)

        #postconditions
        self.assertConsistent()

    def assertConsistent(self):
        assert self.init_opt_point is not None
        assert self.init_opt_point.is_scaled, "need init point scaled"
        
        assert 0 < self.max_num_opt_points
        assert isinstance(self.max_num_opt_points, types.IntType)
        
        assert 0 < self.num_samples_per_active_var
        assert isinstance(self.num_samples_per_active_var, types.IntType)
        
        assert 0.0 < self.stop_radius <= self.init_radius <= self.max_radius <= 1.0
        
        assert 0 < self.loose_thr <= self.tight_thr < 1
        assert 0 < self.gamma1 < 1
        assert 1 < self.gamma2

    def setMaxNumPoints(self, val):
        assert isinstance(val, types.IntType)
        self.max_num_opt_points = val

    def setInitOptPoint(self, val):
        assert isinstance(val, Point)
        self.init_opt_point = val

    def setTargetCost(self, val):
        assert mathutil.isNumber(val)
        self.target_cost = val

    def __str__(self):
        s = "TloSolutionStrategy={"
        s += " max_num_opt_points=%s" % self.max_num_opt_points
        s += "; num_samples_per_active_var=%d" % self.num_samples_per_active_var
        s += "; target_cost=%s" % coststr(self.target_cost)
        s += "; init_radius=%.2e" % self.init_radius
        s += "; max_radius=%.2e" % self.max_radius
        s += "; stop_radius=%.2e" % self.stop_radius
        s += "; loose_thr=%.2e" % self.loose_thr
        s += "; tight_thr=%.2e" % self.tight_thr
        s += "; gamma2=%.2e" % self.gamma2
        s += "; gamma1=%.2e" % self.gamma1
        s += " /TloSolutionStrategy}"
        return s

ALLOWED_ACTIONS = ["SAMPLE_INDS_IN_BALL","INNER_OPT", "REACT_TO_INNER_OPT", "STOP_CONVERGED"]
        
#end magic numbers
#===============================================================

class TemplateTloOptimizer:
    """This class is a simple example of how to embed a TloState
    into an optimizer"""
    
    def __init__(self, ps, ss):
        self.ps = ps
        self.ss = ss
        self.state = None

    def stop(self):
        """Call this in order to externally force a stop"""
        self.state._stop = True

    def optimize(self):
        log.info("Begin opt")
        
        self.state = TloState(self.ps.opt_point_meta, self.ss)

        while True:
            #[at this point: simulate 'inds' and set each ind's 'cost']
            #an example is here, where the cost function is merely quadratic
            inds = self.state.indsWithoutCosts()
            for ind in inds:
                opt_point = self.state.xToScaledPoint(ind.x)
                cost = sum(vi**2 for vi in opt_point.values())
                ind.setCost(cost)

            if self.state.doStop():
                break
            
            self.state.update()

        log.info("Done opt")
            

class TloInd:
    """Merely an association of an opt point with a cost."""
    def __init__(self, x):
        self.x = x          #x[i] is the value of ordered_vars[i], in range [0,1]
        self._cost = None

    def setCost(self, cost):
        """Sets ind's cost as 'cost'"""
        self._cost = cost

    def cost(self):
        """Returns cost"""
        return self._cost

class TloState:
    """Stores the state of a Tlo search.  Each call to update()
    will generate new candidate designs, which can then be simulated
    externally."""
    
    def __init__(self, ps_opm, ss):
        #preconditions
        assert isinstance(ps_opm, PointMeta)
        assert isinstance(ss, TloSolutionStrategy)
        assert isinstance(ss.init_opt_point, Point)
        ss.assertConsistent()

        #main work...
        self.yt_opm = buildCytPointMeta(ps_opm)
        assert len(self.yt_opm.continuousVaryingVarNames()) == len(self.yt_opm)
        
        self.ordered_vars = sorted(self.yt_opm.continuousVaryingVarNames()) #order matters
        
        self.ss = ss
    
        self._stop = False

        self.all_inds = []

        self.action = "SAMPLE_INDS_IN_BALL"

        self.center_ind = self._createInd(self.scaledPointToX(ss.init_opt_point))
        self.radius = ss.init_radius

        self.inner_ind = None
        self.predicted_reduction = None

        #postconditions
        assert self.xToScaledPoint(self.center_ind.x) == ss.init_opt_point
        
            
    def __str__(self):
        s = []
        s += ["STATE:"]
        s += ["center_cost=%.6e" % self.center_ind.cost()]
        if self.all_inds:
            s += ["; best_cost=%.6e" % self.bestInd().cost()]
        s += ["; radius=%.2e" % self.radius]
        s += ["; # opt points=%d (%d in ball)" % (self.numOptPoints(), self.numIndsInBall())]

        #output optimization values too
        s += ["; center_x=%s" % self.center_ind.x[:4]]
        s += ["; center_opt_point=%s" % self.centerOptPoint()]
        
        return "".join(s)

    def numVars(self):
        """Returns number of active opt vars"""
        return len(self.yt_opm.continuousVaryingVarNames())

    def minNumIndsInBall(self):
        """Returns the minimum number of inds needed in the ball in order to build a model"""
        return self.ss.num_samples_per_active_var * self.numVars()

    def numIndsInBall(self):
        """Returns the number of inds in the ball defined by center_x and radius"""
        return len(self.indsInBall())

    def indsInBall(self, radius_multiplier=1.0):
        """Returns a list of the inds that are in the ball defined by center_x and radius.
        Expands radius by 'radius_multiplier'
        """
        return [ind for ind in self.all_inds if self.xInBall(ind.x, radius_multiplier)]

    def xInBall(self, x, radius_multiplier=1.0):
        """Returns True if ix is in the ball defined by center_x and radius.
        Expands radius by 'radius_multiplier'
        """
        return dist01(self.center_ind.x, x) <= (self.radius * radius_multiplier)

    def numOptPoints(self):
        """Returns the number of opt points (inds) covered so far"""
        return len(self.all_inds)

    def bestInd(self):
        """Returns ind with lowest cost.  Needs all inds to have been evaluated."""
        costs = [ind.cost() for ind in self.all_inds]
        assert None not in costs
        return self.all_inds[numpy.argmin(costs)]

    def bestOptPoint(self):
        """Returns best (scaled) opt_point seen so far"""
        return self.xToScaledPoint(self.bestInd().x)
    
    def centerOptPoint(self):
        """Returns center (scaled) opt_point"""
        return self.xToScaledPoint(self.center_ind.x)

    def indsWithoutCosts(self):
        """Returns the list of inds that do not have cost set yet"""
        return [ind for ind in self.all_inds if ind.cost() is None]
    
    def doStop(self):
        """Returns True is this state wants to stop"""
        if self._stop:
            log.info("Stop because user soft stop requested")
            return True

        if self.numOptPoints() >= self.ss.max_num_opt_points:
            log.info("Stop because # opt_points >= maximum allowed (=%d)"%
                     (self.ss.max_num_opt_points))
            return True

        if (self.ss.target_cost is not None) and (self.bestInd().cost() <= self.ss.target_cost):
            log.info("Stop because hit target cost of %.3e" % self.ss.target_cost)
            return True            

        if self.action == "STOP_CONVERGED":
            log.info("Stop because converged")
            return True
            
        return False
            
    #=========================================================================
    #update state!
    def update(self):
        """At the end of a call to this, there should be inds waiting to be evaluated
        and have their cost set"""
        #preconditions
        assert self.action in ALLOWED_ACTIONS

        #
        log.info(str(self))
        
        #===========================================
        #ROUND 1: prepare for inds pending
        #===========================================
        if self.action == "REACT_TO_INNER_OPT":
            log.info("Respond to %s" % self.action)
            
            #compute improvement
            actual_reduction = self.center_ind.cost() - self.inner_ind.cost()
            if self.predicted_reduction <= 0.0:
                improvement_ratio = 0.0
            else:
                improvement_ratio = actual_reduction / self.predicted_reduction # ==rho_k
            log.info("Pred. reduction = %.6e, actual reduction = %.6e, "
                     "improvement ratio = (actual / predicted red.) = %.4e" %
                     (self.predicted_reduction, actual_reduction, improvement_ratio))

            #update center
            if improvement_ratio >= self.ss.loose_thr:
                self.center_ind = self.inner_ind

            #trust-region radius update
            if improvement_ratio >= self.ss.tight_thr:
                log.info("Did well, so grow radius")
                self.radius = min(self.ss.gamma2 * self.radius, self.ss.max_radius)
                self.action = "INNER_OPT"
            elif improvement_ratio >= self.ss.loose_thr:
                log.info("Did so-so, so stay at same radius")
                self.radius = self.radius
                self.action = "INNER_OPT"
            elif improvement_ratio > 0.0:
                log.info("Did poorly, so shrink radius")
                self.radius = self.ss.gamma1 * self.radius 
                self.action = "INNER_OPT"
            else:
                log.info("Did very poorly, shrink a lot and sample more")
                self.radius = self.ss.gamma1 * self.radius 
                self.action = "SAMPLE_INDS_IN_BALL"

            #update corner cases
            if self.numIndsInBall() < self.minNumIndsInBall():
                log.info("After update of trust region, don't have enough inds in ball, so sample more")
                self.action = "SAMPLE_INDS_IN_BALL"

            if self.radius <= self.ss.stop_radius:
                log.info("We will be stopping because radius < min size")
                self.action = "STOP_CONVERGED"
            
            log.info(str(self))

        #===========================================
        #ROUND 2: create inds pending
        #===========================================
        assert self.action != "REACT_TO_INNER_OPT"
        log.info("Respond to %s" % self.action)
        if self.action == "SAMPLE_INDS_IN_BALL":
            num_samples = max(1, self.minNumIndsInBall() - self.numIndsInBall())
            self._newIndsViaSampleInBall(num_samples)
            self.action = "INNER_OPT"

        elif self.action == "INNER_OPT":
            #build model & propose inner ind
            (self.inner_ind, self.predicted_reduction) = self._newIndViaInnerOptimization()
            self.action = "REACT_TO_INNER_OPT"

        elif self.action == "STOP_CONVERGED":
            pass
          
        else:
            raise AssertionError("Unknown label '%s'" % last_action)
        
    #=========================================================================
    #state helper functions
    def _newIndsViaSampleInBall(self, num_inds):
        """Sample 'num_inds' inds in the ball defined by self.center_ind.x and self.radius"""
        log.info("Take %d sample(s) in ball" % num_inds)
        center_x = self.center_ind.x

        #well-spread sampling of stepsizes.  The '-0.5' is so that direction is negative half the time
        S = LHS(self.numVars(), num_inds) / float(num_inds) - 0.5
        new_inds = []
        for i in range(num_inds):
            #s = stepsize.  Regenerate if needed so that magnitude > 0
            s = S[:,i]
            while magnitude(s) == 0.0:
                s = numpy.array([random.random() - 0.5 for j in xrange(self.numVars())])

            #rescale s so that magnitude = 1.0
            s = s / magnitude(s)

            #create new x
            # -the random() part here is so that the sample can be in the ball's interior, not
            #  just boundary
            new_x = center_x + s * self.radius * random.random()

            #verify
            assert self.xInBall(new_x)

            #compute new ind, and remember it
            new_ind = self._createInd(new_x)
            new_inds.append(new_ind)

        return new_inds

    def _newIndViaInnerOptimization(self):
        """
        Does inner optimization by:
         1. Build a model using all inds in the ball defined by (center_ind, radius)
         2. Optimizes on the model to find the lowest-cost x
         3. Creates a new ind at that x 
         4. Returns (new_ind, predicted_reduction)
        """
        #preconditions
        assert self.numIndsInBall() >= self.minNumIndsInBall()

        #main work...
        
        #get training X/y
        # -note that we grab inds that are beyond the usual ball -- we go for twice the radius
        inds = self.indsInBall(radius_multiplier=2.0)
        X = numpy.transpose(numpy.array([ind.x for ind in inds]))
        y = numpy.array([ind.cost() for ind in inds])
        log.info("New ind via inner: build model from %d datapoints, min(y)=%.3e, max(y)=%.3e" %
                 (len(y), min(y), max(y)))

        #build model
        lin_ss = LinearBuildStrategy(y_transforms = ["lin"], target_nmse = 0.01, regularize = True)
        lin_ss.reg.thr = 0.4
        probe_ss = ProbeBuildStrategy(target_train_nmse = 0.02, max_rank=2, lin_ss=lin_ss)
        inner_model = ProbeFactory().build(X, y, probe_ss)

        #optimize on model

        # -cost function
        function = ModelEvaluatorFunction(inner_model, self.center_ind.x, self.radius)

        # -solution strategy
        #   -make it pretty much a hillclimber from starting point of center_ind
        evo_ss = EvoliteSolutionStrategy(popsize = 3, numgen=300, num_mc=0)
        evo_ss.init_xs = [self.center_ind.x]

        #   -_need_ to scale the mutation stepsize with the radius, otherwise steps are too big
        evo_ss.mstd *= self.radius

        # -optvar range
        min_x = numpy.zeros(self.numVars(), dtype=float)
        max_x = numpy.ones(self.numVars(), dtype=float)

        # -run opt
        inner_opt = EvoliteOptimizer(min_x, max_x, evo_ss, function)
        inner_opt.optimize()
        inner_x = inner_opt.best_x

        #compute return data
        inner_ind = self._createInd(inner_x)
        predicted_inner_cost = inner_model.simulate1(inner_ind.x)
        predicted_center_cost = inner_model.simulate1(self.center_ind.x)
        predicted_reduction = predicted_center_cost - predicted_inner_cost
        log.info("New ind via inner: pred. inner cost = %.6e, pred. center cost = %.6e, "
                 "pred. reduction = %.6e" %
                 (predicted_inner_cost, predicted_center_cost, predicted_reduction))

        #return
        return (inner_ind, predicted_reduction)

    def _createInd(self, x):    
        """Creates an ind defined by 1d array x.  Adds to self.all_inds."""
        #preconditions
        assert len(x) == self.numVars()

        #main work
        new_ind = TloInd(x)
        self.all_inds.append(new_ind)
                
        return new_ind

    def scaledPointToX(self, scaled_point):
        """Converts a yt_opm scaled point into 1d array 'x' with range [0,1]"""
        unscaled_point = self.yt_opm.unscale(YTPoint(True, dict(scaled_point)))
        x = numpy.array([unscaled_point[var] for var in self.ordered_vars])
        return x

    def xToScaledPoint(self, x):
        """Converts 1d array 'x' with range [0,1] into a yt_opm scaled point"""
        unscaled_d = dict(zip(self.ordered_vars, x))
        unscaled_point = YTPoint(False, unscaled_d)
        scaled_point = self.yt_opm.scale(unscaled_point)
        return scaled_point
    
          
class ModelEvaluatorFunction:
    """
    @description

        Callable object, for use with EvoliteOptimizer.
    
    @attributes
    
        model -- model -- model which is used for the basis of evaluations.
          Each input dimension to the model is the same as the input
          dimensions of __call__().
        center_x -- 1d array of float -- with radius, defines the allowed trust region
          that we can search in
        radius -- float -- see above
        
    @notes
    
        Callable because it behaves like a function, thanks to use of __call__().
    
    """
    
    def __init__(self, model, center_x, radius):
        self.model = model
        self.center_x = center_x
        self.radius = radius

    def __call__(self, X):
        """
        @description

          Returns an estimate of cost1 for each input (column) vector x in X.
          
        @arguments

          X -- 2d array [input variable #][sample #] - inputs to regressor

        @return

          costs1 -- 1d array [sample #] -- actual costs

        @exceptions

        @notes
        """
        #preconditions
        assert X.shape[1] > 0

        #main work
        X = numpy.asarray(X)

        # -model's cost
        costs1 = self.model.simulate(X)

        # -disallow inds from venturing beyond trust region.  Give them guidance to return to it.
        for i in range(X.shape[1]):
            dist = dist01(self.center_x, X[:,i])
            if dist > self.radius:
                costs1[i] = (dist + 1.0) * 100000 #magic number alert
        
        return costs1
    
  

def dist01(x1, x2):
    """Returns the distance between vectors x1 and x2"""
    return magnitude(x1 - x2)

def magnitude(x):
    """Returns magnitude of vector x"""
    return math.sqrt(sum((xi**2) for xi in x))
