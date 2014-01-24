"""GtoOptimizer

GtoOptimizer = General Trust-region lOcal optimizer.

Uses a trust-region method with modeling to do optimization.

This is like TloOptimizer, except:
-whereas Tlo's inds just store 'cost', Gto's inds store worst-case metric sim results, and compute
 cost as a function of them.
-for inner opt, whereas Tlo builds a single regressor (on cost), Gto builds a regressor for every metric

Minimizes cost.
"""

import copy
from itertools import izip
import logging
import math
import random
import types

import numpy

from adts import *
from engine.EngineUtils import coststr, LHS
from engine.EvoliteOptimizer import EvoliteSolutionStrategy, EvoliteOptimizer
from regressor.LinearModel import LinearBuildStrategy
from regressor.Probe import ProbeBuildStrategy, ProbeFactory
from regressor.Sgb import SgbBuildStrategy, SgbFactory
from util import mathutil
from util.constants import BAD_METRIC_VALUE, INF

log = logging.getLogger("gto")

#===============================================================
#start magic numbers

class GtoSolutionStrategy:

    def __init__(self, weight_per_objective):
        self.init_opt_point = None #Point
        self.max_num_opt_points = 1500
        self.num_samples_per_active_var = 2
        self.target_cost = None

        self.weight_per_objective = weight_per_objective #list with one numeric entry per objective (or None)
        
        self.init_radius = 0.50
        self.max_radius = 0.50
        self.stop_radius = 0.5e-2 #how precise should the final answer be?

        #improvement-ratio thresholds 
        self.loose_thr = 0.01 #eta1
        self.tight_thr = 0.2  #eta2

        #growth rates
        self.gamma1 = 0.667 #growth when we do poorly (want <1, i.e. to actually shrink)
        self.gamma2 = 1.5   #growth when we do well   (want >1)

        #SGB model builder
        self.sgb_ss = SgbBuildStrategy(max_carts=500, learning_rate=0.01, target_trn_nmse=0.02)

        #PROBE model builder
        lin_ss = LinearBuildStrategy(y_transforms=["lin"], target_nmse=0.01, regularize=True)
        lin_ss.reg.thr = 0.4
        self.probe_ss = ProbeBuildStrategy(target_train_nmse = 0.02, max_rank=2, lin_ss=lin_ss)
        
        #postconditions
        self.assertConsistent()

    def assertConsistent(self):
        if self.init_opt_point is not None:
            assert isinstance(self.init_opt_point, Point)
            assert self.init_opt_point.is_scaled, "need init point scaled"
        
        assert 0 < self.max_num_opt_points
        assert isinstance(self.max_num_opt_points, types.IntType)
        
        assert 0 < self.num_samples_per_active_var
        assert isinstance(self.num_samples_per_active_var, types.IntType)

        assert (self.target_cost is None) or mathutil.isNumber(self.target_cost)

        if self.weight_per_objective is not None:
            assert len(self.weight_per_objective) >= 0
            for w in self.weight_per_objective:
                assert w >= 0.0
        
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

    def setTargetWeightPerObjective(self, val):
        assert len(val) >= 0
        self.weight_per_objective = val

    def __str__(self):
        s = "GtoSolutionStrategy={"
        s += " max_num_opt_points=%s" % self.max_num_opt_points
        s += "; init_opt_point is None: %s" % (self.init_opt_point is None)
        s += "; num_samples_per_active_var=%d" % self.num_samples_per_active_var
        s += "; target_cost=%s" % coststr(self.target_cost)
        s += "; init_radius=%.2e" % self.init_radius
        s += "; max_radius=%.2e" % self.max_radius
        s += "; stop_radius=%.2e" % self.stop_radius
        s += "; loose_thr=%.2e" % self.loose_thr
        s += "; tight_thr=%.2e" % self.tight_thr
        s += "; gamma2=%.2e" % self.gamma2
        s += "; gamma1=%.2e" % self.gamma1
        s += "; sgb_ss=%s" % self.sgb_ss
        s += "; probe_ss=%s" % self.probe_ss
        s += " /GtoSolutionStrategy}"
        return s

ALLOWED_ACTIONS = ["SAMPLE_INDS_IN_BALL","INNER_OPT", "REACT_TO_INNER_OPT", "STOP_CONVERGED"]
        
#end magic numbers


#========================================================================
#For Gto, use GtoProblemSetup (below), not ProblemSetup
class GtoProblemSetup:
    """This is a stripped-down problem setup: it just has a list of metrics and an OptPointMeta,
    and has a subset of the functionality of ProblemSetup.
    """
    def __init__(self, metrics, opt_point_meta):
        #preconditions
        assert isinstance(metrics, types.ListType)
        assert isinstance(opt_point_meta, PointMeta)

        #set values
        self.metrics = metrics
        self.opt_point_meta = opt_point_meta

    def flattenedMetrics(self):
        """Returns list of metrics"""
        return self.metrics

    def flattenedMetricNames(self):
        """Returns list of metric names"""
        return [metric.name for metric in self.flattenedMetrics()]

    def numMetrics(self):
        """Returns number of metrics"""
        return len(self.metrics)

    def metricsWithObjectives(self):
        """Returns the metrics that are objectives"""
        return [metric for metric in self.metrics if metric.isObjective()]
    
    def numObjectives(self):
        """Returns number of metrics that are objectives"""
        return len(self.metricsWithObjectives())
    
#========================================================================
class TemplateGtoOptimizer:
    """This class is a simple example of how to embed a GtoState
    into an optimizer"""
    
    def __init__(self, ps, ss, init_opt_point):
        self.ps = ps
        self.ss = ss
        self.init_opt_point = init_opt_point
        self.state = None

    def stop(self):
        """Call this in order to externally force a stop"""
        self.state._stop = True

    def optimize(self):
        log.info("Begin opt")
        
        self.state = GtoState(self.ps, self.ss, self.init_opt_point)

        while True:
            #[at this point: simulate 'inds' and set each ind's 'cost']
            #an example is here, where the cost function is merely quadratic on each metric
            inds = self.state.indsWithoutCosts()
            for ind in inds:
                opt_point = self.state.x01ToScaledPoint(ind.x01)
                wc_metvals = {}
                for metric in self.ps.flattenedMetrics():
                    wc_metvals[metric.name] = sum(vi**2 for vi in opt_point.values())
                ind.setWorstCaseMetricValues(wc_metvals)

            if self.state.doStop():
                break
            
            self.state.update()

        log.info("Done opt")
            

#========================================================================
class GtoInd:
    """This is an association of an opt point with metric_values, from which a cost can
    be computed.  In some ways it is a stripped-down 'Ind' object.
    """
    def __init__(self, x01, ps):
        #preconditions
        assert isinstance(ps, GtoProblemSetup) or isinstance(ps, TestPS)

        #set values
        self.ps = ps       #reference to ps, here for convenience
        
        self.x01 = x01          #x01[i] is the value of ordered_vars[i], in range [0,1]
        self._worst_case_metric_values = None # dict of metric_name : metric_value

    def setWorstCaseMetricValues(self, val):
        """Sets self's _worst_case_metric_values attribute"""
        #preconditions
        assert isinstance(val, types.DictType)
        assert set(val.keys()) == set(self.ps.flattenedMetricNames())

        #main work
        self._worst_case_metric_values = val

    def wasEvaluated(self):
        """Returns True if this ind was evaluated, i.e. has worst-case metric values"""
        return self._worst_case_metric_values is not None

    def worstCaseMetricValue(self, metric_name):
        """Returnst the worst-case value of the specified metric"""
        #preconditions
        assert self._worst_case_metric_values is not None, "cannot call this unless we know metvals"

        #main work
        return self._worst_case_metric_values[metric_name]

    def cost(self, w_per_objective):
        """Returns cost which combines objectives' cost and constraint violation"""
        #preconditions
        assert self._worst_case_metric_values is not None, "cannot call this unless we know metvals"

        #main work...
        violation = self.constraintViolation()
        
        #cost for feasible inds is <0.0, based on maximizing (weighted) margins
        if violation == 0.0:
            total_margin = 0.0
            for (obj, obj_w) in izip(self.ps.metricsWithObjectives(), w_per_objective):
                wc_value = self.worstCaseMetricValue(obj.name)
                assert wc_value is not BAD_METRIC_VALUE, \
                       "should have had violation > 0.0, and never get here"
                margin = obj.margin(wc_value)
                (mn, mx) = (obj.rough_minval, obj.rough_maxval)
                if mx > mn:
                    margin01 = margin / (mx - mn)
                    total_margin = (obj_w * margin01)
                    
            cost = -1.0 * total_margin

        #cost for infeasible inds _is_ constraint violation
        else:
            cost = violation

        return cost 

    def constraintViolation(self):
        """Returns a measure of how much this individual has violated constraints."""
        #preconditions
        assert self._worst_case_metric_values is not None, "cannot call this unless we know metvals"
        
        #corner case
        if self.isBad():
            total_violation = INF
            return total_violation

        #main case...
        total_violation = 0.0
        
        for metric in self.ps.flattenedMetrics():
            metric_value = self.worstCaseMetricValue(metric.name)
            metric_violation = metric.constraintViolation(metric_value)

            #do NOT add fixed cost because it makes the "improvement ratio" calculations less smooth
            #if metric_violation > 0.0:
            #    total_violation += 1.0

            #add variable cost
            (mn, mx) = (metric.rough_minval, metric.rough_maxval)
            if mx > mn:
                violation01 = metric_violation / (mx - mn)
                total_violation += violation01

        return total_violation
    
    def isBad(self):
        """Returns True if any of self's metric values are BAD_METRIC_VALUE.
        Ignores forced_bad.
        """
        #preconditions
        assert self._worst_case_metric_values is not None, "cannot call this unless we know metvals"

        #corner case
        #main work
        for val in self._worst_case_metric_values.itervalues():
            if val == BAD_METRIC_VALUE:
                return True
        return False

#========================================================================
class GtoState:
    """Stores the state of a Gto search.  Each call to update()
    will generate new candidate designs, which can then be simulated
    externally."""
    
    def __init__(self, ps, ss, init_opt_point):
        #condition inputs
        ss.setInitOptPoint(init_opt_point)
        
        #preconditions
        assert isinstance(ps, GtoProblemSetup) or isinstance(ps, TestPS)
        assert isinstance(ss, GtoSolutionStrategy)
        assert isinstance(ss.init_opt_point, Point)
        ss.assertConsistent()
        assert len(ss.weight_per_objective) == ps.numObjectives()

        #main work...
        self.ps = ps
        self.ss = ss
        
        self.opm = ps.opt_point_meta
        for var_meta in ps.opt_point_meta.itervalues():
            assert isinstance(var_meta, ContinuousVarMeta)
            assert var_meta.min_unscaled_value < var_meta.max_unscaled_value
        
        self.ordered_vars = sorted(self.opm.keys()) #order matters
        self.minx = [self.opm[var].min_unscaled_value for var in self.ordered_vars]
        self.maxx = [self.opm[var].max_unscaled_value for var in self.ordered_vars]
    
        self._stop = False

        self.all_inds = []

        self.action = "SAMPLE_INDS_IN_BALL"

        self.center_ind = self._createInd(self.scaledPointToX01(ss.init_opt_point))
        self.radius = ss.init_radius

        self.inner_ind = None
        self.predicted_reduction = None

        #postconditions
        assert mathutil.dictsAlmostEqual(self.x01ToScaledPoint(self.center_ind.x01),
                                         self.ss.init_opt_point, tol = 1.0e-4)
            
    def __str__(self):
        s = []
        s += ["STATE:"]
        s += [" radius=%.2e" % self.radius]
        s += ["; center_%s" % self.indCostStr(self.center_ind)]
        if self.all_inds:
            s += ["; best_%s" % self.indCostStr(self.bestInd())]
        s += ["; # opt points=%d (%d in ball)" % (self.numOptPoints(), self.numIndsInBall())]

        #output optimization values too
        s += ["; center_x01=%s" % self.center_ind.x01[:4]]
        #s += ["; center_opt_point=%s" % self.centerOptPoint()]
        
        return "".join(s)

    def indCostStr(self, ind):
        """Returns, as a string: ind's cost, and possibly other info"""
        return "cost=%.6e " % self.indCost(ind)

    #========================================================================
    def indCost(self, ind):
        """Returns cost of the ind, taking into account the ss's weight per objective"""
        return ind.cost(self.ss.weight_per_objective)

    def numVars(self):
        """Returns number of active opt vars"""
        return len(self.opm)

    def minNumIndsInBall(self):
        """Returns the minimum number of inds needed in the ball in order to build a model"""
        return self.ss.num_samples_per_active_var * self.numVars()

    def numIndsInBall(self):
        """Returns the number of inds in the ball defined by center_x01 and radius"""
        return len(self.indsInBall())

    def indsInBall(self, radius_multiplier=1.0):
        """Returns a list of the inds that are in the ball defined by center_x01 and radius.
        Expands radius by 'radius_multiplier'
        """
        return [ind for ind in self.all_inds if self.x01InBall(ind.x01, radius_multiplier)]

    def x01InBall(self, x01, radius_multiplier=1.0):
        """Returns True if ix is in the ball defined by center_x01 and radius.
        Expands radius by 'radius_multiplier'
        """
        return dist01(self.center_ind.x01, x01) <= (self.radius * radius_multiplier)

    def numOptPoints(self):
        """Returns the number of opt points (inds) covered so far"""
        return len(self.all_inds)

    def bestInd(self):
        """Returns ind with lowest cost.  Needs all inds to have been evaluated"""
        #preconditions
        for ind in self.all_inds:
            assert ind.wasEvaluated()

        #main work
        costs = [self.indCost(ind) for ind in self.all_inds]
        return self.all_inds[numpy.argmin(costs)]

    def bestOptPoint(self):
        """Returns best (scaled) opt_point seen so far"""
        return self.x01ToScaledPoint(self.bestInd().x01)
    
    def centerOptPoint(self):
        """Returns center (scaled) opt_point"""
        return self.x01ToScaledPoint(self.center_ind.x01)

    def indsWithoutCosts(self):
        """Returns the list of inds that do not have cost set yet"""
        return [ind for ind in self.all_inds if not ind.wasEvaluated()]
    
    def doStop(self):
        """Returns True is this state wants to stop"""
        if self._stop:
            log.info("Stop because user soft stop requested")
            return True

        if self.numOptPoints() >= self.ss.max_num_opt_points:
            log.info("Stop because # opt_points >= maximum allowed (=%d)" % (self.ss.max_num_opt_points))
            return True

        if (self.ss.target_cost is not None) and (self.indCost(self.bestInd()) <= self.ss.target_cost):
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
        #ROUND 0: force stop if center == BAD (should only happen if init ind == BAD)
        #===========================================
        if self.center_ind.isBad():
            log.info('Center ind == BAD, so force stop')
            self.action == "STOP_CONVERGED"
        
        #===========================================
        #ROUND 1: prepare for inds pending
        #===========================================
        if self.action == "REACT_TO_INNER_OPT":
            log.info("Respond to %s" % self.action)
            
            #compute improvement
            actual_reduction = self.indCost(self.center_ind) - self.indCost(self.inner_ind)
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
        """Sample 'num_inds' inds in the ball defined by self.center_ind.x01 and self.radius"""
        log.info("Take %d sample(s) in ball" % num_inds)
        center_x01 = self.center_ind.x01

        ##well-spread sampling of stepsizes.  The '-0.5' is so that direction is negative half the time
        S = LHS(self.numVars(), num_inds) / float(num_inds) - 0.5
        new_inds = []
        for i in range(num_inds):
            #s = steps.  Regenerate if needed, so that magnitude > 0
            s = S[:,i]
            while magnitude(s) == 0.0:
                s = numpy.array([random.random() - 0.5 for j in xrange(self.numVars())])

            #rescale s so that magnitude = 1.0
            s = s / magnitude(s)

            #create new x01
            # -rmult is so that the sample can be in the ball's interior, not  just boundary
            rmult = random.random()
            new_x01 = center_x01 + s * self.radius * rmult

            #verify
            assert self.x01InBall(new_x01)

            #rail
            for (i, xi) in enumerate(new_x01):
                new_x01[i] = max(0.0, min(1.0, xi))

            #compute new ind, and remember it
            new_ind = self._createInd(new_x01)
            new_inds.append(new_ind)

        return new_inds

    def _newIndViaInnerOptimization(self):
        """
        Does inner optimization by:
         1. Build a model using all inds in the ball defined by (center_ind, radius)
         2. Optimizes on the model to find the lowest-cost x01
         3. Creates a new ind at that x01 
         4. Returns (new_ind, predicted_reduction)
        """
        #preconditions
        assert self.numIndsInBall() >= self.minNumIndsInBall()

        #main work...
        
        #build inner model
        # -note that we grab inds that are beyond the usual ball via a larger radius
        inds = self.indsInBall(radius_multiplier=2.0)
        good_inds = [ind for ind in inds if not ind.isBad()]
        
        #build a classifier of good vs. BAD
        # -note that it automatically handles corner cases of all 1.0 or all 0.0
        log.info("New ind via inner: build classifier: of %d inds, classify %d good vs. %d BAD" %
                 (len(inds), len(good_inds), len(inds) - len(good_inds)))
        all_X01 = numpy.transpose(numpy.array([ind.x01 for ind in inds]))
        is_bad = numpy.array([(1.0 * ind.isBad()) for ind in inds])
        is_bad_classifier = SgbFactory().build(all_X01, is_bad, self.ss.sgb_ss)

        #build one regressor per metric
        regressor_per_metric = {} #dict of metric_name : regressor
        if len(good_inds) > 0:
            # -build good_X01 (same for each regressor)
            good_X01 = numpy.transpose(numpy.array([ind.x01 for ind in good_inds]))
            
            for (metric_i, metric) in enumerate(self.ps.flattenedMetrics()):
                #build y (== metric values), and determine if classification vs regression
                good_y = numpy.array([ind.worstCaseMetricValue(metric.name) for ind in good_inds])
                classify = (len(set(good_y)) == 2)

                #log
                log.info("New ind via inner: build regressor #%d / %d (%d samples; metric=%s; classify=%s)"
                         % (metric_i+1, self.ps.numMetrics(), len(good_inds), metric.name, classify))

                #build regressor (or classifier)
                #-magic number alert for next several lines
                if classify: #sgb is most appropriate
                    regressor = SgbFactory().build(good_X01, good_y, self.ss.sgb_ss)
                else:        #probe is appropriate
                    regressor = ProbeFactory().build(good_X01, good_y, self.ss.probe_ss)

                #store regressor
                regressor_per_metric[metric.name] = regressor

        #optimize on model

        # -cost function
        function = ModelEvaluatorFunction(is_bad_classifier, regressor_per_metric,
                                          self.center_ind.x01, self.radius, self.ps, self.ss)

        # -solution strategy
        #   -make it pretty much a hillclimber from starting point of center_ind
        evo_ss = EvoliteSolutionStrategy(popsize = 3, numgen=300, num_mc=0)
        evo_ss.init_x01s = [self.center_ind.x01]

        #   -_need_ to scale the mutation stepsize with the radius, otherwise steps are too big
        evo_ss.mstd *= self.radius

        # -optvar range
        min_x01 = numpy.zeros(self.numVars(), dtype=float)
        max_x01 = numpy.ones(self.numVars(), dtype=float)

        # -run opt
        inner_opt = EvoliteOptimizer(min_x01, max_x01, evo_ss, function)
        inner_opt.optimize()
        inner_x01 = inner_opt.best_x

        #compute return data
        inner_ind = self._createInd(inner_x01)
        predicted_inner_cost = function.call1(inner_x01)
        predicted_center_cost = function.call1(self.center_ind.x01)
        predicted_reduction = predicted_center_cost - predicted_inner_cost
        log.info("New ind via inner: pred. inner cost = %.6e, pred. center cost = %.6e, "
                 "pred. reduction = %.6e" %
                 (predicted_inner_cost, predicted_center_cost, predicted_reduction))

        #return
        return (inner_ind, predicted_reduction)

    def _createInd(self, x01):    
        """Creates an ind defined by 1d array x01.  Adds to self.all_inds."""
        #preconditions
        assert len(x01) == self.numVars()

        #main work
        new_ind = GtoInd(x01, self.ps)
        self.all_inds.append(new_ind)
                
        return new_ind

    def scaledPointToX01(self, scaled_point):
        """Converts an opm scaled point into 1d array 'x01' with range [0,1]"""
        assert scaled_point.is_scaled
        unscaled_point = self.opm.unscale(scaled_point)
        x01 = numpy.array([(unscaled_point[var]-mn)/(mx-mn)
                             for (var, mn, mx) in izip(self.ordered_vars, self.minx, self.maxx)])
        return x01

    def x01ToScaledPoint(self, x01):
        """Converts 1d array 'x01' with range [0,1] into a opm scaled point"""
        unscaled_point = Point(False, {})
        for (val, var, mn, mx) in izip(x01, self.ordered_vars, self.minx, self.maxx):
            unscaled_point[var] = mn + val * (mx - mn)
        scaled_point = self.opm.scale(unscaled_point)
        return scaled_point
          
#========================================================================
class ModelEvaluatorFunction:
    """
    @description

        Callable object, for use with EvoliteOptimizer.
    
    @attributes

        regressor_per_metric -- dict of metric_name : regressor.  Use these
          to get estimated values for each metric.   Each input dimension to each regressor
          is the same as the input dimensions of __call__().
        center_x01 -- 1d array of float -- with radius, defines the allowed trust region
          that we can search in
        radius -- float -- see above
        ps --
        ss --
        
    @notes
    
        Callable because it behaves like a function, thanks to use of __call__().
    
    """
    
    def __init__(self, is_bad_classifier, regressor_per_metric, center_x01, radius, ps, ss):
        #preconditions
        assert set(regressor_per_metric.keys()) == set(ps.flattenedMetricNames())
        assert len(center_x01) > 0
        assert mathutil.isNumber(radius)
        assert isinstance(ps, GtoProblemSetup) or isinstance(ps, TestPS)
        assert isinstance(ss, GtoSolutionStrategy)

        #set values
        self.is_bad_classifier = is_bad_classifier
        self.regressor_per_metric = regressor_per_metric
        self.center_x01 = center_x01
        self.radius = radius
        self.ps = ps 
        self.ss = ss

    def call1(self, x01):
        """Like __call__ but works on a single entry x01, rather than a set of entries X01"""
        X01 = numpy.reshape(x01, (len(x01), 1))
        return self.__call__(X01)[0]

    def __call__(self, X01):
        """
        @description

          Returns an estimate of cost1 for each input (column) vector x01 in X01.
          
        @arguments

          X01 -- 2d array [input variable #][sample #] - inputs to regressor

        @return

          costs1 -- 1d array [sample #] -- actual costs

        @exceptions

        @notes
        """
        #preconditions
        assert X01.shape[1] > 0

        #main work
        X01 = numpy.asarray(X01)

        # -regressors' cost
        costs1 = self._regressorCosts(X01)

        # -disallow inds from venturing beyond trust region.  Give them guidance to return to it.
        for i in range(X01.shape[1]):
            dist = dist01(self.center_x01, X01[:,i])
            if dist > self.radius:
                costs1[i] = (dist + 1.0) * 100000 #magic number alert
        
        return costs1

    def _regressorCosts(self, X01):
        #compute metric values
        # note that it only handles metrics that are only modeled by 'regressor_per_metric'
        metric_values_per_ind = [{} for i in xrange(X01.shape[1])] #list of (dict of metname: metval)
        for (metric_name, regressor) in self.regressor_per_metric.iteritems():
            values = regressor.simulate(X01)
            for (ind_i, value) in enumerate(values):
                metric_values_per_ind[ind_i][metric_name] = values[ind_i]

        #compute costs, by creating GtoInds and letting those inds do the dirty work
        cost1s = []
        for (ind_i, metric_values) in enumerate(metric_values_per_ind):
            ind = GtoInd(X01[:,i], self.ps)
            ind.setWorstCaseMetricValues(metric_values)
            cost = ind.cost(self.ss.weight_per_objective)
            cost1s.append(cost)

        #tack on cost according to 'is_bad_classifier'.  Guidelines:
        # -weight must be << trust-region weight
        # -note that the classifier itself has uncertainty, so output is real-valued in range [0.0, 1.0]
        # -don't want to penalize (much) when classifier thinks it's ok (e.g. i_is_bad < 0.5)
        # -do want to penalize a minimum amount when classifier thinks it's bad (when >= 0.5)
        #  but not too much because it may turn out to be good
        # -want to give a gradient to both cases to steer towards good
        is_bad_costs = self.is_bad_classifier.simulate(X01)
        for (i, i_is_bad) in enumerate(is_bad_costs):
            if i_is_bad < 0.5: #magic number
                cost1s[i] = cost1s[i] + 0.1 * i_is_bad #magic number
            else:
                cost1s[i] = cost1s[i] + 1.0 + 5.0 * i_is_bad #magic number

        #done
        return cost1s


#========================================================================
#Utility funcs
def dist01(x1, x2):
    """Returns the distance between vectors x1 and x2"""
    return magnitude(x1 - x2)

def magnitude(x):
    """Returns magnitude of vector x"""
    return math.sqrt(sum((xi**2) for xi in x))


#========================================================================
#Used by unit tests
class TestMetric:
    """This has just enough functionality like Metric to work with TemplateGtoOptimizer
    in unit tests"""
    def __init__(self, name):
        self.name = name
        self.rough_minval = 0.0
        self.rough_maxval = 100.0

    def constraintViolation(self, metric_value):
        return metric_value

class TestPS(GtoProblemSetup):
    """This has just enough functionality like ProblemSetup to work with TemplateGtoOptimizer
    in unit tests"""
    def __init__(self, opt_point_meta):
        metrics = [TestMetric('metric1'), TestMetric('metric2')]
        GtoProblemSetup.__init__(self, metrics, opt_point_meta)

    def metricsWithObjectives(self):
        #overrides metricsWithObjectives in GtoProblemSetup because TestPS uses simpler metrics,
        # i.e. uses TestMetric not Metric
        return []
