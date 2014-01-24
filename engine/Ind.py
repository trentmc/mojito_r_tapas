"""Ind.py

Has classes:
-Ind -- 'individual' in the search: a point in a search space, plus results
-NsgaInd -- an Ind with extra attributes for NSGA-II search

"""
import copy
import cPickle as pickle
import datetime
from itertools import izip
import socket
import time
import types
import random

from adts import *
from adts.Part import replaceAutoNodesWithXXX
from adts.Metric import MAXIMIZE, MINIMIZE
from util import mathutil
from util.ascii import arrayToAscii
from util.constants import AGGR_TEST, BAD_METRIC_VALUE, INF, AGGR_TEST, DOCS_METRIC_NAME
from util.octavecall import plotAndPause


class Ind:
    """
    @description

      An 'individual' in the search: a point in a search space, plus results
      
    @attributes

      sim_requests_made -- dict of [rnd ID][analysis ID][env point ID] : bool --
        for keeping track of if a simulation request has been made
      sim_results -- dict of [rnd ID][metric_name][env point ID] : None/metric_value
        -- for keeping track of completed simulations, and what the value was
      sim_waveforms -- dict of [rnd ID][analysis ID][env point ID] : waveforms_per_ext
        where waveforms_per_ext is either a dict of extension_str : 2d_waveforms_array, or None.
      _ps -- ProblemSetup object -- keep a reference to this in order
        to conveniently compute worst-case metric values, etc

      Goodness-related cached attributes; for each rnd_ID; only stored if fullyEvaluatedAtRndPoint(rnd_ID) == True
      _cached_fully_evaluated -- dict of [rnd_ID] : bool -- will fullyEvaluated() return True?
      _cached_wc_metvals -- dict of [rnd ID][metric_name] : worst_case_metval --
        to speed return of worstCaseMetricValue(rnd ID)
      _cached_is_feasible -- dict of [rnd ID] : is_feasible
      _cached_constraint_violation -- dict of [rnd ID][dataID] : constraintViolation()
      _cached_margins -- dict of [rnd ID] : list_of_float, where cached_margins[rnd ID][i] is
        for objective ps.metricsWithObjectives()[i] (i.e. only refers to metrics with objectives)

      Cached attribute for topology:
      _cached_topo_point -- dict of choicevar_name : choicevar_value where it is a value
        a value of 0, 1, ... when used, and a value of -1 when not used
      
    @notes

      -the sim_requests are keyed by analysis, whereas the results
       are keyed by metric.  This is for reasons of convenient access.
    """
    
    def __init__(self, unscaled_optvals, ps):
        """
        @description

            Constructor.
        
        @arguments
        
            unscaled_optvals -- list of optvar values, in the order of ps.ordered_optvars.
              The point in design space that characterizes the ind.
            ps -- ProblemSetup object -- used to initialize self.sim_requests_made and self.sim_results
        
        @return

          ind -- Ind object
    
        @exceptions
    
        @notes
          
        """
        #preconditions
        if not isinstance(ps, ProblemSetup): raise ValueError
        assert len(unscaled_optvals) == len(ps.ordered_optvars), \
               'unscaled_optvals=%s, ordered_optvars=%s' % (unscaled_optvals, ps.ordered_optvars)
        rnd_IDs = ps.devices_setup.rndIDs()
        assert isinstance(rnd_IDs, types.ListType), (rnd_IDs, type(rnd_IDs))
        for ID in rnd_IDs:
            assert isinstance(ID, types.IntType) or isinstance(ID, types.LongType), (ID, type(ID))
        if ps.doRobust():
            assert len(rnd_IDs) > 1, 'if doing robust, we expect >1 rnd point'
        else:
            assert len(rnd_IDs) == 1, 'if doing nominal, we expect only 1 rnd point'
        assert rnd_IDs[0] == ps.nominalRndPoint().ID, '0th rnd ID must be the ID of ps\' nominal rnd point'
        assert len(rnd_IDs) == len(set(rnd_IDs)), 'rnd IDs must be unique'
        
        #reference to ps (for convenience)
        self._ps = ps
        
        #set 'ID'
        self.ID = self._calculateId()

        #main 'point' in the space
        self.unscaled_optvals = unscaled_optvals

        #for tracking ancestry.  Each of these is list of IDs, via setAncestry()
        self.parent_IDs = None  #just immediate parents
        self.ancestor_IDs = None #all ancestors (incl. parents, grandparents, ..)

        #rnd point IDs
        self.rnd_IDs = rnd_IDs
        
        #sets many attributes for the first time: sim_requests_made,
        # sim_results, sim_waveforms
        self._initializeSimData()
        
        #cached calculations
        self._initializeCachedCalculations()
        
        #cached attributes not related to simulation
        self._cached_topo_point = None

    #========================================================================================================
    #manage ID
    def _calculateId(self):
        # the random number ensures that we don't have the same id for inds that are created on the same machine
        # in rapid succession
        return str("IND-%s-%04d-%s" % (socket.gethostname(), random.randint(0, 9999), str(datetime.datetime.now().time())))

    def shortID(self):
        """Returns a shortened version of self.ID, that aims to keep the ID still unique.
        -makes it parseable when loading into matlab
        -shorter!
        Approach:
        -remove '.esat.kuleuven.be'
        -remove all '.' and ':' and '-' which can mess up parsing.
        -AND removes ALL characters, actually (so that it's _easy_ to load into matlab)

        Example: 'IND-warche.esat.kuleuven.be-12:22:32.311483' => '122232311483'
        """
#         short_ID = copy.copy(self.ID)
#         short_ID = short_ID.replace('esat.kuleuven.be', '')
#         short_ID = short_ID.replace('.', '')
#         short_ID = short_ID.replace('-', '')
#         short_ID = short_ID.replace(':', '')
#         short_ID = short_ID[-12:] #assumes that the string only has 12 numeric characters at the end
        return ("%u" % (self.ID.__hash__() & 0xffffffff))

    def copyWithNewID(self):
        """Return a shallow copy of 'self' but having a new ID"""
        new_ind = Ind(self.unscaled_optvals, self._ps)
        for (key, value) in self.__dict__.iteritems():
            if key != 'ID':
                new_ind.__dict__[key] = value
        return new_ind

    def _doRobust(self):
        return self._ps.doRobust()

    #========================================================================================================
    #lowlevel caching support
    def _initializeSimData(self):
        self.sim_requests_made = {} 
        self.sim_results = {}
        self.sim_waveforms = {}
        for rnd_ID in self.rnd_IDs:
            self.sim_requests_made[rnd_ID] = {} 
            self.sim_results[rnd_ID] = {}
            self.sim_waveforms[rnd_ID] = {}
            
        for an in self._ps.analyses:
            self._initializeSimDataAtAnalysis(an)

    def _initializeSimDataAtAnalysis(self, an):
        for rnd_ID in self.rnd_IDs:
            self.sim_requests_made[rnd_ID][an.ID] = {}
            self.sim_waveforms[rnd_ID][an.ID] = {}
            for env_point in an.env_points:
                self.sim_requests_made[rnd_ID][an.ID][env_point.ID] = False
                self.sim_waveforms[rnd_ID][an.ID][env_point.ID] = None

            for metric in an.metrics:
                self.sim_results[rnd_ID][metric.name] = {}
                for env_point in an.env_points:
                    self.sim_results[rnd_ID][metric.name][env_point.ID] = None
        
    def _initializeCachedCalculations(self):
        self._cached_fully_evaluated = {}
        self._cached_wc_metvals = {}
        self._cached_is_feasible = {}
        self._cached_constraint_violation = {}
        self._cached_margins = {}
        for rnd_ID in self.rnd_IDs:
            self._cached_fully_evaluated[rnd_ID] = False
            self._cached_wc_metvals[rnd_ID] = {}
            self._cached_constraint_violation[rnd_ID] = {}
            self._cached_is_feasible[rnd_ID] = None
            self._cached_margins[rnd_ID] = None
        

    def clearSimulations(self):
        """Clear all simulations on this ind.
        """
        self._initializeSimData()
        self._initializeCachedCalculations()

    def squeeze(self):
        """ removes all attributes that are not mandatory.
            saves space on save.
        """
        self._initializeCachedCalculations()

    #========================================================================================================
    #str override
    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = "Ind={"
        s += 'ID=%s' % self.ID
        s += '; topoSummary=%s' % self.topoSummary()

        if self._doRobust(): num_list = [1, 6, 11, 16, 21, 31] #magic number
        else:                num_list = [1]
        for num_rnd_points in num_list:
            if self.fullyEvaluated(num_rnd_points):
                s += '; (#rnd_pts=%d, is_feasible=%d, wc_metvals=%s)' % \
                     (num_rnd_points, self.isFeasible(num_rnd_points), self.worstCaseMetricValuesStr(num_rnd_points))
                    
        s += '; parent_IDs=%s' % str(self.getParentIDs())
        #s += '; ancestor_IDs=%s' % str(self.getAncestorIDs())

        #s += '; sim_requests_made=FIXME'
        #s += '; sim_results=FIXME'
        
        #s += ' unscaled_optvals=%s' % self.unscaled_optvals
        s += " /Ind}"  
        return s

    #=============================================================================================================
    #manage sim requests
    def reportSimRequest(self, rnd_ID, analysis, env_point):
        """
        @description

          Reports that a request was made at (rnd_ID, analysis, env_point).
          Call this right before, or right after, you make a sim request;
          this ensures that an ind gets fully evaluated, and not doubly either.
        
        @arguments

          rnd_ID -- int --
          analysis -- Analysis -- 
          env_point -- EnvPoint x-- 
        
        @return

          <<none>> but modifies self.sim_requests_made
    
        @exceptions
    
        @notes

          Do not currently use EvalRequest objects.
        """
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs
            assert isinstance(analysis, Analysis)
            assert isinstance(env_point, EnvPoint)
            if self.sim_requests_made[rnd_ID][analysis.ID][env_point.ID]:
                raise ValueError("have previously requested this eval; can't do it again")

        #main work
        self.sim_requests_made[rnd_ID][analysis.ID][env_point.ID] = True

    def simRequestMadeAtNominalAtAllEnvPoints(self, analysis):
        """
            see simRequestMade, but only for nominal and at all env points
        """
        for e in analysis.env_points:
            if not self.simRequestMadeAtNominal(analysis, e):
                return False
        return True

    def simRequestMadeAtNominal(self, analysis, env_point):
        """
            see simRequestMade, but only for nominal
        """
        return self.simRequestMade(self.rnd_IDs[0], analysis, env_point)

    def simRequestMade(self, rnd_ID, analysis, env_point):
        """
        @description

          Reports whether or not a sim request at a particular (analysis, env_point) has been made
        
        @arguments

          rnd_ID -- int
          analysis -- Analysis -- 
          env_point -- EnvPoint -- 
        
        @return

          request_was_made -- bool
    
        @exceptions
    
        @notes

          Do not currently use EvalRequest objects.
        """
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs
            assert isinstance(analysis, Analysis)
            assert isinstance(env_point, EnvPoint)

        #main work
        return self.sim_requests_made[rnd_ID][analysis.ID][env_point.ID]

    #=============================================================================================================
    #manage sim results (incl. lis results and waveforms)
    def getSimResults(self, rnd_ID, analysis, env_point):
        """
        @description

          Returns the simulation results at this (rnd_point, analysis, env_point), i.e. one value per metric.
          Complains if we do not have any (should have checked beforehand with simRequestMade).
        
        @arguments

          rnd_ID -- int --
          analysis -- Analysis -- 
          env_point -- EnvPoint -- 
        
        @return
        
          sim_results -- dict of metric_name : metric_value 
    
        @exceptions
    
        @notes

        """
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs
            assert isinstance(analysis, Analysis)
            assert isinstance(env_point, EnvPoint)
            assert self.simRequestMade(rnd_ID, analysis, env_point)

        #main work
        sim_results = {}
        for metric in analysis.metrics:
            sim_results[metric.name] = self.sim_results[rnd_ID][metric.name][env_point.ID]

        return sim_results
    
    def setSimResults(self, sim_results, rnd_ID, analysis, env_point, waveforms_per_ext=None):
        """
        @description

          Given the sim_results from eval_request, store it
        
        @arguments

          sim_results -- dict of metric_name:metric_value (therefore we do not need the analysis to
            be identified, because every metric name is unique)
          rnd_ID -- int
          analysis -- Analysis -- 
          env_point -- EnvPoint -- 
          waveforms_per_ext -- None, or dict of extension_str : 2d_array_of_waveforms
        
        @return
    
        @exceptions
    
        @notes

          Do not currently use EvalRequest objects.
        """
        #preconditions
        if AGGR_TEST:
            assert isinstance(sim_results, types.DictType)
            assert rnd_ID in self.rnd_IDs
            assert isinstance(analysis, Analysis)
            assert isinstance(env_point, EnvPoint)
            assert (waveforms_per_ext is None) or isinstance(waveforms_per_ext, types.DictType)
            
            if not self.simRequestMade(rnd_ID, analysis, env_point):
                raise ValueError('Have to report the sim request before set results')
            for metric_name in sim_results.iterkeys():
                if self.sim_results[rnd_ID][metric_name][env_point.ID] is not None:
                    raise ValueError('Already have sim_results at (metric %s, rnd pt ID %d, env pt ID %d)' %
                                     (metric_name, rnd_ID, env_point.ID))
            if self.sim_waveforms[rnd_ID][analysis.ID][env_point.ID] is not None:
                raise ValueError('Already have sim_waveforms at (rnd point ID, analysis ID %d, env pt ID %d)' %
                                 (rnd_ID, analysis.ID, env_point.ID))

        #set results
        # -metric values
        for (metric_name, value) in sim_results.iteritems():
            self.sim_results[rnd_ID][metric_name][env_point.ID] = value

        # -waveforms
        if waveforms_per_ext is not None:
            self.sim_waveforms[rnd_ID][analysis.ID][env_point.ID] = waveforms_per_ext

    def overrideSimResults(self, sim_results, rnd_ID, analysis, env_point):
        """Use this only if you really need to.  It ignores whether or not the ind has previously
        been simulated here, and overrides those results.  Alters cache etc accordingly."""
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs
            assert isinstance(analysis, Analysis)
            assert isinstance(env_point, EnvPoint)
            
        #del old
        self.sim_requests_made[rnd_ID][analysis.ID][env_point.ID] = False
        for metric_name in sim_results.iterkeys():
            self.sim_results[rnd_ID][metric_name][env_point.ID] = None

        self._initializeCachedCalculations()

        #set new
        self.reportSimRequest(rnd_ID, analysis, env_point)
        self.setSimResults(sim_results, rnd_ID, analysis, env_point)

    def forceFullyBad(self):
        """Force all sim results to bad, across _all_ rnd IDs"""
        for rnd_ID in self.rnd_IDs:
            self.forceFullyBadAtRndPoint(rnd_ID)

    def forceFullyBadAtRndPoint(self, rnd_ID):
        """
        Puts all sim results to bad (and all sim requests to 'done').
        Therefore, by calling this, a subsequent call to fullyEvaluated()
        will return True.

        Needs to update cache too.
        """
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs

        #main work
        for anID in self.sim_requests_made[rnd_ID].iterkeys():
            for env_point_ID in self.sim_requests_made[rnd_ID][anID].iterkeys():
                self.sim_requests_made[rnd_ID][anID][env_point_ID] = True

        for metric_name in self.sim_results[rnd_ID].iterkeys():
            for env_point_ID in self.sim_results[rnd_ID][metric_name].iterkeys():
                self.sim_results[rnd_ID][metric_name][env_point_ID] = BAD_METRIC_VALUE

        self._initializeCachedCalculations()

    def numRndPointsFullyEvaluated(self):
        """Returns the num rnd points for which this ind has been fully evaluated"""
        for (num_rnd_points, rnd_ID) in enumerate(self.rnd_IDs):
            if not self.fullyEvaluatedAtRndPoint(rnd_ID):
                return num_rnd_points
        return len(self.rnd_IDs)
    
    def fullyEvaluated(self, num_rnd_points):
        """Returns True only if all of the specified rnd corners are fully evaluated"""
        #preconditions
        if AGGR_TEST:
            assert num_rnd_points > 0, "must request _some_ rnd points"
            assert (num_rnd_points == 1) or self._doRobust(), "can't request >1 rnd point if s is only nominal"

        #main work
        for rnd_ID in self.rnd_IDs[:num_rnd_points]:
            if not self.fullyEvaluatedAtRndPoint(rnd_ID):
                return False
        return True
        
    def fullyEvaluatedAtRndPoint(self, rnd_ID):
        """Returns true if this ind has had sim_requests_made at all possible places at rnd point.

        Note that the caching here is different than other caching.  The cache is only used when
        fullyEvaluated() would return True, but not when False.  That's fine, because we get to True quickly,
        and never revert.
        """
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs
            
        #exploit cache?
        if self._cached_fully_evaluated[rnd_ID]:
            return True
        
        #main work
        for requests_at_analysis in self.sim_requests_made[rnd_ID].itervalues():
            for req_made in requests_at_analysis.itervalues():
                #(no need to update cached value, it's still False)
                if not req_made:
                    return False

        #if we get here, then we can also change the cached value to True
        self._cached_fully_evaluated[rnd_ID] = True
        return True
    
    #================================================================================================
    #get metric values & related        
    def nominalWorstCaseMetricValue(self, metric_name):
        return self.worstCaseMetricValue(num_rnd_points=1, metric_name=metric_name)
        
    def worstCaseMetricValue(self, num_rnd_points, metric_name):
        """Return the worst-case metric value across the specified number of rnd points"""
        #preconditions
        if AGGR_TEST:
            assert num_rnd_points > 0, "must request _some_ rnd points"
            assert (num_rnd_points == 1) or self._doRobust(), "can't request >1 rnd point if s is only nominal"

        #main work
        values = [self.worstCaseMetricValueAtRndPoint(rnd_ID, metric_name)
                  for rnd_ID in self.rnd_IDs[:num_rnd_points]]
        assert None not in values, "should have evaluated at each of the rnd points"
        
        metric = self._ps.metric(metric_name)
        wc_value = metric.worstCaseValue(values)

        #done
        return wc_value
        
    def worstCaseMetricValueAtRndPoint(self, rnd_ID, metric_name):
        """Returns worst-case metric value at 'metric_name' and rnd_ID
        which is found by aggregating across all env points at that metric.

        Handles BAD_METRIC_VALUE too
        
        For safety and simplicity, only cache if fully evaluated
         (note that BAD inds return True for fullyEvaluated())
        """
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs
            assert metric_name in self._ps.flattenedMetricNames()
            
        #exploit cache? use 'try' rather than if/then for speed
        try:
            return self._cached_wc_metvals[rnd_ID][metric_name]
        except:
            pass

        #main case: do the computation to find 'wc_metval'
        # -note that the work is done here rather than Metric.worstCaseValue() for two reasons:
        #  1. so that we avoid None values here
        #  2. speed (incl. so that extra data structures are not needed)
        wc_metval = None
        metric = self._ps.metric(metric_name)
        aim = metric._aim
        
        # case: bad value
        if self._metricHasBadValueAtRndPoint(rnd_ID, metric_name):
            wc_metval = BAD_METRIC_VALUE

        # case: minimize
        elif aim == MINIMIZE:
            for metval in self.sim_results[rnd_ID][metric_name].itervalues():
                if metval is None:
                    pass
                elif (wc_metval is None) or (metval > wc_metval):
                    wc_metval = metval

        # case: maximize
        elif aim == MAXIMIZE:
            for metval in self.sim_results[rnd_ID][metric_name].itervalues():
                if metval is None:
                    pass
                elif (wc_metval is None) or (metval < wc_metval):
                    wc_metval = metval

        # case: range
        else:
            for metval in self.sim_results[rnd_ID][metric_name].itervalues():
                if metval is None:
                    pass
                elif wc_metval is None:
                    wc_metval = metval
                    wc_margin = min(wc_metval - metric.min_threshold,
                                    metric.max_threshold - wc_metval)
                else:
                    margin = min(metval - metric.min_threshold,
                                 metric.max_threshold - metval)
                    if margin < wc_margin:
                        wc_metval = metval
                        wc_margin = margin

        #cache
        if self.fullyEvaluatedAtRndPoint(rnd_ID):
            self._cached_wc_metvals[rnd_ID][metric_name] = wc_metval

        #done!
        return wc_metval

    def _metricHasBadValueAtRndPoint(self, rnd_ID, metric_name):
        """Returns True if any of self's results are BAD_METRIC_VALUE
        at the specified metric name and rnd_ID"""
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs
            assert metric_name in self._ps.flattenedMetricNames()
            assert self.sim_results[rnd_ID].has_key(metric_name)

        #metric names
        for metval in self.sim_results[rnd_ID][metric_name].itervalues():
            if metval == BAD_METRIC_VALUE:
                return True
        return False        

    def nominalWorstCaseMetricValuesStr(self):
        return self.worstCaseMetricValuesStr(num_rnd_points=1)
    
    def worstCaseMetricValuesStr(self, num_rnd_points):
        """Returns a string that prints out the metric values in a 'nice' way"""
        return mathutil.niceValuesStr(self.worstCaseMetricValues(num_rnd_points))

    def worstCaseMetricValues(self, num_rnd_points):
        """Returns dict of metric_name : worst_case_metric_value."""
        d = {}
        for metric_name in self._ps.flattenedMetricNames():
            d[metric_name] = self.worstCaseMetricValue(num_rnd_points, metric_name)
        return d

    def nominalEvaluatedWorstCaseMetricValues(self):
        return self.evaluatedWorstCaseMetricValues(num_rnd_points=1)
    
    def evaluatedWorstCaseMetricValues(self, num_rnd_points):
        """Returns dict of metric_name : worst_case_metric_value when the metric
        value is not None, i.e. when it has been evaluated"""
        d = {}
        for metric_name in self._ps.flattenedMetricNames():
            wc_metval = self.worstCaseMetricValue(num_rnd_points, metric_name)
            if wc_metval is not None:
                d[metric_name] = wc_metval
        return d

    #================================================================================================
    #computations on metric values: cost, constraint violation, == BAD, is feasible
    def scalarCost(self, num_rnd_points, w_per_objective, metric_weights=None, metric_bounds = None):
        """Returns cost which combines objective cost and constraint violation, for the given num_rnd_points
        """
        #preconditions
        if AGGR_TEST:
            assert num_rnd_points > 0, "must request _some_ rnd points"
            assert (num_rnd_points == 1) or self._doRobust(), "can't request >1 rnd point if s is only nominal"

        #main work...

        #cost for feasible inds is <0.0, based on maximizing (weighted) margins
        if self.isFeasible(num_rnd_points):
            if metric_bounds:
                cost = 0.0
                for rnd_ID in self.rnd_IDs[:num_rnd_points]:
                    for (obj, obj_w) in izip(self._ps.metricsWithObjectives(), w_per_objective):
                        metricvalue = self.worstCaseMetricValueAtRndPoint(rnd_ID, obj.name)
                        mn, mx = metric_bounds[obj.name][0], metric_bounds[obj.name][1]
                        if mx > mn:
                            if obj.min_threshold == -INF: # minimize
                                metricvalue01 = 1-((metricvalue-mn) / (mx - mn))
                            elif obj.max_threshold == INF: # maximize
                                metricvalue01 = ((metricvalue-mn) / (mx - mn))
                            else:
                                print "%s not an objective" % obj.name
#                             print "for %s: scaling metricvalue from %s to %s" % (obj.name, metricvalue, metricvalue01)
                            cost -= (obj_w * metricvalue01)
                        elif mx == mn:
                            cost -= 1 # FIXME
                        else:
                            print "err: min (%s) > max (%s) for %s" % (mx, mn, obj.name)
            else:
                cost = 0.0
                for rnd_ID in self.rnd_IDs[:num_rnd_points]:
                    for (obj, obj_w) in izip(self._ps.metricsWithObjectives(), w_per_objective):
                        margin = obj.margin(self.worstCaseMetricValueAtRndPoint(rnd_ID, obj.name))
                        mn, mx = obj.rough_minval, obj.rough_maxval
                        if mx > mn:
                            margin01 = margin / (mx - mn)
                            print "for %s: correcting margin from %s to %s" % (obj.name, margin, margin01)
                            cost -= (obj_w * margin01)

        #cost for infeasible inds is >0.0, based on constraint violation
        else:
            cost = self.constraintViolation(num_rnd_points, metric_weights)

        return cost

    def nominalConstraintViolation(self, metric_weights=None, dataID = None):
        return self.constraintViolation(1, metric_weights, dataID)

    def constraintViolation(self, num_rnd_points, metric_weights=None, dataID = None):
        """
          Returns a measure of how much this individual has violated constraints, for the given num_rnd_points.
          Violation = sum of violations across rnd points.
        """
        #preconditions
        if AGGR_TEST:
            assert num_rnd_points > 0, "must request _some_ rnd points"
            assert (num_rnd_points == 1) or self._doRobust(), "can't request >1 rnd point if s is only nominal"

        #corner case
        if self.isBad(num_rnd_points):
            return INF
        
        #main work...
        violation = 0.0
        for rnd_ID in self.rnd_IDs[:num_rnd_points]:
            violation += self.constraintViolationAtRndPoint(rnd_ID, metric_weights, dataID)
            if violation == INF:
                break #early stop
            
        return violation
        
    def constraintViolationAtRndPoint(self, rnd_ID, metric_weights=None, dataID=None):
        """
        @description

          Returns a measure of how much this individual has violated constraints, for the given num_rnd_points.
          Violation is 0.0 if the individual is feasible, and >0 if infeasible.

          Violation has two parts:
          -a fixed value, only dependent on if constraint is met or not
          -a scaled value, based on degree of violation

          The reason for two parts is so that no single constraint overly dominates
          for trying to solve many infeasible constraints (the fixed part), yet
          allowing for differentiation among Inds with the same # constraints violated
          (the scaled part).
        
        @arguments

          rnd_ID -- int -- identifies rnd point
          metric_weights -- dict of metric_name: metric_weight.  If a metric does not have an entry,
            its value is 1.0.  Default of None means 1.0 for all metrics.  Values larger than 1.0 mean
            that the metric gets emphasized more in the sum of violations.
          dataID -- if None, won't cache.  If not None, it will cache such that subsequent calls to this
            routine with this dataID will use the cached value (and ignore metric_weights)
        
        @return

          violation -- float -- 
    
        @exceptions

          If any of this ind's worst-case metric values are BAD_METRIC_VALUE
          then this routine will return 'Inf' (ie ind.isBadAtRndPoint())
          
        @notes

          Caching here doesn't care if the ind is fully evaluated or not,
          because it changes anyway depending on dataID.
        """        
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs
            
        #exploit cache?
        if (dataID is not None) and self._cached_constraint_violation[rnd_ID].has_key(dataID):
            return self._cached_constraint_violation[rnd_ID][dataID]

        #corner case
        if self.isBadAtRndPoint(rnd_ID):
            total_violation = INF
            self._cached_constraint_violation[rnd_ID][dataID] = total_violation
            return total_violation

        #main case...
        total_violation = 0.0
        if metric_weights is None: metric_weights = {}
        
        for metric in self._ps.flattenedMetrics():
            metric_value = self.worstCaseMetricValueAtRndPoint(rnd_ID, metric.name)
            metric_violation01 = metric.constraintViolation01(metric_value)

            if not metric_weights.has_key(metric.name):
                metric_w = 1.0
            else:
                metric_w = metric_weights[metric.name]
                    
            #add fixed cost
            if metric_violation01 > 0.0:
                total_violation += metric_w * 10.0 #magic number alert.

            #add scaled cost
            total_violation += metric_w * metric_violation01

        #cache and return
        if self.fullyEvaluatedAtRndPoint(rnd_ID):
            self._cached_constraint_violation[rnd_ID][dataID] = total_violation
        return total_violation

    def nominalIsBad(self):
        """Returns True if this ind is 'bad' at nominal rnd point"""
        return self.isBadAtRndPoint(self.rnd_IDs[0])

    def isBad(self, num_rnd_points):
        """Returns True if this ind is 'bad' at _any_ of the rnd points specified"""
        for rnd_ID in self.rnd_IDs[:num_rnd_points]:
            if self.isBadAtRndPoint(rnd_ID):
                return True
        return False
    
    def isBadAtRndPoint(self, rnd_ID):
        """
        @description

          An Ind is 'bad' if _any_ of the sim results results back so far have a value of BAD_METRIC_VALUE.
        
        @arguments

          <<none>>
        
        @return

          is_bad -- bool
    
        @exceptions
    
        @notes
        """
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs

        #main work
        for results_at_metric in self.sim_results[rnd_ID].itervalues():
            for val in results_at_metric.itervalues():
                if val == BAD_METRIC_VALUE:
                    return True
        return False

    def isFeasibleAtNominal(self):
        """Returns True if feasible at nominal rnd point"""
        return self.isFeasible(num_rnd_points=1)

    def isFeasible(self, num_rnd_points):
        """Returns True only if this ind is feasible on _all_ the rnd_corners specified"""
        #preconditions
        if AGGR_TEST:
            assert num_rnd_points > 0, "must request _some_ rnd points"
            assert (num_rnd_points == 1) or self._doRobust(), "can't request >1 rnd point if s is only nominal"

        #main work
        for rnd_ID in self.rnd_IDs[:num_rnd_points]:
            if not self.isFeasibleAtRndPoint(rnd_ID):
                return False
        return True

    def isFeasibleOnConstraintsAtRndPoint(self, rnd_ID, metrics):
        """Like isFeasibleAtRndPoint, but only checks on the specified metrics,
        and has no caching"""
        for metric in metrics:
            metric_value = self.worstCaseMetricValueAtRndPoint(rnd_ID, metric.name)
            if not metric.isFeasible(metric_value):
                return False
        return True
        
    def isFeasibleAtRndPoint(self, rnd_ID):
        """
        @description

          Does this individual meet all constraints at the specified rnd_corners?
        
        @arguments

          rnd_ID -- int
        
        @return

          is_feasible -- bool
    
        @exceptions

          Handles BAD_METRIC_VALUEs because Metric handles them
    
        @notes

          Currently assumes that the Ind has been simulated at least
          once on each analysis

          Only caches if the ind has been fully evaluated (for safety)
        """      
        #preconditions
        if AGGR_TEST:
            assert rnd_ID in self.rnd_IDs
            
        #exploit cache?
        if self._cached_is_feasible[rnd_ID] is not None:
            #assert self.fullyEvaluated(), "should only cache if fully eval'd" #turn off for speed
            return self._cached_is_feasible[rnd_ID]

        #main work
        is_feasible = True
        for metric in self._ps.flattenedMetrics():
            metric_value = self.worstCaseMetricValueAtRndPoint(rnd_ID, metric.name)
            if not metric.isFeasible(metric_value):
                is_feasible = False
                break

        #cache
        if self.fullyEvaluatedAtRndPoint(rnd_ID):
            self._cached_is_feasible[rnd_ID] = is_feasible

        #return
        return is_feasible

    #=====================================================================================================
    #pickling support
    def prepareForPickle(self):
        """Clears attributes that cause pickling problems (and therefore network
        transport problems too.)

        Specifically:
        -clears self._ps
        -makes sure there are no BAD_METRIC_VALUE in any results (that won't pickle either)
        """
        self._ps = None

        #postconditions: see if we can actually pickle this ind
        #-'dumps' pickles the object as a string, instead of writing it to a file.
        #-if it fails, then it will raise a PicklingError and specify where the error is
        test_str = pickle.dumps(self)

    def restoreFromPickle(self, ps):
        """Restores the attributes on self that were cleared for pickling.
        """
        self._ps = ps

    #=====================================================================================================
    #set / get point
    def setValues(self, unscaled_values):
        assert len(unscaled_values) == len(self._ps.ordered_optvars) == len(self._ps.embedded_part.point_meta)
        self.unscaled_optvals = unscaled_values
        self._initializeSimData()
        self._initializeCachedCalculations()
        
    def getValues(self):
        return self.unscaled_optvals
    
    def pointSummary(self):
        """Reports a summary of self's point in an easy to read format.
        
        Returns a string
        
        """
        scaled_point = self._ps.scaledPoint( self )
        unscaled_point = dict(zip(self._ps.ordered_optvars, self.unscaled_optvals))
        varnames = self._ps.ordered_optvars
        s = ""
        s += "Current Point\n"
        s += "---------------------------------------------------------------------------------------\n"
        s += "  %3s  %25s  %20s %20s\n" % ('idx', 'Variable name', 'Scaled value', 'Unscaled value' )
        for itm in scaled_point.keys():
            if type(scaled_point[itm]) is float:
                s += " (%3d) %25s: %20e %20e\n" % (varnames.index(itm), itm, scaled_point[itm], unscaled_point[itm])
            elif type(scaled_point[itm]) is int:
                s += " (%3d) %25s: %20d %20d\n" % (varnames.index(itm), itm, scaled_point[itm], unscaled_point[itm])
            else:
                s += " (%3d) %25s: %20s %20s\n" % (varnames.index(itm), itm, str(scaled_point[itm]),
                                                   str(unscaled_point[itm]))
        return s
    
    #=====================================================================================================
    #topology info
    def topoPoint(self):
        """Returns a dict of choicevar : value, where value is 0,1,... depending
        on the choice; OR value of -1 if the choice did not matter.
        """
        #use cached if possible
        if getattr(self, '_cached_topo_point', None) is not None:
            return self._cached_topo_point

        #main work
        emb_part = self._ps.embedded_part
        scaled_point = self._ps.scaledPoint(self)
        vars_used = emb_part.varsUsed(scaled_point)
        choice_vars = emb_part.part.point_meta.choiceVars()

        topo_point = {}
        for (var, unscaled_val) in izip(self._ps.ordered_optvars, self.unscaled_optvals):
            if var in choice_vars:
                if var in vars_used:
                    topo_point[var] = unscaled_val
                else:
                    topo_point[var] = -1

        #save cached
        self._cached_topo_point = topo_point

        #done
        return topo_point

    def topoSummary(self):
        """Reports a summary of self's topology, in a fashion
        that makes it easy to identify the uniqueness of this topology
        vs. other topologies.
        For each choice_variable (in order of sorted choice var names):
        -if variable has no effect, list 'X_'
        -if variable has effect, list its value + '_'
        """
        topo_point = self.topoPoint()

        s = ""
        for var in self._ps.ordered_optvars:
            if topo_point.has_key(var):
                if topo_point[var] >= 0:
                    s += str(topo_point[var])
                else:
                    s += "X"
        return s

    def topoSummaryKey(self):
        """Reports a summary of self's topology, in a fashion
        that makes it easy to identify the uniqueness of this topology
        vs. other topologies.
        Gives a numeric value that is unique for this topology
        For each choice_variable (in order of sorted choice var names):
        -if variable has no effect, list 'X_'
        -if variable has effect, list its value + '_'
        """
        topo_point = self.topoPoint()

        s = "8" # to keep leading zeros
        for var in self._ps.ordered_optvars:
            if topo_point.has_key(var):
                if topo_point[var] >= 0:
                    s += str(topo_point[var])
                else:
                    s += "9"

        return eval(s)
    
    #=====================================================================================================
    #netlist
    def nominalNetlist(self, annotate_bb_info=False, add_infostring=False):
        """Returns the netlist that this ind's unscaled_optvals
        """
        emb_part = self._ps.embedded_part
        old_functions = emb_part.functions #save
        
        emb_part.functions = self._ps.scaledPoint(self)
        variation_data = (self._ps.nominalRndPoint(), EnvPoint(True), self._ps.devices_setup)
        netlist = emb_part.spiceNetlistStr(annotate_bb_info, add_infostring, variation_data)
        
        emb_part.functions = old_functions #restore
        return netlist
    
    #=====================================================================================================
    #novelty info
    def novelty(self):
        """Returns this ind's novelty.  Does not need to have previously
        evaluated novelty"""
        emb_part = self._ps.embedded_part
        scaled_point = self._ps.scaledPoint(self)
        novelty = emb_part.novelty(scaled_point)
        return novelty

    def noveltySummary(self):
        """Gives a string describing how this ind is novel."""

        #set value of 'novelty'.  Use prior data if available.
        novelty = 0
        rnd_ID = self.rnd_IDs[0]
        if self.fullyEvaluated(1):
            metrics = self.worstCaseMetricValues(1)
            for metname, metval in metrics.iteritems():
                if 'novelty' in metname:
                    novelty = metval
                    break

        if (novelty is None) or (novelty == BAD_METRIC_VALUE):
            novelty = self.novelty()

        #corner case: no novelty
        if novelty == 0:
            return "Novelty = 0"

        #main case: have some novelty.
        # -for each novel emb_part
        #    print part (but not sub parts)
        #    print part if no novelty
        emb_part = self._ps.embedded_part
        scaled_point = self._ps.scaledPoint(self)
        
        info_list = emb_part.subPartsInfo(scaled_point, only_novel_subparts = True)

        s = 'Novelty = %d\n' % novelty
            
        s += '---------------------------------------------\n'
        s += 'Embedded parts with novelty:\n'
        unique_novel_parts, unique_novel_part_IDs = [], []
        unique_nonnovel_parts, unique_nonnovel_part_IDs = [], []
        for (sub_emb_part, sub_point, vars_used_by_sub) in info_list:
            s += str(sub_emb_part)
            
            #e.g. of next line'; choice_of_part43 = 368'
            assert isinstance(sub_emb_part.part, FlexPart)
            varmeta = sub_emb_part.part.choiceVarMeta()
            index = sub_point[varmeta.name]
            s += '; %s = %d' % (varmeta.name, index)
            s += '\n'

            #the following line is equivalent to FlexPart.chosenPart(sub_point)
            novel_part = sub_emb_part.part.part_choices[index].part

            #part index=0 is always the original "wrapped" part
            nonnovel_part = sub_emb_part.part.part_choices[0].part
            if novel_part.ID not in unique_novel_part_IDs:
                unique_novel_parts.append(novel_part)
                unique_novel_part_IDs.append(novel_part.ID)
                
                unique_nonnovel_parts.append(nonnovel_part)
                unique_nonnovel_part_IDs.append(nonnovel_part.ID)
        s += '\n'

        s += '---------------------------------------------\n'
        s += 'The novel/nonnovel parts themselves:\n'
        for novel_part, nonnovel_part in zip(unique_novel_parts,
                                             unique_nonnovel_parts):
            s += 'Novel part ID=%d, non-novel equivalent ID=%d\n\n' % \
                 (novel_part.ID, nonnovel_part.ID)
            s += '--------\n'
            s += 'Novel part (ID=%d) details:\n%s\n' % \
                 (novel_part.ID, novel_part.str2())
            s += '--------\n'
            s += 'Nonnovel part (ID=%d) details:\n%s\n' % \
                 (nonnovel_part.ID, nonnovel_part.str2())
            s += '--------\n'
            s += '--------\n'
        s += '\n'
        s += '---------------------------------------------\n'

        return s


    #================================================================================================
    #manage waveforms
    def getSimWaveforms(self, rnd_ID, analysis, env_point):
        """
        @description

          Returns the simulation waveforms at this analysis and env_point.
          Complains if we do not have any (should have checked beforehand
          with simRequestMade).
        
        @arguments

          rnd_ID -- int --
          analysis -- Analysis --
          env_point -- EnvPoint -- 
        
        @return
        
          waveforms_per_ext -- None, or dict of extension_str : 2d_array
            Where extension_str can be: 'sw0' (from dc), 'tr0' (from transient)
    
        @exceptions
    
        @notes

        """
        return self.sim_waveforms[rnd_ID][analysis.ID][env_point.ID]

    def waveformsToFiles(self, output_base):
        """Output all non-none Waveforms to a set of files on disk, and returns
        the filenames.  The waveforms come from two places:
        -reference waveforms come from ps.analyses[i].reference_waveforms
        -this ind's waveforms come from self.getSimWaveforms()
        
        Example:

        Inputs:
        -output_base = '/tmp/best_waveforms'
        -one analysis has ID 10; it's dc so it has 'sw0' waveform file extension
        -one analysis has ID 11; its tran so it has 'tr0' waveform file extension
        -one analysis has ID 12; but it does not have waveform outputs
        -each analysis has two env points with ID 98,99
        -only analysis 11 has reference waveforms

        Outputs files (each has a 2d array of waveforms):
         /tmp/best_waveforms_an10_env98_sw0.txt
         /tmp/best_waveforms_an10_env99_sw0.txt
         /tmp/best_waveforms_reference_an11.txt
         /tmp/best_waveforms_an11_env98_tr0.txt
         /tmp/best_waveforms_an11_env99_tr0.txt

         HACK: for now, it only outputs the nominal rnd point's waveforms
        """
        outfiles = []
        last_ref_waveforms, last_ind_waveforms = None, None
        rnd_ID = self.rnd_IDs[0] #HACK
        for an in self._ps.analyses:
            if an.reference_waveforms is not None:
                filename = output_base + '_reference_an%d.txt' % (an.ID)
                arrayToAscii(filename, an.reference_waveforms)
                outfiles.append(filename)
                last_ref_waveforms = an.reference_waveforms
            
            for e in an.env_points:
                waveforms_per_ext = self.getSimWaveforms(rnd_ID, an, e)
                if waveforms_per_ext:
                    for (ext, waveforms) in waveforms_per_ext.iteritems():
                        if waveforms is not None:
                            filename = output_base + '_an%d_env%d_%s.txt' % (an.ID, e.ID, ext)
                            arrayToAscii(filename, waveforms)
                            outfiles.append(filename)
                            last_ind_waveforms = waveforms

        #to make True is a HACK (but great for debugging!)
        do_plot_waveforms = False
        
        if do_plot_waveforms: 
            #Currently tuned to minimizeNmse, but changeable
            if last_ind_waveforms and last_ref_waveforms:
                x = last_ind_waveforms[0,:];
                yhat = last_ind_waveforms[1,:]
                y = last_ref_waveforms[0,:]
                plotAndPause(x, yhat, x, y, xlabel='Input DC voltage', ylabel='Output DC voltage')

        return outfiles

        
    #=====================================================================================================
    #set / get ancestry
    def setAncestry(self, parents):
        """Set self.parent_IDs, self.ancestor_IDs; given a list of parent inds"""
        self.parent_IDs = []
        self.ancestor_IDs = []
        for parent in parents:
            self.parent_IDs.append(parent.ID)
            self.ancestor_IDs.append(parent.ID)
            if parent.getAncestorIDs() != None:
                self.ancestor_IDs.extend(parent.getAncestorIDs())

        #unique-ify
        self.ancestor_IDs = list(set(self.ancestor_IDs))

    def getParentIDs(self):
        """Return list of parent IDs.  Backwards-compatible."""
        return self.parent_IDs

    def getAncestorIDs(self):
        """Return list of ancestor IDs.  Backwards-compatible."""
        return self.ancestor_IDs

    #=============================================================================================================
    #for nondominated filtering & sorting

    def nominalConstrainedDominates(self, ind_b, metric_weights=None, dataID=None):
        """
        @description

          Returns True if any of the following are True:
          1. ind_a (=self) is feasible and ind_b is not
          2. ind_a and ind_b are both infeasible, but ind_a has a smaller
             overall constraint violation
          3. ind_a and ind_b are both feasible, and ind_a dominates ind_b.

          Remember that 'feasible' means meets _all_ constraints.
          
          If metric_weights are set to None, it ignores constraints, i.e. only considers objectives.
        
        @arguments

          self==ind_a -- Ind object
          ind_b -- Ind object
          metric_weights -- see constraintViolation()
          dataID -- if None, won't cache constraintViolation().
            If not None, it will cache such that subsequent calls to this routine with this dataID
            will use the cached value (and ignore metric_weights)
        
        @return

          a_dominates -- bool
    
        @exceptions
    
        @notes

          Currently assumes that the Ind has been simulated at least once on each analysis 
        """
        assert len(self.rnd_IDs) == 1, "only call this for nominal"
        
        num_rnd_points = 1
        ind_a = self

        feasible_a = ind_a.isFeasibleAtNominal()
        feasible_b = ind_b.isFeasibleAtNominal()

        #case 1
        if feasible_a and not feasible_b:
            return True

        elif not feasible_a and feasible_b:
            return False

        #corner case: ignore constraints
        if (metric_weights is None):
            ## FIXME: borks on BAD_METRIC_VALUE
            return ind_a.nominalDominates(ind_b)

        #case 2
        elif not feasible_a and not feasible_b:
            if (ind_a.constraintViolation(num_rnd_points, metric_weights, dataID) < \
                ind_b.constraintViolation(num_rnd_points, metric_weights, dataID)):
                return True
            else:
                return False

        #case 3
        elif feasible_a and feasible_b and ind_a.nominalDominates(ind_b):
            return True

        #default case
        else:
            return False

    def nominalDominates(self, ind_b):
        """
        @description

          Returns True if ind_a=self dominates ind_b, and False otherwise.

          'Dominates' means that ind_a is better than ind_b in at least
          one objective; and for remaining objectives, at least equal.

          Does not concern itself with metrics that have no objectives
          (ie metric.improve_past_feasible is False)
        
        @arguments

          num_rnd_points -- int in [1, max_num_rnd_points] -- 
          self==ind_a -- Ind object
          ind_b -- Ind object
        
        @return

          a_dominates -- bool
    
        @exceptions

          Can only call this if ind is feasible, because nominalConstrainedDominates()
          should handle the cases where the ind is infeasible
    
        @notes
        """
        assert len(self.rnd_IDs) == 1, "only call this for nominal"
        
        #assert self.isFeasible(num_rnd_points) #turn off for speed
        num_rnd_points = 1
        nom_rnd_ID = self.rnd_IDs[0]
        ind_a = self

        #compute (or retrieve) margins
        if ind_a._cached_margins[nom_rnd_ID] is None:
            ind_a_margins = [
                self._margin_onFeasible_onObjective(ind_a.worstCaseMetricValue(num_rnd_points, metric.name), metric)
                for metric in self._ps._metrics_with_objectives]
        else:
            ind_a_margins = ind_a._cached_margins[nom_rnd_ID]
            
        if ind_b._cached_margins[nom_rnd_ID] is None:
            ind_b_margins = [
                self._margin_onFeasible_onObjective(ind_b.worstCaseMetricValue(num_rnd_points, metric.name),  metric)
                for metric in self._ps._metrics_with_objectives]
        else:
            ind_b_margins = ind_b._cached_margins[nom_rnd_ID]

        #now determine if dominates
        found_better = False
        for (margin_a, margin_b) in izip(ind_a_margins, ind_b_margins):
            if margin_b > margin_a:
                return False
            if (not found_better) and (margin_a > margin_b):
                found_better = True

        #cache if possible
        if (ind_a._cached_margins[nom_rnd_ID] is None) and ind_a.fullyEvaluated(num_rnd_points):
            ind_a._cached_margins[nom_rnd_ID] = ind_a_margins
        if (ind_b._cached_margins[nom_rnd_ID] is None) and ind_b.fullyEvaluated(num_rnd_points):
            ind_b._cached_margins[nom_rnd_ID] = ind_b_margins

        return found_better

    def _margin_onFeasible_onObjective(self, metric_value, metric):
        """Returns margin for metric_value.  Assumes that the metric_value is
        feasible, and that the metric is an objective. (for speed)
        """
        try:
            return min(metric_value - metric.min_threshold, metric.max_threshold - metric_value)
        except:
            import pdb;pdb.set_trace()

class NsgaInd(Ind):
    """
    @description

      Like an 'Ind' but has extra attributes that NSGA-II and ALPS need
      to perform search.
      
    @attributes

      genetic_age -- int -- how many generations the genetic material of this
        ind have been around (NOT the number of generations that _this_ ind
        has been around).  Randomly-generated inds have a genetic_age of 0.
        Inds created from parent(s) have a genetic_age = max genetic_age
        of parent(s) + 1.
      n -- int -- domination count of ind 'p', ie # inds which dominate p
      S -- list of ind -- the inds that this ind dominates (in current iter)
      rank -- int -- rank 0 means it's in the 0th nondominated layer;
        rank 1 means 1st nondominated layer; etc
      distance -- float -- crowding distance; a higher value means
        that the ind has a better chance of being selected because
        it means that the ind is less crowded (and therefore more unique)
      
    @notes
    """
    def __init__(self, unscaled_optvals, ps):
        Ind.__init__(self, unscaled_optvals, ps)

        #Note that genetic age needs to be set later
        self.genetic_age = None

        self.n = None
        self.S = None
        self.rank = None
        self.distance = None

    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = "NsgaInd={"
        s += ' parent_class_info=%s' % Ind.__str__(self)
        s += '; genetic_age=%s' % self.genetic_age
        #s += '; n=%s' % self.n
        #s += '; S=%s' % self.S
        #s += '; rank=%s' % self.rank
        s += '; distance=%s' % self.distance
        s += " /NsgaInd}"  
        return s

    #=====================================================================================================
    #pickling support
    def prepareForPickle(self):
        """Clears attributes that cause pickling problems (and therefore network
        transport problems too.)

        Specifically:
        -clears self.S
        """

        self.S = None

        # propagate to parent class
        Ind.prepareForPickle(self)

    def restoreFromPickle(self, ps):
        """Restores the attributes on self that were cleared for pickling.
        """
        # propagate to parent class
        Ind.restoreFromPickle(self, ps)

    def copyWithNewID(self):
        """Return a shallow copy of 'self' but having a new ID"""
        new_ind = NsgaInd(self.unscaled_optvals, self._ps)
        for (key, value) in self.__dict__.iteritems():
            if key != 'ID':
                new_ind.__dict__[key] = value
        return new_ind

        
