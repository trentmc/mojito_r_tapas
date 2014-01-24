"""EngineUtils.py

Utilities useful for different synthesis engines and components.

Includes:
-AgeLayeredPop
-distributedObjectiveWeights, LHS
-SynthState, loadSynthState
-uniqueIndsByPerformance
-uniqueIndsSubsetByPerformance
-nondominatedFilter, hierNondominatedFilter, nondominatedSort
-orderViaAverageRankingOnFront
-UniqueIDFactory
-coststr
"""
import copy
import cPickle as pickle
from itertools import izip
import logging
import os
import operator
import random
import sys
import types

import numpy

from adts import *
from adts.Metric import MAXIMIZE, MINIMIZE, IN_RANGE
from engine.Ind import Ind
from engine.PopulationSummary import \
     worstCasePopulationSummaryStr, worstCasePopulationSummaryToMatlab, \
     yieldPopulationSummaryStr, yieldPopulationSummaryToMatlab
from regressor.Bottomup import Bottomup, BottomupStrategy
from util import mathutil
from util.constants import BAD_METRIC_VALUE, INF, AGGR_TEST

log = logging.getLogger('engine_utils')

class AgeLayeredPop(list):
    """
    @description

      Holds individuals, stratified by genetic_age.
      
    @attributes

      <list> -- list of list_of_NsgaInd where entry i is for age-level i
      
    @notes
    """
    def __init__(self, R_per_age_layer=None):
        if R_per_age_layer is None:
            R_per_age_layer = []
        list.__init__(self, R_per_age_layer)

    def numAgeLayers(self):
        return len(self)

    def numInds(self):
        n = 0
        for layer_i in xrange(self.numAgeLayers()):
            n += len(self[layer_i])
        return n

    def uniquifyInds(self):
        """Per level, remove any inds that are duplicate in terms of
        performance"""
        for layer_i, R in enumerate(self):
            self[layer_i] = uniqueIndsByPerformance(R)

    def flattened(self):
        """Return all inds as one big flat list"""
        f = []
        for R in self:
            f.extend(R)
        return f

    def hasInds(self):
        """Returns True if there are >0 inds in self."""
        for R in self:
            if len(R) > 0:
                return True
        return False


def numIndsInNestedPop(F):
    """
    @description

      Returns the total # inds in a list of list_of_inds

    @arguments

      F -- list of list_of_inds

    @return

      num_inds -- int -- == len(list_of_inds[0]) + len(list_of_inds[1]) + ... 

    @exceptions

    @notes
    """
    return sum([len(Fi) for Fi in F])


def uniqueIndsByPerformance(inds):
    """
    @description

      Prunes away any inds which are duplicate by performance measures.  Uses just nominal perfs to determine it.

    @arguments

      inds -- list of Ind

    @return

      unique_inds -- list of Ind

    @exceptions

    @notes
    """
    ind_strs = [ind.worstCaseMetricValuesStr(num_rnd_points=1)
                for ind in inds]
    keep_I = mathutil.uniqueStringIndices(ind_strs)
    unique_inds = list(numpy.take(inds, keep_I))
    return unique_inds

def uniqueIndsSubsetByPerformance(Q, P):
    """
    @description

      Prunes away any inds from Q which are duplicate of an ind in
      P or Q by performance measures.

    @arguments

      Q -- list of Ind to be pruned
      P -- list of Ind to also consider when checking uniqueness, but not to return

    @return

      unique_inds -- list of Ind

    @exceptions

    @notes
    """
    # if we put P and Q together, we can make them
    # both unique. If we put P before Q, and both an item
    # form P and Q are present, the one from P will be selected
    # since uniqueStringIndices returns the smallest index.
    # if we then choose only those indices that belong to Q,
    # we have eliminated the duplicates from P and Q
    C = P + Q
    ind_strs = [ind.worstCaseMetricValuesStr()
                for ind in C]
    keep_I = mathutil.uniqueStringIndices(ind_strs)

    #sort the list
    keep_I.sort()
    max_idx = len(P)
    start_idx = 0
    #find the index of the first Q item
    for i in keep_I:
        if i < max_idx:
            start_idx += 1
        else:
            break

    #select the inds
    unique_inds = list(numpy.take(C, keep_I[start_idx:]))
    return unique_inds

def distributedObjectiveWeights(num_objectives, num_weights):
    """Return W [ind #][objective #] as weights used for multi-objective opt (MOEA/D)"""
    W = LHS(num_vars = num_objectives, num_levels = num_weights)
    #normalize to [0,1]
    W = W / float(1)
    for i in range(num_weights):
        W[:,i] = W[:,i]/float(numpy.sum(W[:,i]))
    
    # ensure we have almost unidirectional searches
    # for each objective
    extreme_directions = numpy.identity(num_objectives)
    for i in range(min(num_objectives, num_weights)):
        W[:,i] = extreme_directions[:,i]

    #give every variable a minimum weight of 0.001 (magic number)
    W = 0.001 + (1-0.001*num_objectives) * W

    W = numpy.transpose(W)
    
    # normalize all weights to 2norm=1
    for i in range(len(W[:,0])):
        W[i,:] = mathutil.normalizeVector(W[i,:])

    return W

def LHS(num_vars, num_levels):
    """Returns a matrix of samples W [var #][sample #] using Latin Hypercube Sampling."""
    W = numpy.zeros((num_vars, num_levels))
    var_order = range(num_vars)
    random.shuffle(var_order)
    for var_i in var_order:
        tempvec = range(num_levels)
        random.shuffle(tempvec)
        W[var_i,:] = numpy.array(tempvec)

    return W

def minMaxMetrics(ps, all_inds):
    """
    @description

      Compute min metric value encountered, and max metric value encountered,
      in 'all_inds'

    @arguments

      all_inds -- list of ind

    @return

      minmax_metrics -- dict of metric_name : (min_val, max_val)

    @exceptions

    @notes
    """
    assert isinstance(all_inds, types.ListType), all_inds.__class__
    
    minmax_metrics = {}
    for metric in ps.flattenedMetrics():
        min_f, max_f = INF, -INF
        for ind in all_inds:
            f = ind.nominalWorstCaseMetricValue(metric.name)
            min_f = min(min_f, f)
            max_f = max(max_f, f)
        if min_f == BAD_METRIC_VALUE:
            min_f = metric.rough_minval
        if max_f == BAD_METRIC_VALUE:
            max_f = metric.rough_maxval
        minmax_metrics[metric.name] = (min_f, max_f)

    return minmax_metrics

class SynthState:
    def __init__(self, ps, ss, R_per_age_layer):
        assert isinstance(R_per_age_layer, AgeLayeredPop), R_per_age_layer.__class__
            
        self.ps = ps #Problem Setup
        self.ss = ss #SolutionStrategy

        #R_per_age_layer is children plus parents per age layer;
        # ie R = P + Q for each layer.
        self.R_per_age_layer = R_per_age_layer

        self.generation = 0
        self.tot_num_inds = 0
        self.current_age_layer = 0
        
        #this attribute is helpful to manage how often we do novelty in rnd-gen
        self.num_rnd_gen_rounds_since_novelty = 0
        
        self.num_evaluations_per_analysis = {} #anID : num_evaluations
        for an in self.ps.analyses:
            self.num_evaluations_per_analysis[an.ID] = 0

        #useful for tracking convergence when a single objective (e.g. 2d sphere)
        self.best_0th_metric_value_per_gen = []

        # initialize weights
        self.initWeights()
        
        #lists of nondominated inds
        self.nominal_nondom_inds = [] #list of Ind (fills when nominal opt)
        self.yield_nondom_inds = []  #list of Ind (fills when yield opt)
        self.yield_nondom_lite_inds = [] #list of LiteInd (fills when yield opt)
        
        # the current set is flushed into the 'big' ND set by
        # mergeCurrentToBigNdSet
        self.nominal_nondom_inds_current = [] #list of Ind (fills when nominal opt)
        self.yield_nondom_inds_current = []  #list of Ind (fills when yield opt)
        self.yield_nondom_lite_inds_current = [] #list of LiteInd (fills when yield opt)
           
        self.updateNondominatedInds(R_per_age_layer.flattened())

    def initWeights(self):
        #set attributes for MOEA/D
        # -W [ind #][objective #] contains weights for objective functions (==lambda in MOEA/D paper)
        # -indices_of_neighbors [ind #] contains a list of indices for neighbors (==B in MOEA/D paper)
        num_weights = self.ss.num_inds_per_age_layer
        self.W = distributedObjectiveWeights(self.ps.numObjectives(), num_weights = num_weights)
        log.info("Distributed objective weights: \n%s\n" % self.W)
        self.indices_of_neighbors = []

        #magic number alert...
        if self.ss.num_inds_per_age_layer <= 5:
            num_neighbors = self.ss.num_inds_per_age_layer
        else:
            # divide the space into neighbourhoods that are 5% of the entire space
            # with a minimum of 5
            num_neighbors = int(max(self.ss.num_inds_per_age_layer * 5 / 100, 5))
        
        s = "Neighbors:\n"
        for w_i in xrange(num_weights):
            dists = [mathutil.distance(self.W[w_i,:], self.W[w_j,:]) for w_j in xrange(num_weights)]
            neighbor_I = numpy.argsort(dists)[:num_neighbors] #keep w_i as a neighbor here!
            self.indices_of_neighbors.append(neighbor_I)
            
            s += "W[%3d]: %s\n" % (w_i, neighbor_I)
        log.info(s)

    def numNondominatedIndsStr(self):
        """Returns number of nondominated inds, as a string"""
        if self.ps.doRobust():
            return "there are %d nondominated point(s) from %d Ind(s)" % \
                   (len(self.yield_nondom_lite_inds), len(self.yield_nondom_inds))
        else:
            return "there are %d nondominated Ind(s)" % len(self.nominal_nondom_inds)

    def populationSummaryStr(self, sort_metric, output_all):
        """Returns a string summarizing the population"""
        assert isinstance(output_all, types.BooleanType)

        if output_all:
            return worstCasePopulationSummaryStr(
                self.ps, self.allInds(), sort_metric)
        elif self.ps.doRobust():
            return yieldPopulationSummaryStr(
                self.ps, self.yield_nondom_lite_inds, self.yield_nondom_inds)
        else:
            return worstCasePopulationSummaryStr(
                self.ps, self.nominal_nondom_inds, sort_metric)

    def populationSummaryToMatlab(self, base_file, output_all):
        """Returns a string summarizing the population"""
        assert isinstance(output_all, types.BooleanType)
        
        if self.ps.doRobust():
            return yieldPopulationSummaryToMatlab(
                self.ps, self.yield_nondom_lite_inds, self.yield_nondom_inds, base_file)
        else:
            return worstCasePopulationSummaryToMatlab(
                self.ps, self.nominal_nondom_inds, base_file)

    def nondominatedUpdateIsCostly(self, new_inds):
        """Returns True if calling 'updateNondominatedInds' will be costly"""
        if self.ps.doRobust():
            num_rnd_points = max(self.ss.num_rnd_points_per_layer_for_cost)
            cand_inds = [ind for ind in new_inds if ind.numRndPointsFullyEvaluated() >= num_rnd_points]
            return len(cand_inds) > 0
        else:
            return False

    def mergeCurrentToBigNdSet(self):
        if self.ps.doRobust():
            cand_inds = self.yield_nondom_inds_current
            #NOTE: any changes here should be reflected in nondominatedUpdateIsCostly().
            num_rnd_points = max(self.ss.num_rnd_points_per_layer_for_cost)
            cand_inds = [ind for ind in new_inds if ind.numRndPointsFullyEvaluated() >= num_rnd_points]
            (self.yield_nondom_lite_inds, self.yield_nondom_inds) = yieldNondominatedFilter(
                self.yield_nondom_lite_inds, self.yield_nondom_inds, cand_inds)
            self.yield_nondom_inds_current = []
            self.yield_nondom_lite_inds_current = []
        else: 
            cand_inds = self.nominal_nondom_inds_current
            if cand_inds:
                log.info("update of big non-dominated set")
                new_list = self.nominal_nondom_inds + cand_inds
                pre_count = len(new_list)
                self.nominal_nondom_inds = nondominatedFilter(new_list, None)
                post_count = len(self.nominal_nondom_inds)
                log.info(" update of big non-dominated set finished, kept %s of %s inds" % (post_count, pre_count))
                self.nominal_nondom_inds_current = []
    
    def updateNondominatedInds(self, new_inds, block = True):
        """Updates self's nondominated attributes with new_inds."""
        for ind in new_inds + self.nominal_nondom_inds_current + self.yield_nondom_inds_current:
            assert ind._ps is not None
            
        if self.ps.doRobust():
            #NOTE: any changes here should be reflected in nondominatedUpdateIsCostly().
            num_rnd_points = max(self.ss.num_rnd_points_per_layer_for_cost)
            cand_inds = [ind for ind in new_inds if ind.numRndPointsFullyEvaluated() >= num_rnd_points]
            (self.yield_nondom_lite_inds_current, self.yield_nondom_inds_current) = yieldNondominatedFilter(
                self.yield_nondom_lite_inds_current, self.yield_nondom_inds_current, cand_inds)
        else: 
            cand_inds = [ind for ind in new_inds if ind.isFeasibleAtNominal()]
            if cand_inds:
                log.info("update of current non-dominated set")
                # first filter the children down on their own
                cand_inds = nondominatedFilter(cand_inds, None)
                new_list = self.nominal_nondom_inds_current + cand_inds
                pre_count = len(new_list)
                self.nominal_nondom_inds_current = nondominatedFilter(new_list, None)
                post_count = len(self.nominal_nondom_inds_current)
                log.info(" update of current non-dominated set finished, kept %s of %s inds" % (post_count, pre_count))

    def incrementGeneration(self):
        #update 'generation'
        self.generation += 1

        #update 'best_0th_metric_value_per_gen' (WARNING: it only looks at nominal num_rnd_points=1)
        metric = self.ps.flattenedMetrics()[0]
        best_value = self.R_per_age_layer[0][0].worstCaseMetricValue(1, metric.name)
        for (layer_i, R) in enumerate(self.R_per_age_layer):
            for ind in R:
                if metric.isBetter(ind.worstCaseMetricValue(1, metric.name), best_value):
                    best_value = ind.worstCaseMetricValue(1, metric.name)

        self.best_0th_metric_value_per_gen.append(best_value)
        
    def setCurrentAgeLayer(self, i):
        #update 'generation'
        self.current_age_layer = i

    def updateSynthStateProblemSetup(self, ps):
        self.ps = ps
        for an in self.ps.analyses:
            if not an.ID in  self.num_evaluations_per_analysis.keys():
                self.num_evaluations_per_analysis[an.ID] = 0

    def totalNumEvaluations(self):
        return sum(self.num_evaluations_per_analysis.values())

    def totalNumFunctionEvaluations(self):
        return sum(self.num_evaluations_per_analysis[an.ID]
                   for an in self.ps.analyses
                   if isinstance(an, FunctionAnalysis))

    def totalNumCircuitEvaluations(self):
        return sum(self.num_evaluations_per_analysis[an.ID]
                   for an in self.ps.analyses
                   if isinstance(an, CircuitAnalysis))

    def numAgeLayers(self):
        return self.R_per_age_layer.numAgeLayers()

    def numInds(self):
        return len(self.allInds())
    
    def allInds(self):
        """Returns all inds in union of age-layered pop and nondom inds, without duplicates."""
        inds, IDs = [], set([])
        for subpop in self.R_per_age_layer + [self.nominal_nondom_inds] + [self.yield_nondom_inds]:
            for ind in subpop:
                if ind.ID not in IDs:
                    IDs.add(ind.ID)
                    inds.append(ind)
        return inds

    def getInd(self, target_ind_ID):
        """Returns the ind having the specified (short) ID.  Returns None if not found"""
        assert isinstance(target_ind_ID, types.StringType)
        
        for ind in self.allInds():
            if ind.shortID() == target_ind_ID:
                return ind
        return None
    
    def getCurrentGenerationId(self):
        """
        @description
          generates a unique ID for the current generation. the ID is generated
          such that if a certain ID is smaller than another one, it is either from
          an earlier generation or from a lower age layer

          #FIXME: currently no way to figure out the age layer we're working on
        @arguments


        @return
          id -- Int -- generated as described above
    
        @exceptions
    
        @notes
        
        """
        return self.generation * (self.ss.max_num_age_layers + 1) \
               + self.current_age_layer
    

    def save(self, output_file, test_asserts=True):
        """
        @description

          Pickles self (SynthState) to output_file
        
        @arguments

          output_file -- string
        
        @return

          <<none>> but it has created a file of name 'output_file'
    
        @exceptions
    
        @notes
        """
        #preconditions
        if test_asserts:
            for R in self.R_per_age_layer:
                # NOTE: not true for modified MOEA/D
                # #note that we can have extra if bumped inds get here
                # assert len(R) >= self.ss.num_inds_per_age_layer
                pass

        #
        log.info("Save current state to file '%s': begin" % output_file)
      
        #Prepare to save...
        #-dereference 'ps' because we can't pickle instancemethod objects
        #   (and it would be space-consuming if on every ind)
        ps = self.ps
        self.ps = None

        for R in self.R_per_age_layer:
            for ind in R:
                ind.prepareForPickle()
                ind.squeeze()

        for ind in self.nominal_nondom_inds + self.yield_nondom_inds:
            ind.prepareForPickle()
            ind.squeeze()

        for ind in self.nominal_nondom_inds_current + self.yield_nondom_inds_current:
            ind.prepareForPickle()
            ind.squeeze()

        #-create some extra attributes on 'self' to save:
        #  -ps.embedded_part because it contains novel topo. info
        #  -ps.problem_choice so that we know what problem we had
        #  -ps.devices_setup so that we know what randomness setup we had
        self.embedded_part = ps.embedded_part
        self.problem_choice = ps.problem_choice
        self.devices_setup = ps.devices_setup

        #Do the actual save
        obj_to_save = self
        fid = open(output_file,'w')
        pickle.dump(obj_to_save, fid)
        fid.close()

        #Put everything back in place
        #-put 'ps' back
        self.ps = ps

        for R in self.R_per_age_layer:
            for ind in R:
                ind.restoreFromPickle(ps)

        for ind in self.nominal_nondom_inds + self.yield_nondom_inds:
            ind.restoreFromPickle(ps)
        for ind in self.nominal_nondom_inds_current + self.yield_nondom_inds_current:
            ind.restoreFromPickle(ps)

        #-delete extra attributes on 'self': embedded_part, problem_choice, devices_setup
        delattr(self, 'embedded_part')
        delattr(self, 'problem_choice')
        delattr(self, 'devices_setup')

        #postconditions
        if test_asserts:
            for R in self.R_per_age_layer:
                # NOTE: not true for modified MOEA/D
                # #note that we can have extra if bumped inds get here
                # assert len(R) >= self.ss.num_inds_per_age_layer
                pass

        #Done!
        log.info("Save current state to file '%s': done" % output_file)

def loadSynthState(db_file, ps):
    """
    @description

      Loads a synthesis state from 'db_file'. It can be from any generation.

    @arguments

      db_file -- string -- e.g. my_outfile_path/state_gen0026.db
      ps -- ProblemSetup or None -- the problem setup used when creating the db.
        If None, it uses the ps in the db itself.
      
    @return

      synth_state -- SynthState object

    @exceptions

    @notes

      Uses the input ps as a starting point, but it modifies the ps
      according to things stored in the DB:
      -overwrites ps.embedded_part
      -updates ps.ordered_optvars
      -adds a novelty analysis if needed
      -(plus any other modifications that might happen in Master.__init__
      in order to support novelty)
    """
    log.info("Try to load previous DB '%s'..." % db_file)
    
    #Preconditions, round 1
    log.info("Load step 1/9: simple gate")
    assert isinstance(db_file, types.StringType), db_file.__class__

    #Raw load
    log.info("Load step 2.1/9: load DB '%s' into memory" % db_file)
    assert os.path.exists(db_file), "file does not exist"
    fid = open(db_file,'r')
    synth_state = pickle.load(fid)
    fid.close()
    log.info("Load step 2.2/9")

    #Preconditions, round 2
    if (ps is not None) and synth_state.problem_choice != ps.problem_choice:
        print "\nInput DB's problem choice (%d) does not match the command-line" \
              " problem choice (%d).  Exiting.\n" % \
              (synth_state.problem_choice, ps.problem_choice)
        sys.exit(0)

    #Set ps if needed
    if ps is None:
        log.info("Load step 2.3/9")
        from problems.Problems import ProblemFactory #late import to avoid circular ref.
        log.info("Load step 2.4/9: build ps with problem_choice=%d" % synth_state.problem_choice)
        ps = ProblemFactory().build(synth_state.problem_choice)
        log.info("Load step 2.5/9")
        ps.devices_setup = synth_state.devices_setup
        delattr(synth_state, 'devices_setup')
        log.info("Load step 2.6/9")
        synth_state.updateSynthStateProblemSetup(ps)
        log.info("Load step 2.7/9: Auto-detected problem choice of %d, and built a "
                 " problem setup from it" % synth_state.problem_choice)
    synth_state.ps = ps

    # check weights
    if len(synth_state.W[0,:]) != synth_state.ps.numObjectives() \
         or len(synth_state.W[:,0]) != synth_state.ss.num_inds_per_age_layer:
        log.info("Load step 2.8/9: re-init weights")
        # initialize weights
        synth_state.initWeights()

    #Refine the loaded state    
    #-add 'ps': since we can't fully pickle that, we reinsert it
    log.info("Load step 3/9: reinsert 'ps' into each ind")
    for R in synth_state.R_per_age_layer:
        for ind in R:
            ind.restoreFromPickle(ps)
    for ind in synth_state.nominal_nondom_inds + synth_state.yield_nondom_inds:
        ind.restoreFromPickle(ps)

    if 'nominal_nondom_inds_current' in dir(synth_state) and 'yield_nondom_inds_current' in dir(synth_state):
        for ind in synth_state.nominal_nondom_inds_current + synth_state.yield_nondom_inds_current:
            ind.restoreFromPickle(ps)

    #-don't bother with 'S'
    pass

    #make sure that synth_state.embedded_part and ps.embedded_part align
    if hasattr(synth_state, 'embedded_part') and False:
        #case: novelty-aware DB # FIXME: not true!! also present for trustworthy
        log.info("Load step 4/9: overwrite ps.embedded part")
        ps.embedded_part = synth_state.embedded_part
        ps.updateOrderedOptvarsFromEmbPart()
    else:
        #case: trustworthy DB
        log.info('Load step 4/9: overwrite synth_state.embedded_part')
        synth_state.embedded_part = ps.embedded_part
    ps.validate()

    #
    log.info("Load step 5/9: overwrite synth_state.ps.embedded part")
    synth_state.ps.embedded_part = synth_state.embedded_part

    #-ensure that embedded part supports novelty (ie 'flxwrp' parts exist)
    need_novelty = synth_state.ss.do_novelty_gen
    if need_novelty:
        log.info("Load step 6/9: validate that we have the 'wrapped' parts")
        found_wrapped = False
        for part in ps.embedded_part.flexParts():
            if 'flxwrp' in part.name:
                found_wrapped = True
                break
        assert found_wrapped, "no 'wrapped' parts found. Partnames=%s" % \
               [part.name for part in ps.embedded_part.flexParts()]
    else:
        log.info("Load step 6/9: simple gate")

    # ensure that by the time we are done, synth_state.embedded_part doesn't
    #  have 'embedded_part' attribute
    log.info("Load step 7/9: delete state.embedded_part")
    synth_state.embedded_part = None
    delattr(synth_state, 'embedded_part')

    #reattaching attributes that were removed when pickling
    log.info("Load step 8/9: reattaching attributes")
    # reattach the approxmosmodel
    models = getattr(ps.parts_library.ss, 'approx_mos_models', None)
    if models: 
        ps.embedded_part.reattachAttribute('approx_mos_models', models)

    # fix up old states
    if not 'max_num_inds_for_weighted_topology_improve' in dir(synth_state.ss):
        log.info("set ss.max_num_inds_for_weighted_topology_improve to 20")
        synth_state.ss.max_num_inds_for_weighted_topology_improve = 20

    if not 'do_improve_step' in dir(synth_state.ss):
        log.info("set ss.do_improve_step to False")
        synth_state.ss.do_improve_step = False

    if not 'topology_layers_per_weight' in dir(synth_state.ss):
        if 'modified_moead_inds_per_weight' in dir(synth_state.ss):
            synth_state.ss.topology_layers_per_weight = synth_state.ss.modified_moead_inds_per_weight
        else:
            synth_state.ss.topology_layers_per_weight = 20

    if need_novelty and (not ps.hasNoveltyAnalysis()):
        log.info("Load step 9/9: add 'novelty' goal")
        #note that by adding it to 'ps' it will automatically
        # add it to all other references to ps, i.e. in synth_state.ps
        # and in each ind.
        ps.addNoveltyAnalysis()
    else:
        log.info("Load step 9/9: simple gate")


    #Done!
    log.info("Load: done!")

    return synth_state

class _LiteInd(object):
    """This holds just enough information to compare two LiteInds' dominance, and
    remember which Ind it was derived from.
    """
    # Each LiteInd created get a unique ID
    _ID_counter = 0L
    
    def __init__(self, costs, ind_ID):
        self.costs = costs #e.g. [cost of metric1, cost of metric2, cost of yield]
        self.ind_ID = ind_ID #points to Ind that this LiteInd was derived from
        
        self.str_costs = str(costs) #to be able to identify unique costs
        
        #manage 'ID'
        self._ID = self.__class__._ID_counter
        self.__class__._ID_counter += 1
        
    ID = property(lambda s: s._ID)
    
    def nominalConstrainedDominates(self, ind_b, dummy2, dummy3):
        """Returns True if ind_a=self dominates ind_b, and False otherwise.
        'Dominates' means that ind_a is better than ind_b in at least
        one objective; and for remaining objectives, at least equal.
        """
        ind_a = self

        found_better = False
        for (cost_a, cost_b) in izip(ind_a.costs, ind_b.costs):
            if cost_b < cost_a:
                return False
            if (not found_better) and (cost_a < cost_b):
                found_better = True

        return found_better

def _liteYield(thresholds, values_X, feas_per_rnd):
    """Return the yield estimate, given:
    thresholds -- list of float [objective_i] -- holds constraint thresholds.  Assumes minimize all constraints.
    values_X -- 2d array of float [rnd_point_i][objective_i] -- values encountered for this 'ind'
    feas_per_rnd -- list of bool -- feas_per_rnd[rnd_point_i] is True only if rnd_point_i is
      feasible at all non-objective constraints.
    """
    #preconditions
    if AGGR_TEST:
        assert len(values_X.shape) == 2
        assert values_X.shape[0] > 0
        assert values_X.shape[1] > 0
        assert len(thresholds) == values_X.shape[1]
        assert len(feas_per_rnd) == values_X.shape[0]

    #main work: build up num_feasible count
    (num_rnd_points, num_objectives) = values_X.shape
    num_feasible = 0
    for rnd_point_i in xrange(num_rnd_points):
        if feas_per_rnd[rnd_point_i]:
            is_feasible = True
            for objective_i in xrange(num_objectives):
                if values_X[rnd_point_i][objective_i] > thresholds[objective_i]:
                    is_feasible = False
                    break
            if is_feasible:
                num_feasible += 1

    yld = num_feasible / float(num_rnd_points)
    return yld
        
def extractYieldSpecsTradeoff(ind, prune_period=10000, max_num_combos=10000):
    """
    @description
    
      Given an Ind, return a list of LiteInds that collectively approximate a tradeoff
      between the ps' objectives (treated as specs), and yield.
      Only considers the tradeoff offered by a given rnd_point if it is feasible
      at all other (non-objective) constraints.

    @arguments
    
      ind -- Ind
      prune_period -- int -- governs how often we want to prune the lite inds.  Recommend to use
        the default unless testing.
      max_num_combos -- int -- maximum number of threshold combos.  Beyond 10K it can get
        very time consuming, e.g. 10M combos from 6 objectives will take approx 1h.  Recommend
        to use the default unless testing.

    @return

      nondom_lite_inds -- list of LiteInd
      
    """
    #base data
    ps = ind._ps
    constraints = [metric for metric in ps.flattenedMetrics() if not metric.improve_past_feasible]
    
    metrics = ps.metricsWithObjectives()
    
    num_rnd_points = ind.numRndPointsFullyEvaluated() - 1 #subtract 1 for nominal
    assert num_rnd_points > 0, "cannot operate unless we have some non-nominal rnd points"

    #determine feas_per_rnd
    # -Note how we ignore nominal rnd_ID (the 0th rnd ID)
    # -Note how we only look at non-objective goals (=="constraints" here), except
    #  we look at _all_ goals for BAD values
    feas_per_rnd = []
    for rnd_ID in ind.rnd_IDs[1:num_rnd_points+1]:
        if ind.isBadAtRndPoint(rnd_ID):
            feas = False
        else:
            feas = ind.isFeasibleOnConstraintsAtRndPoint(rnd_ID, constraints)
        feas_per_rnd.append(feas)

    #corner case: no rnd points are feasible
    if True not in feas_per_rnd:
        return []
    
    #determine thresholds_per_metric, cond_values_X
    thresholds_per_metric = [] # [objective_i][unique spec value]
    cond_values_X = numpy.zeros((num_rnd_points, len(metrics)), dtype=float) #[rnd pt i][obj_i]

    for (metric_i, metric) in enumerate(metrics):
        #retrieve cond_values (conditioned values):
        # -replace BAD with 999 (ok because it gets ignored anyway thanks to feas_per_rnd)
        # -make negative if aim is MAXIMIZE
        cond_values = []
        for (ri, rnd_ID) in enumerate(ind.rnd_IDs[1:num_rnd_points+1]):
            value = ind.worstCaseMetricValueAtRndPoint(rnd_ID, metric.name)
            
            if value == BAD_METRIC_VALUE:
                assert not feas_per_rnd[ri], "a BAD value should have been infeasible"
                cond_value = 999.0
            elif metric._aim == MINIMIZE:
                cond_value = value
            elif metric._aim == MAXIMIZE:
                cond_value = -value
            elif metric._aim == IN_RANGE:
                raise NotImplementedError, "IN_RANGE currently not supported"
            else:
                raise ValueError, "unknown metric._aim %s" % metric._aim

            cond_values.append(cond_value)

        #update cond_values_X
        cond_values_X[:, metric_i] = cond_values

        #update thresholds_per_metrics
        thresholds = [value for (value, feas) in izip(cond_values, feas_per_rnd) if feas] #only feas r's
        thresholds = list(set(thresholds)) #uniquify
        thresholds = sorted(thresholds) #sort
        
        thresholds_per_metric.append(thresholds)

    #if needed, prune down the number of combinations if needed
    num_thr_per_metric = [len(thresholds) for thresholds in thresholds_per_metric]
    max_num_thr = max(num_thr_per_metric)
    while numpy.product(num_thr_per_metric) > max_num_combos:
        log.info('extractYieldSpecsTradeoff: prune down from %d combos of specs; num_thr_per_metric=%s' %
                 (numpy.product(num_thr_per_metric), num_thr_per_metric))
        max_num_thr -= 1
        for (i, thresholds) in enumerate(thresholds_per_metric):
            num_old = len(thresholds)
            num_new = min(num_old, max_num_thr-1)
            thresholds_per_metric[i] = Bottomup().cluster1d(thresholds, BottomupStrategy(num_new), True)
            
        num_thr_per_metric = [len(thresholds) for thresholds in thresholds_per_metric]
        
    #build a list of LiteInds by enumerating through the spec, yield combinations
    # -because there can be >10M combinations which raises memory issues,
    #  we have two levels of lists: next_lite_inds and pruned_lite_inds.
    #  next_lite_inds are built up one ind per iteration, but every 'prune_period' iterations
    #  we nondom-prune them and put the results into pruned_lite_inds, then reset next_lite_inds.
    #  And every 'prune_period*100' iterations, we nondom-prune pruned_lite_inds.
    next_lite_inds = [] #list of LiteInd
    pruned_lite_inds = [] #list of LiteInd
    num_thr_per_metric = [len(thresholds) for thresholds in thresholds_per_metric]
    counter = mathutil.MultiBaseCounter(num_thr_per_metric)
    log.info('extractYieldSpecsTradeoff: for Ind ID=%s, check yields of %d combinations of '
             'specs (i.e. candidate lite inds) (num_thr_per_metric=%s)' %
             (ind.shortID(), counter.num_permutations, num_thr_per_metric))
    for (permutation_i, threshold_indices) in enumerate(counter):
        ind_thresholds = [thresholds_per_metric[metric_i][threshold_i]
                          for (metric_i, threshold_i) in enumerate(threshold_indices)]

        yld = _liteYield(ind_thresholds, cond_values_X, feas_per_rnd)
        ind_costs = ind_thresholds + [-yld]

        lite_ind = _LiteInd(ind_costs, ind.ID)
        next_lite_inds.append(lite_ind)

        if permutation_i > 0:
            if ((((permutation_i + 1) % prune_period) == 0) or \
                ((permutation_i + 1) == counter.num_permutations)):
                #the usual nondominated filter calls nominalConstrainedDominates on LiteInd
                pruned_lite_inds += nondominatedFilter(next_lite_inds) 
                next_lite_inds = [] #reset next_lite_inds
                log.info('extractYieldSpecsTradeoff: at Ind ID %s, spec combo #%d / %d; '
                         'len(pruned_lite_inds)=%d' %
                         (ind.shortID(), permutation_i+1, counter.num_permutations, len(pruned_lite_inds)))
            if ((permutation_i + 1) % (prune_period * 100)) == 0:
                pruned_lite_inds = nondominatedFilter(pruned_lite_inds)
                log.info('extractYieldSpecsTradeoff: further pruned pruned_lite_inds to %d inds' %
                         (len(pruned_lite_inds)))

    #final pruning
    nondom_lite_inds = nondominatedFilter(pruned_lite_inds, None)

    log.info('extractYieldSpecsTradeoff: for Ind ID %s, got %d points in '
             'yield-specs tradeoff (from %d total possible points)' %
             (ind.shortID(), len(nondom_lite_inds), counter.num_permutations))

    #done
    return nondom_lite_inds

def yieldNondominatedFilter(prev_lite_inds, prev_inds, new_inds,
                            prune_period=10000, max_num_combos=10000):
    """
    @description

      Returns the subset of inds that are nondominated according to the objectives AND yield
      (yield is treated like an extra objective to maximize).
      
    @arguments

      prev_lite_inds -- list of LiteInd -- previous nondominated LiteInds
      prev_inds -- list of Ind -- previous Inds that the lite_inds point to
      new_inds -- list of Ind -- inds to filter
      prune_period -- see extractYieldSpecsTradeoff
      max_num_combos -- see extractYieldSpecsTradeoff

    @return

      nondom_lite_inds -- list of LiteInd -- updated nondominated LiteInds
      nondom_inds -- list of Ind -- Inds that the updated LiteInds point to

    @exceptions

      -Each ind must be feasible.
      -The problem must be yield (i.e. non-nominal).
      -Each ind must have been fully evaluated on all rnd points

    @notes

      -If any new ind has the same ID as a previous ind, it is ignored
    """
    #preconditions
    if prev_lite_inds:
        assert prev_inds
        assert isinstance(prev_lite_inds[0], _LiteInd)
        assert isinstance(prev_inds[0], Ind)
    if new_inds:
        assert isinstance(new_inds[0], Ind)

    log.info('yieldNondominatedFilter: begin')
    
    #ignore any new_inds that have already been seen
    prev_IDs = [ind.ID for ind in prev_inds]
    new_inds = [ind for ind in new_inds if ind.ID not in prev_IDs]
    
    #add candidate LiteInds, one Ind at a time
    cand_lite_inds = copy.copy(prev_lite_inds)
    for (ind_i, ind) in enumerate(new_inds):
        log.info('yieldNondominatedFilter: extract yield-specs for Ind #%d/%d ID=%s'
                 '; currently have %d cand_lite_inds' %
                 (ind_i+1, len(new_inds), ind.shortID(), len(cand_lite_inds)))
        cand_lite_inds += extractYieldSpecsTradeoff(ind, prune_period, max_num_combos)

    #get nondominated LiteInds from cand LiteInds
    # -note how a usual nondominatedFilter() can be applied 
    # -note how we can avoid the task for a couple corner cases
    if not new_inds:
        nondom_lite_inds = prev_lite_inds
    elif (not prev_inds) and (len(new_inds) == 1):
        nondom_lite_inds = cand_lite_inds
    else:
        log.info('yieldNondominatedFilter: apply nondominatedFilter to %d lite inds: begin' %
                 len(cand_lite_inds))
        nondom_lite_inds = nondominatedFilter(cand_lite_inds, None)

    #find the Inds that the LiteInds reference (typically a many-to-one LiteInd:Ind matching)
    target_IDs = set([lite_ind.ind_ID for lite_ind in nondom_lite_inds])
    nondom_inds = []
    for ind in prev_inds + new_inds:
        if ind.ID in target_IDs:
            nondom_inds.append(ind)
            target_IDs.remove(ind.ID)            

    #done
    log.info('yieldNondominatedFilter: done; returning %d nondom lite inds' % len(nondom_lite_inds))
    return (nondom_lite_inds, nondom_inds)
        
def _uniqueIdFilter(P):
    """Given a list of individuals P, returns a list that has only one copy of each ind (by ID)"""    
    used_IDs = set([])
    unique_P = []
    for ind in P:
        if ind.ID not in used_IDs:
            used_IDs.add(ind.ID)
            unique_P.append(ind)

    return unique_P

def _uniqueLiteCostsFilter(P):
    """If these inds are LiteInds, then filter out inds with identical costs"""
    #handle when not LiteInd
    if not P:
        return []
    if not isinstance(P[0], _LiteInd):
        return P

    #main case
    used_strs = set([])
    unique_P = []
    for ind in P:
        if ind.str_costs not in used_strs:
            used_strs.add(ind.str_costs)
            unique_P.append(ind)

    return unique_P

    
    
def _simpleNondominatedFilter(P, metric_weights=None):
    """
    @description

      Given a list of individuals P, returns the nondominated individuals.
      If metric_weights are set to None, it ignores constraints.
      
    @arguments

      P -- list of Ind -- inds to filter
      metric_weights -- dict of metric_name: metric_weight; details at Ind.constraintViolation

    @return

      nondom_inds -- list of Ind object

    @exceptions

    @notes

    """
    nondom_inds = []
    dataID = UniqueIDFactory().newID()
    for p in P:
        p_is_dominated = False
        for q in P:
            if q.nominalConstrainedDominates(p, metric_weights, dataID):
                p_is_dominated = True
                break
        if not p_is_dominated:
            nondom_inds.append(p)

    #filter down to inds that are unique by ID, and maybe by cost
    nondom_inds = _uniqueIdFilter(nondom_inds)
    nondom_inds = _uniqueLiteCostsFilter(nondom_inds)

    return nondom_inds

def _mergeNondominatedFilter(P, metric_weights=None, recurse_depth=0):
    """Implements nondominatedFilter via merge-sort.
    Uses Trent's merge-sort twist to make nondominated sorting really fast.
    """
    if (len(P) <= 50) or (recurse_depth > 30): #magic number alsert
        return _simpleNondominatedFilter(P, metric_weights)
    else:
        N2 = len(P)/2
        left = P[:N2]
        right = P[N2:]

        #do subdivision
        left = _mergeNondominatedFilter(left, metric_weights, recurse_depth+1)
        right = _mergeNondominatedFilter(right, metric_weights, recurse_depth+1)

        #merge the results
        result = _simpleNondominatedFilter(left+right, metric_weights)
        return result

#-nondominatedFilter is a function reference that can be _mergeNondominatedFilter
# or _simpleNondominatedFilter
#-each approach can be faster than the other, depending on input data
#-interface is identical; see _simpleNondominatedFilter for details
nondominatedFilter = _mergeNondominatedFilter

def hierNondominatedFilter(R_per_age_layer, metric_weights=None):
    """
    @description

      Does a nondominated filter to return a flat list of inds.  But unlike
      other nondominated filters, it takes in an age-layered population (list of ind_list),
      which it leverages for faster filtering.
      
      If metric_weights are set to None, it ignores constraints.

    @arguments

      R_per_age_layer -- AgeLayeredPop -- inds to filter
      metric_weights -- dict of metric_name: metric_weight; details at Ind.constraintViolation

    @return

      nondom_inds -- list of Ind object

    @exceptions

    @notes

    """
    log.info("hierNondominatedFilter: begin (%d inds total)" %
             R_per_age_layer.numInds())
    log.info("Will first do nondom sorts, then nondom-merges")
    
    #do nondominated sort on each layer
    nondom_groups = []
    for layer_i, R in enumerate(R_per_age_layer):
        log.info("Do nondom filter on age layer #%d/%d" %
                 ((layer_i+1), R_per_age_layer.numAgeLayers()))
        nondom_inds = nondominatedFilter(R, metric_weights)
        nondom_groups.append(nondom_inds)

    #pairwise merge until done
    while len(nondom_groups) > 1:
        num_groups = len(nondom_groups)
        log.info("Do another round of nondom-group merging.  %d groups left" %
                 num_groups)
        if num_groups == 3:
            log.info("Merge 3 remaining groups at once")
            nondom_inds = nondominatedFilter(
                nondom_groups[0] + nondom_groups[1] + nondom_groups[2], metric_weights)
            nondom_groups = [nondom_inds]
        else:
            pairs = []
            offset = int(num_groups/2)
            for i in xrange(offset):
                pairs.append((i, i+offset))

            #merge each pair of nondom groups into a single one
            new_nondom_groups = []
            for merge_i, (i,j) in enumerate(pairs):
                log.info(" Do pairwise merge #%d/%d" %
                         (merge_i+1, len(pairs)))
                nondom_inds = nondominatedFilter(
                    nondom_groups[i] + nondom_groups[j], metric_weights)
                new_nondom_groups.append(nondom_inds)

            #handle odd-number straggler
            if float(num_groups/2.0) != float(offset):
                new_nondom_groups.append(nondom_groups[-1])

            #update for the loop
            nondom_groups = new_nondom_groups

    #done
    nondom_inds = nondom_groups[0]
    log.info("hierNondominatedFilter: done; %d inds are nondominated" % len(nondom_inds))
    return nondom_inds

    
def _mergeNondominatedSort(P, max_num_inds=None, max_layer_index=None, metric_weights=None):
    """
    @description

      Sort inds in R into layers of nondomination.  The 0th layer's inds
      are the truly nondominated inds; the 1st layer is the nondominated
      inds if you ignore the 0th layer; the 2nd layer is the nondominated
      inds if you ignore the 0th and 1st layers; etc.
      
      If metric_weights are set to None, it ignores constraints.

    @arguments

      P -- list of Ind -- inds to sort
      max_num_inds -- int -- stop building F once it has this number of
        inds.  Specifying None sets max_num_inds = Inf
      max_layer_index -- int -- stop building F once this layer index
        has been built.  Specifying None sets max_layer_index = Inf.
        Example usage: to retrieve just nondominated set, set this to 0.
      metric_weights -- dict of metric_name: metric_weight; details at Ind.constraintViolation

    @return

      F -- list of nondom_inds_layer where a nondom_inds_layer is a
        list of inds and all inds_layers together make up R.
        E.g. F[2] may have [P[15], P[17], P[4]]

      ALSO: each the inds in P and F have three attributes modified:
        n -- int -- number of inds which dominate the ind
        S -- list of ind -- the inds that this ind dominates
        rank -- int -- 0 means in 0th nondom layer, 1 in 1st layer, etc.

    @exceptions

    @notes
    """
    
    if max_num_inds is None:
        max_num_inds = INF
    if max_layer_index is None:
        max_layer_index = INF

    #corner case
    if len(P) == 0:
        return [[]]

    #main case...
    F = []
    remaining_inds = P
    layer_index = 0
    while len(remaining_inds) > 0:
        next_nondom_inds = _mergeNondominatedFilter(remaining_inds, metric_weights)
        for ind in next_nondom_inds:
            ind.rank = layer_index
        F.append( next_nondom_inds )
            
        next_nondom_IDs = [ind.shortID() for ind in next_nondom_inds]
        remaining_inds = [ind for ind in remaining_inds
                          if ind.shortID() not in next_nondom_IDs]

        #stop if max_layer_index hit
        layer_index += 1
        if layer_index >= max_layer_index:
            break

        #stop if max_num_inds is hit
        if numIndsInNestedPop(F) >= max_num_inds:
            break

    #make sure we don't exceed max_num_inds
    if numIndsInNestedPop(F) > max_num_inds:
        num_extra = numIndsInNestedPop(F) - max_num_inds
        num_keep = max(0, len(F[-1]) - num_extra)
        F[-1] = random.sample(F[-1], num_keep)

    #if the last list_of_inds in F is empty, remove it
    if len(F[-1]) == 0:
        F = F[:-1]

    return F


def _debNondominatedSort(P, max_num_inds=None, max_layer_index=None, metric_weights=None):
    """Implements Deb's NSGA-II approach to nondominated sorting,
    which he refers to as 'fast nondominated sort'.
    """
    if max_num_inds is None:
        max_num_inds = INF
    if max_layer_index is None:
        max_layer_index = INF
        
    F = [[]]
    
    #corner case
    if len(P) == 0:
        return F

    #main case...
    dataID = UniqueIDFactory().newID()
    for p in P:
        p.n = 0 #n is domination count of ind 'p',ie # inds which dominate p
        p.S = [] #S is the set of solutions that 'p' dominates

        for q in P:
            if p.nominalConstrainedDominates(q, metric_weights, dataID):
                p.S += [q]
            elif q.nominalConstrainedDominates(p, metric_weights, dataID):
                p.n += 1

        #if p belongs to 0th front, remember that
        if p.n == 0:
            p.rank = 0
            F[0] += [p]

    i = 0
    while len(F[i]) > 0:
        #stop if max_layer_index is hit
        if i == max_layer_index:
            break
        
        Q = [] #stores members of the next front
        for p in F[i]:
            for q in p.S:
                q.n -= 1
                if q.n == 0:
                    q.rank = i + 1
                    Q += [q]
        i += 1
        F.append(Q)

        #stop if max_num_inds is hit
        if numIndsInNestedPop(F) >= max_num_inds:
            break

    #make sure we don't exceed max_num_inds
    if numIndsInNestedPop(F) > max_num_inds:
        num_extra = numIndsInNestedPop(F) - max_num_inds
        num_keep = max(0, len(F[-1]) - num_extra)
        F[-1] = random.sample(F[-1], num_keep)

    #if the last list_of_inds in F is empty, remove it
    if len(F[-1]) == 0:
        F = F[:-1]

    return F

#nondominatedSort is a function reference that can be _mergeNondominatedSort or _debNondominatedSort.
#-each approach can be faster than the other, depending on input data
#-see _mergeNondominatedSort for the interface
nondominatedSort = _debNondominatedSort


def orderViaCrowdingDistance(inds):
    """
    @description
    
      Order inds (for subsequent selection) via distance-based crowding.
      i.e. Choose inds with highest worst-case distances first.

    @arguments

      inds -- list of Ind -- 

    @return

      ordered_I -- list of int -- the ordering of inds, where ordered_I[0] is best choice,
        [1] is next-best, ..., [-1] is worst

    @exceptions

    @notes
    """
    ordered_I = list(numpy.argsort([-ind.distance for ind in inds]))
    return ordered_I

class UniqueIDFactory:
    _ID_counter = 0L
    
    def newID(self):
        return self.__class__._ID_counter
        self.__class__._ID_counter += 1
        

def coststr(costs):
    """Pretty-print list of costs, or float cost value."""
    #initialization: convert away from tuples
    if isinstance(costs, types.TupleType):
        if len(costs) == 1:
            costs = costs[0]
        else:
            costs = list(costs)

    #main case: decide list vs. single entry
    if isinstance(costs, types.ListType):
        if len(costs) > 12:
            return coststr(costs[:6]) + "..." + coststr(costs[-6:])
        else:
            s = "["
            for (i, cost) in enumerate(costs):
                s += coststr(cost)
                if i < (len(costs)-1):
                    s += ","
            s += "]"
            return s
    else:
        cost = costs
        if cost == BAD_METRIC_VALUE:
            return "BAD_METRIC_VALUE"
        elif cost is None:
            return "None"
        elif mathutil.isNan(cost):
            return "NaN"
        elif cost == 0.0:
            return "0.0"
        elif (0.01 < abs(cost) < 1000.0):
            return "%5.3f" % cost
        else:
            return "%0.3e" % cost
        
class RandomPool(list):
    """
    @description

      Holds random individuals
      
    @attributes

      <list> -- 
      
    @notes
    """
    def __init__(self, ps, do_random_sampling = True):
        """
          ps -- ProblemSetup object --
          do_random_sampling -- bool -- if True, select inds randomly in getInds
        
        """
        list.__init__(self)

        # we need the problem setup to check whether inds are valid
        self.ps = ps #Problem setup the inds have (should have)
        self.problem_choice = ps.problem_choice

        self.do_random_sampling = do_random_sampling
        # keep track of the inds already handed out
        self._inds = {}
        self._inds_taken = {}

        #did the state change since the last save?
        self._int_state_changed = True

    def _rebuildInternalState(self):
        """
        @description

          rebuilds the internal state
        
        """
        self.problem_choice = self.ps.problem_choice

    def availableCount(self):
        return len(self._inds)
            
    def clear(self):
        self._int_state_changed = True
        self._inds = {}
        self._inds_taken = {}
    
    def makeAllIndsSelectable(self):
        """
        @description

          makes all inds selectable again
        
        """
        self._int_state_changed = True
        self._inds.update(self._inds_taken)
        self._inds_taken = {}

    def makeIndsUnselectableHelper(self, ind, unselect_ancestors = False):
        """
        @description
        
          makes inds unselectable. if unselect_ancestors == True, also
          unselects any ancestors
        
        """
        if ind.ID in self._inds:
            self._popInd(ind.ID)
            # unselect ancestors if necessary
            if unselect_ancestors and ind.ancestor_IDs != None:
                for ancestor_ID in ind.ancestor_IDs:
                    if ancestor_ID in self._inds:
                        self.makeIndsUnselectableHelper(self._inds[ancestor_ID], unselect_ancestors)

    def makeIndsUnselectable(self, inds, unselect_ancestors = False):
        """
        @description
        
          makes inds unselectable. if unselect_ancestors == True, also
          unselects any ancestors
        
        """

        self._int_state_changed = True
        
        for ind in inds:
            if (ind.ID in self._inds):
                self.makeIndsUnselectableHelper(ind, True)

        log.info("Unselected %s inds, now have %s random inds (of which %s already used)" % \
                 (len(inds), len(self._inds) + len(self._inds_taken), len(self._inds_taken)) )

    def getIndIDsSortedByTopo(self):
        sorted_by_topo = []
        log.info("summary vector...")
        toposummaries = [(ind.topoSummaryKey(), ind) for ind in self._inds.values()]
        log.info("sort...")
        toposummaries.sort()
        log.info("rebuild...")
        for t in toposummaries:
            sorted_by_topo.append(t[1].ID)
        return sorted_by_topo
    
    def _popInd(self, ind_id):
        """
          pops an ind from the selectable inds and returns it
          puts the ind into the unselectable ind list
        """
        self._int_state_changed = True
        ind = self._inds.pop(ind_id)
        self._inds_taken[ind_id] = ind

        return ind
        
    def loadFromFile(self, db_file):
        """
        @description

          Loads the random set from a file
          appends the loaded inds to the current set of inds
                
        @notes
       
        """
        log.info("Try to load random pool '%s'..." % db_file)
        
        #Preconditions, round 1
        log.info("Load step 1/4: simple gate")
        assert isinstance(db_file, types.StringType), db_file.__class__

        #Raw load
        log.info("Load step 2/4: load pool '%s' into memory" % db_file)
        assert os.path.exists(db_file), "file does not exist"
        fid = open(db_file,'r')
        random_pool = pickle.load(fid)
        fid.close()

        #Preconditions, round 2
        if random_pool.problem_choice != self.problem_choice:
            log.info("\n Saved pool's problem choice (%s) does not match this pool's" \
                     " problem choice (%s).\n" % (random_pool.problem_choice, self.problem_choice))
            raise ValueError, "\n Saved pool's problem choice (%s) does not match this pool's" \
                           " problem choice (%s).\n" % (random_pool.problem_choice, self.problem_choice)

        #copy over the lists
        log.info(" loaded %s inds, of which %s used ones" %\
                 ((len(random_pool._inds) + len(random_pool._inds_taken), len(random_pool._inds_taken))))
        self._inds.update(random_pool._inds)
        self._inds_taken.update(random_pool._inds_taken)
        log.info(" now have %s random inds (of which %s already used)" % \
                 (len(self._inds) + len(self._inds_taken), len(self._inds_taken)) )
    
        #check inds
        log.info("Load step 3/4: check inds")
        for indID in self._inds.keys():
            ind = self._inds[indID]
            if ind.genetic_age > 0:
                log.debug(" ind %s has genetic age %s, resetting to 0" % (indID, ind.genetic_age))
                ind.genetic_age = 0
        for indID in self._inds_taken.keys():
            ind = self._inds_taken[indID]
            if ind.genetic_age > 0:
                log.debug(" ind %s has genetic age %s, resetting to 0" % (indID, ind.genetic_age))
                ind.genetic_age = 0

        #rebuild the internal state
        log.info("Load step 4/4: rebuilding internal state")
        for ind in self._inds.values():
            ind.restoreFromPickle(self.ps)
        for ind in self._inds_taken.values():
            ind.restoreFromPickle(self.ps)
          
        self._rebuildInternalState()

        self._int_state_changed = True

    def saveToFile(self, db_file, only_when_state_changed = False):
        """
        @description

          Saves the random pool to a file
                
        @notes
       
        """
        assert isinstance(db_file, types.StringType), db_file.__class__

        # optional bypass to save disk space
        if only_when_state_changed and not self._int_state_changed:
            log.info("skipping save of random pool since nothing changed since last save.")
            return
        
        # strip unpickelable stuff
        ps = self.ps
        self.ps = None
        for ind in self._inds.values():
            ind.prepareForPickle()
        for ind in self._inds_taken.values():
            ind.prepareForPickle()
        
        log.info("save pool to file '%s'" % db_file)
        log.info("saving %s random inds (of which %s already used)" % \
                 (len(self._inds) + len(self._inds_taken), len(self._inds_taken)) )
        if os.path.exists(db_file):
            log.info("file '%s' already exists, overwriting..." % db_file)

        fid = open(db_file,'w')
        pickle.dump(self, fid)
        fid.close()

        # restore
        self.ps = ps
        for ind in self._inds.values():
            ind.restoreFromPickle(ps)
        for ind in self._inds_taken.values():
            ind.restoreFromPickle(ps)
   
    def putInds(self, inds):
        """
        @description

          Puts a set of random inds into the pool
                
        @notes

          Random inds that are not unique are rejected
       
        """
        self._int_state_changed = True
        log.info("Adding %s inds to pool" % len(inds))
        for ind in inds:
            ind_in_unused = (ind in self._inds)
            ind_in_used = (ind in self._inds_taken)
            if not ind_in_unused and not ind_in_used:
                if ind.genetic_age > 0:
                    log.info("adding random ind with genetic age %s??" % ind.genetic_age)
                self._inds[ind.ID] = ind # do we have to copy here?
            elif ind_in_used:
                log.debug("making ind %s selectable" % str(ind.ID))
                ind_id = ind.ID
                ind = self._inds_taken.pop(ind_id)
                self._inds[ind_id] = ind
            else:
                log.debug("Trying to add an ind that already exists in selectable set: %s" % str(ind.ID))
                #raise ValueError, "Trying to add an ind that already exists: %s" % str(ind.ID)
                        
    def getInds(self, nb_inds_requested):
        """
        @description

          Returns a list of random inds from the pool.
                
        @notes

          During the lifetime of the RandomPool, no two identical
          inds are returned.

          The number of returned inds is not necessarily equal to the
          number requested.
       
        """
        self._int_state_changed = True
        if len(self._inds) < nb_inds_requested:
            nb_to_take = len(self._inds)
        else:
            nb_to_take = nb_inds_requested

        if self.do_random_sampling:
            ind_keys = random.sample(self._inds.keys(), nb_to_take)
        else:
            ## HACK: use the sorted
            ind_keys = self.getIndIDsSortedByTopo()
            if len(ind_keys) >= nb_to_take:
                ind_keys = ind_keys[0:nb_to_take]

        inds = []
        for ind_key in ind_keys:
            inds.append(self._popInd(ind_key))

        log.info("return %s random inds (randomized = %s), still %s unused in pool" %\
                 (len(inds), self.do_random_sampling, len(self._inds)))
        return inds

    def summaryStr(self, sort_metric=None):
        """Returns a string summarizing the database"""
        s = "\n Used inds:\n"
        s += worstCasePopulationSummaryStr(
            self.ps, self._inds_taken.values(), sort_metric)
        s += "\n\nUnused inds:\n"
        s += worstCasePopulationSummaryStr(
            self.ps, self._inds.values(), sort_metric)
        return s


def estimateFrontCost(ps, cands, W, metric_weights):
    N = len(W[:,0])
    best_costs = [] #list of float
    log.debug("Estimating front cost...")

    # use rough metric extremes since those do not change and can be
    # used to compare generations
    metric_bounds = {}
    for metric in ps.flattenedMetrics():
        metric_bounds[metric.name] = (metric.rough_minval, metric.rough_maxval)
    #dbg = "  metric bounds: \n"
    #for obj in self.ps.metricsWithObjectives():
        #dbg += "   %20s: %10s -> %10s\n" % (obj.name, metric_bounds[obj.name][0], metric_bounds[obj.name][1])
    #log.debug(dbg)

    for w_i in range(N):
        costs = [ind.scalarCost(1, W[w_i,:], metric_weights, metric_bounds)
                    for ind in cands]
        best_costs.append(min(costs))

    return numpy.sum(best_costs)

def prepareMOEADLayer(ps, cands, W, metric_weights, max_inds_per_weight=1):
    """
    Prepares an layer for use with MOEA/D.
    
    This means that we find the best inds for each of the weights.
    The side-condition is that all inds have to have a different topology.
    
    returns a list structure:
    [ [best inds for vector 0] [best inds for vector 1] ... ]
    
    note: does not necessarily return max_inds_per_weight inds
    """
    assert len(cands) > 0

    N = len(W[:,0])
    best_costs = [] #list of float
    log.info("Calulating multi-ind MOEA/D layer...")

    # counts at start of layer
    cand_ind_topos = [ind.topoSummary() for ind in set(cands)]

    # find min and max values in this R for each metric
    log.info(" Estimating metric bounds...")
    metric_bounds = minMaxMetrics(ps, cands)
    dbg = " metric bounds: \n"
    for obj in ps.metricsWithObjectives():
        dbg += "  %20s: %10s -> %10s\n" % (obj.name, metric_bounds[obj.name][0], metric_bounds[obj.name][1])
    log.info(dbg)

    # calculate costs per w_i for all candidates
    # and prepare first layer (best ind per weight)
    costs_per_wi = []
    ind_order_per_wi = []
    ind_vector_per_wi = []
    ind_topo_vector_per_wi = []

    for w_i in range(N):
        costs = [ind.scalarCost(1, W[w_i,:], metric_weights, metric_bounds)
                    for ind in cands]
        ind_order_per_wi.append(numpy.argsort(costs))
        costs_per_wi.append(numpy.take(costs, ind_order_per_wi[w_i]))

        # prepare the first layer
        best_I = ind_order_per_wi[w_i][0]
        best_ind = cands[best_I]
        best_costs.append(costs[best_I])

        ind_vector_per_wi.append([best_ind])
        ind_topo_vector_per_wi.append([best_ind.topoSummary()])

    # fill the remainder of the ind vector per weight
    for w_i in range(N):
        ind_order = ind_order_per_wi[w_i]
        for idx in ind_order:
            topo = cands[idx].topoSummary()
            if topo not in ind_topo_vector_per_wi[w_i]:
                ind_vector_per_wi[w_i].append(cands[idx])
                ind_topo_vector_per_wi[w_i].append(topo)
            # stop when enough (efficiency)
            if len(ind_vector_per_wi[w_i]) > max_inds_per_weight:
                break

        # check uniqueness
        assert len(set(ind_topo_vector_per_wi[w_i])) == len(ind_topo_vector_per_wi[w_i])
        assert len(ind_topo_vector_per_wi[w_i]) == len(ind_vector_per_wi[w_i])

        if False: # lots of debug output
            s = "w_i: %d\n" % w_i
            for (idx, ind) in enumerate(ind_vector_per_wi[w_i]):
                s += " %20s" % idx
            s += "\n"
            for (idx, ind) in enumerate(ind_vector_per_wi[w_i]):
                s += " %20s" % ind.shortID()
            s += "\n"
            for (idx, ind) in enumerate(ind_vector_per_wi[w_i]):
                s += " %20s" % ind.topoSummary()
            s += "\n"
            for (idx, ind) in enumerate(ind_vector_per_wi[w_i]):
                s += " %20s" % ind_topo_vector_per_wi[w_i][idx]
            s += "\n"
            for (idx, ind) in enumerate(ind_vector_per_wi[w_i]):
                i = ind_order_per_wi[w_i][idx]
                s += " %20s" % costs_per_wi[w_i][idx]
            s += "\n"

            log.info(s)

    best_inds = set()
    for w_i in range(N):
        best_inds.add(ind_vector_per_wi[w_i][0])
    best_ind_topos = [ind.topoSummary() for ind in best_inds]

    # remove the inds that are not in the surviving set from
    # the per-weight vectors, and cut them to the max_len
    ind_count_per_wi = []
    for w_i in range(N):
        if len(ind_vector_per_wi[w_i]) > max_inds_per_weight:
            ind_vector_per_wi[w_i] = ind_vector_per_wi[w_i][0:max_inds_per_weight]
            ind_topo_vector_per_wi[w_i] = ind_topo_vector_per_wi[w_i][0:max_inds_per_weight]
            costs_per_wi[w_i] = costs_per_wi[w_i][0:max_inds_per_weight]
        ind_count_per_wi.append(len(ind_vector_per_wi[w_i]))

    used_inds = set()
    for inds in ind_vector_per_wi:
        used_inds.update(inds)
    used_ind_topos = [ind.topoSummary() for ind in used_inds]

    all_inds = set(cands)
    kicked_inds = list(all_inds.difference(used_inds))

    log.info("inds per weight (max %d): %s" % (max_inds_per_weight, ind_count_per_wi))

    s = "mod-MOEA/D layer summary:\n"
    for topo in set(cand_ind_topos):
        weightcount = 0
        for w_i in range(N):
            weightcount += ind_topo_vector_per_wi[w_i].count(topo)

        s += "   %20s: used: %3d [best: %3d, weights: %3d], orig: %3d\n" % (topo,
                                                        used_ind_topos.count(topo),
                                                        best_ind_topos.count(topo),
                                                        weightcount,
                                                        cand_ind_topos.count(topo))
    s += "   %20s        %3d        %3d                       %3d\n" % ("",
                                                    len(used_ind_topos),
                                                    len(best_ind_topos),
                                                    len(cand_ind_topos))

    log.info(s)
    log.debug("best cost per weight: %s" % (best_costs))
    # note: bests cost is relative to the population boundaries, can't be used to compare between generations!
    return (ind_vector_per_wi, kicked_inds, costs_per_wi)

def selectBestRankedIndPerTopology(layer, W):
    """
    selects the best-ranked ind for each topology and returns it
    along with the weights

    does not select from the top layer
    """
    N = len(W[:,0])
    max_inds_per_weight = 0
    for w_i in range(N):
        max_inds_per_weight = max(max_inds_per_weight, len(layer[w_i]))

    topo_inds_for_rank = {}
    for rank in range(1, max_inds_per_weight): # don't select any of the first layer ('best') inds
        topo_inds_for_this_rank = {} # dict of [topo] = [(ind, wi) list]
        for w_i in range(N):
            if rank < len(layer[w_i]): # does this weight have this rank
                ind = layer[w_i][rank]
                topo = ind.topoSummary()
                if not topo in topo_inds_for_this_rank.keys():
                    topo_inds_for_this_rank[topo] = []
                topo_inds_for_this_rank[topo].append( (ind, W[w_i,:]) )
        topo_inds_for_rank[rank] = topo_inds_for_this_rank

    selected_topos = set()
    selection = []
    for rank in range(1, max_inds_per_weight):
        topo_inds_for_this_rank = topo_inds_for_rank[rank]
        for topo in topo_inds_for_this_rank.keys():
            if topo not in selected_topos: # if not already selected
                selected = random.choice(topo_inds_for_this_rank[topo])
                selection.append(selected)
                selected_topos.add(topo)

    if True: # debug
        s = "Selection (best rank):\n"
        for (ind, w_i) in selection:
            s += " %20s topo %20s on weight %03d (%s)\n" % (ind.shortID(), ind.topoSummary(), w_i, W[w_i, :])
        log.info(s)

    if len(selection) != len(selected_topos):
        import pdb;pdb.set_trace()
    #assert len(selection) == len(selected_topos)
    return selection

def selectRandomIndPerTopology(layer, W):
    """
    selects one ind for each topology and returns it
    along with the weight corresponding to it.

    does not select from the top layer
    """
    N = len(W[:,0])

    # a dict of [topo] = [ (ind, wi) ...]
    # containing all inds,wi pairs in the layer for a certain topology
    topo_inds = {}
    for w_i in range(N):
        for idx in range(1, len(layer[w_i])): # don't select first layer
            ind = layer[w_i][idx]
            topo = ind.topoSummary()
            if not topo in topo_inds.keys():
                topo_inds[topo] = []
            topo_inds[topo].append( (ind, W[w_i,:]) )

    selection = []
    for topo in topo_inds.keys():
        selected = random.choice(topo_inds[topo])
        selection.append(selected)

    if True: # debug
        s = "Selection (random):\n"
        for (ind, w_i) in selection:
            s += " %20s topo %20s on weight %03d (%s)\n" % (ind.shortID(), ind.topoSummary(), w_i, W[w_i, :])
        log.info(s)

    if len(selection) != len(topo_inds.keys()):
        import pdb;pdb.set_trace()
    #assert len(selection) == len(selected_topos)
    return selection


class TopoCluster():
    """
    
    """
    __ID = 0
    def __init__(self, topo):
        self.ID = TopoCluster.__ID
        TopoCluster.__ID += 1

        self.in_front = False
        self.w_range = set() # neighbourhood
        self.topo = topo
        self.weight_data = {}

    def __str__(self):
        s  = "TopoCluster %4d: " % self.ID
        s += "topo: %20s; " % self.topo
        s += "in_front: %5s; " % self.in_front
        s += "\n"
        s += "neighbourhood: "
        for w in self.w_range:
            s += "%d " % w
        s += "\n"
        s += "w_i:\n"
        for w_i in sorted(self.weight_data.keys()):
            s += " %3d: %20s %2d %s \n" % (w_i, self.weight_data[w_i][0].shortID(), self.weight_data[w_i][1], self.weight_data[w_i][2])
        (ind, wv) = self.getBestIndAndWeight()
        s += "best ind/weight vector: %20s %s\n" % (ind.shortID(), wv)
        return s

    def getInds(self):
        return list(set([v[0] for v in self.weight_data.values()]))
    def getNeighbourhood(self):
        return self.w_range
    def getWiSet(self):
        return self.weight_data.keys()

    def merge(self, topocluster):
        #assert self.topo == topocluster.topo
        if self.topo != topocluster.topo:
            import pdb;pdb.set_trace()
        #s = "for topo %20s merged cluster %s into cluster %s\n" % (self.topo, topocluster.ID, self.ID)
        #s += " self  : %s\n" % str(self)
        #s += " cand  : %s\n" % str(topocluster)
        for new_w_i in topocluster.weight_data.keys():
            assert new_w_i not in self.weight_data.keys()
            self.weight_data[new_w_i] = topocluster.weight_data[new_w_i]
        self.in_front = self.in_front or topocluster.in_front
        self.w_range.update(topocluster.w_range)
        #s += " result: %s" % str(self)
        #log.info(s)

    def canMerge(self, cand):
        # can't be the same cluster
        if self == cand:
            return False
        # require same topology
        if self.topo != cand.topo:
            #log.debug("canMerge: bad topo")
            return False
        # ranges have to have an intersection
        if len(self.w_range.intersection(cand.w_range)) == 0:
            #log.debug("canMerge: No intersection")
            return False

        return True

    def addInd(self, ind, w_i, w_vector, rank):
        """
        add ind
        """
        assert self.topo == ind.topoSummary()
        # a cluster can only have one ind per weight id
        # since it can only have one topology
        self.weight_data[w_i] = (ind, rank, w_vector)

    def updateRange(self, w_i_s):
        self.w_range.update(w_i_s)

    def getBestIndAndWeight(self):
        """
        returns the ind and weight combination that best represents
        this cluster.
        the definition of 'best' is currently "one random"
        
        returns a (ind, w_vect) tuple
        """
        #w_i = random.choice(self.weight_data.keys())
        #return (self.weight_data[w_i][0], self.weight_data[w_i][2])
        
        # construct a weight vector by doing a weighted average on
        # weight vectors of each w_i in the cluster. the weight of the
        # wvect in the sum is the rank
        w_is = []
        ranks = []
        inds = []
        vectors = []
        for w_i in self.weight_data.keys():
            w_is.append(w_i)
            #log.info("w_i %3d: %20s %5s %s" % (w_i, self.weight_data[w_i][0].shortID(), self.weight_data[w_i][1], self.weight_data[w_i][2]))
            inds.append(self.weight_data[w_i][0])
            ranks.append(self.weight_data[w_i][1])
            vectors.append(self.weight_data[w_i][2])

        I = numpy.argsort(ranks)
        w_is_s = numpy.take(w_is, I)
        ranks_s = numpy.take(ranks, I)
        inds_s = numpy.take(inds, I)
        vectors_s = numpy.take(vectors, I)

        # ranks are lower = better
        # we want to rescale this to a ]0,1] range where 0 is 'worse than worst'
        # to use these to scale the weights
        min_rank = ranks_s[0]
        max_rank = ranks_s[-1]

        # rescale
        scaled_rank = numpy.array(ranks_s, dtype=float) - min_rank # shift to 0
        tmp = scaled_rank[-1] + 1.0
        scaled_rank = tmp - scaled_rank # turn 0->10 into 11->1
        scaled_rank = scaled_rank / tmp # turn 11->1 into 1->1/11

        scaled_vectors = []
        best_ranked_inds = set()

        # do the vector weighting
        vec_array = numpy.array(vectors_s)

        for i in range(len(ranks_s)):
            #log.info("sorted w_i %3d: %20s %5s %s" % (w_is_s[i], inds_s[i].shortID(), ranks_s[i], vectors_s[i]))
            if ranks_s[i] == min_rank:
                best_ranked_inds.add(inds_s[i])
            vec_array[i, :] = scaled_rank[i] * vec_array[i, :]
        #log.info("best ranked: %s" % [ind.shortID() for ind in best_ranked_inds])

        # add all scaled vectors
        w_vect = numpy.add.reduce(vec_array)

        # normalize to length 1
        w_vect = mathutil.normalizeVector(w_vect)
        return (random.choice(list(best_ranked_inds)), w_vect)

    def validate(self):
        inds = self.getInds()
        assert len(inds) > 0 # at least one ind per cluster
        our_topos = list(set([ind.topoSummary() for ind in inds]))
        assert len(our_topos) == 1 # exactly one topo per cluster
        assert our_topos[0] == self.topo
        for w_i in self.weight_data.keys():
            assert w_i in self.w_range

def clusterPerTopology(layer, layer_costs, W, indices_of_neighbors):
    """
        clusters a mod-MOEA/D layer population into topology clusters
    """
    N = len(W[:,0])
    do_sanity_checks = True

    all_clusters = set()
    clusters_per_wi_topo = {}

    # calculate normalized costs
    # these indicate relative quality of an ind for a certain direction
    #norm_costs = []
    #for w_i in range(N):
        #w_cost = numpy.array(layer_costs[w_i])
        #w_cost_min = min(w_cost)
        #w_cost_max = max(w_cost)
        #norm_cost = (w_cost - w_cost_min)/(w_cost_max - w_cost_min)
        #norm_costs.append(list(norm_cost))

    # attach initial clusters to each ind
    for w_i in range(N):
        clusters_per_wi_topo[w_i] = {}

        # cand_set has all inds of this weight
        for (rank, cand) in enumerate(layer[w_i]):
            topo = cand.topoSummary()
            if not 'cluster' in dir(cand):
                # haven't we already seen an ind with the same topology?
                # if not, create a cluster for this topology
                if not topo in clusters_per_wi_topo[w_i].keys():
                    clusters_per_wi_topo[w_i][topo] = TopoCluster(topo)
                    all_clusters.add(clusters_per_wi_topo[w_i][topo])

                cand.cluster = clusters_per_wi_topo[w_i][topo] # keep track (temporary)

            # add the ind to this cluster
            cand.cluster.addInd(cand, w_i, W[w_i,:], rank)

            # update the range of the cluster ('neighbourhood')
            cand.cluster.updateRange(indices_of_neighbors[w_i])

        # update the in_front flag
        layer[w_i][0].cluster.in_front = True

    log.info("start merge on %d clusters" % len(all_clusters))
    #for cluster in all_clusters:
        #log.info(" %s" % str(cluster))

    # sanity checks
    if do_sanity_checks:
        all_inds = set()
        for w_i in range(N):
            all_inds.update( layer[w_i] )
        all_topos = set([ind.topoSummary() for ind in all_inds])

        clustered_inds = set()
        for cluster in all_clusters:
            cluster.validate()
            inds1 = set(cluster.getInds())

            for cluster2 in all_clusters:
                if cluster == cluster2:
                    continue
                # inds can only occur in one cluster
                inds2 = set(cluster2.getInds())
                assert len(inds1.intersection(inds2)) == 0
            clustered_inds.update(inds1)
        assert clustered_inds.issubset(all_inds)
        assert clustered_inds.issuperset(all_inds)

    # merge clusters
    to_merge = list(all_clusters)
    did_merge_clusters = True
    while did_merge_clusters:
        i = 0
        did_merge_clusters = False
        clusters_left = len(to_merge)
        while i < clusters_left:
            base = to_merge[i]

            # next vector has all up till base
            new_to_merge = to_merge[0:i+1]
            for j in range(i+1, clusters_left):
                cand = to_merge[j]
                if base.canMerge(cand):
                    base.merge(cand)
                    did_merge_clusters = True
                else:
                    new_to_merge.append(cand)

            # go to next base cluster
            to_merge = new_to_merge
            i += 1
            clusters_left = len(to_merge)
    all_clusters = to_merge
    front_clusters = list([c.in_front for c in all_clusters]).count(True)
    log.info("  merge resulted in %d clusters, of which %d front, %d others" % (len(all_clusters), front_clusters, len(all_clusters)-front_clusters))

    #for cluster in all_clusters:
        #log.info(" %s" % str(cluster))

    # rebuild overview structures
    clusters_per_topo = {}
    clusters_per_wi = {}
    for cluster in all_clusters:
        topo = cluster.topo
        if topo not in clusters_per_topo.keys():
            clusters_per_topo[topo] = []
        clusters_per_topo[topo].append(cluster)

        for w_i in cluster.getWiSet():
            if w_i not in clusters_per_wi.keys():
                clusters_per_wi[w_i] = set()
            clusters_per_wi[w_i].add(cluster)

    #for w_i in range(N):
        #s = "w_i %3d has %3d clusters: %s" %(w_i, len(clusters_per_wi[w_i]), [c.ID for c in clusters_per_wi[w_i]])
        #log.info(s)

    for topo in all_topos:
        s = "topo %20s has %3d clusters: %s" %(topo, len(clusters_per_topo[topo]), [(c.in_front, [ind.shortID() for ind in c.getInds()]) for c in clusters_per_topo[topo]])
        log.info(s)

    # sanity checks
    if do_sanity_checks:
        clustered_inds = set()
        for cluster in all_clusters:
            cluster.validate()
            inds1 = set(cluster.getInds())
            for cluster2 in all_clusters:
                if cluster == cluster2:
                    continue
                # inds can only occur in one cluster
                inds2 = set(cluster2.getInds())
                assert len(inds1.intersection(inds2)) == 0
            clustered_inds.update(inds1)
        assert clustered_inds.issubset(all_inds)
        assert clustered_inds.issuperset(all_inds)

        for topo in all_topos:
            #log.info("topo %20s clusters: %s" % (topo, [c.ID for c in clusters]))
            assert topo in clusters_per_topo.keys()
            clusters = clusters_per_topo[topo]
            for cluster1 in clusters:
                for cluster2 in clusters:
                    if cluster1 == cluster2:
                        continue
                    # cluster weights for the same topo cannot intersect
                    hood1 = cluster1.getNeighbourhood()
                    hood2 = cluster2.getNeighbourhood()
                    assert len(hood1.intersection(hood2)) == 0

    for ind in all_inds:
        assert 'cluster' in dir(ind)
        del ind.__dict__['cluster'] # remove property
    
    return all_clusters