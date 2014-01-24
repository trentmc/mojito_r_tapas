"""Slave.py

-Controlled by Master, doing the Master's bidding
-Typically multiple Slaves are running at once
-From a networking perspective, one can view Slaves as Servers,
 and the Master as the Client.

-Tasks include:
 -Randomly generate ind (and evaluate)
 -Create 2 children from 2 parents (and evaluate)
 -Local optimize

"""
import copy
from itertools import izip
import logging
import os
import random
import time
import types
import socket

import numpy

from adts import *
from engine.Channel import ChannelStrategy, ChannelFactory, Channel, ResultData
import engine.Evaluator as Evaluator
from engine.EngineUtils import coststr
from engine.Ind import NsgaInd
from problems.Problems import ProblemFactory
from engine.SynthSolutionStrategy import SynthSolutionStrategy
from engine.GtoOptimizer import GtoProblemSetup, GtoState
from util import mathutil
from util.constants import AGGR_TEST, BAD_METRIC_VALUE, INF, DOCS_METRIC_NAME

log = logging.getLogger('slave')

class Slave:
    """
    @arguments

      cs -- ChannelSetup object -- describes the communication channel between
                                   master and slaves
            
    @description

    Follows this algorithm...
    
      While True
        Pop a task from Master
        Do the task
        Return results for master
      
    @attributes

      cs -- ChannelStrategy object --
      ps -- None or ProblemSetup object -- usually None, because this can be
        determined from the channel.  But if no real channel, then it needs to be passed in.
      ss -- None or SynthSolutionStrategy object -- just like ps
      
    @notes
    """

    def __init__(self, cs, ps=None, ss=None):
        #preconditions

        #set data...

        #invariant data
        self.ID = str("SLAVE-%s-%s" % (socket.gethostname(), str(time.time()))) #

        self.cs = cs
        self.channel = ChannelFactory(cs).buildChannel(False) #may come out as None, that's ok

        #validate channel
        if self.channel:
            if self.channel.solutionStrategy() == None:
                raise ValueError("Bad solution strategy... is the master running?")

        #set ps, ss
        if self.channel:
            self.ss = self.channel.solutionStrategy()
            self.ps = ProblemFactory().build(self.channel.problemChoice())
            if self.channel.problemIsRobust():
                self.ps.devices_setup.makeRobust()
            log.info(str(self.channel.cs) + "\n")

        else:
            assert ps is not None, "if no channel, then need a ps object"
            assert ss is not None, "if no channel, then need a ss object"

            self.ss = ss
            self.ps = ps

        #set current task
        self.task = None

        #data that is cleared before/after the task, and updated during the task
        self.num_evaluations_per_analysis = {} # dict of anID : int

        #this only gets set to True in unit tests that do profiling.  If True,
        # some places which would normally execute AGGR_TEST are not executed.
        self.doing_profiling = False

        #'num_inds' is only just used to count the number of inds when generating a random ind,
        # so that logging output can be more descriptive
        self.num_inds = 0

        self.channel.reportForService(self.ID)

        self.stop_requested = False

    def setDoingProfiling(self):
        """Will set self.doing_profiling to True.  This should only be called when
        doing profiling."""
        self.doing_profiling = True
        
    def run(self):
        """
        @description

          Runs the slave!  This is the main work routine.
        
        @arguments

          <<none>>
        
        @return

           <<none>> but it continually follows its loop
    
        @exceptions
    
        @notes

        """
        log.info("Begin slave.")
        log.info(self.ps.prettyStr())
        log.info(str(self.ss) + "\n")
        startSimulationServer()

        while True:
            self.run__oneIter()
            if self.doStop():
                break

        stopSimulationServer()
        log.info('Done')
        
    def stop(self):
        """
        @description

          Stops the slave
        
        @arguments

          <<none>>
        
        @return

           <<none>> 
    
        @exceptions
    
        @notes

        """
        stopSimulationServer()
        self.channel.leaveService(self.ID)

    def doStop(self):
        #TODO: add functionality to this, as needed
        return self.stop_requested

    def requestStop(self):
        self.stop_requested = True

    def run__oneIter(self):
        """
        @description

          Run one iteration of:
            Pop a task from Master
            Do the task
            Return results for master
        
        @arguments

          <<none>> but gets tasks from engine if needed
        
        @return

          <<none>> but sets results for engine when possible
    
        @exceptions
    
        @notes
        """
        #update current task if needed
        if self.task is None:
            self.task = self.channel.popTask(self.ID)
            if self.task is None:
                log.info("No tasks available.  Waiting.")
                time.sleep(self.ss.num_seconds_between_slave_task_requests)
                return
            else:
                #clear data to make way for task
                self.num_evaluations_per_analysis = {}

        #if we get here, we have a task to perform.  So spend more effort at it.
        result_data = None
        if self.task.descr == "Generate random ind":
            ind = self._newEvaluatedRandomInd(with_novelty=False)
            if ind is not None:
                result_data = self._createResultData(ind)
                
        elif self.task.descr == "Generate novel random ind":
            ind = self._newEvaluatedRandomInd(with_novelty=True)
            if ind is not None:
                result_data = self._createResultData()
                #FIXME: we need to get engine, and all slaves to update their
                # ps.embedded_part.part with this things.
                result_data.part = self.ps.embedded_part.part

        elif self.task.descr == "Generate child":
            ind = None
            #generate two random indices
            parents = self.task.task_data.parent_set
            nb_parents = len(parents)
            
            #restore the parent inds
            for p in parents:
                p.restoreFromPickle(self.ps)
                
            # while no results (FIXME: deadlock?)
            while ind is None:
                ind = self._newEvaluatedChild(parents[0], parents[1])
            ind.w_i = parents.w_i
                
            result_data = self._createResultData(ind)
            
            for p in parents:
                p.prepareForPickle()
            
            log.debug("Generated new child...")

        elif self.task.descr == "Evaluate ind further":
            ind = self.task.task_data.ind
            ind.restoreFromPickle(self.ps)
            prev_num_rnd = ind.numRndPointsFullyEvaluated()
            new_num_rnd = self.task.task_data.num_rnd_points
            self._evalInd(ind, ind.rnd_IDs[prev_num_rnd:new_num_rnd])
            result_data = self._createResultData(ind)

        elif self.task.descr == "Single simulation":
            d = self.task.task_data
            result_data = ResultData(self.ID, {d.an_ID : 1})
            result_data.sim_results = Evaluator.singleSimulation(self.ps, d.scaled_opt_point, d.rnd_ID, d.an_ID, d.env_ID)

        elif self.task.descr == "Resimulate":
            ind = self.task.task_data.ind
            ind.restoreFromPickle(self.ps)
            ind = self._resimulateInd(ind)
            result_data = self._createResultData(ind)

        elif self.task.descr == "ImproveTopologyForWeight":
            ind = self.task.task_data.ind
            weight_vector = self.task.task_data.weight_vector
            ind.restoreFromPickle(self.ps)
            new_ind = self._improveTopologyForWeight(ind, weight_vector)
            ind.prepareForPickle()
            result_data = self._createResultData(new_ind)

        else:
            raise ValueError("Unknown task.descr of '%s'" % self.task.descr)

        #If we have enough data to finish the task, finish it!
        task = None
        if result_data:
            task = self._completeTask(result_data)

        return task

    def _createResultData(self, ind):
        """Builds and returns a ResultData object, and fills it with:
        -slave_ID, num_inds, num evals
        -input 'ind', conditioned to be pickle-able
        Beyond that, the caller can tack on extra attributes, as desired.
        """
        result_data = ResultData(self.ID, self.num_evaluations_per_analysis)
        ind.prepareForPickle()
        result_data.ind = ind
        return result_data

    def _completeTask(self, result_data):
        """
        1. Creates a ResultFromSlave object and fills it with 'result_data' and 'sim_info'
        2. Clears the way in 'self' for accepting new tasks.

        Returns result_task (not usually needed, but convenient when no Channel)
        """
        #preconditions
        assert isinstance(result_data, ResultData)

        #
        result_task = self.task
        result_task.attachResult(result_data)

        #remove parents to reduce bandwidth
        result_task.task_data.parent_set = None

        if self.channel:
            self.channel.pushResult(result_task)

        #clear data to make way for new tasks
        self.task = None
        self.num_evaluations_per_analysis = {}

        self.num_inds = 0

        return result_task
    
    #======================================================================
    def _createInd(self, unscaled_point):
        """Creates an NsgaInd from 'unscaled_point'.
        Its genetic_age is set to 0, and parents to [], so if one wants
        those changed, they have to do it manually.
        """
        unscaled_optvals = [unscaled_point[var] for var in self.ps.ordered_optvars]
        ind = NsgaInd(unscaled_optvals, self.ps)
        ind.genetic_age = 0
        ind.setAncestry([])
        return ind
    
    #======================================================================
    #Randomly generate ind.
    def _newEvaluatedRandomInd(self, with_novelty):
        """
        @description

          Randomly generate an Ind, and evaluate it.  If it is good,
          returns the ind, else returns None.
        
        @arguments

          novelty -- bool -- inject novelty when creating the inds?
        
        @return

           ind -- Ind -- randomly generated ind.  Already evaluated.  May
             be good or BAD.
    
        @exceptions
    
        @notes

        """
        #preconditions
        self.ps.validate()

        #main work        
        log.info('Gen good random ind, tot num tries=%d\n' % self.num_inds)

        #-save the toplevel part, in case we want to undo novelty-mutate
        # (undoing will give HUGE memory savings)
        #-note: we need to save ps.embedded_part.part rather than just
        # ps.embedded_part, because many places carry a reference to
        # that embedded_part (e.g. most FunctionAnalysis objects'
        # functions, such as EmbeddedPart.novelty()).  If we change
        # the reference then the results will be wrong (ie a defect).
        if with_novelty:
            prev_part = copy.deepcopy(self.ps.embedded_part.part)

        #create the new ind, which includes mutating ps.embedded_part
        # -guarantee novelty, if needed
        testdata = ([], [])
        ind = self._newRandomInd(with_novelty, testdata)
        while with_novelty and ind.novelty() == 0:
            log.info("Try again, because we wanted novel, yet ind is not novel")
            ind = self._newRandomInd(with_novelty, testdata)

        #pick the best one we saw
        ind = self._improveTopologyOnDOCs(ind) #HACK if commented out

        #ensure that it's fully-evaluated
        self._nominalEvalInd(ind)

        log.info("After fixed-topology tries, ind: isBad=%s; %s" % (ind.nominalIsBad(), _indStr(ind)))

        #restore embedded part if needed
        if with_novelty and ind.nominalIsBad():
            delattr(self.ps.embedded_part, 'part') #force memory clear
            self.ps.embedded_part.part = prev_part

        #postconditions
        self.ps.validate()
        if self.ss.do_novelty_gen:
            f = getattr(self.ps.noveltyAnalysis(), 'function', None)
            if f is not None:
                assert id(f.im_self) == id(self.ps.embedded_part)

        #done
        if ind.nominalIsBad():
            return None
        else:
            return ind

    def _improveTopologyForWeight(self, ind, weight):
        """This routine returns a modification of ind, which has changed the
        ind's parameters (but not topology) to improve it according to the
        given weight vector
        """
        log.info("improveTopologyForWeight: begin")

        emb_part = self.ps.embedded_part
        unscaled_point = self.ps.unscaledPoint(ind)
        free_vars = self._freeParameters(unscaled_point)
        if not free_vars:
            log.info("Early exit on improveTopologyInDirection: no free vars")
            return ind

        best_ind = self._hillclimbImproveTopologyForWeight(
            ind, self.ss.max_num_inds_for_weighted_topology_improve, free_vars, weight)

        # make sure it is evaluated properly (analysis might have been skipped due to waterfall)
        self._nominalEvalInd(best_ind)

        log.info("improveTopologyForWeight: done.  Final ind: %s" % _indStr(best_ind))

        return best_ind

    def _showIndMetricPerformance(self, ind, best_ind=None):
        """ show a summary of the metric performance of an ind """
        try:
            s = "\nMetric performance:\n"
            if ind.nominalIsBad():
                s += "  BAD\n"
            else:
                for an in self.ps.analyses:
                    do_ind = ind.simRequestMadeAtNominalAtAllEnvPoints(an)
                    do_best = best_ind.simRequestMadeAtNominalAtAllEnvPoints(an)
                    for metric in an.metrics:
                        if do_ind:
                            metric_value = ind.nominalWorstCaseMetricValue(metric.name)
                        else:
                            metric_value = "NOT EVALUATED"
                        if best_ind and not best_ind.nominalIsBad():
                            if do_best:
                                best_value = best_ind.nominalWorstCaseMetricValue(metric.name)
                            else:
                                best_value = "NOT EVALUATED"
                        else:
                            best_value = BAD_METRIC_VALUE
                        mname = metric.name
                        if len(mname) > 15:
                            mname = mname[0:15]
                        s += "  %15s: " % mname
                        if metric_value == BAD_METRIC_VALUE:
                            s += "our: %15s " % "BAD VALUE"
                        else:
                            if type(metric_value) == float:
                                s += "our: %15g " % metric_value
                            elif type(metric_value) == int:
                                s += "our: %15d " % metric_value
                            else:
                                s += "our: %15s " % str(metric_value)
                        if best_value == BAD_METRIC_VALUE:
                            s += "best: %15s " % "BAD VALUE"
                        else:
                            if type(best_value) == float:
                                s += "best: %15g " % best_value
                            elif type(best_value) == int:
                                s += "best: %15d " % best_value
                            else:
                                s += "best: %15s " % str(best_value)
                        s += " [min: %10g, max: %10g, objective: %5s]\n" % \
                                (metric.min_threshold, metric.max_threshold, metric.improve_past_feasible)
            log.info(s)
        except:
            log.info("\nMetric performance:\n  FAILED")

    def _improveTopologyOnDOCs(self, ind):
        """This routine returns a modification of ind, which has changed the
        ind's parameters (but not topology) to try to make it pass function DOCs,
        simulation DOCs, and even start to meet performance constraints.
        """
        log.info("improveTopologyOnDOCs: begin")

        emb_part = self.ps.embedded_part
        unscaled_point = self.ps.unscaledPoint(ind)
        free_vars = self._freeParameters(unscaled_point)
        if not free_vars:
            log.info("Early exit on improveTopologyOnDOCs: no free vars")
            return ind

        best_ind, best_cost = ind, INF
        
        #Phase I: Get an ind that isn't INF when simulated (ie meets func DOCs, not horrible otherwise)
        sim_ind = ind
        log.info("Phase I: begin (topology: %s)" % (sim_ind.topoSummary()))
        for i in range(self.ss.max_num_inds_phase_I):
            (sim_ind, num_failing) = self._improveTopologyOnFunctionDOCs(
                sim_ind, self.ss.max_num_tries_on_func, free_vars)
            if num_failing == 0:
                sim_cost = self._nominalDOCsCost(sim_ind)

                self._showIndMetricPerformance(sim_ind, best_ind)

                if sim_cost != INF:
                    best_ind, best_cost = sim_ind, sim_cost
                    break
            log.info("Phase I: try to get non-INF sim ind, num_tries = %d / %d, %d failures" %
                     (i+1, self.ss.max_num_inds_phase_I, num_failing))

        if best_cost == INF:
            log.info("Early exit on improveTopology: couldn't get non-INF cost")
            return best_ind 

        #Phase II: Hillclimb to meet simulation DOCs.
        # -even if we haven't met them by when we're done, that's ok
        (best_ind, best_cost) = self._improveTopologyOnSimulationDOCs(
            best_ind, self.ss.max_num_inds_phase_IIa, self.ss.max_num_inds_phase_IIb, free_vars)

        #corner case: if ind is still bad, phase III won't help, so exit early
        if best_ind.nominalIsBad() or (self._nominalDOCsCost(best_ind) is INF):
            return best_ind
        
        #Phase III: hillclimb or gto to meet general constraints
        if self.ss.use_hillclimb_not_gto_phase_III:
            best_ind = self._hillclimbImproveTopologyOnConstraintViolations(
                best_ind, self.ss.max_num_inds_phase_III, free_vars)
        else:
            best_ind = self._gtoImproveTopologyOnConstraintViolations(
                best_ind, self.ss.max_num_inds_phase_III, free_vars)

        log.info("improveTopologyOnDOCs: done.  Final ind: %s" % _indStr(best_ind))

        return best_ind

    def _improveTopologyOnFunctionDOCs(self, ind, max_num_tries_on_func, free_vars):
        """Helper to _improveTopologyOnDOCs -- the functions part"""
        num_global_samples = max_num_tries_on_func / 10 #magic number alert
        
        emb_part = self.ps.embedded_part
        pm = emb_part.part.point_meta
        
        unscaled_point = self.ps.unscaledPoint(ind)
        num_failing = emb_part.numFailingFunctionDOCs(pm.scale(unscaled_point))
        
        last_step_was_improvement = False
        best_unscaled_point, best_num_failing = unscaled_point, num_failing
        for try_i in range(max_num_tries_on_func):
            if best_num_failing == 0:
                break
            
            if last_step_was_improvement:
                unscaled_point = self._repeatChangeFreeVars(
                                         best_unscaled_point,
                                         previous_unscaled_point,
                                         free_vars, 0.1)
            else:
                if try_i < num_global_samples:  random_policy = "uniform_allvars" 
                else:                           random_policy = "uniform_onevar"
                unscaled_point = self._randomChangeFreeVars(best_unscaled_point, random_policy, free_vars,
                                                            self.ss.mutate_stddev_for_DOC_compliance)            

            num_failing = emb_part.numFailingFunctionDOCs(pm.scale(unscaled_point))
            if num_failing <= best_num_failing:
                if num_failing < best_num_failing:
                    log.info("Improved to %d failures..." % (num_failing))
                    last_step_was_improvement = True
                    previous_unscaled_point = best_unscaled_point
                else:
                    last_step_was_improvement = False
                best_unscaled_point, best_num_failing = unscaled_point, num_failing
            else:
                last_step_was_improvement = False

        best_ind = self._createInd(best_unscaled_point)
        #log.info("After %d tries, did we get a circuit that meets function DOCS? %s" %
        #         (try_i + 1, num_failing == 0))
        
        return (best_ind, best_num_failing)

    def _improveTopologyOnSimulationDOCs(self, ind, max_num_inds_a, max_num_inds_b, free_vars):
        """
        -get non-INF sim-DOCS cost
        -try to get sim-DOCs cost = 0.0 via hillclimbing
        """
        #step 1: go for non-inf sim DOCs cost
        log.info("Phase II step 1/2: repeat until sim DOCs cost < Inf")
        (ind, cost) = self._improveTopologyOnSimulationDOCs_Hillclimb(
            ind, max_num_inds_a, free_vars, just_noninf_cost=True)
        
        #corner case: already hit our target, so no more work needed
        if cost == 0.0: 
            log.info("Phase II: can stop because hit cost == 0.0")
            return (ind, cost)

        #corner case: don't have circuit analyses, so return early
        if not self.ps.circuitAnalyses():
            log.info("Phase II: can stop because no circuit analyses")
            return (ind, cost)
        
        #step 2: hillclimb, going for sim-DOCs cost = 0.0
        log.info("Phase II step 2/2: simulation-based hillclimb")
        (best_ind, best_cost) = self._improveTopologyOnSimulationDOCs_Hillclimb(
            ind, max_num_inds_b, free_vars, just_noninf_cost=False)
        
        log.info("Done phase II")
        return (best_ind, best_cost)

    def _improveTopologyOnSimulationDOCs_Hillclimb(self, ind, max_num_inds, free_vars, just_noninf_cost):
        """ Hillclimb to meet simulation DOCs.
        """
        log.info("Phase II: begin hillclimb")
        
        cost = self._nominalDOCsCost(ind)
        best_ind, best_cost = ind, cost
        
        last_step_was_improvement = False
        
        for try_i in range(max_num_inds):
            log.info("Phase II hillclimb ind #%d / %d, DOCs cost=%.5f, best DOCs cost=%.5f" %
                     (try_i+1, max_num_inds, cost, best_cost))

            #maybe stop
            if best_cost == 0.0:
                log.info("Successfully got all function + sim DOCs to pass; ind %s" % _indStr(best_ind))
                break
            
            #maybe early stop
            if just_noninf_cost and (best_cost < INF):
                log.info("Successfully got non-INF function DOCs cost, so stopping")
                break

            if (cost < INF) and (best_cost < INF):
                self._showIndMetricPerformance(ind, best_ind)

            #create new ind
            if last_step_was_improvement:
                log.info("Repeating last step...")
                unscaled_point = self._repeatChangeFreeVars(
                                         self.ps.unscaledPoint(best_ind),
                                         previous_unscaled_point,
                                         free_vars, 1.0)
            else:
                if random.random() < 0.20: random_policy = "uniform_onevar" #magic number alert
                else:                      random_policy = "mutate_allvars"
                unscaled_point = self._randomChangeFreeVars(
                    self.ps.unscaledPoint(best_ind), random_policy, free_vars,
                    self.ss.mutate_stddev_for_DOC_compliance)

            ind = self._createInd(unscaled_point)

            #determine cost
            cost = self._nominalDOCsCost(ind)

            #update best
            if cost <= best_cost: 
                if cost < best_cost: 
                    log.info("Cost improved from %s to %s." % (best_cost, cost))
                    last_step_was_improvement = True
                    previous_unscaled_point = self.ps.unscaledPoint(best_ind)
                else:
                    last_step_was_improvement = False
                best_ind, best_cost = ind, cost
            else:
                last_step_was_improvement = False


        log.info("Phase II: done hillclimb")
        return (best_ind, best_cost)

    def _hillclimbImproveTopologyOnConstraintViolations(self, ind, max_num_inds, free_vars):
        """Hillclimb to minimize constraint violations

        For phase III, the hillclimbing option.
        """
        log.info("Begin phase III: hillclimb to improve topology on constraint violations")
        
        cost = self._nominalConstraintViolationsCost(ind)
        best_ind, best_cost = ind, cost
        
        last_step_was_improvement = False
        
        for try_i in range(max_num_inds):
            log.info("Phase III ind #%d / %d, constraints cost=%.5f, best constraints cost=%.5f" %
                     (try_i+1, max_num_inds, cost, best_cost))

            #maybe stop
            if best_cost == 0.0:
                log.info("Phase II: Hit cost=0.0")
                break

            #create new ind
            if random.random() < 0.10: random_policy = "uniform_onevar" #magic number alert
            else:                      random_policy = "mutate_allvars"
            unscaled_point = self._randomChangeFreeVars(
                self.ps.unscaledPoint(best_ind), random_policy, free_vars,
                self.ss.mutate_stddev_for_constraint_compliance)
                
            ind = self._createInd(unscaled_point)

            #determine cost
            cost = self._nominalConstraintViolationsCost(ind)

            if (cost < INF) and (best_cost < INF):
                self._showIndMetricPerformance(ind, best_ind)

            #update best
            if cost <= best_cost: 
                if cost < best_cost:
                    log.info("Cost improved from %s to %s." % (best_cost, cost))
                    
                    # try running in this direction for a while
                    still_improved = True
#                     initial_step_multiplier = self.ss.initial_step_multiplier
                    initial_step_multiplier = 0.1
                    step_multiplier = initial_step_multiplier
                    previous_unscaled_point = self.ps.unscaledPoint(best_ind)
                    current_unscaled_point = self.ps.unscaledPoint(ind)
                    base_step = self._diffUnscaledPoints(current_unscaled_point, previous_unscaled_point)
                    
                    nbiter = 0
                    
                    while still_improved:
                        nbiter += 1
                        log.info(" Directional search iteration # %s: cost %s" % (nbiter, cost))
                        # create new point
                        next_unscaled_point = self._stepFreeVars(
                                                    current_unscaled_point,
                                                    base_step,
                                                    free_vars, step_multiplier)
                        #create ind and determine cost
                        next_ind = self._createInd(next_unscaled_point)
                        next_cost = self._nominalConstraintViolationsCost(next_ind)

                        if (next_cost < INF) and (cost < INF):
                            self._showIndMetricPerformance(next_ind, ind)
            
                        if next_cost < cost:
                            log.info(" Directional search: cost improved from %s to %s. (step mult: %s)" % (cost, next_cost, step_multiplier))
                            # take larger step
                            step_multiplier = 2.0 * step_multiplier
                            # save the best ind
                            ind = next_ind
                            cost = next_cost
                            still_improved = True
                        else:
                            # the last step failed. It could however be that the
                            # step got too big.
                            if step_multiplier == initial_step_multiplier:
                                # this means two successive failures on the minimum step
                                still_improved = False
                            else:
                                # reset step multiplier
                                step_multiplier = initial_step_multiplier
                                
                                # switch base point to the last good point
                                current_unscaled_point = self.ps.unscaledPoint(ind)
                                
                                # restart the improvement with the base step
                                still_improved = True
                                
                best_ind, best_cost = ind, cost


        log.info("Done Phase III")
        return best_ind      

    def _gtoImproveTopologyOnConstraintViolations(self, ind, max_num_inds, init_free_vars):
        """Gto to minimize constraint violations
        
        For phase III, the gto option.
        """
        log.info("Begin phase III: gto to improve topology on constraint violations")
        
        cost = self._nominalDOCsCost(ind)
        assert cost is not INF, "should not get here with an ind having INF cost"

        #set base info
        # -full_pm is point meta of all variables
        emb_part = self.ps.embedded_part
        full_pm = emb_part.part.point_meta

        # -free_vars is gto's point meta.  Gto needs all its vars to be continuous, so filter
        # -fixed_vars is non-free vars.
        free_vars = [var for var in init_free_vars if isinstance(full_pm[var], ContinuousVarMeta)]
        fixed_vars = mathutil.listDiff(full_pm.keys(), free_vars)

        # -set full/fixed_unscaled_point
        full_unscaled_point = self.ps.unscaledPoint(ind)
        fixed_unscaled_point = Point(False, {})
        for var in fixed_vars:
            fixed_unscaled_point[var] = full_unscaled_point[var]

        # -free_pm is a subset of full_pm
        free_pm = PointMeta({})
        for var in free_vars:
            free_pm[var] = full_pm[var]
            
        #setup gto_ps
        gto_ps = GtoProblemSetup(self._allMetrics(), free_pm)
            
        #setup gto_ss
        gto_ss = copy.copy(self.ss.gto_ss)
        gto_ss.setTargetWeightPerObjective([1.0] * gto_ps.numObjectives())
        gto_ss.setMaxNumPoints(min(gto_ss.max_num_opt_points, max_num_inds))
        gto_ss.setTargetCost(0.0)
        gto_ss.init_radius = 0.02

        #setup gto state
        init_opt_point = free_pm.scale(full_unscaled_point)
        gto_state = GtoState(gto_ps, gto_ss, init_opt_point)

        #main optimization loop
        loop_i = 0
        while True:
            loop_i += 1

            #[at this point: simulate 'gto_inds' and set each gto_ind's 'cost']
            gto_inds = gto_state.indsWithoutCosts()
            for (i, gto_ind) in enumerate(gto_inds):
                #set gto_ind's WC metric values: scale the point, simulate, transfer from sim=>gto
                # -note that it's ok to set BAD_METRIC_VALUEs here
                sim_ind = self._gtoIndToSimInd(gto_ind, gto_state, free_pm, fixed_unscaled_point)

                #simulate on non-transient.  If not BAD, then simulate transient too.
                self._nominalEvalIndOnAllButTransient(sim_ind)
                if not sim_ind.nominalIsBad():
                    self._nominalEvalIndOnTransient(sim_ind)
                if sim_ind.nominalIsBad():
                    sim_ind.forceFullyBad()
                gto_ind.setWorstCaseMetricValues(sim_ind.nominalEvaluatedWorstCaseMetricValues())
                log.info("Phase III (gto): iter #%d, evaluated ind #%d / %d; %s" %
                         (loop_i, i+1, len(gto_inds), gto_state.indCostStr(gto_ind)))

            if gto_state.doStop():
                break

            #main 'smart' step
            gto_state.update()

        #done, so wrapup and return
        best_sim_ind = self._gtoIndToSimInd(gto_state.bestInd(), gto_state, free_pm, fixed_unscaled_point)
        log.info("Done Phase III: gto to improve topology on constraint violations")
        return best_sim_ind

    def _hillclimbImproveTopologyForWeight(self, ind, max_num_inds, free_vars, weight):
        """Hillclimb to minimize weigth cost

        """
        log.info("hillclimb to improve topology on weight: %s" % weight)
        
        cost = self._nominalWeightCost(ind, weight)
        best_ind, best_cost = ind, cost
        
        last_step_was_improvement = False
        
        for try_i in range(max_num_inds):
            log.info("Improve ind try #%d / %d, cost=%.5f, best cost=%.5f" %
                     (try_i+1, max_num_inds, cost, best_cost))

            #create new ind
            if random.random() < 0.10: random_policy = "uniform_onevar" #magic number alert
            else:                      random_policy = "mutate_allvars"
            unscaled_point = self._randomChangeFreeVars(
                self.ps.unscaledPoint(best_ind), random_policy, free_vars,
                self.ss.mutate_stddev_for_constraint_compliance)

            ind = self._createInd(unscaled_point)

            #determine cost
            cost = self._nominalWeightCost(ind, weight)
            if (cost < INF) and (best_cost < INF):
                self._showIndMetricPerformance(ind, best_ind)

            #update best
            if cost < best_cost:
                log.info("Cost improved from %s to %s." % (best_cost, cost))

                # try running in this direction for a while
                still_improved = True
                #initial_step_multiplier = self.ss.initial_step_multiplier
                initial_step_multiplier = 0.1
                step_multiplier = initial_step_multiplier
                previous_unscaled_point = self.ps.unscaledPoint(best_ind)
                current_unscaled_point = self.ps.unscaledPoint(ind)
                base_step = self._diffUnscaledPoints(current_unscaled_point, previous_unscaled_point)

                nbiter = 0
                while still_improved:
                    nbiter += 1
                    log.info(" Directional search iteration # %s: cost %s" % (nbiter, cost))
                    # create new point
                    next_unscaled_point = self._stepFreeVars(
                                                current_unscaled_point,
                                                base_step,
                                                free_vars, step_multiplier)
                    #create ind and determine cost
                    next_ind = self._createInd(next_unscaled_point)
                    next_cost = self._nominalConstraintViolationsCost(next_ind)
                    if (next_cost < INF) and (cost < INF):
                        self._showIndMetricPerformance(next_ind, ind)

                    if next_cost < cost:
                        log.info(" Directional search: cost improved from %s to %s. (step mult: %s)" % (cost, next_cost, step_multiplier))
                        # take larger step
                        step_multiplier = 2.0 * step_multiplier
                        # save the best ind
                        ind = next_ind
                        cost = next_cost
                        still_improved = True
                    else:
                        log.info(" Directional search: cost not improved (step mult: %s)" % (step_multiplier))
                        # the last step failed. It could however be that the
                        # step got too big.
                        if step_multiplier == initial_step_multiplier:
                            # this means two successive failures on the minimum step
                            still_improved = False
                        else:
                            # reset step multiplier
                            step_multiplier = initial_step_multiplier
                            # switch base point to the last good point
                            current_unscaled_point = self.ps.unscaledPoint(ind)
                            # restart the improvement with the base step
                            still_improved = True

                best_ind, best_cost = ind, cost

        # make sure it is evaluated properly (analysis might have been skipped due to waterfall)
        self._nominalEvalInd(best_ind)

        log.info("Done improve")
        return best_ind

    def _gtoIndToSimInd(self, gto_ind, gto_state, free_pm, fixed_unscaled_point):
        """Converts a 'GtoInd' to 'Ind'"""
        free_scaled_point = gto_state.x01ToScaledPoint(gto_ind.x01)
        full_unscaled_point = free_pm.unscale(free_scaled_point)
        for (var, val) in fixed_unscaled_point.iteritems():
            full_unscaled_point[var] = val
        sim_ind = self._createInd(full_unscaled_point)
        return sim_ind

    def _functionDOCsAreFeasible(self, ind):
        """Returns True if ind meets function DOCs, else False"""
        #main case
        scaled_opt_point = self.ps.scaledPoint(ind)
        emb_part = self.ps.embedded_part
        feasible = emb_part.functionDOCsAreFeasible(scaled_opt_point)
        return feasible

    def _nominalDOCsCost(self, ind):
        """This is a cost function for func DOCs and sim DOCs.  (But not general constraints.)"""
        #gate 1: eval on function DOCs.  If not met, return INF.
        if not self._functionDOCsAreFeasible(ind):
            log.debug("_nominalDOCsCost: exit at gate 1: cost = INF because function DOCs infeasible")
            return INF

        self.num_inds += 1
        cost = 0.0
        for an in self.ps.analysesSortedByCost():
            if an.hasDOC():
                self._nominalEvalIndOnAnalysis(an, ind)
                if ind.nominalIsBad():
                    log.debug("_nominalDOCsCost: exit at gate 2: cost = INF because BAD value when evaluating %s" % [an])
                    return INF

                for metric in an.getDOCmetrics():
                    metric_value = ind.nominalWorstCaseMetricValue(metric.name)
                    cost += metric.constraintViolation01(metric_value)

                # if it doesn't have a DOC cost, proceed to the next analysis

        #gate 5: ind is not BAD, so return sim DOCS cost.
        #FIXME: make this part more general for _all_ values!!
        assert self.ps.problem_choice != 15, "make this compliant"

        log.debug("_nominalDOCsCost: exit at gate 5: cost = %.8e" % cost)

        return cost
    
    def _nominalConstraintViolationsCost(self, ind):
        """This is a cost function for func DOCs, sim DOCs,  AND general constraints."""
        #gate 1: meeting DOCs?
        cost_mult = 1000.0 # this should be way higher than the max cost a certain analysis can get
        max_cost = self.ps.maxAnalysisCost()

        docs_cost = self._nominalDOCsCost(ind)
        if docs_cost > 0.0:
            return cost_mult*max_cost + docs_cost

        for an in self.ps.analysesSortedByCost():
            self._nominalEvalIndOnAnalysis(an, ind)
            if ind.nominalIsBad():
                return INF

            base_cost = (max_cost - an.relative_cost) * cost_mult
            extra_cost = 0.0
            for metric in an.metrics:
                metric_value = ind.nominalWorstCaseMetricValue(metric.name)
                violation01 = metric.constraintViolation01(metric_value)
                if self.ss.metric_weights.has_key(metric.name):
                    w = self.ss.metric_weights[metric.name]
                else:
                    w = 1.0
                extra_cost += w * violation01

            # early exit on a low-cost analysis
            # the idea is that lower cost (= effort) analysis end up
            # with higher DOC costs, in a way that will ensure that
            # a more expensive analysis DOC cost will never exceed
            # the cost of any less expensive analysis
            if extra_cost > 0.0:
                return base_cost + extra_cost

            # if it doesn't have a cost, proceed to the next analysis

        # we saw all analysis and they had no cost
        return 0.0

    def _nominalWeightCost(self, ind, weight):
        """Cost of this ind for a given weight vector """
        metric_bounds = {}

        #gate 1: meeting DOCs? 
        docs_cost = self._nominalDOCsCost(ind)
        if docs_cost > 0.0:
            return max(docs_cost, 1000.0 + 1000.0 * docs_cost)

        #gate 2: not BAD on non-transient analyses?
        self._nominalEvalIndOnAllButTransient(ind) #DOCsCost should have done this, but just to be sure...
        if ind.nominalIsBad():
            return INF

        #gate 3: not BAD on transient analyses?
        self._nominalEvalIndOnTransient(ind)
        if ind.nominalIsBad():
            return INF

        # calculate scalar cost
        for metric in self.ps.flattenedMetrics():
            metric_bounds[metric.name] = (metric.rough_minval, metric.rough_maxval)

        return ind.scalarCost(1, weight, self.ss.metric_weights, metric_bounds)

    def _freeParameters(self, unscaled_point):
        """Returns 'free_vars', a list of vars that:
        1. Are 'active', i.e. have max > min
        2. Are 'parameters': vars that are not 'choice' vars, i.e. do not affect the topology
        3. For the topology as defined by the ind's choice vars, they have an effect on the netlist

        (This amounts to the intersection of 1,2,3).
        """
        #preconditions
        assert not unscaled_point.is_scaled

        #base data
        emb_part = self.ps.embedded_part
        point_meta = emb_part.part.point_meta
        scaled_point = point_meta.scale(unscaled_point)
        
        #main work
        active_vars = set(point_meta.varsWithMultipleOptions())
        nonchoice_vars = set(point_meta.nonChoiceVars())
        vars_used = set(emb_part.varsUsed(scaled_point))
        active_nonchoice_vars = list(active_vars & nonchoice_vars & vars_used)

        #done
        return active_nonchoice_vars
    
    def _randomChangeFreeVars(self, unscaled_point, random_policy, free_vars, mutate_stddev):
        """
        @description

          Randomly change the point values without changing topology choice variables.
          
        @arguments

          unscaled_point -- Point
          random_policy -- 'uniform_allvars', 'uniform_onevar', 'mutate_allvars', 'mutate_onevar'
          free_vars -- list of string -- specifies the variables that are non-choice variables
            that actually have an impact on the netlist
          mutate_stddev -- float -- e.g. from ss.mutate_stddev*
          
        @return
        
          new_unscaled_point -- Point
    
        @exceptions
    
        @notes
        """
        #preconditions
        assert random_policy in ["uniform_allvars", "uniform_onevar", "mutate_allvars", "mutate_onevar"]
        assert free_vars, "if we get here we should have vars available"
        assert not unscaled_point.is_scaled

        #main work
        pm = self.ps.embedded_part.part.point_meta

        if "allvars" in random_policy:
            vars_to_change = free_vars
        else:
            vars_to_change = [random.choice(free_vars)]

        new_unscaled_point = Point(False, copy.copy(unscaled_point))
        for var in vars_to_change:
            if "uniform" in random_policy:
                new_val = pm[var].createRandomUnscaledVar(False)
            else:
                new_val = pm[var].mutate(unscaled_point[var], mutate_stddev, False)
            new_unscaled_point[var] = new_val
            
        return new_unscaled_point

    def _diffUnscaledPoints(self, unscaled_point, unscaled_point_prev):
        """
        @description

          Calculates the difference between two unscaled points and unrailed
          
        @arguments

          unscaled_point -- Point as it is now
          unscaled_point_prev -- Point one step before
   
        @return
        
          diff -- Point
    
        @exceptions
    
        @notes
        """
        #preconditions
        assert not unscaled_point.is_scaled
        assert not unscaled_point_prev.is_scaled

        #main work
        diff = {}
        for var in unscaled_point.keys():
            diff[var] = unscaled_point[var] - unscaled_point_prev[var]

        return diff

    def _stepFreeVars(self, unscaled_point, diff, free_vars, step_multiplier = 1.0):
        """
        @description

          Change the point values without changing topology choice variables. Tries to repeat the step
          that was taken to get from unscaled_point_prev to unscaled_point.
          
        @arguments

          unscaled_point -- Point as it is now
          diff -- the difference to apply in the free vars (dict)
          free_vars -- list of string -- specifies the variables that are non-choice variables
            that actually have an impact on the netlist
          step_multiplier -- the multiplier that is applied to the step
          
        @return
        
          new_unscaled_point -- Point
    
        @exceptions
    
        @notes
        """
        #preconditions
        assert free_vars, "if we get here we should have vars available"
        assert not unscaled_point.is_scaled

        #main work
        pm = self.ps.embedded_part.part.point_meta
        new_unscaled_point = Point(False, copy.copy(unscaled_point))
#         log.info("Repeating step...")
        sum_delta = 0
        for var in free_vars:
            # take the step, but railbin the vars
            delta = diff[var]
            sum_delta += delta != 0.0
#             log.info(" %25s %20s => %20s = %s" % (var, unscaled_point_prev[var], unscaled_point[var], delta))
            new_unscaled_point[var] = pm[var].railbinUnscaled(unscaled_point[var] + step_multiplier * delta)
        
        if sum_delta == 0:
            log.info("no step difference:!\n%s\n%s" % (str(unscaled_point_prev), str(unscaled_point)))
            
        return new_unscaled_point

    def _repeatChangeFreeVars(self, unscaled_point, unscaled_point_prev, free_vars, step_multiplier = 1.0):
        """
        @description

          Change the point values without changing topology choice variables. Tries to repeat the step
          that was taken to get from unscaled_point_prev to unscaled_point.
          
        @arguments

          unscaled_point -- Point as it is now
          unscaled_point_prev -- Point one step before
          free_vars -- list of string -- specifies the variables that are non-choice variables
            that actually have an impact on the netlist
          step_multiplier -- the multiplier that is applied to the step
          
        @return
        
          new_unscaled_point -- Point
    
        @exceptions
    
        @notes
        """
        #preconditions
        diff = self._diffUnscaledPoints(unscaled_point, unscaled_point_prev)
        return self._stepFreeVars(unscaled_point, diff, free_vars, step_multiplier)

    def _newRandomInd(self, with_novelty, testdata):
        """
        @description

          Generate a new ind at random.  Well, not quite random because
          if with_novelty is True then it may use a higher-age-layer ind
          as a parent.  Do _not_ evaluate it (leave that to calling routine.)
          Therefore it could be good or bad.
        
        @arguments

          with_novelty -- bool -- inject novelty when creating the ind?
            (tries to get novelty, but to be guaranteed,
             call newRandomInd_NovelIfNeeded instead of this routine)
        
          testdata -- for validation
          
        @return

          new_ind -- Ind object
    
        @exceptions
    
        @notes
        """
        #retrieve base info
        point_meta = self.ps.embedded_part.part.point_meta
        
        #preconditions
        if AGGR_TEST and (not self.doing_profiling):
            # -aggressive test: mess up any other inds?
            (test_parents, test_Q) = testdata
            targ_vars = point_meta.keys()
            for i, ind in enumerate(test_parents + test_Q):
                assert len(targ_vars) == len(ind.unscaled_optvals)

        #main work...
            
        #build new unscaled point 'new_point' (and also set 'parents')
        new_vals, parents = None, None 
        if not with_novelty:
            #always fully random
            log.info("Create (non-novel) random ind from scratch")
            # first step: 
            new_point = point_meta.createRandomUnscaledPoint(with_novelty = False, use_weights = True)
            parents = []
            
        else:
            raise "FIXME"
            #since novel, allowed to have a parent from a higher age layer
            #choose an age layer.  So first select the layer
            if len(self.state.R_per_age_layer[0]) == 0:
                #corner case: no inds at all yet, forced to layer 0
                age_layer_i = 0
            else:
                #main case: have inds, choose layer based on biases
                #calc num_age_layers, being wary if highest layer is empty
                num_age_layers = max([i for i,R in enumerate(self.state.R_per_age_layer) if len(R) > 0])
                age_layer_i = mathutil.randIndex(self.ss.getNovelBiasPerLayer()[:num_age_layers])

            #layer is selected, so now create the opt point (maybe with parent)
            if age_layer_i == 0:
                log.info("Create novel random ind from scratch")
                new_point = point_meta.createRandomUnscaledPoint(with_novelty = False, use_weights = True) #NOTE: keep with_novelty to False!!
                parents = []
            else:
                ind = random.choice(self.state.R_per_age_layer[age_layer_i])
                parents = [ind]
                new_point = self.ps.unscaledPoint(ind)
                log.info("Create novel random ind with parent/hint ID=%s from age layer %d" % (ind.ID, age_layer_i))
            
            #add a novel part to the library
            factory = RandomPartFactory(self.ps.parts_library)
            (var_name, choice_value) = factory.build(
                self.ps.embedded_part, new_point)
            
            assert var_name in new_point.keys()

            #force ind to use that part
            new_point[var_name] = choice_value

        assert new_point is not None
        assert parents is not None

        #build ind
        new_ind = self._createInd(new_point)
        new_ind.setAncestry(parents)

        #postconditions
        if AGGR_TEST and (not self.doing_profiling):
            targ_vars2 = point_meta.keys()
            validateVarLists(targ_vars, targ_vars2, 'targ_vars', 'vars after')
                      
        #done
        return new_ind

    #============================================================================
    #Vary parents.  Includes crossover and mutation.
    def _newEvaluatedChild(self, par1, par2):
        """
        @description

          Varies par1 and par2 via mutation or crossover, and either an evaluated child or None.
          
          Repeats generating children as necessary until it is 'accepted', i.e. if:
          -its netlist is different than both parents
          -its simulation results are not 'bad'
          -its 'nice' metric string is different either parent's string,
           and different than any string in the input 'tabu_perfs'
         
        @arguments

          par1 -- Ind -- first parent
          par2 -- Ind -- second parent
        
        @return

          success -- bool -- True if two children were generated with fewer than 'max_num_rounds'
          child -- Ind or None -- offspring
    
        @exceptions
    
        @notes

          We only compare the nominal simulations to determine if performances are different.  Simpler, cheaper.
        """
        #extract num rnd points to eval child at
        # -Note that it is OK for parents to have been evaluated at fewer rnd points, because
        #  they may have come from a lower layer
        num_rnd_points = self.task.task_data.num_rnd_points
        
        #number of rounds at generating children.
        # If this is exceeded, stops and returns success=False.
        max_num_rounds = 500 #magic number alert
        tabu_perfs = [ind.nominalWorstCaseMetricValuesStr()
                      for ind in self.task.task_data.parent_set]

        # would this be to validate that children are unique?
        testdata = (self.task.task_data.parent_set, [])
        
        log.debug('Vary parents to get two good, unique children: begin')
        
        par1_nom_perf = par1.nominalWorstCaseMetricValuesStr()
        par2_nom_perf = par2.nominalWorstCaseMetricValuesStr()
        
        #log.debug('Vary parents: round #%d, num_inds=%d' % (vary_round, self.num_inds))

        #note: _varyParents gives children with netlists that are different than either parent's netlist
        child = self._varyParents(par1, par2, testdata)

        #filter 1: did we successfully vary parents to create a unique child?
        if child is None:
            return None

        self.num_inds += 1

        #filter 2: simulate at nominal, and test
        self._nominalEvalInd(child)
        child_nom_perf = child.nominalWorstCaseMetricValuesStr()
        perfs_same = (child_nom_perf == par1_nom_perf) or (child_nom_perf == par2_nom_perf) or \
                     (child_nom_perf in tabu_perfs)
        if child.nominalIsBad():
            log.info('Do not keep child b/c bad nominal simulation results')
            return None
        
        if perfs_same:
            log.info('Do not keep child b/c (nominal) performances not unique')
            return None

        #filter 3: simulate at non-nominal, and test
        self._evalInd(child, child.rnd_IDs[1:num_rnd_points])
        if child.isBad(num_rnd_points):
            log.info('Do not keep child b/c bad non-nominal simulation results')
            return None

        #success
        log.info("Success: keep child")
        return child
        
    def _varyParents(self, par1, par2, testdata):
        """
        @description

          Varies par1 and par2 via mutation or crossover, and returns 1 child
          Repeats variation as necessary so ensure that netlist is different.
        
        @arguments

          par1 -- Ind -- first parent
          par2 -- Ind -- second parent

          testdata -- pass this through for validation at mutation/xover level
        
        @return

          child -- Ind or None -- newly generated ind #1, or None if not success
    
        @exceptions
    
        @notes
        """
        #preconditions
        assert par1 is not None
        assert par2 is not None
        
        #choose how much to vary each ind
        num_vary1 = mathutil.randIndex(self.ss.num_vary_biases)

        #maybe do crossover before mutating
        do_crossover = random.random() < self.ss.prob_crossover
        if do_crossover:
            child = self._crossoverInds(par1, par2, testdata)
            num_vary1 = max(0, num_vary1 - 2)
        else:
            child = par1

        #Always mutate.  Keep mutating until the child is different.
        # (There is huge chance of a same child due to the structure of GRAIL)
        child = self._mutateInd(child, num_vary1)
        (success, child) = self._mutateUntilDifferent(child, par1.nominalNetlist())
        
        if success:
            #set genetic age, parent_IDs, ancestor_IDs
            if do_crossover:
                child.genetic_age = max(par1.genetic_age, par2.genetic_age)
                child.setAncestry([par1, par2])
            else:
                child.genetic_age = par1.genetic_age
                child.setAncestry([par1])

            return child
        else:
            return None

    def _mutateUntilDifferent(self, ind, reference_netlist):
        """Mutate ind until its netlist is different than reference_netlist.
        Note that it does not continually build on the mutates, i.e.
        we do _not_ wander throughout the neutral space because
        that risks damaging the hidden building blocks.

        Returns (True, ind) or (False, None) where 1st entry indicates success.
        """
        num_tries = 0
        while True:
            #avoid infinite loop; and excessive effort
            if num_tries > 200:
                return (False, None)

            num_vary = mathutil.randIndex(self.ss.num_vary_biases)
            mut_ind = self._mutateInd(ind, num_vary)
            if mut_ind.nominalNetlist() != reference_netlist:
                return (True, mut_ind)
                                      
        return (False, ind)

    def _crossoverInds(self, par1, par2, testdata):
        """
        @description

          Cross over parent1 and parent2 to create two children, and
          tries to preserve building blocks to the best extent possible.

          Algorithm:
          -in par1, pick a sub-part (or sub-sub-part, etc)
          -determine all the variables that affect that part
          -give par2 those vars' values, and take par2's values in return
        
        @arguments

          par1 -- Ind -- first parent
          par2 -- Ind -- second parent

          testdata -- for validation
        
        @return

          child -- Ind -- newly generated ind
    
        @exceptions
    
        @notes

          The genetic_age attribute of the children is still None. Leave
          setting that to higher-level routines.

          Note that some of the 'affecting' variables will affect places other
          than the part too, i.e. there is crosstalk among trees.  That's
          ok, while this isn't perfect, it's far better than the fully-naive
          uniform crossover which has 100% crosstalk.
        """
        #preconditions
        targ_vars = self.ps.ordered_optvars
        (parents, Q) = testdata

        #retrieve parent info
        par1_vals = par1.unscaled_optvals
        par2_vals = par2.unscaled_optvals
        
        #preconditions round 2
        if AGGR_TEST:
            # -parents test
            assert len(par1_vals) == len(targ_vars)
            assert len(par2_vals) == len(targ_vars)

            # -aggressive test: mess up any other inds?
            for i, ind in enumerate(parents + Q):
                assert len(ind.unscaled_optvals) == len(targ_vars)
        
        
        #choose a sub-part, and find which vars affect it (including
        # alternate topology sub-implementations)
        # -get info
        emb_part = self.ps.embedded_part
        par1_scaled_point = self.ps.scaledPoint(par1)
        info_list = emb_part.subPartsInfo(par1_scaled_point)

        # -make choice
        cand_info_list = []
        for sub_part_info in info_list:
            (sub_part, sub_point, vars_affecting) = sub_part_info
            validateIsSubset(vars_affecting, par1_scaled_point.keys(),
                             'affecting', 'par1_scaled_point.keys()')
            if len(vars_affecting) > 0:
                cand_info_list.append(sub_part_info)

        assert len(cand_info_list) > 0, "vars should affect some parts"
        sub_part_info = random.choice(cand_info_list)
        (sub_emb_part, sub_scaled_point, vars_affecting) = sub_part_info
        if sub_emb_part is not None: name = sub_emb_part.part.name
        else:                        name = 'None'
        log.debug('Crossover: sub_emb_part=%s, vars_affecting=%s' % (name, vars_affecting))

        #build up values list for each child
        child_vals = []
        for (var, par1_val, par2_val) in izip(targ_vars, par1_vals, par2_vals):
            if var in vars_affecting:
                child_vals.append(par1_val)
            else:
                child_vals.append(par2_val)

        #build children
        child = NsgaInd(child_vals, self.ps)

        #postconditions
        if AGGR_TEST:
            # -accidentally changed the point meta?
            targ_vars2 = self.ps.ordered_optvars
            validateVarLists(targ_vars, targ_vars2,
                             'original target vars', 'target vars after')

            # -inds created ok?
            assert len(child.unscaled_optvals) == len(targ_vars)

            # -aggressive test: mess up any other inds?
            for i, ind in enumerate(parents + Q):
                assert len(ind.unscaled_optvals) == len(targ_vars)

        #done
        return child
        
    def _mutateInd(self, parent_ind, num_mutates):
        """
        @description

          Applies 'num_mutates' mutations to parent_ind.
        
        @arguments

          parent_ind -- Ind_object -- ind to be mutated
          num_mutates -- int -- num consecutive mutates to apply

          testdata -- for validation
        
        @return

          child_ind --  Ind object --
    
        @exceptions
    
        @notes

          The genetic_age attribute of the child is still None. Leave
          setting that to higher-level routines.
        """
        #retrieve base info
        emb_part = self.ps.embedded_part
        point_meta = emb_part.part.point_meta
        active_vars = point_meta.varsWithMultipleOptions()

        #this is what we'll be modifying, and will eventually build child_ind with
        unscaled_point = self.ps.unscaledPoint(parent_ind)

        #repeat the following 'num_mutates' times.  (Usually 1 time, but sometimes more)
        for mutate_i in range(num_mutates):
            #part 1: maybe mutate choice var (no novelty!)
            if random.random() < self.ss.prob_mutate_1_choice_var:
                choice_vars = point_meta.choiceVars()
                active_choice_vars = mathutil.listIntersect(active_vars, choice_vars)
                if active_choice_vars:
                    var = random.choice(active_choice_vars)
                    unscaled_point[var] = point_meta[var].mutate(
                        unscaled_point[var], self.ss.mutate_stddev_during_evolution, False)

            #part 2: mutate one or more 'free vars' (active, non-choice vars that current topo. uses)
            free_vars = self._freeParameters(unscaled_point)
            if random.random() < 0.10: #magic number alert
                random_policy = "mutate_onevar"
            else:
                random_policy = "mutate_allvars"
            
            unscaled_point = self._randomChangeFreeVars(unscaled_point, random_policy, free_vars,
                                                        self.ss.mutate_stddev_during_evolution)

        #build Ind
        child = self._createInd(unscaled_point)
        
        #done
        return child
 
    #============================================================================
    # redo the simulations
    def _resimulateInd(self, ind):
        #log.info("simulating ind %s" % str(ind))
        ind.clearSimulations()
        self._evalInd(ind, ind.rnd_IDs[:self.task.task_data.num_rnd_points])
        return ind

    #==========================================================================
    #evaluate
    def _nominalEvalIndOnDOCs(self, ind):
        """Evaluates 'ind' on all function + simulation DOCs, at nominal rnd point"""
        scaled_opt_point = self.ps.scaledPoint(ind)
        for an in self.ps.analyses:
            if an.hasDOC():                    
                for e in an.env_points:
                    self._nominalEvalIndAtAnalysisEnvPoint(ind, an, e, scaled_opt_point)

    def _nominalEvalIndOnAnalysis(self, an, ind):
        """Evaluates 'ind' on the specific analysis, at nominal rnd point"""
        scaled_opt_point = self.ps.scaledPoint(ind)
        for e in an.env_points:
            self._nominalEvalIndAtAnalysisEnvPoint(ind, an, e, scaled_opt_point)

    def _nominalEvalIndOnAllButTransient(self, ind):
        """Evaluates 'ind' on everything except for transient analyses (which are expensive), at nominal rnd point"""
        scaled_opt_point = self.ps.scaledPoint(ind)
        for an in self.ps.analyses:
            if not an.hasSimulationType('tran'):
                for e in an.env_points:
                    self._nominalEvalIndAtAnalysisEnvPoint(ind, an, e, scaled_opt_point)
                    
    def _nominalEvalIndOnTransient(self, ind):
        """Evaluates 'ind' on JUST transient analyses (which are expensive), at nominal rnd point"""
        scaled_opt_point = self.ps.scaledPoint(ind)
        for an in self.ps.analyses:
            if an.hasSimulationType('tran'):
                for e in an.env_points:
                    self._nominalEvalIndAtAnalysisEnvPoint(ind, an, e, scaled_opt_point)

    def _allMetrics(self):
        """Returns all metrics except for ones in transient analysis"""
        return [metric
                for an in self.ps.analyses
                for metric in an.metrics]

    def _nominalEvalInd(self, ind):
        """Evaluate an ind on all analyses and env points, on the nominal rnd point"""
        self._evalInd(ind, [self.ps.nominalRndPoint().ID])
        
    def _evalInd(self, ind, rnd_IDs):
        """Evaluate an ind on all analyses and env points, on the specified rnd IDs"""
        #do the actual evaluation
        new_num_evals_per_an = Evaluator.evalInd(self.ps, ind, rnd_IDs)

        #udpate sim count
        for (anID, new_n) in new_num_evals_per_an.iteritems():
            if not self.num_evaluations_per_analysis.has_key(anID):
                self.num_evaluations_per_analysis[anID] = 0
            self.num_evaluations_per_analysis[anID] += new_n
                

    def _nominalEvalIndAtAnalysisEnvPoint(self, ind, analysis, env_point,
                                          prescaled_ind_opt_point=None, save_lis_results=False):
        """Pass through to Evaluator.nominalEvalIndAtAnalysisEnvPoint, but also updates sim count.
        """
        #udpate sim count
        if not self.num_evaluations_per_analysis.has_key(analysis.ID):
            self.num_evaluations_per_analysis[analysis.ID] = 0
        self.num_evaluations_per_analysis[analysis.ID] += 1
        
        #do the actual evaluation
        retval = Evaluator.nominalEvalIndAtAnalysisEnvPoint(
            self.ps, ind, analysis, env_point, prescaled_ind_opt_point, save_lis_results=save_lis_results)

        return retval

class LisCostFunction:
    """
    Returns the cost of 'query_point' on a sensitivity-based linear model characterized by:
    x = (center_point, perturb_point)
    y = (center_lis_results, perturb_lis_results_per_var).
    
    Used for phase II.
    """
    def __init__(self, center_point, perturb_point, center_lis_results, perturb_lis_results_per_var, ps):
        """
        center_point -- dict of var_name : var_value
        perturb_point -- dict of var_name : var_value -- like center point, but free_vars perturbed
        center_lis_results -- dict of [lis__device_name__measure_name] : lis_value
        perturb_lis_results_per_var -- dict of [var_name][lis__device_name__measure_name] : lis_value
        ps -- ProblemSetup
        """
        #preconditions
        assert center_point.ID != perturb_point.ID

        #set values
        self.center_point = center_point
        self.perturb_point = perturb_point
        self.perturb_lis_results_per_var = perturb_lis_results_per_var
        self.ps = ps

        # -only store the center_lis_results that are numeric
        self.center_lis_results = {}
        for (doc_name, doc_value) in center_lis_results.iteritems():
            if mathutil.isNumber(doc_value):
                self.center_lis_results[doc_name] = doc_value
                
    def __call__(self, query_point):

        #compute query_lis_results
        query_lis_results = {} # dict of 'lis__device_name__measure_name' : lis_value
        for (doc_name, base_doc_value) in self.center_lis_results.iteritems():
            doc_value = base_doc_value
            for (var, perturb_lis_results) in self.perturb_lis_results_per_var.iteritems():
                #corner case
                if not perturb_lis_results.has_key(doc_name):
                    return INF
                
                x1 = self.center_point[var]
                x2 = self.perturb_point[var]

                y1 = base_doc_value
                y2 = perturb_lis_results[doc_name]

                slope = (y2 - y1) / (x2 - x1)
                x = query_point[var]
                doc_value += slope * (x - x1)

            query_lis_results[doc_name] = doc_value
            
        #compute simulationDOCsCost from query_lis_results
        # -note that simulationDOCsCost() uses ps.embedded_part.functions which
        # was computed
        cost = self.ps.embedded_part.simulationDOCsCost(query_lis_results)
        
        #done
        return cost
    
def _indStr(ind):
    """Compact way to print out some info about an ind"""
    #return "ID=%s (values=%s)" % (ind.ID, ind.getValues()) #turn on this line for debugging
    return "ID=%s" % ind.ID
