"""Master.py

This is the main controller of the search, and the slaves.
It dispatches work for slaves, and retrieves their results and
tracks overall progress.

From a networking perspective, one can view Slaves as Servers,
and the Master as the Client.

The search algorithm is a combination of:
  -ALPS (ref Hornby, GECCO 2006)
  -NSGA-II (ref Deb, IEEE Trans EC 2003) 
  -MOEA/D (ref Zhang and Li, IEEE Trans 11(6) December 2007)
  -A-Teams -- asynchronous teams implementation for master-slave (ref Talukdar) 

"""
import copy
from itertools import izip
import logging
import os
import random
import shutil
import sys
import time
import types

import numpy

from adts import *
import engine.Channel as Channel
import engine.EngineUtils as EngineUtils
import engine.Evaluator as Evaluator
from engine.Slave import Slave
from engine.PopulationSummary import worstCasePopulationSummaryStr
from engine.SynthSolutionStrategy import SynthSolutionStrategy, updateSynthStrategy
from util import mathutil
from util.constants import *

log = logging.getLogger('master')

class Master(object):
    """
    @description

      Synthesizes circuits.
      
    @attributes

      cs -- ChannelSetup object -- describes the communication mechanism
                                   between master and slave(s)
      ps -- ProblemSetup object -- describes the search space and goals
      ss -- SynthSolutionStrategy object -- strategy parameters
      pop -- Pop -- list of Inds, which is effectively the current state
      output_dir -- string -- directory of where results are stored
      channel -- Channel -- how master and slave communicate.
      restart_file -- string -- name of state file, where to
        continue a previous run from (None if not wanted)
      random_pool_file -- string -- name of random pool file
     
    @notes
    """

    def __init__(self, cs, ps, ss, output_dir, restart_file, random_pool_file = None):
        """
        @description

          Constructor.

        """
        #preconditions
        assert isinstance(ps, ProblemSetup) or (ps is None)
        assert isinstance(ss, SynthSolutionStrategy) or (ss is None)
        assert isinstance(output_dir, types.StringType) or (output_dir is None)
        assert isinstance(restart_file, types.StringType) or (restart_file is None)
        assert isinstance(random_pool_file, types.StringType) or (random_pool_file is None)
        
        # the SS can be none only if the restart file is not none
        assert isinstance(ss, SynthSolutionStrategy) or isinstance(restart_file, types.StringType)
        
        # the PS can be none only if the restart file is not none
        assert isinstance(ps, ProblemSetup) or isinstance(restart_file, types.StringType)  

        #main work...
        
        #get 'loaded_state' if possible
        if restart_file is None:
            loaded_state = None
        else:
            #load previous state
            restart_file = os.path.abspath(restart_file)
            try:
                loaded_state = EngineUtils.loadSynthState(restart_file, ps)
            except:
                log.error("\n\nCould not open restart_file=%s.  Exiting.\n" % restart_file)
                sys.exit(0)

        #remember if past DB knew of robust, novelty
        past_had_robust = (loaded_state is not None) and loaded_state.ps.doRobust()
        past_had_novelty = (loaded_state is not None) and loaded_state.ss.do_novelty_gen
        
        #if no SS was specified we have to get it from the loaded state
        if ss is None:
            ss = loaded_state.ss
            updateSynthStrategy(ss)

        # if we didn't specify a PS we have to get it from the loaded state
        if ps == None:
            ps = loaded_state.ps

        #catch corner case: robust constraint
        if (loaded_state is not None) and (loaded_state.ps.doRobust() != ps.doRobust()):
            log.error("\n\nCannot go from a robust run to non-robust run, or vice-versa  Exiting.\n")
            sys.exit(0)

        #catch corner case: novelty constraint
        if past_had_novelty and (not ss.do_novelty_gen):
            log.error("\n\nCannot do non-novelty continuing from a novelty DB.  Exiting.\n")
            sys.exit(0)

        target_num_inds_per_age_layer = ss.num_inds_per_age_layer

        #need to guarantee that this search has _some_ 100% trustworthy inds
        # to work with.  
        if loaded_state is None:
            #case: no past DB, and doing novelty.  Check!
            if ss.do_novelty_gen:
                if ss.num_rnd_gen_rounds_between_novelty == 0:
                    log.warning("Input ss had # rnd gen rounds between novelty set to 0, but we need to guarantee _some_"
                                " trustworthy inds; therefore re-set it to 1")
                    ss.num_rnd_gen_rounds_between_novelty = 1
            #case: no past DB, but always 100% trustworthy.
            else:
                pass
        #case: past DB, which is guaranteed compliant
        else:
            pass

        #update ps to be novelty-ready (ps.embedded_part, ps.analyses)
        ps.validate()
        if past_had_novelty:
            #we'll be using loaded_state.ps which is already novelty-ready
            assert loaded_state is not None
        
        elif ss.do_novelty_gen:                
            self._ensurePossPartsToAddAreDeclaredAndCached(ps)

            #prepare the library for novelty
            ps.parts_library.wrapEachNonFlexPartWithFlexPart()
            
            #add a goal to minimize novelty
            ps.addNoveltyAnalysis()
        else:
            pass
        old_optvars = copy.copy(ps.ordered_optvars)
        ps.updateOrderedOptvarsFromEmbPart()

        #initialize state, possibly from previous run
        if loaded_state is None:
            #create a new state
            self.state = EngineUtils.SynthState(ps, ss, EngineUtils.AgeLayeredPop())
        else:
            self.state = loaded_state

            #reassign ss, ps
            self.state.ps = ps
            self.state.ss = ss

            #enable conversion from non-novelty to novelty db
            if past_had_novelty or (not ss.do_novelty_gen):
                log.info("Do not need to update DB from non-novelty to novelty")
                #:NOTE: this is already done by virtue of loading self.state
                # which has 'ps' and 'ss' attributes, and 'ps'
                # has 'embedded_part' attribute.
                pass
            
            else:
                log.info("Need to update DB from non-novelty to novelty...")

                #add measure of 'metric_novelty' for each ind
                novelty_an = ps.noveltyAnalysis()
                    
                #update num evals
                all_inds = self.state.R_per_age_layer.flattened()
                all_es = novelty_an.env_points
                num_evals = len(all_inds) * len(all_es)
                self.state.num_evaluations_per_analysis[novelty_an.ID] = num_evals

                #calc added_vars; validate
                pm = ps.embedded_part.part.point_meta
                new_optvars = pm.keys()
                typ_ind = all_inds[0]
                added_vars = mathutil.listDiff(new_optvars, old_optvars)
                for var_added in added_vars:
                    assert pm[var_added].isChoiceVar()
                    assert "choice_of" in var_added
                    
                #fill in missing ind info
                for (ind_i, ind) in enumerate(all_inds):
                    log.info("update ind #%d/%d" % (ind_i+1, len(all_inds)))

                    #set ps
                    ind._ps = ps
                        
                    #set choice variable values
                    unscaled_d = dict(zip(old_optvars, ind.unscaled_optvals))
                    for var_added in added_vars:
                        #it's always a choice_var for a flex-wrapping part
                        unscaled_d[var_added] = 0
                    ind.unscaled_optvals = [unscaled_d[var] for var in ps.ordered_optvars]
                            
                    #make space for sim requests and results
                    ind._initializeSimDataAtAnalysis(novelty_an)

                    #get novelty actually measured in requests and results
                    scaled_pt = ps.scaledPoint(ind)
                    for e in novelty_an.env_points:
                        Evaluator.evalIndAtAnalysisEnvPoint(self.ps, ind, novelty_an, e, scaled_pt)

            #maybe change popsize
            # -can have a smaller popsize but not a larger popsize
            loaded_size = len(self.state.R_per_age_layer[0])
            new_size = target_num_inds_per_age_layer
            
            self.state.ss.num_inds_per_age_layer = target_num_inds_per_age_layer
        
            log.info("Incoming DB had %d inds per layer;  new run targets %d "
                     "inds per layer" % (loaded_size/2, new_size/2))
#             if new_size == loaded_size:
#                 #nothing to do
#                 pass
#             elif new_size < loaded_size:
#                 raise "FIXME for MOEA/D"
#             else:
#                 raise AssertionError(
#                     "New run must have # inds per age layer <= prev DB's")
#             
        #clear up output directory
        if output_dir is not None:
            self.output_dir = os.path.abspath(output_dir)
            if self.output_dir[-1] != "/":
                self.output_dir += "/"
            if os.path.exists(self.output_dir):
                log.warning("Output path '%s' already exists; will rewrite" %
                            self.output_dir)
#                 shutil.rmtree(self.output_dir)
            else:
                os.mkdir(self.output_dir)
        else:
            self.output_dir = None

        #create and populate the random pool
        self.random_pool = EngineUtils.RandomPool(self.ps, True)
        if random_pool_file != None:
            log.info("Loading random pool from %s", random_pool_file)
            self.random_pool.loadFromFile(random_pool_file)
            
            # make sure the random inds that are used to generate the loaded
            # database are marked as unselectable in the random pool
            # all other inds should be selectable
            self.random_pool.makeAllIndsSelectable()
            log.info("Deselecting already used random inds in this pool...")
            inds = self.state.R_per_age_layer.flattened()
            self.random_pool.makeIndsUnselectable(inds, unselect_ancestors = True)

        #if we had a restart file, re-save it
        #-good for record-keeping of an output directory, to tighten 
        # chain between prev DB and new results
        #-includes corner case: avoid having to do the time-consuming
        # novelty-synchronizing step again 
        #-includes corner case: ensure that restart file's info wasn't lost
        # due to clearing up the output directory (if restart file was in output
        # directory)
        if self.output_dir is not None and restart_file is not None:
            self.saveState()

        self.channel = Channel.ChannelFactory(cs).buildChannel(True)
        if self.channel is not None:
            self.channel.registerMaster(ss, ps.problem_choice, ps.doRobust())

        #have 'cs', 'ss' attribute for convenience
        self.ss = self.state.ss
        self.cs = cs

        #_unused_results holds results from tasks that we haven't used yet.
        # -only add to these via self.addUnusedResults(), so that
        #  sim info gets updated at the same time
        self._unused_results = [] #list of TaskForSlave
        if self.channel:
            #flush channel's results from previous runs
            prior_results = self.channel.popFinishedTasks() 
            log.info("Flushed %d results that were still on channel from a previous run" %
                     len(prior_results))
            self.channel.reset()
        
        #the master ID
        self.ID = str("MASTER-" + str(time.time())) #
        
        #postconditions
        ps.validate()
        assert self.state.ss.do_novelty_gen == self.ss.do_novelty_gen \
               == ss.do_novelty_gen

    #for easier reference, make 'ps' look like attributes of self
    def __getPs(self):
        return self.state.ps
    def __setPs(self, new_ps):
        raise AssertionError("not allowed to set Master.ps")
    ps = property(__getPs, __setPs)

    def _ensurePossPartsToAddAreDeclaredAndCached(self, ps):
        lib = ps.parts_library
        lib.wire()
        lib.resistor()
        lib.capacitor()
        lib.mosDiode()
        lib.biasedMos()
        lib.RC_series()
        lib.twoBiasedMoses()
        lib.stackedCascodeMos()

    def run(self):
        """
        @description

          Runs the synthesis engine!  This is the main work routine.
        
        @arguments

          <<none>>
        
        @return

           <<none>> but it continually generates output files to output_dir
    
        @exceptions
    
        @notes

          Some of the files that it generates to output_dir/ are:
            state_gen0001.db -- init results; ie num generations complete = 1
            state_gen0002.db -- results when num generations complete = 2
            state_gen0003.db -- ...
        """
        log.info("Begin.")
        log.info(self.ps.prettyStr())
        log.info(str(self.ss) + "\n")

        while True:
            self.run__oneGeneration()
            self.saveState()
            if self.doStop():
                break
        log.info("Done")

    def saveState(self):
        """
        -Save self.state to 'state_current.db' (overwrites past state, to save disk space)
        -Every so often, save self.state to 'state_genXXXX.db' as backup
        -Save random pool to 'state_rndpool.db'
        -Save best_waveforms.txt (currently commented out)
        """
        if self.output_dir is not None:
            #save random pool
            self.random_pool.saveToFile(self.output_dir + "state_rndpool.db")
            
            #output to current_state.db
            self.state.save(self.output_dir + "state_current.db")
            
            #maybe output to 'state_genXXXX.db'
            if (self.state.generation < 4) or ((self.state.generation % 1) == 0):
                ##HACK: the if statement should be gone. this results in one NDset update every 10 gens
                ##hence not all DB's have a consistent nd-set
                if ((self.state.generation % 10) == 0):
                    # merge the current nondom set into the "big" nondom set
                    # avoids too large sorting times during the evolution
                    self.state.mergeCurrentToBigNdSet()
                # save the state
                self.state.save(self.output_dir + "state_gen%04d.db" % self.state.generation)

            #output waveforms of best ind
#             best_ind = self.state.bestInd()
#             outfiles = best_ind.waveformsToFiles(self.output_dir + "best_waveforms")
#             log.info("Output these waveform files for best ind: %s" % outfiles)

        else:
            debug.info("Not saving state since no output directory specified")
            
    def saveStateToFile(self, file):
        """Save the current state to the specified file"""
        self.state.save(file)
        
    def resimulate(self, random_pool_only=False):
        """Resimulates every ind of the current state according to the current self.ps"""
        log.info("Begin resimulate.")
        log.info(self.ps.prettyStr())
        log.info(str(self.ss) + "\n")

        if random_pool_only:
            log.info("Resimulating random pool..." )
            self.random_pool.makeAllIndsSelectable()
            random_inds = list(self.random_pool._inds.values())
            new_inds = self._resimulateIndsViaChannel(random_inds, 1)
            new_inds = EngineUtils.uniqueIndsByPerformance(new_inds)
            new_pool = EngineUtils.RandomPool(self.random_pool.ps, self.random_pool.do_random_sampling)
            new_pool.putInds(new_inds)
            self.random_pool = new_pool
        else:
            for (layer_i, R) in enumerate(self.state.R_per_age_layer):
                self.state.setCurrentAgeLayer(layer_i)
    
                log.info("Resimulating age layer %s..." % str(layer_i) )
                target_num_rnd_points = self._numRndPointsAtLayerForEval(layer_i)
    
                new_inds = self._resimulateIndsViaChannel(R, target_num_rnd_points)
                self.state.R_per_age_layer[layer_i] = new_inds
    
            log.info("Resimulating current non-dominated set..." )
            self.state.setCurrentAgeLayer(layer_i+1)
            target_num_rnd_points = self._numRndPointsAtLayerForEval(0)
    
            nominal_nondom_inds_current = self._resimulateIndsViaChannel(self.state.nominal_nondom_inds_current, target_num_rnd_points)
            start_len = len(nominal_nondom_inds_current)
            feasible_inds = []
            for ind in nominal_nondom_inds_current:
                if ind.isFeasibleAtNominal():
                    feasible_inds.append(ind)
            log.info(" %s/%s inds were feasible after resimulation" % (len(feasible_inds), start_len))
            self.state.nominal_nondom_inds_current = EngineUtils.nondominatedFilter(feasible_inds, None)
            log.info(" %s/%s inds survived nondominated filter" % (len(self.state.nominal_nondom_inds_current), len(feasible_inds)))
    
            log.info("Resimulating external non-dominated set..." )
            self.state.setCurrentAgeLayer(layer_i+2)
            target_num_rnd_points = self._numRndPointsAtLayerForEval(0)
    
            nominal_nondom_inds = self._resimulateIndsViaChannel(self.state.nominal_nondom_inds, target_num_rnd_points)
            start_len = len(nominal_nondom_inds)
            feasible_inds = []
            for ind in nominal_nondom_inds:
                if ind.isFeasibleAtNominal():
                    feasible_inds.append(ind)
            log.info(" %s/%s inds were feasible after resimulation" % (len(feasible_inds), start_len))
            self.state.nominal_nondom_inds = EngineUtils.nondominatedFilter(feasible_inds, None)
            log.info(" %s/%s inds survived nondominated filter" % (len(self.state.nominal_nondom_inds), len(feasible_inds)))

        log.info("Done")

    def run__oneGeneration(self):
        """
        @description

          Run one evolutionary generation
        
        @arguments

          <<none>> but uses self.state
        
        @return

          <<none>> but updates self.state
    
        @exceptions
    
        @notes
        """
        N = self.ss.num_inds_per_age_layer
        R_per_age_layer = self.state.R_per_age_layer
        
        log.info("=============================================")
        log.info("=============================================")
        log.info("Gen=%d (# age layers=%d): begin" % (self.state.generation, R_per_age_layer.numAgeLayers()))
            
        if self.state.generation % self.ss.age_gap == 0:
            log.info("This is an age_gap generation, so grow one more layer, and generate new random inds at layer 0")
            
            #update age-layers population structure based on ages
            # -just having an empty 'R' is ok because _updateR() will fill
            #  it in by include the lower layer when choosing cand_parents
            if R_per_age_layer.numAgeLayers() < self.ss.max_num_age_layers:
                R_per_age_layer.append([])
        
            #randomly generate new inds for age layer 0 (a la ALPS)
            # (let the previous inds from layer 0 have one last chance by
            #  putting them into level 1)
            if R_per_age_layer.numAgeLayers() > 1:
                R_per_age_layer[1] += R_per_age_layer[0]
            with_novelty = self.ss.do_novelty_gen and \
                           (self.state.num_rnd_gen_rounds_since_novelty >= self.ss.num_rnd_gen_rounds_between_novelty)
            with_novelty = with_novelty or self.ss.always_with_novelty

            if self.ss.moea_per_age_layer[0] == 'MOEA/D':
                log.info("Preparing initial MOEA/D layer")
                # a moed layer needs less inds to start with
                # but we can use all we have. the layer prep phase will
                # select the best available. the remainder is put back into the pool
                nb_available = self.random_pool.availableCount()
                nb_inds_to_take = int(max(self.ss.num_initial_inds_per_layer/10, nb_available))
                rand_inds = self.generateRandomInds(nb_inds_to_take, with_novelty)

                assert len(rand_inds) == len(EngineUtils.uniqueIndsByPerformance(rand_inds)),\
                    "the randomly-generated inds were not unique by performance"

                nb_inds_per_weight = self.ss.topology_layers_per_weight
                (inds_kept, inds_kicked, rnd_costs) = EngineUtils.prepareMOEADLayer(self.ps, rand_inds, self.state.W, self.ss.metric_weights, nb_inds_per_weight)
                initial_layer_set = set()
                for s in inds_kept:
                    initial_layer_set.update(s)
                initial_layer = list(initial_layer_set)
                self.random_pool.putInds(list(set(rand_inds).difference(initial_layer_set)))

                log.info("Of %s random inds, %s inds were used in the initial layer." % \
                        (len(rand_inds), len(initial_layer)))
            elif self.ss.moea_per_age_layer[0] == 'NSGA-II':
                log.info("Preparing initial NSGA-II layer")
                rand_inds = self.generateRandomInds(N/2, with_novelty)
                assert len(rand_inds) == len(EngineUtils.uniqueIndsByPerformance(rand_inds)),\
                    "the randomly-generated inds were not unique by performance"
                initial_layer = rand_inds
            else:
                raise "unknown EA specified: %s" % self.ss.moea_per_age_layer[0]

            R_per_age_layer[0] = initial_layer

            if with_novelty:
                self.state.num_rnd_gen_rounds_since_novelty = 0
            else:
                self.state.num_rnd_gen_rounds_since_novelty += 1

            log.info("Start with initial layer:\n%s\n" % worstCasePopulationSummaryStr(self.ps, R_per_age_layer[0]))

        #MAIN WORK: one layer at a time, select and create children to get new R
        # Note how elder_inds from level i bump up to level i+1.
        new_R_per_age_layer = EngineUtils.AgeLayeredPop()
        inds_kicked_out_prev_layer = [] #these inds get 'one last chance' before being bumped
        num_age_layers = R_per_age_layer.numAgeLayers()
        for age_layer_i in range(num_age_layers):
#             import pdb;pdb.set_trace()
            log.info("---------------------------------------------------------")
            log.info("---------------------------------------------------------")
            s = "Gen=%d: age_layer=%d (num layers=%d): select and create children, using %s" % \
                (self.state.generation, age_layer_i, num_age_layers, self.ss.moea_per_age_layer[age_layer_i])
            self.state.setCurrentAgeLayer(age_layer_i)
            log.info(s + ": begin")
            
            dbg = "R[lyr%2d]   : " % age_layer_i
            inds = R_per_age_layer[age_layer_i]
            inds.sort()
            if inds:
                for ind in inds:
                    dbg += "%12s " % ind.shortID()
            log.debug(dbg)
           
            (new_R, inds_kicked_out_cur_layer) = self._updateR(R_per_age_layer, age_layer_i)
            
            dbg = " => new_R   : "
            inds = new_R
            inds.sort()
            for ind in inds:
                dbg += "%12s " % ind.shortID()
            log.debug(dbg)

            dbg = " => kicked  : "
            inds = inds_kicked_out_cur_layer
            inds.sort()
            for ind in inds:
                dbg += "%12s " % ind.shortID()
            log.debug(dbg)
            
            if (age_layer_i < (num_age_layers - 1)):
                # no need to copy these inds since they are not present in the
                # previous layer anymore
                new_R += inds_kicked_out_prev_layer
            new_R_per_age_layer.append(new_R)

            dbg = " new R[lyr%2d]   : " % age_layer_i
            inds = new_R_per_age_layer[age_layer_i]
            inds.sort()
            if inds:
                for ind in inds:
                    dbg += "%12s " % ind.shortID()
            log.debug(dbg)
            
            log.info(s + ": done")

            #prepare for re-loop
            inds_kicked_out_prev_layer = inds_kicked_out_cur_layer
            
        #age the inds
        aged_IDs = set([])
        for (layer_i, R) in enumerate(new_R_per_age_layer):
            for ind in R:
                if ind.ID not in aged_IDs:
                    ind.genetic_age += 1
                    aged_IDs.add(ind.ID)

        #update self.state
        self.state.R_per_age_layer = new_R_per_age_layer

        #further-evaluate inds as needed (to sync up with required num rnd points)
        self._furtherEvaluateAgeLayers()
        
        log.info("=========================================================")
        log.info("=========================================================")
        #log.info("Wrapping up gen=%d; %s: %s" %
                 #(self.state.generation, self.state.numNondominatedIndsStr(),
                  #self.state.populationSummaryStr(None, False)))
        log.info("Wrapping up gen=%d; %s" %
                 (self.state.generation, self.state.numNondominatedIndsStr()))
        log.info("Gen=%d: done" % self.state.generation)
        self._assertNumEvalsConsistent()
        
        self.state.incrementGeneration()
        log.info("Best_metval0_per_gen = %s" % self.state.best_0th_metric_value_per_gen)

        #clean the results for stuff we don't need
        self.cleanupUnusedResults(self.state.getCurrentGenerationId())

    def _updateR(self, R_per_age_layer, layer_i): 
        """
        @description
        
          Update R = R_per_age_layer[layer_i] using the following steps:
            1. Chooses candidate parents using ALPS rules on layer i and age.
            2. Creates children
        
          uses the EA specified for this age layer
          
        @arguments

          R_per_age_layer -- list of list_of_NsgaInd -- one list per age layer.
          layer_i -- int -- the age layer of interest
        
        @return

          updated_R -- list of NsgaInd -- R, but updated
          inds_kicked_out -- list of NsgaInd -- inds that were in R but could
            not be considered as parents because they were too old
    
        @exceptions
    
        @notes
        """        
        #base data
        if self.ss.moea_per_age_layer[layer_i] == 'MOEA/D':
            if layer_i > 0 and self.ss.moea_per_age_layer[layer_i-1] == 'NSGA-II':
                log.info("updating interface layer NSGA-II => MOEA/D")
                # MOEA/D + ALPS assumes that the weights (i.e. 'directions') for the previous layer
                # are similar to the current layer. However with a layer_i-1 from NSGA this is not
                # true.
                lower_layer_not_from_moead = True
            else:
                lower_layer_not_from_moead = False
            (R, inds_kicked_out, best_costs) = self._updateR_MOEAD(R_per_age_layer, layer_i, lower_layer_not_from_moead)
            self.doStatusOutput(layer_i, R, best_costs)

        elif self.ss.moea_per_age_layer[layer_i] == 'NSGA-II':
            #compute metric ranges
            minmax_metrics = EngineUtils.minMaxMetrics(self.ps, R_per_age_layer.flattened())
            (R, inds_kicked_out) = self._updateR_NSGAII(R_per_age_layer, layer_i, minmax_metrics)
            self.doStatusOutput(layer_i, R)
            
        else:
            raise "unknown EA specified for layer %s: %s" % (layer_i, self.ss.moea_per_age_layer[layer_i])
        
        return (R, inds_kicked_out)

    def _updateR_MOEAD(self, R_per_age_layer, layer_i, lower_layer_not_from_moead=False, do_improve_step=True): 
        """
        @description
        
          Update R = R_per_age_layer[layer_i] using the following steps:
            1. Chooses candidate parents using ALPS rules on layer i and age.
            2. Creates children
        
            uses modified MOEA/D:
                - there are N weight vectors in objective space
                PRE:
                - to each weight vector we assign the best performing individual
                - we also assign M-1 next best individuals to the weight vector, with
                   the constraint that all inds assigned to one weigth vector should
                   have a different topology.
                - for each weight vector we select X neighbours.
                GEN:
                - for each vector, we use the inds from the weight vector and it's 
                  neighbours to generate offspring.
                - the parents and neighbors are saved into one set.
                POST:
                - the output set is constructed by using the same rules as the PRE
                  phase. The number of individuals is limited to the popsize.

            modified MOEA/D (2):
              End goal is to keep topological diversity in the population. The main
              target is to ensure that topologies do not die simply because they are
              difficult to optimize. Hence we have to limit the evolution of easy 
              topologies and encourage the evolution of difficult ones.

              Ideas:
                - do not allow a topology to die when it would mean that the population
                  loses diversity. note that it is not sufficient to keep the ind in the
                  population. we also have to ensure that it progresses.
                - limit the amount of inds for one topology:
                    * for each topology:
                        - T = select all inds with that topo
                        - NDT = nondom-sort T
                    * while not pop_full:
                        for each t in topologies:
                            I = select first ind from NDT(T)
                            add I to population
                    as long as nb_topos < popsize this might work
                    it creates a "shelled" approach
                - instead of throwing all inds for one weight vector into one
                  big parent set, do this for each 'layer'. where a layer is 
                  the second best different-topo ind for that vector.

        @arguments

          R_per_age_layer -- list of list_of_NsgaInd -- one list per age layer.
          layer_i -- int -- the age layer of interest
          lower_layer_not_from_moead -- bool -- indicates whether the previous layer is
              a MOEA/D layer. if not the layer will be initialized before use.
        
        @return

          updated_R -- list of NsgaInd -- R, but updated
          inds_kicked_out -- list of NsgaInd -- inds that were in R but could
            not be considered as parents because they were too old
    
        @exceptions
    
        @notes
           initializing a layer means assigning the best ind from the input layer for each
           weigth vector.
        """        
        #base data
        N = self.ss.num_inds_per_age_layer
        num_layers = R_per_age_layer.numAgeLayers()

        nb_inds_per_weight = self.ss.topology_layers_per_weight

        # gather the inds that are potential candidates
        all_cands = set()
        all_cands.update(R_per_age_layer[layer_i])
        if layer_i > 0:
            all_cands.update(R_per_age_layer[layer_i-1])

        # kick out the inds that are too old
        our_cands = [cand for cand in all_cands
                                   if self.ss.allowIndInAgeLayer(layer_i, cand.genetic_age, num_layers)]
        if not our_cands:
            import pdb;pdb.set_trace()
        #assert len(our_cands) > 0

        # prepare layered MOEA/D set
        (base_layer, base_inds_kicked, base_costs) = EngineUtils.prepareMOEADLayer(self.ps, our_cands, self.state.W, self.ss.metric_weights, nb_inds_per_weight)

        ind_weight_pairs_to_improve = []

        # select a set of to-be-improved inds
        if self.ss.do_improve_step:
            #ind_weight_pairs_to_improve = EngineUtils.selectBestRankedIndPerTopology(base_layer, self.state.W)
            #ind_weight_pairs_to_improve = EngineUtils._selectRandomIndPerTopology(base_layer, self.state.W)
            clusters = EngineUtils.clusterPerTopology(base_layer, base_costs, self.state.W, self.state.indices_of_neighbors)
            s = "Front clusters:\n"
            for cluster in clusters:
                if not cluster.in_front:
                    (ind, weight_vector) = cluster.getBestIndAndWeight()
                    log.debug("cluster %s gives ind %s for weight %s" % (cluster.ID, ind.shortID(), weight_vector))
                    ind_weight_pairs_to_improve.append( (ind, weight_vector) )
                else:
                    s += " [%20s] %3d inds: %s\n" % (cluster.topo, len(cluster.getInds()), [ind.shortID() for ind in cluster.getInds()])
            log.info(s)
        else:
            ind_weight_pairs_to_improve = []

        #generate candidate parents for each w_i
        cand_sets = [] #list of ind_list
        for w_i in range(N):
            neighbor_I = self.state.indices_of_neighbors[w_i]
            cand_set = set()
            for idx in neighbor_I:
                base_cands = base_layer[idx]
                cand_set.update([cand for (idx, cand) in enumerate(base_cands)
                                   if self.ss.allowIndInAgeLayer(layer_i, cand.genetic_age, num_layers)
                                   and idx < self.ss.topology_layers_to_evolve_on])

            if not cand_set:
                import pdb;pdb.set_trace()
            cand_sets.append(list(cand_set))

        dbg = " MOEA/D parent set: " 
        inds = set()
        for cand_set in cand_sets:
            inds.update([ind for ind in cand_set])
        inds = list(inds)
        inds.sort()
        if inds:
            for ind in inds:
                dbg += "%12s " % ind.shortID()
        log.debug(dbg)

        #generate parents for each w_i
        parent_sets = [] #list of ParentSet
        for w_i in range(N):
            cand_set = cand_sets[w_i]
            parent_set = ParentSet([random.choice(cand_set), random.choice(cand_set)], w_i)
            parent_sets.append(parent_set)

        #create children & evaluate for each w_i
        # -actual call
        children = self.varyParentsToGetChildren(parent_sets, self._numRndPointsAtLayerForEval(layer_i))

        # -are children in the correct order?
        assert len(children) == N
        for target_w_i in range(N):
            assert children[target_w_i].w_i == target_w_i

        # do the improve step if requested
        improved = []
        if self.ss.do_improve_step:
            log.info("Improve step...")

            log.info(" Estimating metric bounds...")
            # use rough metric extremes as used by the local optimizer
            metric_bounds = {}
            for metric in self.ps.flattenedMetrics():
                metric_bounds[metric.name] = (metric.rough_minval, metric.rough_maxval)
            dbg = "  metric bounds: \n"
            for obj in self.ps.metricsWithObjectives():
                dbg += "   %20s: %10s -> %10s\n" % (obj.name, metric_bounds[obj.name][0], metric_bounds[obj.name][1])
            log.info(dbg)

            # construct a list of inds to improve
            inds_to_improve = []
            weights_to_improve_on = []
            initial_costs = []

            improve_children = True
            improve_parents = False
            if improve_children:
                # improve those children that are an improvement over their parents
                # the rationale is that these still show some potential for improvement
                for w_i in range(N):
                    best_costs_of_parents = min([ind.scalarCost(1, self.state.W[w_i,:], self.ss.metric_weights, metric_bounds)
                                                  for ind in parent_set])
                    cost_of_child = children[w_i].scalarCost(1, self.state.W[w_i,:], self.ss.metric_weights, metric_bounds)
                    if cost_of_child < best_costs_of_parents:
                        log.debug("Child %s improved parent cost from %f to %f, doing local optimize..." % \
                                  (children[w_i].ID, best_costs_of_parents, cost_of_child))
                        inds_to_improve.append(children[w_i])
                        weights_to_improve_on.append(self.state.W[w_i, :])
                        initial_costs.append(cost_of_child)
                    else:
                        log.debug("Child %s did not improve parent cost (%f to %f)..." % \
                                  (children[w_i].ID, best_costs_of_parents, cost_of_child))

                log.info("%s/%s children were better than parents, doing local optimize..." % (len(inds_to_improve), len(children)))

            elif improve_parents: # pick one random ind for every weight
                for w_i in range(N):
                    inds_to_improve.append(random.choice(cand_sets[w_i]))
                    weights_to_improve_on.append(self.state.W[w_i, :])

            # the earlier selected ones
            for (ind, weights) in ind_weight_pairs_to_improve:
                # this can safely be overwritten since it's
                # not used in the population and the inds in the
                # selection are all different (all have different topo)
                ind.w_i = 0
                inds_to_improve.append(ind)
                weights_to_improve_on.append(weights)
                initial_costs.append(ind.scalarCost(1, weights, self.ss.metric_weights, metric_bounds))

            improved = self.improveIndsForWeights(inds_to_improve, weights_to_improve_on)

            log.debug("Improved: %s" % ["%s" % ind.ID for ind in improved])

            improved_inds = []
            # delete those that are not improved
            for (idx, ind) in enumerate(improved):
                if ind in inds_to_improve:
                    log.debug("Ind %s (%s) was not improved." % (ind.ID, ind.topoSummary()))
                else:
                    improved_inds.append(ind)
                    orig_ind = inds_to_improve[idx]
                    weights = weights_to_improve_on[idx]
                    cost = ind.scalarCost(1, weights, self.ss.metric_weights, metric_bounds)
                    log.info("Ind %20s => %20s (%s => %s) cost %10f => %10f" % \
                               (orig_ind.shortID(), ind.shortID(),
                                orig_ind.topoSummary(), ind.topoSummary(),
                                initial_costs[idx], cost))
            log.info("%s/%s inds were improved by local optimization" % (len(improved_inds), len(inds_to_improve)))

        # throw all inds (parents, children and improved) into one big set
        all_cands = set()
        for w_i in range(N):
            all_cands.update(cand_sets[w_i] + [children[w_i]])
        log.info("Age layer %d summed front cost initial        : %f" % (layer_i, EngineUtils.estimateFrontCost(self.ps, our_cands, self.state.W, self.ss.metric_weights)))
        log.info("Age layer %d summed front cost with children  : %f" % (layer_i, EngineUtils.estimateFrontCost(self.ps, our_cands+children, self.state.W, self.ss.metric_weights)))
        all_cands.update(improved_inds)
        R = list(all_cands)
        log.info("Age layer %d summed front cost final          : %f" % (layer_i, EngineUtils.estimateFrontCost(self.ps, R, self.state.W, self.ss.metric_weights)))

        #set inds_kicked_out due to age
        inds_kicked_out = [ind for ind in R_per_age_layer[layer_i]
                           if not self.ss.allowIndInAgeLayer(layer_i, ind.genetic_age, num_layers)]

        # queue all newly generated inds for addition to the nondominated set
        self.state.updateNondominatedInds(children + improved_inds)

        #done
        return (R, inds_kicked_out, None)

    def _updateR_NSGAII(self, R_per_age_layer, layer_i, minmax_metrics): 
        """
        @description
        
          Update R = R_per_age_layer[age_layer_i] using the following steps:
            1. Chooses candidate parents using ALPS rules on layer i.
            2. Nondominated-sorts the candidate parents
            3. Selects parents using NSGA-II style selection
            4. Creates offspring from parents
            5. Returns offspring + parents
        
        @arguments

          R_per_age_layer -- list of list_of_NsgaInd -- one list per age layer.
          age_layer_i -- int -- the age layer of interest
          minmax_metrics -- metrics bounds -- see minMaxMetrics for details
        
        @return

          updated_R -- list of NsgaInd -- R, but updated
          inds_kicked_out -- list of NsgaInd -- inds that were in R but could
            not be considered as parents because they were too old
    
        @exceptions
    
        @notes
        """
        N = self.ss.num_inds_per_age_layer
        num_layers = R_per_age_layer.numAgeLayers()

        #get candidate parents
        cand_parents = [cand for (cand_i, cand) in enumerate(R_per_age_layer[layer_i])
                         if self.ss.allowIndInAgeLayer(layer_i, cand.genetic_age, num_layers)]

        # allow inds from previous layer too
        # these are created with a new ID such that they are also aged properly
        if layer_i > 0:
            cand_parents += [cand.copyWithNewID() for (cand_i, cand) in enumerate(R_per_age_layer[layer_i-1])
                            if self.ss.allowIndInAgeLayer(layer_i-1, cand.genetic_age, num_layers)]

        dbg = " NSGA-II parent set: " 
        inds = set()
        inds.update([ind for ind in cand_parents])
        inds = list(inds)
        inds.sort()
        if inds:
            for ind in inds:
                dbg += "%12s " % ind.shortID()
        log.info(dbg)

        #cand_F = F[0] + F[1] + ... = nondominated layers
        # -keep _all_ the inds available here!!
        # -layer 0 should have N*2 cand. parents (last gen's par. & children)(except kicked-out)
        # -layer >0 should have N*4 cand. parents, because it includes lower layer too
        cand_F = EngineUtils.nondominatedSort(
                                cand_parents, max_num_inds=None,
                                metric_weights = self.ss.metric_weights)

        #output state
#         self.doStatusOutput(age_layer_i, cand_F)

        #fill parent population P
        # :FIXME: :DEFECT: cand_F has the same num inds as P!!!!!
        P = self._nsgaSelectInds(cand_F, N/2, minmax_metrics)

#         #use selection, mutation, and crossover to create a new child pop Q
#         # -note that the new pop gets evaluated within
#         # :FIXME: :DEFECT: does selection too!
#         parents = []
#         while len(parents) < len(P):
#             ind_a, ind_b = random.choice(P), random.choice(P)
# 
#             #first try selecting based on domination
#             if ind_a.rank < ind_b.rank:   parents.append(ind_a)
#             elif ind_b.rank < ind_a.rank: parents.append(ind_b)
# 
#             #if needed, select based on distance
#             elif ind_a.distance > ind_b.distance:  parents.append(ind_a)
#             else:                                  parents.append(ind_b)

        #generate parent sets
        # simply selects random inds from the parent set
        parent_sets = [] #list of ParentSet
        for w_i in range(N/2):
            parent_set = ParentSet([random.choice(P), random.choice(P)], w_i)
            parent_sets.append(parent_set)

        #with parents, generate children via variation
        # -actual call
        children = self.varyParentsToGetChildren(parent_sets, self._numRndPointsAtLayerForEval(layer_i))
        
        # queue the children for addition to the nondominated set
        self.state.updateNondominatedInds(children)

        #combine parent and offspring population
        R = P + children

        inds_kicked_out = [ind for ind in R_per_age_layer[layer_i]
                           if not self.ss.allowIndInAgeLayer(layer_i, ind.genetic_age, num_layers)]
        
        return R, inds_kicked_out
        
    def _nsgaSelectInds(self, F, target_num_inds, minmax_metrics):
        """
        @description

          Selects 'target_num_inds' using nondominated-layered_inds 'F'
          -mostly according to NSGA-II's selection algorithm (which basically
           says take the 50% of inds in the top nondominated layers).
          -but if all inds are in pareto front, then select based on ARF
        
        @arguments

          F -- list of nondom_inds_layer where a nondom_inds_layer is a list
            of inds.  E.g. the output of nondominatedSort().
          target_num_inds -- int -- number of inds to select.  
        
        @return
    
        @exceptions

          target_num_inds must be <= total number of inds in F.
    
        @notes
        """
        #preconditions
        try:
            assert target_num_inds <= EngineUtils.numIndsInNestedPop(F)
        except:
            import pdb;pdb.set_trace()

        log.info("_nsgaSelectInds: Begin.  Have %d inds, must choose %d." %
                 (EngineUtils.numIndsInNestedPop(F), target_num_inds))
        
        N = target_num_inds
        P, i = [], 0
        while True:
            #set 'distance' value to each ind in F[i]
            self.crowdingDistanceAssignment(F[i], minmax_metrics)
            
            #stop now if this next layer would overfill (have to select more smartly)
            if (len(P) + len(F[i])) > N: break

            #include ith nondominated front in the parent pop P
            P += F[i]

            #stop now if full
            if len(P) == N: break
            
            #check the next front for inclusion
            i += 1

        #fill up the rest of P with elements of F[i].
        #-if at 0th (nondominated) layer (and feasible), via ARF
        #-if at other layers, select via crowding
        num_left = N - len(P)
        if num_left == 0:
            log.info("Did not have to cut across a layer to choose parents.")
        else:
            metrics_with_objectives = self.ps.metricsWithObjectives()
            #Turned OFF ARF because it will overly bias towards a single region of metric space
            if False: #(i == 0) and (len(metrics_with_objectives) > 3) and self._allIndsFeasible(F[0]):
                log.info("Invoke Average Ranking (ARF) to select from remaining %d inds" % len(F[0]))
                ordered_I = EngineUtils.orderViaAverageRankingOnFront(metrics_with_objectives, F[0])
                self._invoked_ARF = True
            else:
                log.info("Invoke crowding to select from remaining %d inds" % len(F[i]))
                ordered_I = EngineUtils.orderViaCrowdingDistance(F[i])
                
            F[i] = list(numpy.take(F[i], ordered_I))
            P += F[i][:num_left]

        log.info("_nsgaSelectInds: Done")

        return P

    def crowdingDistanceAssignment(self, layer_inds, minmax_metrics):
        """
        @description

          Assign a crowding distance to each individual in list of inds
          at a layer of F.
        
        @arguments

          layer_inds -- list of Ind
          minmax_metrics -- dict of metric_name : (min_val, max_val) as
            computed across _all_ inds, not just across layer_inds
        
        @return

          <<none>> but alters the 'distance' attribute of each individual
           in layer_inds
    
        @exceptions
    
        @notes
        """
        #corner case
        if len(layer_inds) == 0:
            return

        #initialize distance for each ind to 0.0
        for ind in layer_inds:
            ind.distance = 0.0

        #increment distance for each ind on a metric-by-metric basis
        for metric in self.ps.flattenedMetrics():
            #retrieve max and min; if max==min then this metric won't
            # affect distance calcs
            (met_min,met_max) = minmax_metrics[metric.name]
            assert met_min > -INF, "can't scale on inf"
            assert met_max < INF,  "can't scale on inf"
            if met_min == met_max:
                continue

            #sort layer_inds and metvals, according to metvals 
            metvals = [ind.nominalWorstCaseMetricValue(metric.name)
                       for ind in layer_inds]
            I = numpy.argsort(metvals)
            layer_inds = numpy.take(layer_inds, I)
            metvals = numpy.take(metvals, I)

            #ensure that boundary points are always selected via dist = Inf
            layer_inds[0].distance = INF
            layer_inds[-1].distance = INF

            #all other points get distance set based on nearness to
            # ind on both sides (scaled by the max and min seen on metric)
            for i in range(1,len(layer_inds)-1):
                d = abs(metvals[i+1] - metvals[i-1]) / (met_max - met_min)
                layer_inds[i].distance += d

    def _numRndPointsAtLayerForCost(self, layer_i):
        """Returns the number of rnd points that layer_i needs to use to determine cost"""
        if self.ps.doRobust():
            return self.ss.num_rnd_points_per_layer_for_cost[layer_i]
        else:
            return 1

    def _numRndPointsAtLayerForEval(self, layer_i):
        """Returns the number of rnd points that layer_i needs to evaluate at"""
        if self.ps.doRobust():
            return self.ss.num_rnd_points_per_layer_for_eval[layer_i]
        else:
            return 1

    def doStop(self):
        """
        @description

          Stop?
        
        @arguments
        
        @return

          stop -- bool
    
        @exceptions
    
        @notes
        """
        #tlm - possible new stop criterion is max num simulations

        if self.state.generation > self.ss.max_num_gens:
            log.info("Stop: num_generations (%d) > max" % self.state.generation)
            return True
        
        return False
            
    def doStatusOutput(self, age_layer_i, R, costs = None):
        """
        @description

          Logs search status for this age layer

        @arguments

          age_layer_i -- int
          R -- list of Ind
          costs -- list of float -- one entry per ind
          
        """
        #preconditions
        if costs != None:
            assert len(R) == len(costs)
        
        #main work
        state = self.state
        
        s = []

        s += ["Gen=%d / age_layer=%d: " % (state.generation, age_layer_i)]
        
        s += ["; front cost: %f" % EngineUtils.estimateFrontCost(self.ps, R, self.state.W, self.ss.metric_weights)]
        
        s += ["; tot_#_evals=%d (%d on funcs, %d on sim)" %
              ( state.totalNumEvaluations(), state.totalNumFunctionEvaluations(),
                state.totalNumCircuitEvaluations() )]
        
        s += ["; #evals_per_func_analysis={"]
        for an in self.ps.functionAnalyses():
            s += ["%s:%d," % (an.ID, state.num_evaluations_per_analysis[an.ID])]
        s += ["}"]
        
        s += ["; #evals_per_sim_analysis={"]
        for an in self.ps.circuitAnalyses():
            s += ["%s:%d," % (an.ID, state.num_evaluations_per_analysis[an.ID])]
        s += ["}\n"]
        
        log.info("".join(s))

        if costs:
            s = []
            s += ["best cost for each scalar sub-problem=["]
            for (i, cost) in enumerate(costs):
                s += ["%.3e" % cost]
                if i < (len(costs) - 1):
                    s += [", "]
            s += ["]\n"]
            log.info("".join(s))

        log.info("All Inds of age_layer_i=%d, showing worst-case values: %s" %
                 (age_layer_i, worstCasePopulationSummaryStr(self.ps, R)))

        return

    def addUnusedResults(self, tasks_with_results):
        """
        @description
        
          Adds tasks_with_results to self._unused_results, and updates the sim info, e.g. inds count and # sims count.

        @arguments

          tasks_with_results -- list of Channel.TaskForSlave, each having attribute 'result_data'
        
        @return

          <<none>> but updates self._unused_results and self.state
    
        @exceptions
    
        @notes
        """
        #preconditions
        for result_task in tasks_with_results:
            assert isinstance(result_task, Channel.TaskForSlave)
            assert isinstance(result_task.result_data, Channel.ResultData)

        #main work
        for result_task in tasks_with_results:
            if result_task.master_ID == self.ID:
                #update self._unused_results
                self._unused_results.append(result_task)

                #update self.state.num_evaluations_per_analysis
                for (anID, num_evals_at_an) in \
                        result_task.result_data.num_evaluations_per_analysis.iteritems():
                    self.state.num_evaluations_per_analysis[anID] += num_evals_at_an
            else:
                log.info("discarding result intended for master %s" % str(result_task.master_ID))

    def popFromUnusedResults(self, target_num_results, descr, generation_id, input_w_I_left):
        """
        Try to pop up to 'target_num_results' results from self._unused_results that match 'descr'.
        Returns a list of TaskForSlave.

        Only returns inds that pass the following filters:
        1. adding the ind keeps len(returned inds) <= target_num_results
        2. result.descr matches input 'descr', e.g. 'Generate random ind' or 'Generate child'
        3. result.task_data.generation_id matches 'generation_id', OR generation_id is None
        4. result.result_data.ind.w_i exists and is in w_I_left, OR w_I_left is None

        Note that in the popped results, there will never be duplicate values for w_i.
        """
        used_results, unused_results = [], []
        w_I_left = copy.copy(input_w_I_left)
        for result in self._unused_results: #result is a TaskForSlave
            #if the result passes filters 1-4, add it
            if (len(used_results) < target_num_results) and \
                   (result.descr == descr) and \
                   ((generation_id == None) or (result.task_data.generation_id == generation_id)) and \
                   ((w_I_left == None) or (getattr(result.result_data.ind, 'w_i', None) in w_I_left)):
                used_results.append(result)

                #update w_I_left (because we don't want to have duplicate results for a given w_i)
                if (w_I_left is not None):
                    w_I_left.remove(result.result_data.ind.w_i)

            else:
                unused_results.append(result)
                
        self._unused_results = unused_results
        return used_results
    
    def cleanupUnusedResults(self, min_generation_id = None):
        """
        Remove the simulation results that are not relevant
        from the queue.
        """
        new_results = []
        for result in self._unused_results:
            if result.task_data.master_id != self.ID:
                log.info("Removing task with bad master id %s" % result.task_data.master_id)
                continue
            if (min_generation_id != None) and (result.task_data.generation_id != None) and \
                   (result.task_data.generation_id < min_generation_id):
                log.info("Removing old task (%s<%s)" % (result.task_data.generation_id,min_generation_id))
                continue

            new_results.append(result)
            
        self._unused_results = new_results

    def getIndsFromResults(self, results):
        """Returns the inds from 'results' (which is a list of TaskForSlave).
        -Needs to restore each ind from pickling.
        """
        inds = []
        for r in results:
            r.result_data.ind.restoreFromPickle(self.ps)
            inds.append(r.result_data.ind)
            
        return inds
    
    def _updateIndsAndwI(self, results, prev_inds, prev_w_I_left):
        """Using the input list of results, return updated copies of inds and w_I_left.

        @arguments
          results -- list of TaskForSlave -- results returned from slaves
          prev_inds -- list of Ind
          prev_w_I_left -- set of w_i, or None -- if 'Generate child' then this indicates
            which w_i's have been not been found yet.  None for other task descriptions.

        @return
          inds -- list of Ind -- updated version of prev_inds
          w_I_left -- set of w_i, or None -- updated version of prev_w_I_left

        @notes
          -w_i is an int in 0, 1, ..., popsize-1 corresponding to scalar cost function i.
          -the input data structures are _not_ altered
          
        @exceptions
          Need to pre-filter the results such that each result's w_i value is in prev_w_I_left,
            and there are no duplicate w_i values.

        """
        new_inds = self.getIndsFromResults(results)
        inds = copy.copy(prev_inds) + new_inds
        
        if prev_w_I_left is None:
            w_I_left = None
        else:
            w_I_left = copy.copy(prev_w_I_left)
            for new_ind in new_inds:
                assert new_ind.w_i in w_I_left, "did not properly pre-filter results for w_I_left"
                w_I_left.remove(new_ind.w_i)

        #postconditions
        assert len(inds) == (len(results) + len(prev_inds))
        if w_I_left is not None:
            assert len(w_I_left) == (len(prev_w_I_left) - len(results))

        #done
        return (inds, w_I_left)

    def _furtherEvaluateAgeLayers(self):
        """Further-evaluated any inds in self.state.R_per_age_layer which
        are not up-to-date with required num rnd points"""

        #determine which inds to evaluate further, and how many rnd points are needed
        # -note how because we go in ascending layer_i, the dict will handle duplicates
        #  with the highest value of num_rnd_points
        num_rnd_per_ID = {} # ind_ID : target_num_rnd_points
        ind_per_ID = {}     # ind_ID : ind
        for (layer_i, R) in enumerate(self.state.R_per_age_layer):
            target_num_rnd_points = self._numRndPointsAtLayerForEval(layer_i)
            for ind in R:
                if ind.numRndPointsFullyEvaluated() < target_num_rnd_points:
                    num_rnd_per_ID[ind.ID] = target_num_rnd_points
                    ind_per_ID[ind.ID] = ind

        #create list of tasks
        task_data_list = []     
        for (ID, target_num_rnd_points) in num_rnd_per_ID.iteritems():
            task_data = self._baseTaskData(self.state.getCurrentGenerationId())
            task_data.ind = ind_per_ID[ID]
            task_data.num_rnd_points = target_num_rnd_points
            task_data_list.append(task_data)
        
        if task_data_list:
            log.info("Further evaluate %d inds in age layers: begin" % len(task_data_list))
            
            # -main call to Slaves; restore from pickling
            evaluated_inds = self.generateInds(task_data_list, descr='Evaluate ind further')
            for ind in evaluated_inds:
                ind.restoreFromPickle(self.ps)

            # -update R_per_age_layer 
            self.replaceIndsAccordingToID(evaluated_inds)

            # -update nondominated inds (and keep slaves busy if appropriate)
            self._updateNondominatedInds(evaluated_inds)
            
            log.info("Further evaluate %d inds in age layers: done" % len(task_data_list))
        else:
            log.info("Don't need to further evaluate any inds in age layers")

    def _updateNondominatedInds(self, inds):
        """Update state's nondominated inds, and keep slaves busy if the update will be costly"""
        #maybe keep the slaves busy
        if self.state.nondominatedUpdateIsCostly(inds) and self.channel:
            self.pushRandomGenerateIndsOnChannel()

        #main work
        self.state.updateNondominatedInds(inds)

    def _assertNumEvalsConsistent(self):
        """Raise an exception if an ind does not have >= the target num rnd points"""
        for (layer_i, R) in enumerate(self.state.R_per_age_layer):
            for ind in R:
                assert ind.numRndPointsFullyEvaluated() >= self._numRndPointsAtLayerForEval(layer_i)

    def replaceIndsAccordingToID(self, new_inds):
        """Update any ind in self.state.R_per_age_layer with a new_ind if IDs match."""
        #make quick-access dict of new_ID => ind
        new_ind_per_ID = {}
        for ind in new_inds:
            new_ind_per_ID[ind.ID] = ind
        new_IDs = new_ind_per_ID.keys()

        #go through each ind of each layer, and replace it with new one if IDs align
        new_R_per_age_layer = EngineUtils.AgeLayeredPop([])
        for R in self.state.R_per_age_layer:
            new_R = []
            for ind in R:
                if ind.ID in new_IDs:
                    new_R.append(new_ind_per_ID[ind.ID])
                else:
                    new_R.append(ind)
            new_R_per_age_layer.append(R)
            
        self.state.R_per_age_layer = new_R_per_age_layer

    def varyParentsToGetChildren(self, parent_sets, num_rnd_points_to_eval_children):
        """Returns a list of child inds based on varying parent_sets
        Note that the parents do not need to have all the rnd points that the children need.
        """
        #Part I: Nominal-generate children
        log.info('Nominal-generate %d children: begin' % len(parent_sets))

        # -build task data
        task_data_list = []
        for parent_set in parent_sets:
            task_data = self._baseTaskData(self.state.getCurrentGenerationId())
            for ind in parent_set:
                ind.prepareForPickle()
            task_data.parent_set = parent_set
            task_data.num_rnd_points = 1 #i.e. nominal-eval
            task_data_list.append(task_data)

        # -call to slaves
        children = self.generateInds(task_data_list, descr='Generate child')

        # -restore from pickling
        for ind in children:
            ind.restoreFromPickle(self.ps)
        for parent_set in parent_sets:
            for ind in parent_set:
                ind.restoreFromPickle(self.ps)

        log.info('Nominal-generate %d children: done' % len(children))

        #Part II: Fully eval children
        log.info('Fully evaluate %d children: begin' % len(children))
        if num_rnd_points_to_eval_children > 1:
            task_data_list = []     
            for ind in children:
                task_data = self._baseTaskData(self.state.getCurrentGenerationId())
                task_data.ind = ind
                task_data.num_rnd_points = num_rnd_points_to_eval_children #i.e. full eval
                task_data_list.append(task_data)

            # -call to slaves
            children = self.generateInds(task_data_list, descr='Evaluate ind further')

            # -restore from pickling
            for ind in children:
                ind.restoreFromPickle(self.ps)
        log.info('Fully evaluate %d children: done' % len(children))

        # -re-order children according to w_i
        w_I = [child.w_i for child in children]
        children = [children[i] for i in numpy.argsort(w_I)]

        #done
        return children

    def improveIndsForWeights(self, inds, weight_vectors):
        """Returns a list of inds that are improved versions of the given inds,
        according to the provided weight vectors.

        for each ind there should be a weight vector
        """
        assert len(inds) == len(weight_vectors)

        log.info('Improve %d inds: begin' % len(inds))

        # -build task data
        task_data_list = []
        for idx, ind in enumerate(inds):
            task_data = self._baseTaskData(self.state.getCurrentGenerationId())
            task_data.ignore_zombification = False
            task_data.ID = idx
            task_data.ind = ind
            task_data.ind.prepareForPickle()
            task_data.weight_vector = weight_vectors[idx]
            task_data_list.append(task_data)

        # -call to slaves
        improved = self.generateInds(task_data_list, descr='ImproveTopologyForWeight')

        # -restore from pickling
        for ind in improved:
            ind.restoreFromPickle(self.ps)
        for ind in inds:
            ind.restoreFromPickle(self.ps)

        # figure out what inds have improved, and set ancestry
        for (idx, old_ind) in enumerate(inds):
            new_ind = improved[idx]
            if old_ind != new_ind:
                new_ind.setAncestry([old_ind])
                # magic number alert: how much older do these get?
                new_ind.genetic_age = old_ind.genetic_age

        log.info('Improve %d inds: done' % len(improved))

        #done
        return improved

    def generateRandomInds(self, target_num_inds, with_novelty):
        """Generate and return the specified number of inds"""
        # -main work
        inds = []
        remaining_num_inds = target_num_inds - len(inds)
        while remaining_num_inds > 0:
            new_inds = self.random_pool.getInds(remaining_num_inds)
            inds.extend(new_inds)

            remaining_num_inds = target_num_inds - len(inds)
            log.info("Took %s random inds from the random pool, now at %s/%s" % \
                     (len(new_inds), len(inds), target_num_inds))
            
            if remaining_num_inds > 0:
                log.info("Queue creation of %s random inds" % (remaining_num_inds))
                # we didn't have enough, so explicitly queue generation of more random inds
                task_data = self._randomIndTaskData()
                task_data_list = [task_data for i in range(remaining_num_inds)]
                self.generateInds(task_data_list, self._randomIndTaskDescr(with_novelty))

        # -update nondominated inds (and keep slaves busy if appropriate)
        self.state.updateNondominatedInds(inds)

        return inds

    def generateInds(self, task_data_list, descr):
        """
        @description

          Generate good inds according to task description.  Stay in this
          loop until the target_num_inds is hit.
        
        @arguments

          task_data_list -- list of TaskData objects -- 
          descr -- one of ALL_TASK_DESCRIPTIONS in Channel.py, e.g. 
              'Generate random ind', 'Generate child', 'Evaluate ind further', etc
        
        @return

           inds -- list of Ind -- 
    
        @exceptions
    
        @notes
        
          Random inds are put into the random pool, such that they are kept for later
          use.  The random inds are not returned to avoid bypassing the random pool, since
          that keeps track of the already used inds.
          
        """
        #preconditions
        self.ps.validate()
    
        #do the rest of the work in either local or channel...
        if self.channel is None:
            inds = self._generateIndsViaLocal(task_data_list, descr)
        else:
            inds = self._generateIndsViaChannel(task_data_list, descr)

        # if we had as task to generate random inds, we should take them and put them
        # in the random pool
        if descr == "Generate random ind":
            log.info("Put %s random inds into the random pool" % (len(inds)))
            self.random_pool.putInds(inds)

        # also put all remaining random inds (regardless of our current task)
        # into the random pool such that they are saved for later use
        # grab the other inds too FIXME: this is not clean, should be done better
        results = self.popFromUnusedResults(100000, "Generate random ind", None, None)
        random_inds = self.getIndsFromResults(results)
        log.info("Put %s excess random inds into the random pool" % (len(random_inds)))
        self.random_pool.putInds(random_inds)

        # random inds should be taken from the pool
        if descr == "Generate random ind":
            return []
        else:
            return inds
    
    def _generateIndsViaLocal(self, task_data_list, descr):
        """Implementation of generateInds() by creating the Slaves on-the-fly
        and doing it locally.  This is very useful for unit tests."""
        slave = Slave(self.cs, self.ps, self.ss) 

        results = []
        for task_data in task_data_list:
            #create task
            task = Channel.TaskForSlave(self.ID, descr, task_data)

            #"push" task to slave
            slave.task = task

            #run slave to complete the task
            result_task = None
            num_loops = 0
            while result_task is None:
                result_task = slave.run__oneIter()
                num_loops += 1; assert num_loops < 1000000, "hit infinite loop"

            #update results
            results.append(result_task)

        #update self, and get inds
        assert len(results) == len(task_data_list)
        self.addUnusedResults(results)
        results = self.popFromUnusedResults(results, descr, None, None)

        inds = self.getIndsFromResults(results)

        return inds

    def _generateIndsViaChannel(self, task_data_list, descr):
        """Implement generateInds() using Channel"""
        if descr == "Generate random ind":
            return self._generateRandomIndsViaChannel(task_data_list)
        elif descr == "Generate child":
            return self._generateChildrenViaChannel(task_data_list)
        elif descr == "Evaluate ind further":
            return self._evaluateIndsFurtherViaChannel(task_data_list)
        elif descr == "ImproveTopologyForWeight":
            return self._improveIndsViaChannel(task_data_list)
        elif descr == "Single simulation":
            raise AssertionError("Not directly supported; call with 'Evaluate inds further' instead")
        elif descr == "Resimulate":
            raise AssertionError("Call resimulateIndsViaChannel() instead of this")
        else:
            raise AssertionError("description '%s' not supported" % descr)

    def _generateRandomIndsViaChannel(self, task_data_list):
        """Helper to  _generateIndsViaChannel, for descr = 'Generate random ind'"""
        #corner case
        target_num_inds = len(task_data_list)
        if target_num_inds == 0:
            return []
        
        # -push all requests
        descr = 'Generate random ind'
        tasks = [Channel.TaskForSlave(self.ID, descr, task_data)
                 for task_data in task_data_list]
        self.channel.pushTasks(tasks)

        # -initialize output: 'inds'
        inds = [] #all inds found, whether they are for the target descr or randomly-generated

        # -add some previous data to 'inds' 
        popped_results = self.popFromUnusedResults(target_num_inds, descr, None, None)
        (inds, dummy) = self._updateIndsAndwI(popped_results, inds, None)

        # -loop until we've hit our target number of inds
        while True:            
            #update unused results from channel
            new_results = self.channel.popFinishedTasks()
            log.info("Got %d new results from channel" % len(new_results))
            if new_results:
                self.addUnusedResults(new_results)
                
                #update 'inds' from unused results
                num_left = target_num_inds - len(inds)
                new_results_for_us = self.popFromUnusedResults(num_left, descr, None, None)
                (inds, dummy) = self._updateIndsAndwI(new_results_for_us, inds, None)
                log.info("Of the new results, %d are for '%s'" % (len(new_results_for_us), descr))

            #log
            num_left = target_num_inds - len(inds)
            log.info("Have %d / %d inds for: %s" % (len(inds), target_num_inds, descr))

            #maybe stop
            if num_left == 0:
                log.info("Stop because got target number of inds")
                break

            #keep the slaves busy, including zombie-catching
            if (len(self.channel.tasksWaiting()) == 0):
                tasks_to_feed_on_wait = 2 #magic number alert
                tasks = [Channel.TaskForSlave(self.ID, descr, random.choice(task_data_list))
                         for i in range(tasks_to_feed_on_wait)]
                log.info(" Still require %d results, so feed %d extra '%s' tasks into queue..." % \
                         (num_left, len(tasks), descr))
                self.channel.pushTasks(tasks)
                
            #wait before relooping
            time.sleep(self.ss.num_seconds_between_master_result_requests_random)

        #postconditions
        assert len(inds) == target_num_inds

        #done!
        return inds
    
    def _generateChildrenViaChannel(self, task_data_list):
        """Helper to  _generateIndsViaChannel, for descr = 'Generate child'
        -Needs to track which w_I (sub-goals) have been done, and which are left.
        """    
        #corner case
        target_num_inds = len(task_data_list)
        if target_num_inds == 0:
            return []

        # -push all requests
        descr = 'Generate child'
        tasks = [Channel.TaskForSlave(self.ID, descr, task_data)
                 for task_data in task_data_list]
        self.channel.pushTasks(tasks)

        # -initialize output: 'inds'
        inds = [] #all inds found, whether they are for the target descr or randomly-generated

        # -initialize 'w_I_left', which is w_i's for sub-goals in 
        #  Need w_I_left to be emptied (ie all w_i's found) 
        w_I_left = set([task_data.parent_set.w_i for task_data in task_data_list])

        # the countdown for task requeueing
        countdown = self.ss.num_seconds_before_child_generation_requeue \
            / self.ss.num_seconds_between_master_result_requests_nonrandom

        # -loop until we've hit our target number of inds
        while True:            
            #update unused results from channel
            new_results = self.channel.popFinishedTasks()
            log.info("Got %d new results from channel" % len(new_results))
            if new_results:
                self.addUnusedResults(new_results)
                
                #update 'inds' & 'w_I_left' from unused results
                num_left = target_num_inds - len(inds)
                new_results_for_us = self.popFromUnusedResults(
                    num_left, descr, self.state.getCurrentGenerationId(), w_I_left)
                (inds, w_I_left) = self._updateIndsAndwI(new_results_for_us, inds, w_I_left)
                log.info("Of the new results, %d are for '%s', at this gen, and in w_I_left" % \
                         (len(new_results_for_us), descr))

            #log
            num_left = target_num_inds - len(inds)
            log.info("Have %d / %d inds for: %s" % (len(inds), target_num_inds, descr))
            w_I_found = [task_data.parent_set.w_i for task_data in task_data_list
                         if task_data.parent_set.w_i not in w_I_left]
            log.info('w_I_found: %s, w_I_left: %s' % (sorted(w_I_found), sorted(w_I_left)))
            assert len(w_I_left) == num_left

            #maybe stop
            if num_left == 0:
                log.info("Stop because got target number of inds")
                break

            #keep the slaves busy, including zombie-catching
            if (len(self.channel.tasksWaiting()) == 0):
                # only add new tasks after a certain holdoff period
                # since otherwise the difficult child generation doesn't
                # get a chance in multi-cpu setups
                if countdown <= 0:
                    tasks = [Channel.TaskForSlave(self.ID, descr, task_data)
                            for task_data in task_data_list
                            if task_data.parent_set.w_i in w_I_left]
                    log.info(" Still require %d results, so feed %d extra '%s' tasks into queue..." % \
                            (num_left, len(tasks), descr))
                    self.channel.pushTasks(tasks)

                    # reinitialize the countdown
                    countdown = self.ss.num_seconds_before_child_generation_requeue \
                                / self.ss.num_seconds_between_master_result_requests_nonrandom
                else:
                    if len(new_results) == 0:
                        log.info(" Still require %d results, %d seconds until requeue..." % \
                                (num_left, countdown * self.ss.num_seconds_between_master_result_requests_nonrandom))
                        countdown -= 1
                    else:
                        # reinitialize the countdown
                        countdown = self.ss.num_seconds_before_child_generation_requeue \
                                    / self.ss.num_seconds_between_master_result_requests_nonrandom

            #wait before relooping
            time.sleep(self.ss.num_seconds_between_master_result_requests_nonrandom)

        # HACK: add some gen random ind tasks to speed up age gaps
        # we know that once every self.ss.age_gap generations
        # we require self.ss.num_inds_per_age_layer new random inds
        # 
        # so we make sure that every generation we add some random
        # ind generation tasks in order to have most of these present
        # in the random pool at the time they are needed. This keeps
        # the slaves busy
        #
        # however, the random pool still has a certain amount of inds
        # available.
        #
        # we only add 25% in order not to swamp the slaves.
        #
        if self.ss.moea_per_age_layer[0] == 'MOEA/D':
            total_required_at_gap = self.ss.num_inds_per_age_layer/10
        else:
            total_required_at_gap = self.ss.num_inds_per_age_layer
        rnd_inds_required_next_gap = total_required_at_gap \
                                     - self.random_pool.availableCount()
        if rnd_inds_required_next_gap < 0:
            rnd_inds_required_next_gap = 0

        # distribute the tasks over the different generations
        nom_rnd_inds_per_gen = total_required_at_gap / self.ss.age_gap
        nb_rnd_inds = rnd_inds_required_next_gap / self.ss.age_gap
        # limit the number of injected tasks per generation
        # to minimal 1% of the nominal amount per gen, with 
        # a minimum of 1
        # maximal 50% of the nominal amount
        if nb_rnd_inds <= nom_rnd_inds_per_gen * 1/100:
            nb_rnd_inds = nom_rnd_inds_per_gen * 1/100
        elif nb_rnd_inds > nom_rnd_inds_per_gen * 50/100:
            nb_rnd_inds = nom_rnd_inds_per_gen * 50/100

        log.info("Advance queueing %s random generation tasks" % nb_rnd_inds)
        rnd_tasks = []
        rnd_descr = 'Generate random ind'
        rnd_task_data = self._randomIndTaskData()
        for i in range(nb_rnd_inds):
            rnd_tasks.append(Channel.TaskForSlave(self.ID, rnd_descr, rnd_task_data))
        self.channel.pushTasks(rnd_tasks)

        #postconditions
        assert len(inds) == target_num_inds

        #done!
        return inds
    
    def _evaluateIndsFurtherViaChannel(self, task_data_list):
        """Helper to  _generateIndsViaChannel, for descr = 'Evaluate ind further'
        This operates with better throughput by breaking down tasks into one simulation at a time,
        i.e. by {opt point, rnd point, analysis, env point}.
        """
        #preconditions
        ind_IDs = [task_data.ind.shortID() for task_data in task_data_list]
        assert len(ind_IDs) == len(set(ind_IDs)), "can't have duplicate inds in task list"

        #corner case
        if len(task_data_list) == 0:
            return [] 

        #--Algorithm--
        #make fast-access ind dict
        #initialize key-sim data
        #send out all sim requests
        #while results remaining:
        #  gather results and incorporate them
        #  if idle slaves:
        #    send out more (duplicate) sim requests, biasing to requests having fewer sims
        #fill in each ind with all sim results

        #make fast-access ind dict
        opt_point_per_ind_ID = {} # ind_ID : scaled_opt_point
        ind_per_ind_ID = {} # ind_ID : ind
        for task_data in task_data_list:
            ind = task_data.ind
            opt_point_per_ind_ID[ind.shortID()] = self.ps.scaledPoint(ind)
            ind_per_ind_ID[ind.shortID()] = ind

        #initialize key-sim data
        # -this tracks which sim requests to make (by counting), and which are done (deleted when done)
        # -note that we locally simulate functions, which is faster than sending over network
        num_sim_requests_made = {} #[(ind_ID, rnd_ID, an_ID, env_ID)] : int
        for task_data in task_data_list:
            ind, num_rnd_points = task_data.ind, task_data.num_rnd_points
            for rnd_ID in ind.rnd_IDs[:num_rnd_points]:
                for an in self.ps.analyses:
                     for e in an.env_points:                         
                         if not ind.simRequestMade(rnd_ID, an, e):
                             if isinstance(an, FunctionAnalysis):
                                 ind.reportSimRequest(rnd_ID, an, e)
                                 local_results = Evaluator.singleSimulation(
                                     self.ps, opt_point_per_ind_ID[ind.shortID()], rnd_ID, an.ID, e.ID)
                                 ind.setSimResults(local_results, rnd_ID, an, e)
                             else:
                                 num_sim_requests_made[(ind.shortID(), rnd_ID, an.ID, e.ID)] = 0
                     
        # -initialize output
        sim_results = {} #[(ind_ID, rnd_ID, an_ID, env_ID)] : (dict of metric_name : metric_value)
        
        #send out all sim requests
        tasks = []
        for key in num_sim_requests_made.keys():
            (ind_ID, rnd_ID, an_ID, env_ID) = key
            d = self._baseTaskData(self.state.getCurrentGenerationId())
            d.scaled_opt_point = opt_point_per_ind_ID[ind_ID]
            d.ind_ID, d.rnd_ID, d.an_ID, d.env_ID = ind_ID, rnd_ID, an_ID, env_ID
            task = Channel.TaskForSlave(self.ID, 'Single simulation', d)
            tasks.append(task)
            num_sim_requests_made[key] += 1
        target_num_tasks = len(tasks)
        
        log.info("In evaluating %d inds further, we have %d 'Single simulation' tasks" %
                 (len(task_data_list), target_num_tasks))
        if target_num_tasks == 0: return [] #corner case

        self.channel.pushTasks(tasks)

        #while results remaining
        while True:
            #gather results and incorporate them
            new_results = self.channel.popFinishedTasks()
            log.info("Got %d new results from channel" % len(new_results))
            if new_results:
                self.addUnusedResults(new_results)
                tasks = self.popFromUnusedResults(
                    100000, 'Single simulation', self.state.getCurrentGenerationId(), None)
                log.info("Of the new results, %d are for 'Evaluate further' at this gen" % len(tasks))
                for task in tasks: #task is TaskForSlave
                    #update key-sim data (ignore if we already have task's results, e.g. due to dup. request)
                    d = task.task_data #d is TaskData
                    key = (d.ind_ID, d.rnd_ID, d.an_ID, d.env_ID)
                    if num_sim_requests_made.has_key(key):
                        del num_sim_requests_made[key]
                        sim_results[key] = task.result_data.sim_results
                        
            #log
            num_left = target_num_tasks - len(sim_results)
            log.info("Have %d / %d results for: 'Evaluate further'" % (len(sim_results), target_num_tasks))

            #maybe stop
            if num_left == 0:
                log.info("Stop because got target number of results")
                break

            #keep the slaves busy, including zombie-catching
            # -add the single-simulation tasks that have been requested the least
            if (len(self.channel.tasksWaiting()) == 0):
                #set chosen_keys
                tasks_to_feed_on_wait = 5 #magic number alert
                keys = num_sim_requests_made.keys()
                counts = [num_sim_requests_made[key] for key in keys]
                I = numpy.argsort(counts)
                chosen_keys = [keys[i] for i in I[:tasks_to_feed_on_wait]]

                #build tasks
                tasks = []
                for key in chosen_keys:
                    (ind_ID, rnd_ID, an_ID, env_ID) = key
                    d = self._baseTaskData(self.state.getCurrentGenerationId())
                    d.scaled_opt_point = opt_point_per_ind_ID[ind_ID]
                    d.ind_ID, d.rnd_ID, d.an_ID, d.env_ID = ind_ID, rnd_ID, an_ID, env_ID
                    task = Channel.TaskForSlave(self.ID, 'Single simulation', d)
                    tasks.append(task)
                    num_sim_requests_made[key] += 1

                #send off tasks
                log.info(" Still require %d results, so feed %d extra 'Single simulation' "
                         "tasks into queue..." % (num_left, len(tasks)))
                self.channel.pushTasks(tasks)
                
            #wait before relooping
            time.sleep(self.ss.num_seconds_between_master_result_requests_nonrandom)

        #we now have all the results, so insert them back into inds
        assert target_num_tasks == len(sim_results)
        for key in sim_results.keys():
            (ind_ID, rnd_ID, an_ID, env_ID) = key
            ind = ind_per_ind_ID[ind_ID]
            analysis = self.ps.analysis(an_ID)
            env_point = analysis.envPoint(env_ID)
            ind.reportSimRequest(rnd_ID, analysis, env_point)
            ind.setSimResults(sim_results[key], rnd_ID, analysis, env_point)
        inds = ind_per_ind_ID.values()

        #postconditions
        for task_data in task_data_list:
            assert task_data.ind.numRndPointsFullyEvaluated() >= task_data.num_rnd_points

        supplied_ind_IDs = [task_data.ind.shortID() for task_data in task_data_list]
        for result_ind in inds:
            assert result_ind.shortID() in supplied_ind_IDs

        #done!
        return inds

    def _improveIndsViaChannel(self, task_data_list):
        """Helper to  _generateIndsViaChannel, for descr = 'Generate child'
        -Needs to track which w_I (sub-goals) have been done, and which are left.
        """
        #corner case
        target_num_inds = len(task_data_list)
        if target_num_inds == 0:
            return []

        # -push all requests
        descr = 'ImproveTopologyForWeight'
        tasks = [Channel.TaskForSlave(self.ID, descr, task_data)
                 for task_data in task_data_list]
        self.channel.pushTasks(tasks)

        # -initialize output: 'inds'
        inds = {} #all inds found, whether they are for the target descr or randomly-generated

        # -initialize 'w_I_left', which is w_i's for sub-goals in 
        #  Need w_I_left to be emptied (ie all w_i's found) 
        ids_left = set([task_data.ID for task_data in task_data_list])
        ids_found = set()

        # the countdown for task requeueing
        countdown = self.ss.num_seconds_before_child_generation_requeue \
            / self.ss.num_seconds_between_master_result_requests_nonrandom

        new_results_for_us = []
        # -loop until we've hit our target number of inds
        while True:
            #update unused results from channel
            new_results = self.channel.popFinishedTasks()
            log.info("Got %d new results from channel" % len(new_results))
            if new_results:
                self.addUnusedResults(new_results)

                #update 'inds' & 'w_I_left' from unused results
                num_left = target_num_inds - len(inds)
                new_results_for_us = self.popFromUnusedResults(
                    num_left, descr, self.state.getCurrentGenerationId(), None)

                for r in new_results_for_us:
                    if r.task_data.ID in ids_left:
                        old_ind = r.task_data.ind
                        new_ind = r.result_data.ind
                        log.debug("got new ind %s for task ID %s" % (new_ind.ID, r.task_data.ID))
                        inds[r.task_data.ID] = new_ind
                        ids_left.remove(r.task_data.ID)
                        ids_found.add(r.task_data.ID)

                log.info("Of the new results, %d are for '%s', at this gen" % \
                         (len(new_results_for_us), descr))

            #log
            num_left = target_num_inds - len(inds)
            log.info("Have %d / %d inds for: %s" % (len(inds), target_num_inds, descr))
            log.info('IDs_found: %s, IDs_left: %s' % (sorted(ids_found), sorted(ids_left)))
            assert len(ids_left) == num_left

            #maybe stop
            if num_left == 0:
                log.info("Stop because got target number of inds")
                break

            #keep the slaves busy, including zombie-catching
            if (len(self.channel.tasksWaiting()) == 0):
                # only add new tasks after a certain holdoff period
                # since otherwise the difficult child generation doesn't
                # get a chance in multi-cpu setups
                if countdown <= 0:
                    tasks = [Channel.TaskForSlave(self.ID, descr, task_data)
                             for task_data in task_data_list
                             if task_data.ID in ids_left]
                    log.info(" Still require %d results, so feed %d extra '%s' tasks into queue..." % \
                            (num_left, len(tasks), descr))
                    self.channel.pushTasks(tasks)

                    # reinitialize the countdown
                    countdown = self.ss.num_seconds_before_child_generation_requeue \
                                / self.ss.num_seconds_between_master_result_requests_nonrandom
                else:
                    if len(new_results_for_us) == 0:
                        log.info(" Still require %d results, %d seconds until requeue..." % \
                                (num_left, countdown * self.ss.num_seconds_between_master_result_requests_nonrandom))
                        countdown -= 1
                    else:
                        # reinitialize the countdown
                        countdown = self.ss.num_seconds_before_child_generation_requeue \
                                    / self.ss.num_seconds_between_master_result_requests_nonrandom

            #wait before relooping
            time.sleep(self.ss.num_seconds_between_master_result_requests_nonrandom)

        # reorder the inds such that improved inds have the same index as their 'parent'
        reordered_inds = []
        try:
            for task_data in task_data_list:
                old_ind = task_data.ind
                new_ind = inds[task_data.ID]
                if new_ind.ID == old_ind.ID:
                    reordered_inds.append(old_ind)
                else:
                    reordered_inds.append(new_ind)
        except:
            import pdb;pdb.set_trace()
        #postconditions
        assert len(reordered_inds) == target_num_inds

        #done!
        return reordered_inds

    def _resimulateIndsViaChannel(self, inds, target_num_rnd_points=1):
        """Implementation of resimulateInds() by using the Channel's help."""
        
        #preconditions
        self.ps.validate()

        #main work...
        target_num_inds = len(inds)
        if target_num_inds == 0:
            return []

        result_inds = []
        unfinished_tasks = []

        task_id = 0
        for ind in inds:
            task_data = self._baseTaskData(self.state.getCurrentGenerationId())
            task_data.ignore_zombification = False
            
            task_data.ind = ind
            task_data.ind.prepareForPickle()
            task_data.num_rnd_points = target_num_rnd_points

            task = Channel.TaskForSlave(self.ID, "Resimulate", task_data)
            task.task_id = task_id
            task_id += 1

            unfinished_tasks.append(task)

            # push task 
            self.channel.pushTasks([task])

        while True:
            #update unused results from channel
            fresh_results = self.channel.popFinishedTasks()
            self.addUnusedResults(fresh_results)
            log.info("Got %d new results from channel" % len(fresh_results))
            
            if fresh_results:
                #update inds from unused results
                new_results_for_us = self.popFromUnusedResults(
                    len(unfinished_tasks), "Resimulate", self.state.getCurrentGenerationId(), None)
                # match the incoming results with a sent task in order to
                # keep track of unfinished tasks
                accepted_results = []
                for result in new_results_for_us:
                    accepted = False
                    
                    #unefficient but what the hell
                    for unfinished_task in unfinished_tasks:
                        if result.task_id == unfinished_task.task_id:
                            unfinished_tasks.remove(unfinished_task)
                            log.debug("accepting result: %s" % str(result))
                            accepted_results.append(result)
                            break
                        
                    else:
                        log.debug("rejecting result: %s" % str(result))
                        
                # get the accepted results
                new_inds = self.getIndsFromResults(accepted_results)
                result_inds.extend(new_inds)
                log.info("Of the new results, %d are for '%s' at this gen" % (len(new_inds), "Resimulate"))

            elif len(self.channel.tasksWaiting()) == 0:
                log.info("No results and no tasks waiting.")
                target_num_new = len(unfinished_tasks)
                log.info("Still require %d results, feed the unfinished tasks back into the channel..." % \
                         (target_num_new))

                self.channel.pushTasks(unfinished_tasks)

            log.info("Have %d / %d inds" % (len(result_inds), target_num_inds))
            
            if len(unfinished_tasks) == 0:
                log.info("Stop because got all inds back")
                break
            else:
                time.sleep(self.ss.num_seconds_between_master_result_requests_nonrandom)

        for ind in result_inds:
            ind.restoreFromPickle(self.ps)

        return result_inds

    def pushRandomGenerateIndsOnChannel(self):
        """Ask channel to start generating random inds, but don't wait for them to come back"""
        assert self.channel
        target_num_inds = 10 #magic number alert -- related to number of cpus available
        task_data = self._randomIndTaskData()
        descr = self._randomIndTaskDescr(False)
        tasks = [Channel.TaskForSlave(self.ID, descr, task_data)
                 for ind_i in xrange(target_num_inds)]
        self.channel.pushTasks(tasks)
        log.info("Pushed %d random-ind generation tasks onto channel, but not waiting for their return"
                 % target_num_inds)
    
    def _randomIndTaskData(self):
        """Returns a TaskData specifying to generate a random ind"""
        task_data = self._baseTaskData(None)
        task_data.ignore_zombification = True
        return task_data

    def _randomIndTaskDescr(self, with_novelty):
        """Returns either 'Generate novel random ind' or 'Generate random ind'"""
        if with_novelty:
            descr = "Generate novel random ind"
            raise NotImplementedError, "broken by random pool"
        else:
            descr = "Generate random ind"
        return descr

    def _baseTaskData(self, current_generation_id):
        """Use this to create a starting TaskData, which Master will expect it has.
        On top of the base, any attributes can be added.
        Set current_generation_id to None if we don't want the result to be
        generation-dependent (e.g. generating random inds).
        """
        task_data = Channel.TaskData()
        task_data.generation_id = current_generation_id
        task_data.master_id = self.ID
        return task_data

class ParentSet(list):
    """Is a list of parent Inds, but also remembers what w_i they belong to.
    """
    def __init__(self, parents, w_i):
        #list of parent Inds
        list.__init__(self, parents)

        #index into weight vec
        self.w_i = w_i 
