"""SynthSolutionStrategy

This is the ss used by Master and Slave.

Includes:
-SynthSolutionStrategy
-ChannelStrategy (an attribute in SynthSolutionStrategy)
"""

import logging
import types

from engine.GtoOptimizer import GtoSolutionStrategy

log = logging.getLogger('master')

def updateSynthStrategy(old_ss):
    """
    For each parameter that old_ss does not have, it sets using default values.
    Helps backwards compatibility (loading old databases).
    Does the modification in-place.
    """
    new_ss = SynthSolutionStrategy(old_ss.do_novelty_gen, old_ss.num_inds_per_age_layer, old_ss.age_gap)
    old_varnames = old_ss.__dict__.keys()
    for new_varname in new_ss.__dict__.iterkeys():
        if new_varname not in old_varnames:
            log.info("Added new attribute '%s' to old solution strategy" % new_varname)
            setattr(old_ss, new_varname, getattr(new_ss, new_varname))

class SynthSolutionStrategy: 
    """
    @description

      Holds 'magic numbers' related to strategy of running synthesis engine
      
    @attributes

      See implementation.
      
    @notes
    """

    def __init__(self, do_novelty_gen, num_inds_per_age_layer, age_gap=20):
        """
        @description

          Constructor.
        
        @arguments

          do_novelty_gen -- bool -- try creating novel topologies too?
          num_inds_per_age_layer -- int --
          age_gap -- int -- how many generations between creating new random inds?
        
        @return

          SynthSolutionStrategy object
    
        @exceptions
    
        @notes

        """
        #preconditions
        assert isinstance(do_novelty_gen, types.BooleanType)
        assert isinstance(num_inds_per_age_layer, types.IntType)
        assert num_inds_per_age_layer > 0
        assert isinstance(age_gap, types.IntType)
        assert age_gap > 0

        #set values...

        #turn on/off novelty-gen
        self.do_novelty_gen = do_novelty_gen

        #if True, will always force novelty when generating random inds
        self.always_with_novelty = False 
        
        #plot first two objectives each gen?
        self.do_plot = False

        #population size PER LAYER                            #[default, ok range]
        self.num_inds_per_age_layer = num_inds_per_age_layer  #[100, 5 .. 1e5]
        # the amount of inds to generate on age gaps. note that MOEA/D has as side-effect
        # that a certain amount of random inds are not used.
        self.num_initial_inds_per_layer = self.num_inds_per_age_layer
        
        #ALPS settings
        # -overall pop size = num_inds_per_age_layer * num_layers
        self.max_num_age_layers = 10  #[10]
        self.age_gap = age_gap        #[20]
         
        # define the EA algorithm per layer
        self.moea_per_age_layer = ['MOEA/D' for i in range(self.max_num_age_layers)]
#         self.moea_per_age_layer = ['NSGA-II' for i in range(self.max_num_age_layers)]
#        self.moea_per_age_layer[0] = 'NSGA-II'
#         self.moea_per_age_layer[1] = 'NSGA-II'
        
        # the number of topos to keep track of per weight. (currently fixed to be of different topology)
        self.topology_layers_per_weight = 20
        self.topology_layers_to_evolve_on = 20

        #set max_age_per_layer.
        self.layer_scheme = 'linear' #'linear' or 'polynomial'
        self.max_age_per_layer = self._maxAgePerLayer(self.layer_scheme, self.age_gap, self.max_num_age_layers)

        #every time we randomly generate a new 0th-age-layer pop, we can
        # either generate inds with novelty or without.  This governs the rate.
        self.num_rnd_gen_rounds_between_novelty = 0 #[1, 0..3]
        self.novel_bias_per_layer = self.getNovelBiasPerLayer()

        #maximum number of generations 
        self.max_num_gens = 100000

        #prob of doing crossover in addition to mutates
        ##FIXME: make this per-age-layer?
        self.prob_crossover = 0.50          #[0.50, 0.1 .. 0.9]

        #prob of mutating 1 choice var
        self.prob_mutate_1_choice_var = 0.10

        #in mutating, stddev of gaussian distribution (assuming that min=0.0 and max=1.0)
        self.mutate_stddev_for_DOC_compliance = 0.05 
        self.mutate_stddev_for_constraint_compliance = 0.03
        self.mutate_stddev_during_evolution = 0.01 

        #[bias to vary using 1 operator, using 2, using 3, ...]
        self.num_vary_biases = [100.0, 15.0, 5.0, 3.0, 2.0, 2.0, 2.0, 1.0]

        #slave-specific attributes
        self.num_seconds_between_slave_task_requests = 1
        self.num_seconds_between_master_result_requests_random = 5    #for random-generation tasks
        self.num_seconds_between_master_result_requests_nonrandom = 1 #for other tasks

        # we wait for a certain amount of time before we requeue child generation tasks
        # otherwise difficult children are not generated
        self.num_seconds_before_child_generation_requeue = 30
        
        #For emphasizing different metrics when they are still infeasible.
        #  If not None, it's a dict of metric_name : metric_weight.  If
        #  a metric does not have an entry, its value is 1.0.  Default of
        #  None means 1.0 for all metrics.  Values larger than 1.0 mean
        #  that the metric gets emphasized more in the sum of violations
        #  Note that master.py gives DOCS_METRIC_NAME a weight of 50.0
        self.metric_weights = {'pole2_margin':3.0}

        #strategies for optimizers in generating feasible individuals within Slave
        self.gto_ss = GtoSolutionStrategy(weight_per_objective=None)

        #robustness-specific attributes (IF the problem has robustness)
        # -implements structural homotopy: more effort nearer to the top
        # -a layer_i needs to have enough evaluations to be considered for layer_i+1
        # -the 0th corner is always nominal.  The other 30 are MC samples (only use the 30 for est. yield)
        # WARNING: this _only_ currently works when there are 10 age layers!!! (HACK)
        self.num_rnd_points_per_layer_for_cost = [1, 1, 4, 4, 7, 7, 10, 10, 21, 21, 31]
        self.num_rnd_points_per_layer_for_eval =    [1, 4, 4, 7, 7, 10, 10, 21, 21, 31, 31]
        
        #Settings for slave's random generation of inds
        self.max_num_tries_on_func = 100
        self.max_num_inds_phase_I = 30
        self.max_num_inds_phase_IIa = 100
        self.max_num_inds_phase_IIb = 50 
        self.max_num_inds_phase_III = 3000
        self.use_hillclimb_not_gto_phase_III = True

        self.do_improve_step = True
        self.max_num_inds_for_weighted_topology_improve = 100

    def setTinyRobust(self):
        """Only turn this on for testing"""
        self.age_gap = 3
        self.max_num_age_layers = 4
        self.max_age_per_layer = self._maxAgePerLayer(self.layer_scheme, self.age_gap, self.max_num_age_layers)
        self.num_rnd_points_per_layer_for_cost = [1, 1, 3, 5]
        self.num_rnd_points_per_layer_for_eval =    [1, 3, 5, 5]
        
        self.max_num_tries_on_func = 2000
        self.max_num_inds_phase_I = 10
        self.max_num_inds_phase_IIa = 0
        self.max_num_inds_phase_IIb = 0 
        self.max_num_inds_phase_III = 0
        
    def setMaxNumGenerations(self, max_num_gens):
        self.max_num_gens = max_num_gens

    def getNovelBiasPerLayer(self):
        """
        Returns self.novel_bias_per_layer, with backwards-compatibility.

        On attribute: 
        When we generate novel inds at age_layer 0, we are allowed to use
         parents from other age layers.  This sets the bias to those levels.
        In general, have strong bias to oldest layers, but also a spike for
         level 0; if level 0 then we draw from random.
        """
        if not hasattr(self, 'novel_bias_per_layer'):
            self.novel_bias_per_layer = [
                layer_i**2
                for layer_i in range(self.max_num_age_layers)]
            self.novel_bias_per_layer[0] = max(1, self.novel_bias_per_layer[0]/5.0)
        
        return self.novel_bias_per_layer

    def _maxAgePerLayer(self, scheme, age_gap, max_num_age_layers):
        """Returns a list of ages (ints) based on scheme of polynomial or linear;
        age_gap, and max_num_age_layers.

        Note: highest active layer has no age limit, regardless of what
        this method returns.
        """
        if scheme == 'polynomial':
            return [(n**2 + 1) * age_gap  for n in range(max_num_age_layers)]
        elif scheme == 'linear':
            return [(n + 1) * age_gap     for n in range(max_num_age_layers)]
        else:
            raise "Unknown scheme '%s'" % scheme

    def allowIndInAgeLayer(self, age_layer_i, ind_genetic_age, num_active_layers):
        """Returns True if this layer is the top age layer, or
        if the genetic_age of the ind is < the max age allowed for this layer
        NOTE: genetic age starts at 0
        """
        if age_layer_i == num_active_layers-1:
            return True
#         elif ind_genetic_age <= self.max_age_per_layer[age_layer_i]:
        elif ind_genetic_age < self.max_age_per_layer[age_layer_i]:
            return True
        else:
            return False
        
    def __str__(self):
        """
        @description

          Override str()
          
        """
        s = []
        s += ["SynthSolutionStrategy={"]
        s += [" do_novelty=%s" % self.do_novelty_gen]
        s += ["; always_with_novelty=%s" % self.always_with_novelty]
        s += ["; num_inds_per_age_layer=%d" % self.num_inds_per_age_layer]
        s += ["; max_num_age_layers=%d" % self.max_num_age_layers]
        s += ["; overall popsize=%d" % 
             (self.num_inds_per_age_layer * self.max_num_age_layers)]
        s += ["; age_gap=%d" % self.age_gap]
        s += ["; layer_scheme=%s" % self.layer_scheme]
        s += ["; max_age_per_layer=%s" % self.max_age_per_layer]
        if self.do_novelty_gen:
            s += ["; num_rnd_gen_rounds_between_novelty=%d" % self.num_rnd_gen_rounds_between_novelty]
            s += ["; novel_bias_per_layer = %s" % str(self.getNovelBiasPerLayer())]
        s += ["; max_num_gens=%d" % self.max_num_gens]
        s += ["; prob_crossover=%.3f" % self.prob_crossover]
        s += ["; prob_mutate_1_choice_var=%.3f" % self.prob_mutate_1_choice_var]
        s += ["; mutate_stddev_for_DOC_compliance=%.4f" % self.mutate_stddev_for_DOC_compliance]
        s += ["; mutate_stddev_for_constraint_compliance=%.4f" % self.mutate_stddev_for_constraint_compliance]
        s += ["; mutate_stddev_during_evolution =%.4f" % self.mutate_stddev_during_evolution ]
        s += ["; num_vary_biases=%s" % self.num_vary_biases]
        s += ["; num_seconds_between_slave_task_requests=%g" % self.num_seconds_between_slave_task_requests]
        s += ["; num_seconds_between_master_result_requests_random=%g" % self.num_seconds_between_master_result_requests_random]
        s += ["; num_seconds_between_master_result_requests_nonrandom=%g" % self.num_seconds_between_master_result_requests_nonrandom]
        s += ["; metric_weights=%s" % self.metric_weights]
        s += ["; gto_ss=%s" % self.gto_ss]
        s += ["; slave-max_num_tries_on_func=%s" % self.max_num_tries_on_func]
        s += ["; slave-max_num_inds_phase_I=%s" % self.max_num_inds_phase_I]
        s += ["; slave-max_num_inds_phase_IIa=%s" % self.max_num_inds_phase_IIa]
        s += ["; slave-max_num_inds_phase_IIb=%s" % self.max_num_inds_phase_IIb]
        s += ["; slave-max_num_inds_phase_III=%s" % self.max_num_inds_phase_III]
        s += ["; slave-use_hillclimb_not_gto_phase_III=%s" % self.use_hillclimb_not_gto_phase_III]
        s += [" /SynthSolutionStrategy}"]
        return "".join(s)
