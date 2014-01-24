"""Evaluator.py

Has routines:
-nominalEvalInd(ps, ind)
-evalInd(ps, ind, rnd_IDs)
-nominalEvalIndAtAnalysisEnvPoint(..)
-evalIndAtRndPointAnalysisEnvPoint(..)
-singleSimulation(ps, scaled_opt_point, rnd_ID, an_ID, env_ID)
"""
import logging
import types

from adts import *
from Ind import Ind
from util import mathutil
from util.constants import BAD_METRIC_VALUE, DOCS_METRIC_NAME

log = logging.getLogger('evaluate')

def nominalEvalInd(ps, ind):
    nom_rnd_ID = ps.nominalRndPoint().ID
    return self.evalInd(ps, ind, [nom_rnd_ID])
    
def evalInd(ps, ind, rnd_IDs):
    """
    @description

      Evaluates this ind on all analysis/env_point combos, at specified rnd points,
      stores results on the ind.

      Evaluates on the functionAnalyses first.  If any of them
      come out to be infeasible, then forces the ind to 'BAD' and
      does not run the circuit (simulation) analyses.  That
      could be from area analysis, from FunctionDOCs, or otherwise.

      If during simulation, a metric comes out BAD, then
      it forces the ind to be BAD as well.

    @arguments

      ps -- ProblemSetup
      ind -- Ind object -- ind to evaluate

    @return

       new_num_evaluations_per_analysis -- dict of analysisID : int
       
       <<plus>> it modifies the ind's internal data

    @exceptions

    @notes
    """
    #preconditions
    assert isinstance(ps, ProblemSetup)
    assert isinstance(ind, Ind)
    assert isinstance(rnd_IDs, types.ListType)

    #
    log.info('******************************************************************')
    log.info('******************************************************************')
    log.info('******************************************************************')
    log.info('Evaluate ind ID=%s on %d rnd_IDs: %s: begin' % (ind.ID, len(rnd_IDs), rnd_IDs))

    scaled_opt_point = ps.scaledPoint(ind)

    new_num_evals_per_an = {}
    for an in ps.analyses:
        new_num_evals_per_an[an.ID] = 0

    #functions first
    for rnd_ID in rnd_IDs:
        for analysis in ps.functionAnalyses():
            for env_point in analysis.env_points:
                if not ind.simRequestMade(rnd_ID, analysis, env_point):
                    new_num_evals_per_an[analysis.ID] += 1
                    sim_results = evalIndAtRndPointAnalysisEnvPoint(
                        ps, ind, rnd_ID, analysis, env_point, scaled_opt_point, True)

                    for (metric_name, metric_value) in sim_results.iteritems():
                        #early exit on BAD if function is infeasible (exception: when there are no circuit analyses)
                        if (not ps.metric(metric_name).isFeasible(metric_value)) and (ps.circuitAnalyses()):
                            log.info("Force ind to BAD because function '%s' is infeasible" % metric_name)
                            ind.forceFullyBad()
                            return new_num_evals_per_an

    #do circuit simulation only if still feasible after funcs
    for (rnd_i, rnd_ID) in enumerate(rnd_IDs):
        log.info('------------------------------------------------------------------')
        log.info('------------------------------------------------------------------')
        log.info('Evaluate ind (ind_ID=%s) on rnd point #%d/%d (rnd_ID=%d)' % (ind.ID, rnd_i+1, len(rnd_IDs), rnd_ID))
        for analysis in ps.circuitAnalyses():
            for env_point in analysis.env_points:
                if not ind.simRequestMade(rnd_ID, analysis, env_point):
                    new_num_evals_per_an[analysis.ID] += 1
                    sim_results = evalIndAtRndPointAnalysisEnvPoint(
                        ps, ind, rnd_ID, analysis, env_point, scaled_opt_point, True)

                    if mathutil.hasBAD(sim_results.values()):
                        log.info("Force ind to BAD because BAD_METRIC_VALUE found during simulation")
                        ind.forceFullyBad()

                        return new_num_evals_per_an
            
    log.info("This ind evaluates to 'good'")
        
    scaled_point = ps.scaledPoint(ind)
    
    if log.isEnabledFor(logging.DEBUG):
        log.debug('  scaled_point:  %s', scaled_point)
                 

    return new_num_evals_per_an


def nominalEvalIndAtAnalysisEnvPoint(ps, ind, analysis, env_point, prescaled_ind_opt_point=None,
                                     lightweight=False, save_lis_results=False):
    nom_rnd_ID = ps.nominalRndPoint().ID
    return evalIndAtRndPointAnalysisEnvPoint(
        ps, ind, nom_rnd_ID, analysis, env_point, prescaled_ind_opt_point, lightweight, save_lis_results)
    
def evalIndAtRndPointAnalysisEnvPoint(ps, ind, rnd_ID, analysis, env_point, prescaled_ind_opt_point=None,
                                      lightweight=False, save_lis_results=False):
    """
    @description

      Simulates this ind on the given analysis/env_point/rnd_point combo; stores results on the ind.

      (Does nothing if the evaluation has previously been done)

    @arguments

      ind -- Ind -- ind to evaluate
      rnd_ID -- rnd_ID
      analysis -- Analysis
      env_point -- EnvPoint
      prescaled_ind_opt_point -- ps.scaledPoint(ind)
        -- can pass in so that we don't have to recompute if multiple calls
      lightweight -- bool -- if True, it will some assertions
      save_lis_results -- bool -- if True, it will save the lis results on
        the ind.  Otherwise, it won't.  It usually doesn't need to because
        the single DOC metric value is computed from these.

    @return

       MAINLY, IT... modifies the ind's internal data

       ALSO:
       sim_results -- dict of metric_name : metric_value -- the
         simulation values actually found

    @exceptions

    @notes
    """
    #preconditions
    if not lightweight:
        assert isinstance(ps, ProblemSetup)
        assert isinstance(ind, Ind)
        assert isinstance(rnd_ID, types.IntType) or isinstance(rnd_ID, types.LongType)
        assert rnd_ID in ind.rnd_IDs
        assert isinstance(analysis, Analysis)
        assert isinstance(env_point, EnvPoint)
        assert isinstance(prescaled_ind_opt_point, Point) or (prescaled_ind_opt_point is None)
    
    #corner case: have already evaluated here
    if ind.simRequestMade(rnd_ID, analysis, env_point):
        assert not save_lis_results, "cannot save_lis_results if this ind was already evaluated"
        return ind.getSimResults(rnd_ID, analysis, env_point)

    #remember the request
    ind.reportSimRequest(rnd_ID, analysis, env_point)

    #create 'scaled_opt_point'
    if prescaled_ind_opt_point is None:
        scaled_opt_point = ps.scaledPoint(ind)
        assert scaled_opt_point.is_scaled
    else:
        assert prescaled_ind_opt_point.is_scaled
        scaled_opt_point = prescaled_ind_opt_point

    #main simulation call
    sim_results = singleSimulation(ps, scaled_opt_point, rnd_ID, analysis.ID, env_point.ID)

    #set results
    ind.setSimResults(sim_results, rnd_ID, analysis, env_point)

    return sim_results

def singleSimulation(ps, scaled_opt_point, rnd_ID, an_ID, env_ID):
    """Invokes a single simulation at this (opt point, rnd point, analysis, env point).
    Returns the sim_results as metric_name : metric_value"""
    #get rnd_point, analysis, env_point
    rnd_point = ps.devices_setup.getRndPoint(rnd_ID)
    analysis = ps.analysis(an_ID)
    env_point = analysis.envPoint(env_ID)

    #evaluate on function, or circuit simulator
    if isinstance(analysis, FunctionAnalysis):
        #build opt_env_point
        opt_env_point = Point(True, {})
        opt_env_point.update(scaled_opt_point)
        opt_env_point.update(env_point)

        #call the function, e.g. part.functionDOCsAreFeasible.  Some functions can take a rnd point; others can't.
        try:
            scalar_value = analysis.function(opt_env_point, rnd_point)
        except:
            scalar_value = analysis.function(opt_env_point)

        #set results
        sim_results = {analysis.metric.name : scalar_value}

    elif isinstance(analysis, CircuitAnalysis):
        #compute netlist
        variation_data = (rnd_point, env_point, ps.devices_setup)
        netlist = analysis.createFullNetlist(ps.embedded_part, scaled_opt_point, variation_data)
        
        #call simulator
        (sim_results, lis_results, waveforms_per_ext) = analysis.simulate(netlist)

        #set DOCs metric
        assert not sim_results.has_key(DOCS_METRIC_NAME)
        if analysis.hasDOC():
            have_bad = False
            for metric_value in sim_results.itervalues():
                if metric_value == BAD_METRIC_VALUE:
                    have_bad = True
                    break
            if have_bad:
                sim_results[DOCS_METRIC_NAME] = BAD_METRIC_VALUE
            else:
                cost = ps.embedded_part.simulationDOCsCost(lis_results)
                sim_results[DOCS_METRIC_NAME] = cost
                log.info('%s = %.3f' % (DOCS_METRIC_NAME, cost))

        #NOTE: ignoring waveforms_per_ext because it causes out-of-disk-space errors
        #NOTE: don't bother returning lis_results because we've already extracted DOCs cost, so
        # no real need to keep, and it adds complexity
        #...so we just return sim_results
        
    else:
        raise AssertionError("Unknown analysis class: %s" % str(analysis.__class__))

    return sim_results


