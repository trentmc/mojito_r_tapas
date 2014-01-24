"""VarInfluenceUtils.py

These are utilities for general handling of variable influences:

Calculating influences:
-confidenceIntervals() can give confidence intervals on variable influences
 for _any_ regressor.  Note that some regressors supply their
 own routine which would be more accurate / faster; so use this when
 those don't exist.
-meanStderrs() gives (mean, stderr) descriptions of relative influence
-meanStderrsToConfidenceIntervals() converts meanStderrs output
 to confidenceIntervals output.  Can be useful when a regressor
 can compute its meanStderr but then we want confidence intervals (eg Rfor).

Note:the influence calculations may be cleaner if in an abstract Regressor class.

Printing influences:
-influenceStr() outputs a string describing (mean) relative
 influences and cumulative influence, looking nearly identical to MSTAT output.
-boundedInfluenceStr() is like influenceStr() but also gives
 the upper/lower confidence bound for each variable
"""

import logging
import copy
import math
import random
import os

import numpy

from Isle import IsleModel, isleXterms
from util import mathutil

log = logging.getLogger('var_infl')


    
    
def meanStderrs(model, X, y, num_scrambles=20, force_scramble=False):

    """
    @description

      This is a generalized approach to get influence per var of any regressor
      model. For each var, returns an estimate of its relative influence
      (as a mean, and standard error of the mean) tuple.

      Even _if_ the model supplies routines for (mean,stderr) or even just mean,
      we _still_ use this algorithm.
    
      Strategy:
      -Q: If we scramble an input variable m's values, does output change much?
      -If no, then variable 'm' doesn't have much impact
      -If yes, then variable 'm' has impact
      -Relative impact of var m = mean( [abs_nmse_diff_for_scramble_i
                                         for each scramble_i] )
      -Stddev in rel impact of var m = stddev( [abs_nmse_diff_for_scramble_i
                                                for each scramble_i] )

      Since for each scramble, we simuluate across a whole X.  If model
      simulation time is non-negligible, then this routine can be slow
      (e.g. it's slow for Rfor, but not bad for smaller regressors
      like LinearModel and Mars.  But Rfor provides its own faster
      routine to do this anyway!).

    @arguments

      model -- ANY regressor!! :)
      X -- 2d array -- input points. [input var #][sample point #]
      y -- 1d array -- one output corresponding to each input [sample point #]
      num_scrambles -- int -- the # times we scramble the inputs, from which we
        take subseqent mean and stderr calcs from

    @return

      mean_stderr_tuples -- list of (infl_mean, infl_stderr) tuples;
        one tuple per var.

    @exceptions

    @notes

      Note that X and y should ideally be the test data (vs. training data)
      sum(infl_means) == 1.0.  infl_stderrs are normalized accordingly..
      
      Concern: will stderr actually reduce if we have more samples in X / y?
      (because we want our confidence intervals to tighten!)
      Answer:
      Less_stderr if (More abs_err samples OR
                      abs_err samples are more consistent)
      and therefore if ... (More num_scrambles OR __more data__!)
    """
    #corner case: trivial calculation
    n, N = X.shape
    if n == 1:
        return [(1.0,0.0)]
    
    min_mean = 0.0   #keep at zero
    min_stderr = 0.0 # " "
        
    if not force_scramble:
        #corner case: the model supplies its own routine
        try:
            mean_stderrs = model.influencePerVarMeanStderrs(X)
            mean_stderrs = _railMinimumsOfMeanStderrs(min_mean, min_stderr,
                                                      mean_stderrs)
            # Rfor, LinearModel, ConstantModel
            log.debug('Model provides its own infl. means/stderrs.')
            return mean_stderrs
        except Exception:
            log.debug('Model does not provide its own infl. means/stderrs. '
                      'Scrambling variables.')
            
    #else...
    
    yhat = model.simulate(X)
    e = mathutil.nmse(y, yhat, min(y), max(y))

    #diffs[var_m][scramble_k] 
    diffs = numpy.zeros((n, num_scrambles), dtype=float)

    vars_to_scramble = range(n)
    
    #for each var to scramble, create a list of the diffs
    # -special case for IsleModels in order to avoid always simulating all xterms
    #  (BIG speed difference)
    scrambled_row_I = range(N) #only create the vector once
    min_y, max_y = min(y), max(y)
    if isinstance(model, IsleModel):
        log.debug("Since model is an isle, do scrambling on X11, not X")
        X11 = mathutil.scaleEachRowToPlusMinus1(X, model.minX, model.maxX)
        Xterms11 = isleXterms(X11, model.bases11)
        scrambled_Xterms11 = copy.copy(Xterms11)
        bases_per_var = {} # var : list_of_bases
        for var_m in vars_to_scramble:
            bases_per_var[var_m] = [base_i
                                    for base_i, base in enumerate(model.bases11)
                                    if base.hasVar(var_m)]
    else:
        scrambled_X = copy.copy(X) 
        
    for scramble_k in range(num_scrambles):
        log.debug("Do scramble #%d/%d" % (scramble_k+1, num_scrambles))
        
        # -scramble in the same ways for each var, which will reduce
        #  variation (due to blocking).  Makes a BIG difference :)
        random.shuffle(scrambled_row_I)
        
        for var_i, var_m in enumerate(vars_to_scramble):
            #compute scrambled_yhat by temporary scrambling of a row in X or X11
            if isinstance(model, IsleModel):
                for base_i in bases_per_var[var_m]:
                    scrambled_Xterms11[base_i] = numpy.take( \
                        Xterms11[base_i,:], scrambled_row_I )
                scrambled_yhat = model.lin_model.simulate( scrambled_Xterms11 )
                for base_i in bases_per_var[var_m]:
                    scrambled_Xterms11[base_i] = Xterms11[base_i,:]
            else:
                scrambled_X[var_m,:] = numpy.take(X[var_m,:], scrambled_row_I)
                scrambled_yhat = model.simulate(scrambled_X)
                scrambled_X[var_m,:] = X[var_m,:]

            #comparison nmse
            scrambled_e = mathutil.nmse(y, scrambled_yhat, min_y, max_y)

            #NOT abs(scrambled_e - e) b/c scrambled_e will usually be >e,
            # and if it isn't:
            # -the difference _shouldn't_ be reflected in the mean calcs;
            #  hence the max(0,diff) there
            # -but it should be reflected in the stderr calcs
            diff = scrambled_e - e 

            diff = diff / max(e, 1.0e-5) #normalize
            diffs[var_m][scramble_k] = diff

            
    means = numpy.array([math.sqrt( max(0.0, mathutil.average(diffs[v,:])))
                           for v in range(n)])
    denom = sum(means)
    if denom == 0.0: denom = 1.0 #avoid divide-by-zero in later lines
    means = means / denom
    
    #compute std errs
    mult = 3.0 #1.0 is no safety margin, >1 is a margin (this is a HACK!!!)
    sqN = math.sqrt(num_scrambles)
    stderrs = numpy.zeros(n, dtype=float)
    
    for var_m in vars_to_scramble:
        stderrs[var_m] = mult*math.sqrt(mathutil.stddev(diffs[var_m,:]))/denom/sqN

    #rail
    mean_stderrs = zip(means, stderrs)
    mean_stderrs = _railMinimumsOfMeanStderrs(min_mean, min_stderr, mean_stderrs)
    
    #done
    log.debug('Done meanStderrs()')
    
    #return a list of tuples
    return mean_stderrs



def _railMinimumsOfMeanStderrs(min_mean, min_stderr, mean_stderrs):
    """
    @description

      Have a min mean impact and stderr for _every_ var

    @arguments

      mean_stderr_tuples -- list of (float,float)

    @return

      railed_mean_stderr_tuples -- list of (float,float)

    @exceptions

    @notes
    """
    means = numpy.array([mean for mean,stderr in mean_stderrs], dtype=float)
    stderrs = numpy.array([stderr for mean,stderr in mean_stderrs],
                            dtype=float)
    
    n = len(means)
    for var_m in range(n):
        means[var_m] = max(means[var_m], min_mean)
        stderrs[var_m] = max(stderrs[var_m], min_stderr)
        
    denom = sum(means)
    if denom == 0.0: denom = 1.0 #avoid divide-by-zero in later lines
    means = means / denom
    
    return zip(means, stderrs)
    
    
def influenceStr(influence_per_var, varnames, print_xi = True,
                 print_zero_infl_vars = False):
    """
    @description

      Prints out relative influence, cumulative influence, and varname for
      each var up to when we hit 100.0% (rounded off) total influence.

      Note that this output has been designed to look like MSTAT output.

      Example output (when no lower/upper; those add two columns):

          #    Var      Cum    Parameter
              Contr    Contr   Name
              ( % )    ( % )
          0    16.60    16.6   M0__nsmm_delr_nsub_nmos_bsim3v3 (x21)
          1    16.00    32.7   M0__nsmm_dela_vfb_nmos_bsim3v3 (x22)
          2    16.00    48.6   M1__nsmm_delr_nsub_nmos_bsim3v3 (x8)
          3    15.50    64.1   M1__nsmm_dela_vfb_nmos_bsim3v3 (x7)
          4    15.00    79.1   M0__nsmm_dela_vtl_nmos_bsim3v3 (x20)
          5    14.60    93.7   M1__nsmm_dela_vtl_nmos_bsim3v3 (x0)
          6     3.30    97.0   M0__nsmm_delr_ubref_nmos_bsim3v3 (x19)
          7     3.00   100.0   M1__nsmm_delr_ubref_nmos_bsim3v3 (x30)

      See the unit tests for more examples.

    @arguments

      influence_per_var -- (1d array or list of float) 
      varnames -- list of strings -- one string for each var, in
        same order as the model's inputs when simulating

    @return

      influence_per_var_string -- string object -- see description.
      
    @exceptions

    @notes        

      If relative influences do not sum up to 1.0, it will normalize them.
      
      For speed, we build up 's' as a list rather than a string
    """
    #Corner case
    if len(influence_per_var) == 0:
        return ''

    s = ['\n']

    #Main case
    influence_per_var, s2 = _conditionInfluencePerVar(influence_per_var)
    s += [s2]
    
    I = numpy.argsort(-1.0*influence_per_var)
    
    s += ['%3s   %6s   %5s   %s\n' % ("#", " Var ", "Cumul", "Parameter")]
    s += ['%3s   %6s   %5s   %s\n' % ("", "Contr", "Contr", "Name")]
    s += ['%3s   %6s   %5s   %s\n' % ("", "( % )", "( % )", "")]
    
    cum_infl = 0.0
    for index, i in enumerate(I):
        cum_infl += influence_per_var[i]
        varname = str(varnames[i])
        if print_xi:
            varname += ' (x%d)' % i
        s += ['%3d   %6.2f   %5.1f   %s\n' % \
              (index,
               100.0*influence_per_var[i],
               100.0*cum_infl,
               varname)]
        
        #maybe stop as soon as we get to zero / near-zero influences
        if not print_zero_infl_vars and str(cum_infl*100.0)[:3] == '100':
            break
        
    return ''.join(s)


def _conditionInfluencePerVar(influence_per_var):
    """
    @description

      Conditions influence_per_var as follows:
      1. If there's a non-numeric entry in influence_per_var
         then all entries get set to 0.0
      2. Normalizes it 
      3. Ensures it's a 1d array
      
    @arguments

      unconditioned_influence_per_var -- list or 1d array of int/float 

    @return

      conditioned_influence_per_var -- 1d array, according to description
      extra_string_output -- extra info about how the conditioning went,
        that the calling routine may wish to use
      
    @exceptions

    @notes      
      
      Helper function for modelInfluence{WithBounds}Str
    """
    s = []
    if min(influence_per_var) < 0:
        raise ValueError("Influence value <0 found: %s" % influence_per_var)

    sum_infls = sum(influence_per_var)
    influence_per_var = numpy.asarray(influence_per_var, dtype=float)
    if sum_infls > 0: 
        influence_per_var = influence_per_var / sum_infls #normalize

    extra_string_output = ''.join(s)

    return influence_per_var, extra_string_output
    
