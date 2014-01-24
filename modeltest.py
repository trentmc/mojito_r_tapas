#!/usr/bin/env python2.4
##!/usr/bin/env python 

import logging
import os
import random
import sys

import numpy

from adts import *
from regressor import RegressorUtils
from regressor.Caff import CaffBuildStrategy, CaffFactory
from regressor.Lut import LutStrategy, LutFactory
from regressor.Luc import LucStrategy, LucFactory
#from regressor.Probe import ProbeBuildStrategy, ProbeFactory
from regressor.Sgb import SgbBuildStrategy, SgbFactory
from util import mathutil
from util.ascii import *

#set up logging
logging.basicConfig()
logging.getLogger("ascii").setLevel(logging.INFO)
logging.getLogger("caff").setLevel(logging.INFO)
logging.getLogger("lin").setLevel(logging.WARNING)
#logging.getLogger("lin").setLevel(logging.DEBUG)
logging.getLogger("luc").setLevel(logging.DEBUG)
logging.getLogger("lut").setLevel(logging.DEBUG)
logging.getLogger("sgb").setLevel(logging.DEBUG)
logging.getLogger("var_infl").setLevel(logging.DEBUG)
#logging.getLogger("probe").setLevel(logging.INFO)

#set help message
help = """
Usage: modeltest REGR_TYPE FILEBASE_OR_NUM [INPUT_VAR_TYPE TARGET_VARNAME]

 REGR_TYPE -- luc, lut, caff, sgb, ..
 INPUT_VAR_TYPE -- one of:
   'metrics' (map metrics => target topovar, or all-but-target-metrics => target metric)
   'topovars' (topovars => target metric),
   'allvars' (allvars => target metric)
 FILEBASE_OR_NUM -- Examples:
  -0 - sin(x)
  -1 - walter fu
  -2 - walter lfgain
  -3 - walter offsetn
  -4 - walter pm
  -5 - walter srn
  -6 - walter srp
  -/users/micas/tmcconag/novelty_results/results_three_objs
  - ..
  
"""
    
#===========================================================
#===========================================================

#set magic numbers : begin
target_nmse = 0.01 #0.0 #0.05
num_scrambles = 50 #20

sgb_max_carts = 500 #500
sgb_learn_rate = 0.02 #0.02

perc_test = 0.25
caffeine_do_mobj = True
caffeine_max_num_nonlinear_bases = 4 #15
caffeine_popsize = 100
caffeine_pop_mult = 5
caffeine_max_numgen = 200
probe_rank = 1

#set magic numbers : done

def twoDimArray(vec):
    X = numpy.zeros((1,len(vec)), dtype=float)
    X[0,:] = vec
    return X
    
if __name__== "__main__":            
    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [3, 5]:
        print help
        sys.exit(0)

    #yank out the args
    regr_type = sys.argv[1]
    assert regr_type in ["luc", "lut", "caff", "sgb"]

    #(currently) hardcoded
    filebase_or_num = sys.argv[2]

    # Report what we're working with
    print "Regressor_type = %s, filebase_or_num = %s" % \
          (regr_type, filebase_or_num)
        
    #set train_X, train_y, test_X, test_y, input_varnames
    if filebase_or_num == "0":
        assert num_args == 3
        print "Training data is function #0: sin(x)"
        x = numpy.arange(0.0, 2*3.1415, 0.01)
        y = numpy.sin(x)
        X = numpy.reshape(x, (1, len(x)))
        input_varnames = ["x0"]
        (train_X, train_y, test_X, test_y) = \
                  RegressorUtils.generateTrainTestData(X, y, perc_test)

    elif filebase_or_num in ["1","2","3","4","5","6"]:
        assert num_args == 3
        num = int(filebase_or_num)
        perf_metrics = ["fu", "lfgain", "offsetn", "pm", "srn", "srp"]
        perf_metric = perf_metrics[num - 1]
        print "Training data is function #%d: walter %s" % (num, perf_metric)
        filebase = "/users/micas/tmcconag/regressor_data/walterdata/"
        train_X = asciiTo2dArray(filebase + "train_X.txt")
        train_y = asciiTo2dArray(filebase + "train_" + perf_metric + ".txt")[0,:]
        test_X = asciiTo2dArray(filebase + "test_X.txt")
        test_y = asciiTo2dArray(filebase + "test_" + perf_metric + ".txt")[0,:]
        input_varnames = asciiRowToStrings(filebase + "varnames.txt")

        #
        assert train_X.shape[0] == test_X.shape[0]
        assert train_X.shape[1] == len(train_y) == train_y.shape[0]
        assert test_X.shape[1] == len(test_y) == test_y.shape[0]
    
        #maybe logscale
        if perf_metric == "fu":
            print "Warning: setting y to log(fu), not fu"
            train_y = numpy.log10(train_y)
            test_y = numpy.log10(test_y)
        
    else:
        #this section builds a mapping of topology_vars => metric
        # metrics => topology var
        assert num_args == 5
        input_var_type = sys.argv[3]
        target_varname = sys.argv[4]

        assert input_var_type in ['metrics', 'topovars', 'allvars']
        
        filebase = filebase_or_num
        metrics_filebase = filebase + "_points.unscaled"
        print "Training data filebase is: %s" % filebase

        #load data 
        print "Load training data..."
        all_metric_vars = asciiRowToStrings(filebase  + '_metrics.hdr')
        all_metric_X = numpy.transpose(asciiTo2dArray(filebase + '_metrics.val')) #[metric][sample]
        
        all_topo_vars = asciiRowToStrings(filebase  + '_topos.hdr')
        all_topo_X = numpy.transpose(asciiTo2dArray(filebase + '_topos.val')) #[topovar][sample]
        
        all_unscaled_vars = asciiRowToStrings(filebase  + '_points.unscaled.hdr')
        all_unscaled_X = numpy.transpose(asciiTo2dArray(filebase + '_points.unscaled.val'))
        
        objective_vars = asciiRowToStrings(filebase  + '_objectives.hdr')
        objective_I = [i for (i, metname) in enumerate(all_metric_vars) if metname in objective_vars]
        objective_X = numpy.take(all_metric_X, objective_I, 0)

        #remove 'indID' from data before worrying about it each time.  It's 0th entry / row
        all_topo_vars = all_topo_vars[1:]
        all_unscaled_vars = all_unscaled_vars[1:]
        all_topo_X = numpy.take(all_topo_X, range(1, all_topo_X.shape[0]), 0)
        all_unscaled_X = numpy.take(all_unscaled_X, range(1, all_unscaled_X.shape[0]), 0)

        #build X, y, input_varnames
        if (input_var_type == 'metrics') and (target_varname in all_topo_vars): #metrics => topo_var
            y = all_topo_X[all_topo_vars.index(target_varname),:]
            input_varnames = objective_vars
            X = objective_X
            
        if (input_var_type == 'metrics') and (target_varname in objective_vars): #metrics => metric
            target_i = objective_vars.index(target_varname)
            input_I = [i for (i, var) in enumerate(objective_vars) if var != target_varname]
            
            y = objective_X[target_i,:]
            input_varnames = [var for var in objective_vars if var != target_varname]
            X = numpy.take(objective_X, input_I, 0)
            
        elif input_var_type == 'topovars': #topovars => metric
            assert target_varname in all_metric_vars
            y = all_metric_X[all_metric_vars.index(target_varname), :]
            input_varnames = all_topo_vars
            X = all_topo_X
                
        elif input_var_type == 'allvars': #topovars => metric
            assert target_varname in all_metric_vars
            y = all_metric_X[all_metric_vars.index(target_varname), :]
            input_varnames = all_unscaled_vars
            X = all_unscaled_X
        else:
            raise AssertionError(input_var_type)
        
        train_X, train_y = X, y
        test_X, test_y = X, y
        
    #
    min_y = min(train_y) #don"t include test_y so that same results as evo.
    max_y = max(train_y) # ""
    min_x = min(mathutil.minPerRow(train_X), mathutil.minPerRow(test_X))
    max_x = max(mathutil.maxPerRow(train_X), mathutil.maxPerRow(test_X))
    
    #build regressor...
    print "Have %d training samples and %d test samples" % (len(train_y), len(test_y))
    print "Build regressor..."
    
    regressor = None #fill this in
    nondom_regressors = None #maybe fill this in
    if regr_type == "lut":
        ss = LutStrategy()
        ss.bandwidth = 0.001
        regressor = LutFactory().build(train_X, train_y, ss)

    elif regr_type == "luc":
        ss = LucStrategy()
        regressor = LucFactory().build(train_X, train_y, ss)
                
    elif regr_type == "caff":
        ss = CaffBuildStrategy(caffeine_do_mobj,
                               caffeine_max_num_nonlinear_bases,
                               caffeine_popsize,
                               caffeine_pop_mult,
                               caffeine_max_numgen,
                               target_nmse)
        regressor, nondom_regressors = \
                   CaffFactory().build(train_X, train_y, input_varnames, ss)

    elif regr_type == "probe":
        ss = ProbeBuildStrategy(probe_rank)
        regressor = ProbeFactory().build(train_X, train_y, ss)
        
    elif regr_type == "sgb":
        ss = SgbBuildStrategy(max_carts=sgb_max_carts, learning_rate=sgb_learn_rate,
                              target_trn_nmse=target_nmse)
        regressor = SgbFactory().build(train_X, train_y, ss)
        #regressor = SgbFactory().build(X, y, ss, test_cols)
        #regressor = SgbFactory().build(X, y, ss)

    else:
        raise AssertionError("Unknown regressor type: %s" % regr_type)

    print "Regressor is built"
    print "Regressor: %s" % regressor
    print "Now test regressor"

    #simulate
    #yhat = regressor.simulate(X)
    train_yhat = regressor.simulate(train_X)
    test_yhat = regressor.simulate(test_X)

    #calc nmse
    train_nmse = mathutil.nmse(train_yhat, train_y, min_y, max_y)
    test_nmse = mathutil.nmse(test_yhat, test_y, min_y, max_y)
    print "Train nmse=%.10f" % train_nmse
    print "Test nmse=%.10f" % test_nmse

    #plot output
    from util.octavecall import plotAndPause
    #plotAndPause(X[0,:], y, X[0,:], yhat)
    #plotAndPause(test_X[0,:], test_y, test_X[0,:], test_yhat)

    if regr_type == 'sgb':
        #print relative influence of variables on the output
        from regressor.VarInfluenceUtils import meanStderrs, influenceStr
        mean_stderr_tuples = meanStderrs(regressor, train_X, train_y, num_scrambles=num_scrambles,
                                         force_scramble=True)
        infls = [mean for (mean, stderr) in mean_stderr_tuples]
        s = influenceStr(infls, input_varnames, print_xi=True, print_zero_infl_vars=True)
        print ""
        print "Relative influence on '%s':" % target_varname
        print s

        #dump impacts to a .csv file
        # -first column is var names
        # -second column is relative impact per var
        I = numpy.argsort(infls) #sort in ascending order
        sorted_infls = [infls[i] for i in I]
        sorted_input_varnames = [input_varnames[i] for i in I]
        s = ""
        for (i, (var, infl)) in enumerate(zip(sorted_input_varnames, sorted_infls)):
            s += "%s, %.6f\n" % (var, infl)

        filename = "%s_impacts_on_%s.csv" % (input_var_type, target_varname)
        stringToAscii(filename, s)
        files_created = [filename]

        #if applicable, print relative influence of topo. vars vs. sizing & biasing vars
        if (num_args > 3) and (input_var_type == 'allvars'):
            infl_topo, infl_other = 0.0, 0.0
            for (i, var) in enumerate(all_unscaled_vars):
                if var in all_topo_vars:
                    infl_topo += infls[i]
                else:
                    infl_other += infls[i]

            s = influenceStr(
                [infl_topo, infl_other],
                ['all %d topology variables' % len(all_topo_vars),
                 'all %d sizing & biasing variables' % (len(all_unscaled_vars) - len(all_topo_vars))],
                print_xi=False, print_zero_infl_vars=True)
            print s

            #dump these to a .csv file too
            s = ""
            s += "topology variables, %.6f\n" % infl_topo
            s += "sizing variables, %.6f\n" % infl_other

            filename = "%s_impacts_on_%s__summary.csv" % (input_var_type, target_varname)
            stringToAscii(filename, s)
            files_created.append(filename)


        print "Created files:"
        for filename in files_created:
            print "  " + filename


    if nondom_regressors:
        print 'Have %d nondom_regressors' % len(nondom_regressors)

        #build data
        #train_nmses = twoDimArray([r.nmse for r in nondom_regressors])
        train_nmses  = twoDimArray([
            mathutil.nmse(r.simulate(train_X), train_y, min_y, max_y)
            for r in nondom_regressors])
        caffbase = 'caff_%s_mapping_to_%s' % (input_var_type, target_varname)
        train_nmses_file = '%s_train_nmses.txt' % caffbase
        
        test_nmses  = twoDimArray([
            mathutil.nmse(r.simulate(test_X), test_y, min_y, max_y)
            for r in nondom_regressors])
        test_nmses_file = '%s_test_nmses.txt' % caffbase
        
        complexities = twoDimArray([r.complexity for r in nondom_regressors])
        complexities_file = '%s_complexities.txt' % caffbase

        expressions = ['%d: %s\n\n' % (i,r)
                       for (i, r) in enumerate(nondom_regressors)]
        expressions_file = '%s_expressions.txt' % caffbase
        

        #save to disk
        arrayToAscii(train_nmses_file, train_nmses)
        arrayToAscii(test_nmses_file, test_nmses)
        arrayToAscii(complexities_file, complexities)
        stringsToAscii(expressions_file, expressions)
        print "Created files:"
        print train_nmses_file
        print test_nmses_file
        print complexities_file
        print expressions_file
