"""PopulationSummary.py

Includes:
-{worstCase,yield}populationSummaryStr
-{worstCase,yield}populationSummaryToMatlab
"""
from itertools import izip

import numpy

from adts.Metric import MAXIMIZE, MINIMIZE, IN_RANGE
from util.ascii import stringToAscii
from util.constants import BAD_METRIC_VALUE


def worstCasePopulationSummaryStr(ps, inds, sort_metric=None):
    """
    @description

      Returns a string describing a population, with one ind per row, giving worst-case
      performance measures.  Not yield-aware.

    @arguments

      ps -- ProblemSetup
      inds -- list of Ind
      sort_metric -- string or None -- if not None, will sort
        the inds by 'sort_metric'.  Else sorts the inds by op_point.ID

    @return

      s -- string

    @exceptions

    @notes
    """
    #corner case
    if len(inds) == 0:
        return "(No inds)"

    #main case...
    big_s = ['\n']

    # -sort inds according to either ID or sort_metric
    if sort_metric is None:
        #tups = [(ind.shortID(), i) for (i, ind) in enumerate(inds)]
        #tups.sort()
        #I = [i for (ID, i) in tups]
        I = range(len(inds))
    else:
        I = numpy.argsort([ind.worstCaseMetricValue(ind.numRndPointsFullyEvaluated(), sort_metric)
                             for ind in inds])
    inds = numpy.take(inds, I)
     
    sp_per_metric = {}
    for analysis_index, analysis in enumerate(ps.analyses):
        for metric in analysis.metrics:
            sp_per_metric[metric.name] = str(max(9,min(len(metric.name), 14)) + 1)

    topsum_str = 'TopoSummary'
    topsum_sp = str(max(len(topsum_str), len(inds[0].topoSummary())))

    # -output
    #   -row of 'an_index=XX'
    row_s = '%30s %10s %5s ' % ("","","")
    exec("row_s += '%" + topsum_sp + "s' % ''")
    for analysis_index, analysis in enumerate(ps.analyses):
        for metric in analysis.metrics:
            an_str = 'anIdx=%d' % analysis_index
            exec("row_s += '%" + sp_per_metric[metric.name] + "s' % an_str")
    big_s += [row_s + '\n']

    #   -row of 'IND_ID   metricname1   metricname2 ...'
    row_s = '%30s %10s %5s ' % ('Ind_ID', 'GeneticAge', 'Feas')
    exec("row_s += '%" + topsum_sp + "s' % topsum_str")
    for analysis_index, analysis in enumerate(ps.analyses):
        for metric in analysis.metrics:
            metname_str = metric.name[-14:]
            exec("row_s += '%" + sp_per_metric[metric.name]+ "s' % metname_str")
    big_s += [row_s + '\n']

    #   -for each ind, a row of '<ID_value> <metric1_value> <metric2_value> ...'
    for ind in inds:
        num_rnd_points = ind.numRndPointsFullyEvaluated()
        row_s = '%30s %10d %5s ' % (ind.shortID(), ind.genetic_age, ind.isFeasible(num_rnd_points))
        exec("row_s += '%" + topsum_sp + "s' % ind.topoSummary()")
        for analysis_index, analysis in enumerate(ps.analyses):
            for metric in analysis.metrics:
                wc_value =  ind.worstCaseMetricValue(num_rnd_points, metric.name)
                if wc_value == BAD_METRIC_VALUE:
                    val_str = '%s' % BAD_METRIC_VALUE
                else:
                    val_str = '%g' % wc_value
                exec("row_s += '%" + sp_per_metric[metric.name]+ "s' % val_str")
        big_s += [row_s + '\n']
        
    big_s += ['\n']

    big_s += ['More info:\n']
    big_s += ['  Problem number: %d\n' % ps.problem_choice]

    objs = ps.metricsWithObjectives()
    big_s += ['  %d Objectives: ' % len(objs)]
    for (i, obj) in enumerate(objs):
        if i > 0: big_s += [', ']
        big_s += ['%s %s' % (obj.aimStr(), obj.name)] 
    big_s += ['\n']
    
    constraints = ps.flattenedMetrics() #even objectives have constraints
    big_s += ['  %d Constraints: ' % len(constraints)]
    for (i, constraint) in enumerate(constraints):
        if i > 0: big_s += [', ']
        big_s += ['%s' % constraint.prettyStr()]
    big_s += ['\n']

    topo_summaries = [ind.topoSummary() for ind in inds]
    big_s += ['  Of the %d individuals (%d unique ones), there are %d unique topologies.\n' %
              (len(inds), len(set(inds)), len(set(topo_summaries)))]
    
    return ''.join(big_s)

def worstCasePopulationSummaryToMatlab(ps, inds, base_file):
    """
    @description

      Dumps into matlab files a description of a population, giving worst-case
      performance measures.  Not yield-aware.

    @arguments

      ps -- ProblemSetup
      inds -- list of Ind
      base_file -- string -- base filename, which will output files readable by matlab:
      -Metrics in BASE_FILE_metrics.val (data), BASE_FILE_metrics.hdr (row of metric names)
      -Unscaled Points in BASE_FILE_points.unscaled.val (data), BASE_FILE_points.unscaled.hdr (var names) 
      -Scaled Points in BASE_FILE_points.scaled.val (data), BASE_FILE_points.scaled.hdr (var names) 
      -Topology descriptions in BASE_FILE_topos.val (data), BASE_FILE_topos.hdr (row of choice var names)

    @return

      nothing

    @exceptions

    @notes
    """
    #=============================================================================================
    #metrics files... {hdr, val}
    metrics_base_file = base_file + "_metrics"

    #build ".hdr" file
    metrics_s = "IND_ID, GENETIC_AGE, FEASIBLE, TopoSummary"
    for analysis_index, analysis in enumerate(ps.analyses):
        for metric in analysis.metrics:
            metrics_s += ",%s" % metric.name[:14]
    metrics_s += "\n"

    stringToAscii("%s.hdr" % (metrics_base_file), metrics_s)

    #build ".val" file
    # -for each ind, a row of: id, age, feas, toposummary, metric1, metric2, ..., metricn
    metrics_s = ""
    for ind in inds:
        num_rnd_points = ind.numRndPointsFullyEvaluated()
        topo_summary = "9" + ind.topoSummary() #make sure that it doesn't lose leading zeros
        topo_summary = topo_summary.replace("_","").replace("X","8") #make it parseable
        row_s = "%s,%d,%d,%s" % (ind.shortID(), ind.genetic_age, ind.isFeasible(num_rnd_points), topo_summary)
        for analysis_index, analysis in enumerate(ps.analyses):
            for metric in analysis.metrics:
                row_s += ",%g" % ind.worstCaseMetricValue(num_rnd_points, metric.name)
        metrics_s += row_s + "\n"

    stringToAscii("%s.val" % (metrics_base_file), metrics_s)

    #=============================================================================================
    #notposs files... {val}
    notposs_base_file = base_file + "_notposs"

    #build ".hdr" file -- note that this is identical to the *_objectives.hdr file (here for convenience)
    notposs_s = ""
    for metric in ps.metricsWithObjectives():
        notposs_s += ",%s" % metric.name[:14]
    notposs_s += "\n"

    stringToAscii("%s.hdr" % notposs_base_file, notposs_s)
    
    #build ".val" file
    # -for each ind, a row of: objective1_value, objective2_value, ...., objectiveN_value
    notposs_s = ""
    for ind in inds:
        num_rnd_points = ind.numRndPointsFullyEvaluated()
        for metric in ps.metricsWithObjectives():
            value = metric.slightlyBetterValue(ind.worstCaseMetricValue(num_rnd_points, metric.name))
            notposs_s += ",%g" % value
        notposs_s += "\n"

    stringToAscii("%s.val" % (notposs_base_file), notposs_s)

    #=============================================================================================
    #build *_objectives.hdr -- a list of the objectives
    objectives_base_file = base_file + "_objectives"
    
    objectives_s = ""
    for metric in ps.metricsWithObjectives():
        objectives_s += ",%s" % metric.name[:14]
    objectives_s += "\n"

    stringToAscii("%s.hdr" % objectives_base_file, objectives_s)
    
    #=============================================================================================
    #points files... {hdr, val} x {scaled, unscaled}
    points_base_file = base_file + "_points"

    #build ".hdr" files
    unscaled_s = ""; scaled_s  = ""
    unscaled_s += "IND_ID,"; scaled_s += "IND_ID,"

    for (i, name) in enumerate(ps.ordered_optvars):
        unscaled_s += "%s" % (name); scaled_s += "%s" % (name)
        if i < (len(ps.ordered_optvars)-1):
            unscaled_s += ","; scaled_s += ","
    unscaled_s += "\n"; scaled_s += "\n"

    stringToAscii("%s.unscaled.hdr" % (points_base_file), unscaled_s)
    stringToAscii("%s.scaled.hdr"   % (points_base_file), scaled_s)

    #build ".val" files
    unscaled_s = ""; scaled_s = ""
    for ind in inds:
        scaled_point = ps.scaledPoint(ind)
        unscaled_s += "%s," % ind.shortID(); scaled_s += "%s," % ind.shortID()
        for (i, name) in enumerate(ps.ordered_optvars):
            unscaled_s += "%g" % ind.unscaled_optvals[i]
            scaled_s += "%g" % scaled_point[name]
            if i < (len(ps.ordered_optvars)-1):
                unscaled_s += ","; scaled_s += ","
        unscaled_s += "\n"; scaled_s += "\n"

    stringToAscii("%s.unscaled.val" % (points_base_file), unscaled_s)
    stringToAscii("%s.scaled.val" % (points_base_file), scaled_s)
    
    #=============================================================================================
    #topology files... {hdr, val}
    topos_base_file = base_file + "_topos"
    
    choice_vars = ps.embedded_part.part.point_meta.choiceVars()
    choice_vars = [name for name in ps.ordered_optvars if name in choice_vars] #order them
    
    #build ".hdr" file
    topos_s = ""
    topos_s += "IND_ID,"
    for (i, name) in enumerate(choice_vars):
        topos_s += "%s" % (name)
        if i < (len(choice_vars)-1):
            topos_s += ","
    topos_s += "\n"

    stringToAscii("%s.hdr" % (topos_base_file), topos_s)
    
    #build ".val" file
    topos_s = ""
    for ind in inds:
        topos_s += "%s," % ind.shortID()
        
        topo_point = ind.topoPoint()
        for (i, var) in enumerate(choice_vars):
            topos_s += "%g" % topo_point[var] #note that '-1' means "don't care"
            if i < (len(choice_vars)-1):
                topos_s += ","
        topos_s += "\n"
        
    stringToAscii("%s.val" % (topos_base_file), topos_s)

    #=============================================================================================
    #topology files... {hdr, val}
    topos_base_file = base_file + "_topotrace"

    #build ".val" file
    topos_s = ""
    for ind in inds:
        topos_s += "%s," % ind.shortID()
        topos_s += "%s" % str(ind.topo_trace)
        topos_s += "\n"
        
    stringToAscii("%s.val" % (topos_base_file), topos_s)

def yieldPopulationSummaryStr(ps, lite_inds, inds):    
    """
    @description

      Returns a string describing a population, with one LiteInd per row, giving spec vs. yield
      measures.  Sorts by yield.

    @arguments

      ps -- ProblemSetup
      lite_inds -- list of LiteInd
      inds -- list of Ind, which are referenced by lite_inds

    @return

      s -- string

    @exceptions

    @notes
    """
    #corner case
    if len(inds) == 0:
        return "(No inds)"

    #main case...
    big_s = ['\n']

    # -sort inds according to negative yield (the last measure), i.e. to descending order of yield
    neg_yields = [lite_ind.costs[-1] for lite_ind in lite_inds]
    I = numpy.argsort(neg_yields)
    lite_inds = numpy.take(lite_inds, I)

    obj_metnames = ['yield'] + [metric.name for metric in ps.metricsWithObjectives()]
     
    sp_per_metric = {} # metric_name : spaced_metric_name
    for metname in obj_metnames:
        sp_per_metric[metname] = str(max(11,min(len(metname), 14)) + 1)

    topsum_str = 'TopoSummary'
    topsum_sp = str(max(len(topsum_str), len(inds[0].topoSummary())))

    # -output
    #   -row of 'an_index=XX'
    row_s = '%18s %18s %10s ' % ("","","")
    exec("row_s += '%" + topsum_sp + "s' % ''")
    for metname in obj_metnames:
        if metname == 'yield': an_str = ''
        else:                  an_str = 'anIdx=%d' % ps.analysisIndexOfMetric(metname)
        exec("row_s += '%" + sp_per_metric[metname] + "s' % an_str")
    big_s += [row_s + '\n']

    #   -row of 'IND_ID   metricname1   metricname2 ...'
    row_s = '%18s %18s %10s ' % ('LiteID', 'Ind_ID', 'GeneticAge')
    exec("row_s += '%" + topsum_sp + "s' % topsum_str")
    for metname in obj_metnames:
        metname_str = metname[-14:]
        exec("row_s += '%" + sp_per_metric[metname]+ "s' % metname_str")
    big_s += [row_s + '\n']

    #   -for each LiteInd, a row of:
    #    '<lite_ind_ID> <ind_ID> <genetic_age> <metric1_value=yield> <metric2_value> ...'
    ID_to_ind = _indToIndID(inds)
    for lite_ind in lite_inds:
        ind = ID_to_ind[lite_ind.ind_ID]
        
        row_s = '%18s %18s %10d ' % (lite_ind.ID, ind.shortID(), ind.genetic_age)
        exec("row_s += '%" + topsum_sp + "s' % ind.topoSummary()")
        for (cost, metname) in izip([lite_ind.costs[-1]] + lite_ind.costs[:-1], obj_metnames):
            val = _costToVal(ps, metname, cost)
            if metname == 'yield': val_str = '%.3f' % abs(val) #the 'abs' fixes printing -0.000
            else:                  val_str = coststr2(val) 
            exec("row_s += '%" + sp_per_metric[metname]+ "s' % val_str")
        big_s += [row_s + '\n']
        
    big_s += ['\n']

    big_s += ['More info:\n']
    big_s += ['  Problem number: %d\n' % ps.problem_choice]
    
    objs = ps.metricsWithObjectives()
    big_s += ['  %d Objectives: maximize yield' % (len(objs)+1)]
    for (i, obj) in enumerate(objs):
        big_s += [', %s %s' % (obj.aimStr(), obj.name)] 
    big_s += ['\n']
    
    constraints = [metric for metric in ps.flattenedMetrics() if not metric.improve_past_feasible]
    big_s += ['  %d Constraints: ' % len(constraints)]
    for (i, constraint) in enumerate(constraints):
        if i > 0: big_s += [', ']
        big_s += ['%s' % constraint.prettyStr()]
    big_s += ['\n']
    
    topo_summaries = [ind.topoSummary() for ind in inds]
    big_s += ['  In the %d points and %d Inds, there are %d unique topologies.\n' %
              (len(lite_inds), len(inds), len(set(topo_summaries)))]
    
    return ''.join(big_s)

def yieldPopulationSummaryToMatlab(ps, lite_inds, inds, base_file):
    """
    @description

      Dumps into matlab files a description of a population,
      with one LiteInd per row, giving spec vs. yield measures.  Sorts by yield.  

    @arguments

      ps -- ProblemSetup
      lite_inds -- list of LiteInd
      inds -- list of Ind
      base_file -- string -- base filename, which will output files readable by matlab:
      -Metrics in BASE_FILE_metrics.val (data), BASE_FILE_metrics.hdr (row of metric names)
      -Unscaled Points in BASE_FILE_points.unscaled.val (data), BASE_FILE_points.unscaled.hdr (var names) 
      -Scaled Points in BASE_FILE_points.scaled.val (data), BASE_FILE_points.scaled.hdr (var names) 
      -Topology descriptions in BASE_FILE_topos.val (data), BASE_FILE_topos.hdr (row of choice var names)

    @return

      nothing

    @exceptions

    @notes
    """
    #=============================================================================================
    #metrics files... {hdr, val}
    metrics_base_file = base_file + "_metrics"

    obj_metnames = ['yield'] + [metric.name for metric in ps.metricsWithObjectives()]
    ID_to_ind = _indToIndID(inds)
    
    #build ".hdr" file
    metrics_s = "LITEID, IND_ID, GENETIC_AGE, TopoSummary"
    for metname in obj_metnames:
        metrics_s += ",%s" % metname[:14]
    metrics_s += "\n"

    stringToAscii("%s.hdr" % (metrics_base_file), metrics_s)

    #build ".val" file
    # -for each ind, a row of: id, age, feas, toposummary, metric1, metric2, ..., metricn
    metrics_s = ""
    for lite_ind in lite_inds:
        ind = ID_to_ind[lite_ind.ind_ID]
        
        topo_summary = "9" + ind.topoSummary() #make sure that it doesn't lose leading zeros
        topo_summary = topo_summary.replace("_","").replace("X","8") #make it parseable
        row_s = "%d,%s,%d,%s" % (lite_ind.ID, ind.shortID(), ind.genetic_age, topo_summary)
        for (cost, metname) in izip([lite_ind.costs[-1]] + lite_ind.costs[:-1], obj_metnames):
            val = _costToVal(ps, metname, cost)
            row_s += ",%g" % val
        metrics_s += row_s + "\n"

    stringToAscii("%s.val" % (metrics_base_file), metrics_s)

    #=============================================================================================
    #notposs files... {val}
    notposs_base_file = base_file + "_notposs"

    #build ".hdr" file -- note that this is identical to the *_objectives.hdr file (here for convenience)
    notposs_s = ""
    for (i, metname) in enumerate(obj_metnames):
        if i > 0: notposs_s += ","
        notposs_s += "%s" % metname[:14]
    notposs_s += "\n"

    stringToAscii("%s.hdr" % notposs_base_file, notposs_s)
    
    #build ".val" file
    # -for each ind, a row of: objective1_value, objective2_value, ...., objectiveN_value
    notposs_s = ""
    for lite_ind in lite_inds:
        ind = ID_to_ind[lite_ind.ind_ID]
        i = 0
        for (cost, metname) in izip([lite_ind.costs[-1]] + lite_ind.costs[:-1], obj_metnames):
            val = _costToVal(ps, metname, cost)
            if metname == 'yield': val = val + 0.01 
            else:                  val = metric.slightlyBetterValue(val)
            if i > 0: notposs_s += ","
            notposs_s += "%g" % val
            i += 1
            
        notposs_s += "\n"

    stringToAscii("%s.val" % (notposs_base_file), notposs_s)

    #=============================================================================================
    #build *_objectives.hdr -- a list of the objectives
    objectives_base_file = base_file + "_objectives"
    
    objectives_s = ""
    for (i, metname) in enumerate(obj_metnames):
        if i > 0:
            objectives_s += ","
        objectives_s += "%s" % metname[:14]
    objectives_s += "\n"

    stringToAscii("%s.hdr" % objectives_base_file, objectives_s)
    
    #=============================================================================================
    #points files... {hdr, val} x {scaled, unscaled}
    points_base_file = base_file + "_points"

    #build ".hdr" files
    unscaled_s = ""; scaled_s  = ""
    unscaled_s += "LiteID,"; scaled_s += "LiteID,"
    unscaled_s += "IND_ID,"; scaled_s += "IND_ID,"

    for (i, name) in enumerate(ps.ordered_optvars):
        unscaled_s += "%s" % (name); scaled_s += "%s" % (name)
        if i < (len(ps.ordered_optvars)-1):
            unscaled_s += ","; scaled_s += ","
    unscaled_s += "\n"; scaled_s += "\n"

    stringToAscii("%s.unscaled.hdr" % (points_base_file), unscaled_s)
    stringToAscii("%s.scaled.hdr"   % (points_base_file), scaled_s)

    #build ".val" files
    unscaled_s = ""; scaled_s = ""
    for lite_ind in lite_inds:
        ind = ID_to_ind[lite_ind.ind_ID]
        scaled_point = ps.scaledPoint(ind)
        unscaled_s += "%d," % lite_ind.ID; scaled_s += "%d," % lite_ind.ID
        unscaled_s += "%s," % ind.shortID(); scaled_s += "%s," % ind.shortID()
        for (i, name) in enumerate(ps.ordered_optvars):
            unscaled_s += "%g" % ind.unscaled_optvals[i]
            scaled_s += "%g" % scaled_point[name]
            if i < (len(ps.ordered_optvars)-1):
                unscaled_s += ","; scaled_s += ","
        unscaled_s += "\n"; scaled_s += "\n"

    stringToAscii("%s.unscaled.val" % (points_base_file), unscaled_s)
    stringToAscii("%s.scaled.val" % (points_base_file), scaled_s)
    
    #=============================================================================================
    #topology files... {hdr, val}
    topos_base_file = base_file + "_topos"
    
    choice_vars = ps.embedded_part.part.point_meta.choiceVars()
    choice_vars = [name for name in ps.ordered_optvars if name in choice_vars] #order them
    
    #build ".hdr" file
    topos_s = ""
    topos_s += "LiteID,"
    topos_s += "IND_ID,"
    for (i, name) in enumerate(choice_vars):
        topos_s += "%s" % (name)
        if i < (len(choice_vars)-1):
            topos_s += ","
    topos_s += "\n"

    stringToAscii("%s.hdr" % (topos_base_file), topos_s)
    
    #build ".val" file
    topos_s = ""
    for lite_ind in lite_inds:
        ind = ID_to_ind[lite_ind.ind_ID]
        topos_s += "%d," % lite_ind.ID
        topos_s += "%s," % ind.shortID()
        
        topo_point = ind.topoPoint()
        for (i, var) in enumerate(choice_vars):
            topos_s += "%g" % topo_point[var] #note that '-1' means "don't care"
            if i < (len(choice_vars)-1):
                topos_s += ","
        topos_s += "\n"
        
    stringToAscii("%s.val" % (topos_base_file), topos_s)



def _costToVal(ps, metname, cost):
    if metname == 'yield': aim = MAXIMIZE
    else:                  aim = ps.metric(metname)._aim

    if aim == MAXIMIZE:    val = -cost
    elif aim == MINIMIZE:  val = cost
    elif aim == IN_RANGE:  raise NotImplementedError("build me")
    else:                  raise AssertionError("unknown metric aim")

    return val

def _indToIndID(inds):
    """Given a list of inds, returns a dict of ind_ID : ind"""
    d = {}
    for ind in inds:
        d[ind.ID] = ind
    return d

def coststr2(x):
    if x == 0.0:
        return "0.0"
    elif (0.01 < abs(x) < 1000.0):
        return "%9.4f" % x #could be 9.5 
    else:
        return "%9.4e" % x #could be 9.5 
