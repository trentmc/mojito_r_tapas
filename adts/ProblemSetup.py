"""ProblemSetup.py

Holds:
-ProblemSetup

"""
from itertools import izip
import logging

from adts.Analysis import FunctionAnalysis, CircuitAnalysis
from adts.Point import Point, EnvPoint
from adts.DevicesSetup import DevicesSetup
from util import mathutil

log = logging.getLogger('problems')

        
class ProblemSetup:
    """
    @description

      Holds all the information necessary to attack problem.
      At the core, it's:
      -the search space
      -the goals
      -rnd problem setup
      
    @attributes
      
      embedded_part -- EmbeddedPart object -- describes top-level Part along with its connections.
        Note that the 'functions' attribute of this embedded_part will change with every Ind;
        upon instantiation here, it's ok to set each function value to None.
      analyses -- list of Analysis objects -- hold info about goals
      parts_library -- Library instance -- e.g. SizesLibrary or OpLibrary
      devices_setup -- DevicesSetup -- info about generating rnd points
      problem_choice -- int 
      ordered_optvars -- ordered list of optvar names so that each ind's opt_point can
        be stored as a list of values, rather than a dict storing var names (for mem reasons)
      
    @notes
    """
    
    def __init__(self, embedded_part, analyses, parts_library, devices_setup=None):
        """
        @description
        
        @arguments

          embedded_part -- see class description
          analyses -- see class description
          parts_library -- see class description
          devices_setup -- see class description (if None is specified, it creates a default one)
        
        @return

          ProblemSetup object
    
        @exceptions
    
        @notes
        
        """
        #preconditions
        #assert isinstance(embedded_part, EmbeddedPart) #turn off assertion to remove dependencies...
        assert embedded_part is not None                #...but still have a lightweight check
        if len(analyses) == 0:
            raise ValueError("Need >0 analyses")
        an_IDs = []
        for analysis in analyses:
            if analysis.ID in an_IDs:
                raise ValueError('found duplicate analysis ID: %s, %s' %
                                 (analysis.ID, an_IDs))
            an_IDs.append(analysis.ID)
        metric_names = set([])
        for an in analyses:
            for metric in an.metrics:
                assert metric.name not in metric_names
                metric_names.add(metric.name)

        #set values
        self.embedded_part = embedded_part
        self.analyses = analyses
        self.parts_library = parts_library
        if devices_setup is None:
            self.devices_setup = DevicesSetup('UMC180')
        else:
            self.devices_setup = devices_setup
        self.problem_choice = None #set later

        #have an ordered list of optvar names so that each ind's "point" can
        # merely be a list of values, rather than a dict storing var names
        pm = self.embedded_part.part.point_meta
        self.ordered_optvars = sorted(pm.keys())

        #fast-access dicts
        self._metric_name_to_metric = None
        self._flattened_metrics = None
        self._flattened_metric_names = None
        self._metrics_with_objectives = None
        self._doc_metric_names = None
        self._updateFastAccessDicts()

        tmp = sorted([(an.relative_cost, an) for an in self.analyses])
        self._analysis_sorted_by_cost = [t[1] for t in tmp]

        #postconditions
        pm = self.embedded_part.part.point_meta
        assert None not in pm.keys()
        num_rounds = 5
        for round_i in range(num_rounds):
            for with_novelty in [False, True]:
                uo = pm.createRandomUnscaledPoint(with_novelty)
                so = pm.scale(uo)

    def stripToSpecifiedMetrics(self, metric_names):
        """Strip all of self's metrics that are not in 'metric_names'"""
        for analysis in self.analyses:
            analysis.metrics = [metric for metric in analysis.metrics if metric.name in metric_names]
        
        self._updateFastAccessDicts()

    def stripAllButDOCs(self, include_circuit_analysis):
        """Remove all analyses except for ones with DOCs in them.
        If include_circuit_analysis, then it includes circuit analysis DOCs and function DOCs;
        otherwise it just includes function DOCs"""
        self.analyses = [an for an in self.DOCAnalyses()
                         if (isinstance(an, FunctionAnalysis) or include_circuit_analysis)]
        
        self._updateFastAccessDicts()

    def DOCMetricNames(self):
        """Returns a list of all metric names that have 'DOC' in them.  Can be func or sim."""
        return self._doc_metric_names

    def DOCAnalyses(self):
        """Returns a list of the analyses of self that have DOCs.  Can be func or sim."""
        return [an for an in self.analyses
                if an.hasDOC()]

    def nominalRndPoint(self):
        return self.devices_setup.nominalRndPoint()

    def _updateFastAccessDicts(self):
        #fast-access dict for metrics
        self._metric_name_to_metric = {}
        for analysis in self.analyses:
            for metric in analysis.metrics:
                self._metric_name_to_metric[metric.name] = metric

        #fast-access for flattened metrics
        self._flattened_metrics = [metric
                                   for an in self.analyses
                                   for metric in an.metrics]

        #fast-access for flattened metric names
        self._flattened_metric_names = [metric.name for metric in self._flattened_metrics]

        #fast-access for metrics with objectives
        self._metrics_with_objectives = [metric for metric in self._flattened_metrics
                                         if metric.improve_past_feasible]

        #fast-access for metrics with DOCs
        self._doc_metric_names = [name for name in self._flattened_metric_names
                                  if "DOC" in name]

        tmp = sorted([(an.relative_cost, an) for an in self.analyses])
        self._analysis_sorted_by_cost = [t[1] for t in tmp]

    def validate(self):
        """Raises an exception if 'self' is inconsistent in some way.
        Current checks:
        -ordered_optvars == embedded part's optvars
        """
        assert self.ordered_optvars == sorted(self.embedded_part.part.point_meta.keys())
        
    def setProblemChoice(self, problem_choice):
        """Typically, only problems/Problems.py needs to call this."""
        self.problem_choice = problem_choice

    def updateOrderedOptvarsFromEmbPart(self):
        self.ordered_optvars = sorted(self.embedded_part.part.point_meta.keys())
        self.validate()

    def updateOptPointMetaFlexPartChoices(self, broadening_means_novel):
        """Calls EmbeddedPart.updateOptPointMetaFlexPartChoices (more details there),
        and also updates self.ordered_optvars correspondingly
        """
        self.validate()
        vars_before = self.ordered_optvars
        self.embedded_part.updateOptPointMetaFlexPartChoices(broadening_means_novel)
        self.updateOrderedOptvarsFromEmbPart()
        vars_after = self.ordered_optvars
        if len(vars_after) > len(vars_before):
            log.warning("Needed to add new vars to account for all choice vars: %s" %
                        mathutil.listDiff(vars_after, vars_before))

    def unscaledPoint(self, ind):
        """Return an unscaled Point from the incoming ind's list of unscaled var values
        """
        d = {}
        for (var, val) in izip(self.ordered_optvars, ind.unscaled_optvals):
            d[var] = val
        unscaled_point = Point(False, d)
        return unscaled_point
                
    def scaledPoint(self, ind):
        """Return a scaled Point from the incoming ind's list of unscaled var values
        """
        scaled_point = self.embedded_part.part.point_meta.scale(self.unscaledPoint(ind))
        return scaled_point

    def addAnalysis(self, an):
        """Add analysis 'an' to self.analyses, safely"""
        assert an.ID not in [other_an.ID for other_an in self.analyses]
        self.analyses.append(an)
        self._updateFastAccessDicts()

    def functionAnalyses(self):
        """
        @description

          Returns the list of analyses that are FunctionAnalysis objects
          
        """
        return [analysis for analysis in self.analyses
                if isinstance(analysis, FunctionAnalysis)]

    def circuitAnalyses(self):
        """
        @description

          Returns the list of analyses that are CircuitAnalysis objects
          
        """
        return [analysis for analysis in self.analyses
                if isinstance(analysis, CircuitAnalysis)]

    def maxAnalysisCost(self):
        """
        returns the cost of the most expensive analysis
        """

        # no analysis
        if len(self._analysis_sorted_by_cost) == 0:
            return 0.0
        return self._analysis_sorted_by_cost[-1].relative_cost

    def analysesSortedByCost(self):
        """
        returns a list of analysis, ordered by cost
        """
        return self._analysis_sorted_by_cost

    def doRobust(self):
        """Returns if the problem requires that we do robust.  I.e. is there random variation in the devices_setup"""
        return self.devices_setup.doRobust()

    def addNoveltyAnalysis(self):
        """Add a goal to minimize novelty.  Relies on self.embedded_part
        to offer a function called 'novelty'."""
        if self.hasNoveltyAnalysis():
            log.warning("Tried to add novelty analysis, but already have it")
        else:
            novelty_an = FunctionAnalysis(self.embedded_part.novelty, [EnvPoint(True)],
                                          float('-Inf'), 1000, True, 0, 5)
            self.addAnalysis(novelty_an)
            log.info("Added novelty analysis to PS")
        
    def hasNoveltyAnalysis(self):
        """Returns True if one of self's analyses has a metric that
        contains 'novelty' in its name"""
        return (self.noveltyAnalysis() is not None)

    def noveltyAnalysis(self):
        """Returns the analysis in 'self' that measures novelty;
        and None if there is not such an analysis.
        Test via: 'novelty' in any metric name.
        """
        for an in self.analyses:
            for metric in an.metrics:
                if 'novelty' in metric.name:
                    return an
        return None

    def metric(self, metric_name):
        """
        @description

          Returns the metric corresponding to 'metric_name'
        
        @arguments

          metric_name -- string
        
        @return

          metrics -- Metric object
    
        @exceptions
    
        @notes
        
        """
        if self._metric_name_to_metric.has_key(metric_name):
            return self._metric_name_to_metric[metric_name]
        else:
            raise ValueError("No metric with name '%s' found" % metric_name)

    def numMetrics(self):
        """Returns total number of metrics"""
        return len(self.flattenedMetricNames())

    def numObjectives(self):
        """Returns the number of metrics that have objectives"""
        return len(self._metrics_with_objectives)

    def metricsWithObjectives(self):
        """Returns the subset of flattened metrics that have objectives.  Order matters."""
        return self._metrics_with_objectives

    def analysisIndexOfMetric(self, metric_name):
        """Returns the analysis index corresponding to metric_name
        (i.e. self.analyses[analysis_index] has the target metric)"""
        for (an_index, an) in enumerate(self.analyses):
            for metric in an.metrics:
                if metric.name == metric_name:
                    return an_index
        raise AssertionError("no analysis found for metric_name=%s" % metric_name)

    def analysis(self, target_an_ID):
        """Returns the analysis having target_an_ID; raises exception if not found"""
        for an in self.analyses:
            if an.ID == target_an_ID:
                return an
        raise AssertionError
    
    def flattenedMetrics(self):
        """
        @description

          Returns list of metrics, flattened across analyses.  Order matters.
        
        @arguments

          <<none>>
        
        @return

          metrics_list -- list of Metric
    
        @exceptions
    
        @notes

        """
        return self._flattened_metrics

    def flattenedMetricNames(self):
        """
        @description

          Returns list of metric _names_, flattened across analyses
        
        @arguments

          <<none>>
        
        @return

          metrics_list -- list of string
    
        @exceptions
    
        @notes
        
        """
        return self._flattened_metric_names
                
        
    def prettyStr(self):
        """
        @description

          Pretty description of each metric.
          
        """
        s = '\n\nProblem Setup:\n'
        s += '   Embedded_part = %s\n\n' % self.embedded_part.part.name
        
        s += '   Point_meta:\n%s\n' % self.embedded_part.part.point_meta
        
        num_con, num_obj = 0,0
        for analysis in self.analyses:
            s += 'Analysis ID = %d:\n' % analysis.ID
            for metric in analysis.metrics:
                s += '   Constraint: ' + metric.prettyStr() + '\n'
                num_con += 1
                num_obj += metric.improve_past_feasible
            s += '\n'
        s += 'Total # constraints = %d; total # objectives = %d\n' % \
             (num_con, num_obj)
        return s
        
        
    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = ''
        s += '\nProblemSetup={\n'
        s += ' problem_choice = %s\n' % self.problem_choice
        s += ' doRobust? %s\n' % self.doRobust()
        s += ' embedded_part = %s\n' % self.embedded_part.part.name
        s += ' point_meta = %s\n' % self.embedded_part.part.point_meta
        s += ' # analyses = %s' % len(self.analyses)
        s += ' devices_setup = %s' % self.devices_setup
        s += '\n\nProblemSetup.analyses={'
        for (i, analysis) in enumerate(self.analyses):
            s += '\n\nProblemSetup.analyses[%d]: \n%s' % (i, analysis)
        s += '\n\n/ProblemSetup.Analyses}\n'
        s += '\n/ProblemSetup}\n'
        return s
        
 
