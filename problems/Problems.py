"""Problems.py

Includes:
-ProblemFactory

"""
import logging
import os

import numpy

from adts import *
from SizesLibrary import Point18SizesLibrary, SizesLibraryStrategy, SizesLibrary
from OpLibrary import OpLibraryStrategy, OpLibrary
from OpLibrary2 import OpLibrary2
from OpLibraryEMC import OpLibraryEMC
from MetricCalculators import NmseOnTargetWaveformCalculator, NmseOnTargetShapeCalculator, TransientWaveformCalculator
from util.constants import PROBLEM_DESCRIPTIONS, DOCS_METRIC_NAME, INF

log = logging.getLogger('problems')

WAVEFORM_NUMBER_WIDTH = 11

TYPICAL_DOC_MEASURES = ['region','model','vgs', 'vth', 'vds', 'vdsat','sat_violation']

DISABLE_TRANSIENT_GLOBALLY = False

#Note: to change the MOS models, change constants.NMOS_TEMPLATE_MODEL, PMOS_TEMPLATE_MODEL

#==============================================================================================
#define some metric templates so that we don't have to continually redefine them below
# -they are all constraints here, but then we can add objectives via 'setAsObjective()'
# -their specifications have initial settings here, but can change later with 'setMin/MaxThreshold()'

#typical ac metrics...
METRIC__DOCS         = Metric(DOCS_METRIC_NAME, -INF, 1e-5, False, 0, 10)
METRIC__gain         = Metric('gain', 20, INF, False, 10, 80)
METRIC__phase0       = Metric('phase0', -30, 30, False, -50, 50)
METRIC__phasemargin  = Metric('phasemargin', 65, 180, False, 50, 180)
METRIC__gbw          = Metric('gbw', 1.0e6, INF, False, 0, 1e9)
METRIC__pole1fr      = Metric('pole1fr', 0.0, INF, False, 0, 1e9)
METRIC__pole2fr      = Metric('pole2fr', 0.0, INF, False, 0, 1e9)
METRIC__pole2_margin = Metric('pole2_margin', 1.0, INF, False, 0, 1)
METRIC__pwrnode      = Metric('pwrnode', -INF, 100.0e-3,False, 0, 100e-3)
METRIC__fbmnode      = Metric('fbmnode', -INF, 50.0e-3, False, 0, 2)
METRICS__ac_metrics_big =   [METRIC__DOCS, METRIC__gain, METRIC__phase0, METRIC__phasemargin,
                             METRIC__gbw, METRIC__pole1fr, METRIC__pole2fr, METRIC__pole2_margin,
                             METRIC__pwrnode, METRIC__fbmnode]
METRICS__ac_metrics_small = [METRIC__DOCS, METRIC__gain, METRIC__phase0, METRIC__phasemargin,
                             METRIC__gbw, 
                             METRIC__pwrnode, METRIC__fbmnode]

#typical transient metrics...
METRIC__dynamic_range = Metric('dynamic_range', 0.1, INF, False, 0.1, 1.7) #=max(yhat) -min(yhat) of waveform
METRIC__slewrate = Metric('slewrate', 1e4, INF, False, 1e3, 1e10) #=min(slewrate_pos, slewrate_neg) 
METRIC__slewrate_log = Metric('slewrate_log', 4, INF, False, 4, 10) #=min(slewrate_pos, slewrate_neg) 

# -the following measures help guarantee the fitted waveform is ok
#METRIC__nmse              = Metric('nmse', -INF, 0.20, False, 0, 1)
#METRIC__correlation       = Metric('correlation', 0.70, INF, False, 0, 1)
#METRIC__ymin_before_ymax  = Metric('ymin_before_ymax', 1.0, INF, False, 0, 1)
#METRIC__ymin_after_ymax   = Metric('ymin_after_ymax', 1.0, INF, False, 0, 1) 

METRICS__tran_metrics_new = [METRIC__dynamic_range, METRIC__slewrate, METRIC__slewrate_log]
                             #METRIC__nmse, METRIC__correlation,
                             #METRIC__ymin_before_ymax, METRIC__ymin_after_ymax]

METRIC__srneg    = Metric('srneg',-INF,0,False,-3e7, 0)
METRIC__srpos    = Metric('srpos',0,INF,False, 0,3e7)
METRIC__outmax   = Metric('outmax',-10,10,False,-10,10)
METRIC__outmin   = Metric('outmin',-10,10,False,-10,10)       
METRIC__outswing = Metric('outswing',1.0,INF,False,0,2)
METRICS__tran_metrics_old = [METRIC__srneg, METRIC__srpos, METRIC__outmax, METRIC__outmin, METRIC__outswing]

METRIC__ITF = Metric('i(rload)/ibias',0.99,1.01,False,0.9,1.1)
METRIC__damping = Metric('damping',-60,-20,False,-40,-10)
METRIC__curnode = Metric('curnode',0.99e-4,1.01e-4,False,0.9e-4,1.1e-4)
METRIC__pwrnode = Metric('pwrnode',-INF,5.4e-4,False,3.6e-4,5.4e-4)
METRIC__dvout = Metric('dvout',-0.001,0.001,False,-0.01,0.01)
METRICS__cmEMC_ac = [METRIC__DOCS,METRIC__ITF,METRIC__damping,METRIC__curnode,METRIC__pwrnode]
METRICS__cmEMC_tran = [METRIC__dvout]

#area analysis & objective to minimize
def addArea(analyses, emb_part, devices_setup, log_scale = False, objective = True):
    """In-place add the analysis & objective of area."""
    if log_scale:
        if objective:
            minval = -INF
        else:
            minval = -14
        an = FunctionAnalysis(emb_part.areaLog10, [EnvPoint(True)], minval, -4.0, objective, -12.0, -4.0, 'area_log')
    else:
        if objective:
            minval = -INF
        else:
            minval = 1e-14
        an = FunctionAnalysis(emb_part.area, [EnvPoint(True)], minval, 1e-4, objective, 1e-12, 1e-4, 'area')
    emb_part.devices_setup = devices_setup #emb_part needs to see devices_setup
    analyses.append(an)

#relative costs
AC_COST = 2.0
TRAN_COST = 5.0

#==============================================================================================
#helper functions for changing metrics in a list: => objective; min_threshold, max_threshold
def setAsObjective(metrics_list, metric_name):
    """Finds the metric in metrics_list that matches 'metric_name', and sets it to an objective.
    Be careful to use this before creating the full problem setup, because ps stores the objectives
    as a separate list.
    """
    getMetricFromList(metrics_list, metric_name).setAsObjective()

def setMinThreshold(metrics_list, metric_name, min_threshold):
    getMetricFromList(metrics_list, metric_name).min_threshold = min_thresold
    
def setMaxThreshold(metrics_list, metric_name, max_threshold):
    getMetricFromList(metrics_list, metric_name).max_threshold = max_thresold
            
def getMetricFromList(metrics_list, metric_name):
    for metric in metrics_list:
        assert isinstance(metric, Metric), (str(metric), metric.__class__)
        if metric.name == metric_name:
            return metric
    raise AssertionError




#==============================================================================================
#main class...
class ProblemFactory:
    """
    @description
    
      ProblemFactory builds ProblemSetup objects for different problems.
      
    @attributes
      
    @notes
    """
    
    def __init__(self):
        pass

    def problemDescriptions(self):
        """Outputs a string describing problems"""
        #Note: when this class changes its problem descriptions, must also
        # update PROBLEM_DESCRIPTIONS.  The reason they are separate is
        # so that the descriptions do not have any of the dependencies that
        # Problems has.
        return PROBLEM_DESCRIPTIONS

    def build(self, problem_choice, extra_args=None):
        """
        @description

          Builds a ProblemSetup based on the input 'problem_choice'.
        
        @arguments

          problem_choice -- int -- to select problem.  See problemDescriptions().
        
        @return

          ps -- ProblemSetup object --
        
        @exceptions
    
        @notes
          
        """
        problem = None #fill this in
        if problem_choice == 1:
            problem = self.maximizePartCount_Problem('UMC180')
        elif 2<= problem_choice <= 9:
            problem = self.maxPartCount_minArea_Problem('UMC180', problem_choice)
        elif problem_choice == 10:
            problem = self.tenObjectives_FuncProblem('UMC180')
        elif problem_choice == 11:
            problem = self.tenObjectives_CircuitProblem('UMC180', 'OpLibrary')
            
        elif problem_choice == 12:
            problem = self.justDOCs_CircuitProblem('UMC180', False)
        elif problem_choice == 13:
            problem = self.justDOCs_CircuitProblem('UMC180', True)
            
        elif problem_choice == 15:
            problem = self.twoDimSphere_Problem('UMC180')
        
        elif problem_choice == 31:
            problem = self.WL_ssViAmp1_Problem('UMC180')
        elif problem_choice == 33:
            problem = self.OP_ssViAmp1_Problem('UMC180', 'OpLibrary')
        elif problem_choice == 34:
            problem = self.OP_ssViAmp1_Problem('UMC90', 'OpLibrary')
            
        elif problem_choice == 39:
            problem = self.OP_ssViAmp1_Problem('UMC180', 'OpLibrary2')
            
        
        elif problem_choice == 41:
            problem = self.WL_dsViAmp_Problem('UMC180', DS=True, DSS=False, DDS=False)
        elif problem_choice == 42:
            problem = self.OP_dsViAmp_Problem('UMC180', DS=True, DSS=False, DDS=False, libname='OpLibrary')
        elif problem_choice == 43:
            problem = self.OP_dsViAmp_LOG_Problem('UMC180', DS=True, DSS=False, DDS=False, libname='OpLibrary')
        elif problem_choice == 44:
            problem = self.OP_dsViAmp_Problem('UMC180', DS=True, DSS=False, DDS=False, libname='OpLibrary2')
            
        elif problem_choice == 51:
            problem = self.WL_dsViAmp_Problem('UMC180', DS=False, DSS=True, DDS=False)
        elif problem_choice == 52:
            problem = self.OP_dsViAmp_Problem('UMC180', DS=False, DSS=True, DDS=False, libname='OpLibrary')
        elif problem_choice == 53:
            problem = self.OP_dsViAmp_LOG_Problem('UMC180', DS=False, DSS=True, DDS=False, libname='OpLibrary')
        elif problem_choice == 54:
            problem = self.OP_dsViAmp_Problem('UMC180', DS=False, DSS=True, DDS=False, libname='OpLibrary2')
            
        elif problem_choice == 61:
            problem = self.WL_dsViAmp_Problem('UMC180', DS=True, DSS=True, DDS=False)
        elif problem_choice == 62:
            problem = self.OP_dsViAmp_Problem('UMC180', DS=True, DSS=True, DDS=False, libname='OpLibrary')
        elif problem_choice == 63:
            problem = self.OP_dsViAmp_LOG_Problem('UMC180', DS=True, DSS=True, DDS=False, libname='OpLibrary')
        elif problem_choice == 64:
            problem = self.OP_dsViAmp_LOG_Problem('UMC90', DS=True, DSS=True, DDS=False, libname='OpLibrary')
        elif problem_choice == 65:
            problem = self.OP_dsViAmp_LOG_Problem('UMC180', DS=True, DSS=True, DDS=False, libname='OpLibrary', test=1)
        elif problem_choice == 69:
            problem = self.OP_dsViAmp_Problem('UMC180', DS=True, DSS=True, DDS=False, libname='OpLibrary2')
            
        elif problem_choice == 71:
            problem = self.WL_dsViAmp_Problem('UMC180', DS=True, DSS=True, DDS=True)
     	elif problem_choice == 72:
            problem = self.OP_dsViAmp_Problem('UMC180', DS=True, DSS=True, DDS=True, libname='OpLibrary')     
            
	elif problem_choice == 81:
            (dc_sweep_start_voltage, dc_sweep_end_voltage, target_waveform) = extra_args
            problem = self.minimizeNmseOnTargetWaveform(
                'UMC180', dc_sweep_start_voltage, dc_sweep_end_voltage, target_waveform)	
	elif problem_choice == 82:
            problem = self.minimizeNmseOnTargetShape('UMC180', 'hockey')
        
	elif problem_choice == 83:
            problem = self.minimizeNmseOnTargetShape('UMC180', 'bump', False)
        
	elif problem_choice == 84:
            problem = self.minimizeNmseOnTargetShape('UMC180', 'bump', True)
            
        elif problem_choice==100:
            problem = self.OP_currentmirrorEMC_Problem('UMC180')
            
        elif problem_choice == 101:
            problem = self.OP_ssViAmp1mod_Problem('UMC180', 'OpLibrary')

        else:
            raise AssertionError('unknown problem choice: %d' % problem_choice)


        part = problem.embedded_part.part
        log.info("Schema for problem's part '%s': \n%s" %
                 (part.name, part.schemas()))
        schemas = part.schemas()
        schemas.merge()
        log.info("Merged schemas: %s" % schemas)
        log.info("Number of topologies for part '%s' = %d" %
                 (part.name, part.numSubpartPermutations()))

        #make sure all choice vars are propagated upwards as needed
        #We can safely turn this off if our library is properly defined.
        #problem.updateOptPointMetaFlexPartChoices(broadening_means_novel=False)
            
        #store 'problem_choice' in problem too
        problem.setProblemChoice(problem_choice)
        problem.validate()
        
        return problem

    def _buildOpLib(self, lib_ss, libname):
        if libname == 'OpLibrary':
            return OpLibrary(lib_ss)
        elif libname == 'OpLibrary2':
            return OpLibrary2(lib_ss)
        else:
            raise ValueError, 'bad library name specified: %s' % libname

    def maximizePartCount_Problem(self, process):
        """
        @description
        
          This is a simple non-simulator problem which tries to maximize
          the number of atomic parts used in a dsViAmp1
        
        @arguments

          process
        
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes
          
        """
        devices_setup = DevicesSetup(process)
        library = Point18SizesLibrary()
        part = library.dsViAmp1()
        
        connections = part.unityPortMap()
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        an = FunctionAnalysis(embedded_part.numAtomicParts, [EnvPoint(True)],
                              1, INF, True, 0, 100)
        ps = ProblemSetup(embedded_part, [an], library, devices_setup)
        
        return ps
    
    def maxPartCount_minArea_Problem(self, process, problem_choice):
        """
        @description
        
          This is a simple bi-objective non-simulator problem:
          -try to maximize # atomic parts
          -try to minimize transistor area (ignore R and C areas)
          -plus meet an arbitrary functionDOC of W>L
        
        @arguments

          problem_choice -- 2 for smallest .. 9 for largest circuit
          
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes
          
        """
        devices_setup = DevicesSetup(process)
        library = Point18SizesLibrary()

        #to mos3, add dummy function DOC: constraint of get w > l
#         metric = Metric('W_minus_L', 0.00001, INF, False, 0.0, 10.0e-6)
#         function = '(W-L)'
#         doc = FunctionDOC(metric, function)
        
#         mos3 = library.mos3()
#         mos3.addFunctionDOC(doc)

        #main problem setup...
        if problem_choice == 2:
            part = library.mos3()
            min_num_parts = 1
        elif problem_choice == 3:
            part = library.saturatedMos3()
            min_num_parts = 1
        elif problem_choice == 4:
            part = library.levelShifter()
            min_num_parts = 1
        elif problem_choice == 5:
            part = library.ddViInput()
            min_num_parts = 1
        elif problem_choice == 6:
            part = library.dsViAmp1()
            min_num_parts = 10
        elif problem_choice == 7:
            part = library.dsViAmp2_SingleEndedMiddle_VddGndPorts()
            min_num_parts = 10
        elif problem_choice == 8:
            ps = self.OP_dsViAmp_Problem('UMC180', False, True, False, 'OpLibrary')
            part = ps.embedded_part.part
            min_num_parts = 10
        elif problem_choice == 9:
            ps = self.OP_dsViAmp_Problem('UMC180', False, True, False, 'OpLibrary')
            part = self.ps.embedded_part.part
            min_num_parts = 6 #10
        else:
            raise ValueError("Unknown problem choice '%d'" % problem_choice)
        
        connections = part.unityPortMap()
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        analyses = []

        #maximize num atomic parts
        an0 = FunctionAnalysis(embedded_part.numAtomicParts, [EnvPoint(True)],
                               min_num_parts, INF, True, 0, 100)
        analyses.append(an0)

        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup)

        #meet functionDOCs
        #:NOTE: tlm - turn this off for novelty (hurts op pt driven search)
        #an2 = FunctionAnalysis(embedded_part.functionDOCsAreFeasible,
        #                       [EnvPoint(True)],
        #                       0.99, INF, False, 0, 1)
        #analyses.append(an2)

        #build ps
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        
        return ps
    
    def tenObjectives_FuncProblem(self, process):
        """
        @description
        
          This is a simple non-simulator problem which has 10 objectives:
          -minimize var1
          -maximize var1
          -minimize var2
          -maximize var2
          -...
          -minimize var5
          -maximize var5

          Where the vars are the first 5 non-choice vars in a dsViAmp1.
          
          Good for testing the massively multiobjective side of things.
        
        @arguments

          <<none>>          
        
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes
          
        """
        devices_setup = DevicesSetup(process)
        library = Point18SizesLibrary()
        part = library.dsViAmp1()
        
        connections = part.unityPortMap()
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        es = [EnvPoint(True)]

        #have mostly choice vars because they have few values each, therefore when
        # turned into objectives there will be a lot of overlap.  But have just enough
        # other metrics to give each ind a unique performance vector; do this with one
        # non-choice var
        selected_vars = sorted(part.point_meta.choiceVars())[:9] + \
                        sorted(part.point_meta.nonChoiceVars())[:1]
        analyses = []
        for var in selected_vars:
            mn, mx = part.point_meta[var].min_unscaled_value, part.point_meta[var].max_unscaled_value
            var_value_func = _simulationFunc(var)
            if mn < mx: 
                analyses.append(
                    FunctionAnalysis(var_value_func, es, mn-1.0,  INF, True,
                                     mn, mx, name_override='maximize__%s' % var))
        
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)

        return ps

    def tenObjectives_CircuitProblem(self, process, libname):
        """
        @description
        
          This is a simple simulator problem which has 10 objectives.
          It's based on WL_dsViAmp_Problem DS, and just converts all its
          metrics to be objectives.
          Good for testing the massively multiobjective side of things.
                  
        @arguments

          <<none>>          
        
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes
          
        """
        ps = self.OP_ssViAmp1_Problem(process, libname)
        for an in ps.analyses:
            for metric in an.metrics:
                metric.improve_past_feasible = True
        return ps

    def justDOCs_CircuitProblem(self, process, include_circuit_analysis):        
        #start with problem 42: OP_dsViAmp1
        ps = self.build(42)

        #get rid of all but the function & simulation DOCs analyses
        ps.stripAllButDOCs(include_circuit_analysis)

        #postconditions
        if include_circuit_analysis:
            assert len(ps.analyses) == 2
            assert len(ps.DOCMetricNames()) == 2
        else:
            assert len(ps.analyses) == 1
            assert len(ps.DOCMetricNames()) == 1
                    
        return ps

    def twoDimSphere_Problem(self, process):
        """
        @description
        
          This is a simple non-simulator problem which tries to minimize
          the cost of a 2-d sphere: f(x) = x^2.

          It is nice for testing convergence of algorithms.
        
        @arguments

          <<none>>          
        
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes
          
        """
        devices_setup = DevicesSetup(process)
        library = Point18SizesLibrary()

        #has two variables 'x1' and 'x2', each in range [-10.0, +10.0]
        part = library.sphere2dPart() 
        
        connections = part.unityPortMap()
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)
        function = _sphereFunc()
        feas_threshold = 0.01 #set to 1e6 to not invoke yt in rand gen; or to 0.01 to invoke yt in slave
        an = FunctionAnalysis(function, [EnvPoint(True)], -INF, feas_threshold, True, 0, 100,
                              name_override='sum_x_sq')
        ps = ProblemSetup(embedded_part, [an], library, devices_setup)
        
        return ps

    def WL_dsViAmp_Problem(self, process, DS, DSS, DDS):
        """
        @description
        
          Amplifier problem, for double-ended input / single-ended output.
          Many goals, including slew rate, gain, power, area, ...
        
        @arguments
        
          DS -- bool -- include 1-stage amp?
          DSS -- bool -- include 2-stage single-ended-middle amp?
          DDS -- bool -- include 2-stage differential-middle amp?
        
        @return

          ps -- ProblemSetup object
    
        @exceptions

          Need to include at least one of the stages.
    
        @notes
        """
        #validate inputs
        if not (DS or DSS or DDS):
            raise ValueError('must include at least one of the choices')
        
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = SizesLibraryStrategy(devices_setup)
        library = SizesLibrary(lib_ss)

        #choose main part
        if DS and not DSS and not DDS:
            part = library.dsViAmp1_VddGndPorts()
        elif not DS and DSS and not DDS:
            part = library.dsViAmp2_SingleEndedMiddle_VddGndPorts()
        elif not DS and not DSS and DDS:
            part = library.dsViAmp2_DifferentialMiddle_VddGndPorts()
        elif not DS and DSS and DDS:
            part = library.dsViAmp2_VddGndPorts()
        elif DS and DSS and DDS:
            part = library.dsViAmp_VddGndPorts()
        else:
            raise ValueError("this combo of DS/DSS/DSS not supported yet")

        #build embedded part
        # -dsViAmp1_VddGndPorts has ports: Vin1, Vin2, Iout, Vdd, gnd
        
        #the keys of 'connections' are the external ports of 'part'
        #the value corresponding to each key must be in the test_fixture_strings
        # that are below
        connections = {'Vin1':'ninp', 'Vin2':'ninn', 'Iout':'nout',
                       'Vdd':'ndd', 'gnd':'gnd'}

        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        #we'll be building this up
        analyses = []

        #-------------------------------------------------------
        #shared info between analyses
        # (though any of this can be analysis-specific if we'd wanted
        #  to set them there instead)
        max_simulation_time = 5 #in seconds
        
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/miller2/'
        
        simulator_options_string = """
.include %ssimulator_options.inc
""" % cir_file_path
        
        #-------------------------------------------------------
        #build dc analysis
        if False:
            pass

        #-------------------------------------------------------
        #build ac analysis
        if True:
            d = {'pCload':5e-12,
                 'pVdd':devices_setup.vdd(),
                 'pVdcin':0.9,
                 'pVout':0.9,
                 'pRfb':1.000e+09,
                 'pCfb':1.000e-03}
            ac_env_points = [EnvPoint(True, d)]
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin
Vinac		ninp		ninpdc	AC=1 SIN(0 1 10k)

* feedback loop for dc biasing of output stage

Efb1	nlpfin	gnd	nout	gnd	1
Rfb	nlpfin	nlpfout	pRfb
Cfb	nlpfout	gnd	pCfb

.param pFb=10
Efb2	ninn	gnd	volts='MAX(0,MIN(1.8, (V(nlpfout)-pVout) * pFb + pVdcin ))'

* this measures the amount of feedback biasing there is
EFBM fbmnode gnd volts='ABS((V(nlpfout)-pVout) * pFb)'

* simulation statements

.op
.ac	dec	50	1.0e0	10.0e9

* pole-zero analysis
.pz v(nout) Vinac

* Frequency-domain measurements
.measure ac ampl       max vdb(nout) at=0
.measure ac inampl max vdb(ninp,ninn) at=0
.measure ac gain PARAM='ampl-inampl'
.measure ac phase FIND vp(nout) WHEN vdb(nout)=0 CROSS=1
.measure ac phasemargin PARAM='180+phase'
.measure ac GBW WHEN vdb(nout)=0 CROSS=1
.measure ac phase0 FIND vp(nout) at=1e1

.measure ac pole1 WHEN vp(nout)=135 CROSS=1
.measure ac pole2 WHEN vp(nout)=45 CROSS=1

* power measurement
EPWR1 pwrnode gnd volts='-pVdd*I(Vdd)'


"""
            #build list of metrics
            ac_metrics = METRICS__ac_metrics_big[:]
            setAsObjective(ac_metrics, 'gbw')

            sim = Simulator(
                {
#                 'ma0':['gain','phasemargin','phase0','gbw','pole1','pole2'],
                'ma0':['gain','phasemargin','phase0','gbw'],
                'ic0':['pwrnode','fbmnode'],
                'lis':[DOCS_METRIC_NAME,'pole1fr','pole2fr','pole2_margin']
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                TYPICAL_DOC_MEASURES)
            ac_an = CircuitAnalysis(ac_env_points, ac_metrics, sim, AC_COST)
            analyses.append(ac_an)
            
        #-------------------------------------------------------
        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup)

        #note that we will later add 'novelty' as an OBJECTIVE too
        
        #-------------------------------------------------------
        #add function DOCs analysis
        #:NOTE: tlm - do NOT need to turn this off for novelty because
        # it still helps WL based search (unlike op pt based search)
        funcDOCs_an = FunctionAnalysis(
            embedded_part.functionDOCsAreFeasible, [EnvPoint(True)], 0.99, INF, False, 0, 1.0)
        analyses.append(funcDOCs_an)
        
        #-------------------------------------------------------
        #build transient analysis
        if False and not DISABLE_TRANSIENT_GLOBALLY:
            tran_env_points = [EnvPoint(True)]
            test_fixture_string = """
*input waveform case 1
VIN0 n_vin0 n_vss PWL(0.0 0.65 1.0e-3 0.75)

VDD  n_vdd n_vss DC 1.8V
VGND n_vss 0 DC 0.0V

*for input waveform case 1 and case 2
*.tran tstep    tstop   <tstart> <tmaxstep>

*proper version: (101 points)
.tran  0.01e-3  1.0e-3  0.0      0.01e-3

*HACK to-be-fast version (21 points)
*.tran  0.05e-3  1.0e-3  0.0      0.05e-3

*what to print
.print tran V(n_vin0)
.print tran V(n_vout0)
"""
            #FIXME: add more metrics?
            tran_metrics = [METRIC__slewrate]
            
            output_file_num_vars = {'tr0':None} #FIXME
            output_file_start_line = {'tr0':None} #FIXME
            metric_calculators = {'slewrate':None} #FIXME
            sim = Simulator({'tr0':'slewrate'},
                            cir_file_path,
                            max_simulation_time,
                            simulator_options_string,
                            test_fixture_string,
                            TYPICAL_DOC_MEASURES,
                            output_file_num_vars,
                            output_file_start_line,
                            WAVEFORM_NUMBER_WIDTH,
                            metric_calculators)            
            tran_an = CircuitAnalysis(tran_env_points, tran_metrics, sim, 'tran', TRAN_COST)
            analyses.append(tran_an)

        #-------------------------------------------------------
        #finally, build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps

    def OP_dsViAmp_Problem(self, process, DS, DSS, DDS, libname):
        """
        @description
        
          Amplifier problem, for double-ended input / single-ended output.
          Many goals, including slew rate, gain, power, area, ...
        
          Operating point driven
          
        @arguments
        
          DS -- bool -- include 1-stage amp?
          DSS -- bool -- include 2-stage single-ended-middle amp?
          DDS -- bool -- include 2-stage differential-middle amp?
        
        @return

          ps -- ProblemSetup object
    
        @exceptions

          Need to include at least one of the stages.
    
        @notes
        """
        #validate inputs
        if not (DS or DSS or DDS):
            raise ValueError('must include at least one of the choices')
        
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = OpLibraryStrategy(devices_setup)
        library = self._buildOpLib(lib_ss, libname)
        
        #choose main part
        if DS and not DSS and not DDS:
            part = library.dsViAmp1_VddGndPorts()
        elif not DS and DSS and not DDS:
            part = library.dsViAmp2_SingleEndedMiddle_VddGndPorts()
        elif not DS and not DSS and DDS:
            part = library.dsViAmp2_DifferentialMiddle_VddGndPorts()
        elif not DS and DSS and DDS:
            part = library.dsViAmp2_VddGndPorts()
#         elif DS and DSS and DDS:
        elif DS and DSS: # FIXME: DDS is not ready yet
            part = library.dsViAmp_VddGndPorts()
        else:
            raise ValueError("this combo of DS/DSS/DSS not supported yet")

        #build embedded part
        # -dsViAmp1_VddGndPorts has ports: Vin1, Vin2, Iout, Vdd, gnd
        
        #the keys of 'connections' are the external ports of 'part'
        #the value corresponding to each key must be in the test_fixture_strings
        # that are below
        connections = {'Vin1':'ninp', 'Vin2':'ninn', 'Iout':'nout', 'Vdd':'ndd', 'gnd':'gnd'}

        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        #we'll be building this up
        analyses = []

        #-------------------------------------------------------
        #shared info between analyses
        # (though any of this can be analysis-specific if we'd wanted
        #  to set them there instead)        
        max_simulation_time = 50 #in seconds
        
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/miller2/'
        
        simulator_options_string = """
.include %ssimulator_options.inc
""" % cir_file_path
        
        #-------------------------------------------------------
        #build dc analysis
        if False:
            pass

        #-------------------------------------------------------
        #build ac analysis
        if True:
            d = {'pCload':1e-12,
                 'pVdd':devices_setup.vdd(),
                 'pVdcin':0.9,
                 'pVout':0.9,
                 'pRfb':1.000e+09,
                 'pCfb':1.000e-03,
                 'pTemp':25,
                 }
            ac_env_points = [EnvPoint(True, d)]
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin
Vinac		ninp		ninpdc	AC=1 SIN(0 1 10k)

* feedback loop for dc biasing of output stage

Efb1	nlpfin	gnd	nout	gnd	1
Rfb	nlpfin	nlpfout	pRfb
Cfb	nlpfout	gnd	pCfb

.param pFb=10
Efb2	ninn	gnd	volts='MAX(0,MIN(1.8, (V(nlpfout)-pVout) * pFb + pVdcin ))'

* this measures the amount of feedback biasing there is
EFBM fbmnode gnd volts='ABS((V(nlpfout)-pVout) * pFb)'

* simulation statements

.op
.ac	dec	50	1.0e0	10.0e9

* pole-zero analysis
.pz v(nout) Vinac

* Frequency-domain measurements
.measure ac ampl       max vdb(nout) at=0
.measure ac inampl max vdb(ninp,ninn) at=0
.measure ac gain PARAM='ampl-inampl'
.measure ac phase FIND vp(nout) WHEN vdb(nout)=0 CROSS=1
.measure ac phasemargin PARAM='180+phase'
.measure ac GBW WHEN vdb(nout)=0 CROSS=1
.measure ac phase0 FIND vp(nout) at=1e1

.measure ac pole1 WHEN vp(nout)=135 CROSS=1
.measure ac pole2 WHEN vp(nout)=45 CROSS=1

* power measurement
EPWR1 pwrnode gnd volts='-pVdd*I(Vdd)'


"""
            #build list of metrics
            ac_metrics = METRICS__ac_metrics_big[:]
            setAsObjective(ac_metrics, 'gain')
            setAsObjective(ac_metrics, 'gbw')
            setAsObjective(ac_metrics, 'pwrnode')

            sim = Simulator(
                {
#                 'ma0':['gain','phasemargin','phase0','gbw','pole1','pole2'],
                'ma0':['gain','phasemargin','phase0','gbw'],
                'ic0':['pwrnode','fbmnode'],
                'lis':[DOCS_METRIC_NAME,'pole1fr','pole2fr','pole2_margin']
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                TYPICAL_DOC_MEASURES)
            ac_an = CircuitAnalysis(ac_env_points, ac_metrics, sim, AC_COST)
            analyses.append(ac_an)
            
        #-------------------------------------------------------
        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup)

        #note that we will later add 'novelty' as an OBJECTIVE too
        
        #-------------------------------------------------------
        #add function DOCs analysis
        #:NOTE: tlm - turn this off for novelty (hurts op pt driven search)
        funcDOCs_an = FunctionAnalysis(
            embedded_part.functionDOCsAreFeasible, [EnvPoint(True)], 0.99, INF, False, 0, 1)
        analyses.append(funcDOCs_an)
        
        #-------------------------------------------------------
        #build transient analysis
        if True and not DISABLE_TRANSIENT_GLOBALLY:
            tran_env_points = ac_env_points
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin
VinTran         ninp            ninpdc  PULSE(-0.2 0.2 0.2e-6 100p 100p 0.5e-6 1e-6)

Enin2	        ninn	        gnd	volts = 'pVdcin-V(ninp, ninpdc)'

* simulation statements

.tran 100p 1e-6

* output the voltage waveforms 'vinp' (input) and 'nout' (output)
.print tran V(nout)


"""
            transient_extension = 'tr0'
            metrics_calculator = TransientWaveformCalculator(
                0, 1, chop_initial=True)

            tran_metrics = METRICS__tran_metrics_new[:]
            setAsObjective(tran_metrics, 'dynamic_range')
            setAsObjective(tran_metrics, 'slewrate')

            #for nout in .print commands ('time' variable comes for free)
            output_file_num_vars = {transient_extension:1} 

            #set this by: examine .tr0 file, and supply which line the numbers
            # start at.  Start counting at 0.
            # One output var => 4.
            # Two output vars => 5.
            # ...
            output_file_start_line = {transient_extension:4}
         
            sim = Simulator(
                {
                'tr0':metrics_calculator.metricNames(),
                #'lis':[DOCS_METRIC_NAME],
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                [],
                output_file_num_vars,
                output_file_start_line,
                WAVEFORM_NUMBER_WIDTH,
                metrics_calculator,
                )
            tran_an = CircuitAnalysis(tran_env_points, tran_metrics, sim, TRAN_COST)
                                                  
            analyses.append(tran_an)
## """
##             #FIXME: add more metrics?
##             tran_metrics = METRICS__tran_metrics_old[:]
                                      
##             sim = Simulator(
##                 {
##                 'ma0':['srneg','srpos','outmax','outmin'],
##                 },
##                 cir_file_path,
##                 max_simulation_time,
##                 simulator_options_string,
##                 test_fixture_string,
##                 TYPICAL_DOC_MEASURES)
##             tran_an = CircuitAnalysis(tran_env_points, tran_metrics, sim, TRAN_COST)
                                                  
##             analyses.append(tran_an)

        #-------------------------------------------------------
        #finally, build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps

    def OP_dsViAmp_LOG_Problem(self, process, DS, DSS, DDS, libname, test = None):
        """
        @description
        
          Amplifier problem, for double-ended input / single-ended output.
          Many goals, including slew rate, gain, power, area, ...
        
          Operating point driven
          log10 scaled GBW
          
        @arguments
        
          DS -- bool -- include 1-stage amp?
          DSS -- bool -- include 2-stage single-ended-middle amp?
          DDS -- bool -- include 2-stage differential-middle amp?
        
        @return

          ps -- ProblemSetup object
    
        @exceptions

          Need to include at least one of the stages.
    
        @notes
        """
        #validate inputs
        if not (DS or DSS or DDS):
            raise ValueError('must include at least one of the choices')
        
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = OpLibraryStrategy(devices_setup)
        library = self._buildOpLib(lib_ss, libname)

        #choose main part
        if DS and not DSS and not DDS:
            part = library.dsViAmp1_VddGndPorts()
        elif not DS and DSS and not DDS:
            part = library.dsViAmp2_SingleEndedMiddle_VddGndPorts()
        elif not DS and not DSS and DDS:
            part = library.dsViAmp2_DifferentialMiddle_VddGndPorts()
        elif not DS and DSS and DDS:
            part = library.dsViAmp2_VddGndPorts()
#         elif DS and DSS and DDS:
        elif DS and DSS: # FIXME: DDS is not ready yet
            if not test:
                part = library.dsViAmp_VddGndPorts()
            else:
                part = library.dsViAmp_VddGndPorts_TST()
        else:
            raise ValueError("this combo of DS/DSS/DSS not supported yet")

        #build embedded part
        # -dsViAmp1_VddGndPorts has ports: Vin1, Vin2, Iout, Vdd, gnd
        
        #the keys of 'connections' are the external ports of 'part'
        #the value corresponding to each key must be in the test_fixture_strings
        # that are below
        connections = {'Vin1':'ninp', 'Vin2':'ninn', 'Iout':'nout', 'Vdd':'ndd', 'gnd':'gnd'}

        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        #we'll be building this up
        analyses = []

        #-------------------------------------------------------
        #shared info between analyses
        # (though any of this can be analysis-specific if we'd wanted
        #  to set them there instead)        
        max_simulation_time = 50 #in seconds
        
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/miller2/'
        
        simulator_options_string = """
.include %ssimulator_options.inc
""" % cir_file_path
        
        #-------------------------------------------------------
        #build dc analysis
        if False:
            pass

        #-------------------------------------------------------
        #build ac analysis
        if True:
            d = {'pCload':1e-12,
                 'pVdd':devices_setup.vdd(),
                 'pVdcin':0.9,
                 'pVout':0.9,
                 'pRfb':1.000e+09,
                 'pCfb':1.000e-03,
                 'pTemp':25,
                 }
            ac_env_points = [EnvPoint(True, d)]
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin
Vinac		ninp		ninpdc	AC=1 SIN(0 1 10k)

* feedback loop for dc biasing of output stage

Efb1	nlpfin	gnd	nout	gnd	1
Rfb	nlpfin	nlpfout	pRfb
Cfb	nlpfout	gnd	pCfb

.param pFb=10
Efb2	ninn	gnd	volts='MAX(0,MIN(1.8, (V(nlpfout)-pVout) * pFb + pVdcin ))'

* this measures the amount of feedback biasing there is
EFBM fbmnode gnd volts='ABS((V(nlpfout)-pVout) * pFb)'

* simulation statements

.op
.ac	dec	50	1.0e0	10.0e9

* pole-zero analysis
.pz v(nout) Vinac

* Frequency-domain measurements
.measure ac ampl       max vdb(nout) at=0
.measure ac inampl max vdb(ninp,ninn) at=0
.measure ac gain PARAM='ampl-inampl'
.measure ac phase FIND vp(nout) WHEN vdb(nout)=0 CROSS=1
.measure ac phasemargin PARAM='180+phase'
.measure ac GBW WHEN vdb(nout)=0 CROSS=1
.measure ac gbw_log param='log(GBW)/log(10)'
.measure ac phase0 FIND vp(nout) at=1e1

.measure ac pole1 WHEN vp(nout)=135 CROSS=1
.measure ac pole2 WHEN vp(nout)=45 CROSS=1

* power measurement
EPWR1 pwrnode gnd volts='-pVdd*I(Vdd)'
EPWR2 pwr_log gnd volts='log(-pVdd*I(Vdd))/log(10)'


"""
            #build list of metrics
            ac_metrics = METRICS__ac_metrics_big[:]
            ac_metrics.append(Metric('gbw_log', 6, INF, False, 0, 10))
            ac_metrics.append(Metric('pwr_log', -INF, -1, False, -6, -1))
            
            setAsObjective(ac_metrics, 'gain')
            setAsObjective(ac_metrics, 'pwr_log')
            setAsObjective(ac_metrics, 'gbw_log')

            sim = Simulator(
                {
#                 'ma0':['gain','phasemargin','phase0','gbw','pole1','pole2'],
                'ma0':['gain','phasemargin','phase0','gbw', 'gbw_log'],
                'ic0':['pwrnode','pwr_log','fbmnode'],
                'lis':[DOCS_METRIC_NAME,'pole1fr','pole2fr','pole2_margin']
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                TYPICAL_DOC_MEASURES)
            ac_an = CircuitAnalysis(ac_env_points, ac_metrics, sim, AC_COST)
            analyses.append(ac_an)
            
        #-------------------------------------------------------
        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup, log_scale=True, objective=True)
        addArea(analyses, embedded_part, devices_setup, log_scale=False, objective=False)

        #note that we will later add 'novelty' as an OBJECTIVE too
        
        #-------------------------------------------------------
        #add function DOCs analysis
        #:NOTE: tlm - turn this off for novelty (hurts op pt driven search)
        funcDOCs_an = FunctionAnalysis(
            embedded_part.functionDOCsAreFeasible, [EnvPoint(True)], 0.99, INF, False, 0, 1)
        analyses.append(funcDOCs_an)
        
        #-------------------------------------------------------
        #build transient analysis
        if True and not DISABLE_TRANSIENT_GLOBALLY:
            tran_env_points = ac_env_points
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin

"""
            tmargin = 1e-9 # one nanosec
            v_hi = 0.3
            v_low = -0.3
            td = 0.2e-6
            tfall = 100.0e-12
            trise = tfall
            min_slewrate = 1.0e6
            # the pulse has to be wide enough for a min-slew amp to settle
            pwidth = (devices_setup.vdd() / min_slewrate) + tmargin
            period = pwidth * 2.0
            test_fixture_string += "VinTran ninp ninpdc  PULSE(%g %g %g %g %g %g %g)\n" % \
                                        (v_low, v_hi, td, trise, tfall, pwidth, period)

            test_fixture_string += """

Enin2	        ninn	        gnd	volts = 'pVdcin-V(ninp, ninpdc)'

* to load the output slightly, such that leakage current
* doesn't mask the results too much
Vload           nout_ref gnd DC=pVout
Rload nout nout_ref R=1e6

* simulation statements
"""

            test_fixture_string += ".tran 100p %g" % (tmargin + td + period + tmargin)
            test_fixture_string += """
* output the voltage waveforms 'vinp' (input) and 'nout' (output)
.print tran V(nout)

"""
            transient_extension = 'tr0'
            metrics_calculator = TransientWaveformCalculator(
                0, 1, chop_initial=True)

            tran_metrics = METRICS__tran_metrics_new[:]
            # NOTE: might be better if we specify a center voltage to measure the SR/DR around
            # to the TransientWaveformCalculator
            offset_max_rel_error = 0.2 # allow 20% error in the centering of the output level
            offset_max = d['pVout'] * (1.0 + offset_max_rel_error)
            offset_min = d['pVout'] * (1.0 - offset_max_rel_error)
            tran_metrics.append(Metric('offset', offset_min, offset_max, False, 0.0, devices_setup.vdd()))

            pwidth_max_rel_error = 0.5 # 50% error on the pulse width
            pwidth_max = pwidth * (1.0 + pwidth_max_rel_error)
            pwidth_min = pwidth * (1.0 - pwidth_max_rel_error)
            tran_metrics.append(Metric('pulse_width', pwidth_min, pwidth_max, False, 0.0, period))

            setAsObjective(tran_metrics, 'dynamic_range')
            setAsObjective(tran_metrics, 'slewrate_log')

            #for nout in .print commands ('time' variable comes for free)
            output_file_num_vars = {transient_extension:1} 

            #set this by: examine .tr0 file, and supply which line the numbers
            # start at.  Start counting at 0.
            # One output var => 4.
            # Two output vars => 5.
            # ...
            output_file_start_line = {transient_extension:4}
         
            sim = Simulator(
                {
                'tr0':metrics_calculator.metricNames(),
                #'lis':[DOCS_METRIC_NAME],
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                [],
                output_file_num_vars,
                output_file_start_line,
                WAVEFORM_NUMBER_WIDTH,
                metrics_calculator,
                )
            tran_an = CircuitAnalysis(tran_env_points, tran_metrics, sim, TRAN_COST)
                                                  
            analyses.append(tran_an)
## """
##             #FIXME: add more metrics?
##             tran_metrics = METRICS__tran_metrics_old[:]
                                      
##             sim = Simulator(
##                 {
##                 'ma0':['srneg','srpos','outmax','outmin'],
##                 },
##                 cir_file_path,
##                 max_simulation_time,
##                 simulator_options_string,
##                 test_fixture_string,
##                 TYPICAL_DOC_MEASURES)
##             tran_an = CircuitAnalysis(tran_env_points, tran_metrics, sim, TRAN_COST)
                                                  
##             analyses.append(tran_an)

        #-------------------------------------------------------
        #finally, build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps
                        
    def WL_ssViAmp1_Problem(self, process):
        """
        @description
        
          Amplifier problem, for single ended input / single-ended output.
          Many goals, including slew rate, gain, power, area, ...
        
        @arguments

          <<none>>          
        
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes
        """
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = SizesLibraryStrategy(devices_setup)
        library = SizesLibrary(lib_ss)
        
        #build embedded part
        # -ssViAmp1_VddGndPorts has ports: Vin, Iout, Vdd, gnd
        part = library.ssViAmp1_VddGndPorts()

        #the keys of 'connections' are the external ports of 'part'
        #the value corresponding to each key must be in the test_fixture_strings
        # that are below
        connections = {'Vin':'ninp', 'Iout':'nout', 'Vdd':'ndd', 'gnd':'gnd'}

        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        #we'll be building this up
        analyses = []

        #-------------------------------------------------------
        #shared info between analyses
        # (though any of this can be analysis-specific if we'd wanted
        #  to set them there instead)
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/ssvi1/'
        max_simulation_time = 5 #in seconds
        simulator_options_string = """
.include %ssimulator_options.inc
""" % cir_file_path

        #-------------------------------------------------------
        #build ac analysis
        if True:
            d = {
                 'pCload':5e-12,
                 'pVdd':devices_setup.vdd(),
                 'pVdcin':0.9,
                 'pVout':0.9,
                 'pRfb':1.000e+09,
                 'pCfb':1.000e-03}
            ac_env_points = [EnvPoint(True, d)]
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin
Vinac		ninpdc		ninp1	AC=1
Vintran 	ninp1 		ninpx 	DC=0 PWL(
+ 0     0
+ 0.1n   -0.2
+ 10.0n  -0.2 
+ 10.1n  0.2 
+ 30.0n  0.2
+ 30.1n  -.2 )

* feedback loop for dc biasing 
Vout_ref	nvout_ref	gnd	pVout
Efb1	nfbin	gnd	nout	nvout_ref	1.0e2
Rfb	nfbin	nfbout	pRfb
Cfb	nfbout	gnd	pCfb
Efb2	n1	gnd	nfbout	gnd	1.0
Efb3	ninp_unlim	n1	ninpx	gnd	1.0
Efb4	ninp	gnd	volts='MAX(-1.8,MIN(1.8,V(ninp_unlim)))'

* this measures the amount of feedback biasing there is
EFBM fbmnode gnd volts='ABS(V(ninp)-V(ninpdc))'

* simulation statements

.op
*.DC TEMP 25 25 10
.ac	dec	50	1.0e0	10.0e9
* pole-zero analysis
.pz v(nout) Vinac

.probe ac V(nout)
.probe ac V(ninp)
.probe ac V(*)

*.tran 100p 50n
.probe tran V(nout)
.probe tran V(ninp)
.probe tran V(*)

* Frequency-domain measurements
.measure ac ampl       max vdb(nout) at=0
.measure ac inampl max vdb(ninp) at=0
.measure ac gain PARAM='ampl-inampl'
.measure ac phase FIND vp(nout) WHEN vdb(nout)=0 CROSS=1
.measure ac phasemargin PARAM='phase+180'
.measure ac GBW WHEN vdb(nout)=0 CROSS=1
.measure ac phase0 FIND vp(nout) at=1e1



* power measurement
EPWR1 pwrnode gnd volts='-pVdd*I(Vdd)'

"""
            ac_metrics = METRICS__ac_metrics_small[:]
            setAsObjective(ac_metrics, 'gain')
            setAsObjective(ac_metrics, 'pwrnode')
        
            sim = Simulator({'ma0':['gain','phase0','phasemargin','gbw'],
                             'ic0':['pwrnode','fbmnode'],
                             'lis':[DOCS_METRIC_NAME]},
                            cir_file_path,
                            max_simulation_time,
                            simulator_options_string,
                            test_fixture_string,
                            TYPICAL_DOC_MEASURES)
                            
            ac_an = CircuitAnalysis(ac_env_points, ac_metrics, sim, AC_COST)
            analyses.append(ac_an)
            
        #-------------------------------------------------------
        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup)
        
        #-------------------------------------------------------
        #add function DOCs analysis
        funcDOCs_an = FunctionAnalysis(
            embedded_part.functionDOCsAreFeasible, [EnvPoint(True)], 0.99, INF, False, 0, 1)
        analyses.append(funcDOCs_an)
        
        #-------------------------------------------------------
        #finally, build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps
        
    def OP_ssViAmp1_Problem(self, process, libname):
        """
        @description
        
          Amplifier problem, for single ended input / single-ended output.
          Many goals, including slew rate, gain, power, area, ...
        
          Operating point driven.

          == Problem 32.
          
        @arguments

          libname -- 'OpLibrary' or 'OpLibrary2'
          process -- 'UMC180' or 'UMC90'
        
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes
        """
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = OpLibraryStrategy(devices_setup)
        library = self._buildOpLib(lib_ss, libname)
          
        #build embedded part
        # -ssViAmp1_VddGndPorts has ports: Vin, Iout, Vdd, gnd
        part = library.ssViAmp1_VddGndPorts_Fixed()

        #the keys of 'connections' are the external ports of 'part'
        #the value corresponding to each key must be in the test_fixture_strings
        # that are below
        connections = {'Vin':'ninp', 'Iout':'nout', 'Vdd':'ndd', 'gnd':'gnd'}

        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        #we'll be building this up
        analyses = []

        #-------------------------------------------------------
        #shared info between analyses
        # (though any of this can be analysis-specific if we'd wanted
        #  to set them there instead)
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/ssvi1/'
        max_simulation_time = 5 #in seconds
        simulator_options_string = """
.include %ssimulator_options.inc
""" % cir_file_path

        #-------------------------------------------------------
        #build ac analysis
        if True:
            d = {
                 'pCload':5e-12,
                 'pVdd':devices_setup.vdd(),
                 'pVdcin':0.9,
                 'pVout':0.9,
                 'pRfb':1.000e+09,
                 'pCfb':1.000e-03,
                 'pTemp':25,
                 }
            d2 = {
                 'pCload':5e-12,
                 'pVdd':devices_setup.vdd(),
                 'pVdcin':0.9,
                 'pVout':0.9,
                 'pRfb':1.000e+09,
                 'pCfb':1.000e-03,
                 'pTemp':50,
                 }
            ac_env_points = [EnvPoint(True, d)]#,EnvPoint(True, d2)]
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin
Vinac		ninpdc		ninp1	AC=1
Vintran 	ninp1 		ninpx 	DC=0 PWL(
+ 0     0
+ 0.1n   -0.2
+ 10.0n  -0.2 
+ 10.1n  0.2 
+ 30.0n  0.2
+ 30.1n  -.2 )

* feedback loop for dc biasing 
Vout_ref	nvout_ref	gnd	pVout
Efb1	nfbin	gnd	nout	nvout_ref	1.0e2
Rfb	nfbin	nfbout	pRfb
Cfb	nfbout	gnd	pCfb
Efb2	n1	gnd	nfbout	gnd	1.0
Efb3	ninp_unlim	n1	ninpx	gnd	1.0
Efb4	ninp	gnd	volts='MAX(0,MIN(pVdd,V(ninp_unlim)))'

* this measures the amount of feedback biasing there is
EFBM fbmnode gnd volts='ABS(V(ninp)-V(ninpdc))'

* simulation statements

.op

* temperature analysis
.temp pTemp

*.DC TEMP 25 25 10
.ac	dec	50	1.0e0	10.0e9
* pole-zero analysis
.pz v(nout) Vinac

.probe ac V(nout)
.probe ac V(ninp)
.probe ac V(*)

*.tran 100p 50n
.probe tran V(nout)
.probe tran V(ninp)
.probe tran V(*)

* Frequency-domain measurements
.measure ac ampl       max vdb(nout) at=0
.measure ac inampl max vdb(ninp) at=0
.measure ac gain PARAM='ampl-inampl'
.measure ac phase FIND vp(nout) WHEN vdb(nout)=0 CROSS=1
.measure ac phasemargin PARAM='phase+180'
.measure ac GBW WHEN vdb(nout)=0 CROSS=1
.measure ac phase0 FIND vp(nout) at=1e1

* power measurement
EPWR1 pwrnode gnd volts='-pVdd*I(Vdd)'

"""
            ac_metrics = METRICS__ac_metrics_small[:]
            setAsObjective(ac_metrics, 'gain')
            setAsObjective(ac_metrics, 'pwrnode')
   
            #if we use a .lis output like 'region' or 'vgs' even once in
            # order to constrain DOCs, list it here
            # (if you forget a measure, it _will_ complain)

            #possible doc measures (lis measures):
            # 'region' -- output in .lis will be one of: 'Linear', 'Saturat', 'Cutoff'
            # 'vgs'    -- output in .lis will be a float.  Typically positive for NMOS, negative for PMOS.
            # 'vth'     --  ""
            # 'vdsat' --  ""

            sim = Simulator({'ma0':['gain','phase0','phasemargin','gbw'],
                             'ic0':['pwrnode','fbmnode'],
                             'lis':[DOCS_METRIC_NAME]},
                            cir_file_path,
                            max_simulation_time,
                            simulator_options_string,
                            test_fixture_string,
                            TYPICAL_DOC_MEASURES)
                            
            ac_an = CircuitAnalysis(ac_env_points, ac_metrics, sim, AC_COST)
            analyses.append(ac_an)
        #-------------------------------------------------------
        #build transient analysis
        if True and not DISABLE_TRANSIENT_GLOBALLY:
            tran_env_points = ac_env_points
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin
Vinac		ninpdc		ninp1	AC=1
Vintran 	ninp1 		ninpx 	DC=0 PWL(
+ 0     0
+ 0.01n   -0.2
+ 100.0n  -0.2 
+ 100.01n  0.2 
+ 200.0n  0.2
+ 200.01n  -.2 )

* feedback loop for dc biasing 
Vout_ref	nvout_ref	gnd	pVout
Efb1	nfbin	gnd	nout	nvout_ref	1.0e2
Rfb	nfbin	nfbout	pRfb
Cfb	nfbout	gnd	pCfb
Efb2	n1	gnd	nfbout	gnd	1.0
Efb3	ninp_unlim	n1	ninpx	gnd	1.0
Efb4	ninp	gnd	volts='MAX(0,MIN(pVdd,V(ninp_unlim)))'

* simulation statements

.op

* temperature analysis
.temp pTemp

.tran 100p 300n

* output the voltage waveforms 'vinp' (input) and 'nout' (output)
.print tran V(nout)


"""
            transient_extension = 'tr0'
            metrics_calculator = TransientWaveformCalculator(0, 1, chop_initial=True)

            tran_metrics = METRICS__tran_metrics_new[:]
            setAsObjective(tran_metrics, 'dynamic_range')
            setAsObjective(tran_metrics, 'slewrate')
            
            #for nout in .print commands ('time' variable comes for free)
            output_file_num_vars = {transient_extension:1} 

            #set this by: examine .tr0 file, and supply which line the numbers
            # start at.  Start counting at 0.
            # One output var => 4.
            # Two output vars => 5.
            # ...
            output_file_start_line = {transient_extension:4}
         
            sim = Simulator(
                {
                'tr0':metrics_calculator.metricNames(),
                #'lis':[DOCS_METRIC_NAME],
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                [],
                output_file_num_vars,
                output_file_start_line,
                WAVEFORM_NUMBER_WIDTH,
                metrics_calculator,
                )
            tran_an = CircuitAnalysis(tran_env_points, tran_metrics, sim, AC_COST)
                                                  
            analyses.append(tran_an)

        #-------------------------------------------------------
        #build old-style transient analysis
        if False and not DISABLE_TRANSIENT_GLOBALLY:
            tran2_env_points = ac_env_points
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin
Vinac		ninpdc		ninp1	AC=1
Vintran 	ninp1 		ninpx 	DC=0 PWL(
+ 0     0
+ 0.01n   -0.2
+ 100.0n  -0.2 
+ 100.01n  0.2 
+ 200.0n  0.2
+ 200.01n  -.2 )

* feedback loop for dc biasing 
Vout_ref	nvout_ref	gnd	pVout
Efb1	nfbin	gnd	nout	nvout_ref	1.0e2
Rfb	nfbin	nfbout	pRfb
Cfb	nfbout	gnd	pCfb
Efb2	n1	gnd	nfbout	gnd	1.0
Efb3	ninp_unlim	n1	ninpx	gnd	1.0
Efb4	ninp	gnd	volts='MAX(0,MIN(pVdd,V(ninp_unlim)))'

* simulation statements

.op

* temperature analysis
.temp pTemp

.tran 100p 300n

.measure tran srneg deriv v(nout) when v(nout)='pVout*0.95' CROSS=3
.measure tran srpos deriv v(nout) when v(nout)='pVout*1.05' CROSS=1

.measure tran outmax find V(nout) at=199.9n 
.measure tran outmin find V(nout) at=99.9n

.measure tran 'outswing' param='outmax-outmin'

*.probe tran V(*)

"""
            #FIXME: add more metrics?
            tran2_metrics = METRICS__tran_metrics_old[:]
         
            doc_measures = []
            #doc_measures = TYPICAL_DOC_MEASURES
                                      
            sim = Simulator(
                {
                'mt0':['srneg','srpos','outmax','outmin','outswing'],
                #'lis':[DOCS_METRIC_NAME],
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                doc_measures)
            tran2_an = CircuitAnalysis(tran2_env_points, tran2_metrics, sim, TRAN_COST)
                                                  
            analyses.append(tran2_an)
                        
        #-------------------------------------------------------
        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup)
       
        #-------------------------------------------------------
        #add function DOCs analysis
        funcDOCs_an = FunctionAnalysis(
            embedded_part.functionDOCsAreFeasible, [EnvPoint(True)],  0.99, INF, False, 0, 1)
        analyses.append(funcDOCs_an)
        
        #-------------------------------------------------------
        #finally, build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps
        
    def OP_ssViAmp1mod_Problem(self, process, libname):
        """
        @description
        
          Amplifier problem, for single ended input / single-ended output.
          Many goals, including slew rate, gain, power, area, ...
        
          Operating point driven
          
          The amplifier representation is restricted such that there is no
          way to represent an active load amp. This results in low gain amps,
          but gain is one of the goals. Active load amps however have higher gain,
          hence the novelty should be able to generate these.
          
          The second objective is power, since that determines the gain for a ss amp.
          However, with an actively loaded amp, you can have a lot more gain with the
          same power consumption. Mojito-n should hence come up with a biasedmos to replace
          the resistor.
          
        @arguments

          <<none>>          
        
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes
        """
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = OpLibraryStrategy(devices_setup)
        library = self._buildOpLib(lib_ss, libname)
        
        #build embedded part
        # -ssViAmp1_VddGndPorts has ports: Vin, Iout, Vdd, gnd
        part = library.ssViAmp1mod_VddGndPorts_Fixed()

        #the keys of 'connections' are the external ports of 'part'
        #the value corresponding to each key must be in the test_fixture_strings
        # that are below
        connections = {'Vin':'ninp', 'Iout':'nout', 'Vdd':'ndd', 'gnd':'gnd'}

        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        #we'll be building this up
        analyses = []

        #-------------------------------------------------------
        #shared info between analyses
        # (though any of this can be analysis-specific if we'd wanted
        #  to set them there instead)
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/ssvi1/'
        max_simulation_time = 5 #in seconds
        simulator_options_string = """
.include %ssimulator_options.inc
""" % cir_file_path

        #-------------------------------------------------------
        #build ac analysis
        if True:
            d = {
                 'pCload':5e-12,
                 'pVdd':devices_setup.vdd(),
                 'pVdcin':0.9,
                 'pVout':0.9,
                 'pRfb':1.000e+09,
                 'pCfb':1.000e-03,
                 'pTemp':25,
                 }
            d2 = {
                 'pCload':5e-12,
                 'pVdd':devices_setup.vdd(),
                 'pVdcin':0.9,
                 'pVout':0.9,
                 'pRfb':1.000e+09,
                 'pCfb':1.000e-03,
                 'pTemp':50,
                 }
            ac_env_points = [EnvPoint(True, d)]
            test_fixture_string = """
Cload	nout	gnd	pCload

* biasing circuitry

Vdd		ndd		gnd	DC=pVdd
Vss		gnd		0	DC=0
Vindc		ninpdc		gnd	DC=pVdcin
Vinac		ninpdc		ninp1	AC=1
Vintran 	ninp1 		ninpx 	DC=0 PWL(
+ 0     0
+ 0.1n   -0.2
+ 10.0n  -0.2 
+ 10.1n  0.2 
+ 30.0n  0.2
+ 30.1n  -.2 )

* feedback loop for dc biasing 
Vout_ref	nvout_ref	gnd	pVout
Efb1	nfbin	gnd	nout	nvout_ref	1.0e2
Rfb	nfbin	nfbout	pRfb
Cfb	nfbout	gnd	pCfb
Efb2	n1	gnd	nfbout	gnd	1.0
Efb3	ninp_unlim	n1	ninpx	gnd	1.0
Efb4	ninp	gnd	volts='MAX(0,MIN(pVdd,V(ninp_unlim)))'

* this measures the amount of feedback biasing there is
EFBM fbmnode gnd volts='ABS(V(ninp)-V(ninpdc))'

* simulation statements

.op

* temperature analysis
.temp pTemp

*.DC TEMP 25 25 10
.ac	dec	50	1.0e0	10.0e9
* pole-zero analysis
.pz v(nout) Vinac

.probe ac V(nout)
.probe ac V(ninp)
.probe ac V(*)

*.tran 100p 50n
.probe tran V(nout)
.probe tran V(ninp)
.probe tran V(*)

* Frequency-domain measurements
.measure ac ampl       max vdb(nout) at=0
.measure ac inampl max vdb(ninp) at=0
.measure ac gain PARAM='ampl-inampl'
.measure ac phase FIND vp(nout) WHEN vdb(nout)=0 CROSS=1
.measure ac phasemargin PARAM='phase+180'
.measure ac GBW WHEN vdb(nout)=0 CROSS=1
.measure ac phase0 FIND vp(nout) at=1e1

* power measurement
EPWR1 pwrnode gnd volts='-pVdd*I(Vdd)'

"""
            ac_metrics = METRICS__ac_metrics_small[:]
            setAsObjective(ac_metrics, 'gain')
            setAsObjective(ac_metrics, 'gbw')
            setAsObjective(ac_metrics, 'pwrnode')
   
            sim = Simulator({'ma0':['gain','phase0','phasemargin','gbw'],
                             'ic0':['pwrnode','fbmnode'],
                             'lis':[DOCS_METRIC_NAME]},
                            cir_file_path,
                            max_simulation_time,
                            simulator_options_string,
                            test_fixture_string,
                            TYPICAL_DOC_MEASURES)
                            
            ac_an = CircuitAnalysis(ac_env_points, ac_metrics, sim, AC_COST)
            analyses.append(ac_an)

        #-------------------------------------------------------
        #add function DOCs analysis
        funcDOCs_an = FunctionAnalysis(
            embedded_part.functionDOCsAreFeasible, [EnvPoint(True)], 0.99, INF, False, 0, 1)
        analyses.append(funcDOCs_an)
        
        #-------------------------------------------------------
        #finally, build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps
        
      
    def minimizeNmseOnTargetWaveform(self, process, dc_sweep_start_voltage, dc_sweep_end_voltage,
                                     target_waveform):
        """
        @description
        
          Returns a problem in which we're trying to minimize nmse
          to the input 'target_waveform'.
        
        @arguments

          dc_sweep_start_voltage -- float -- input start voltage
          dc_sweep_end_voltage -- float -- input end voltage
          target_waveform -- 1d array -- the target waveform
            desired.  Also specifies the number of values to use in the
            input dc sweep.
               
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes

          There is great flexibility in the choice of part; currently
          it is just a single mos (which means that it can't fit things well!)
        """
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = SizesLibraryStrategy(devices_setup)
        library = SizesLibrary(lib_ss)

        # part
        # -lots of choices here!
        part = library.mos4()
           
        #build part
 
        #connections = part.unityPortMap()
        connections = {'G':'ninp', 'D':'nout',
                       'S':'ndd', 'B':'gnd'}
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        #we'll be building this up
        analyses = []

        #-------------------------------------------------------
        #shared info between analyses
        # (though any of this can be analysis-specific if we'd wanted
        #  to set them there instead)
        max_simulation_time = 1.5 #in seconds
        
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/nmse/'
        
        simulator_options_string = """
.include %ssimulator_options.inc
""" % cir_file_path
        
        #-------------------------------------------------------
        #build dc analysis
        if True:
            d = {'pVdd':devices_setup.vdd(),
		 'pVinac':0 
                 }
            dc_env_points = [EnvPoint(True, d)]

            step_voltage = (dc_sweep_end_voltage - dc_sweep_start_voltage) / \
                           float(len(target_waveform))
            
            test_fixture_string = """


* biasing circuitry
Vdd		ndd		gnd	DC=pVdd
Vindc		ninpdc		gnd	DC=5
Vinac		ninpdc		ninp	AC=pVinac

* simulation statements

.dc	Vindc	%f	%f	%f		

* output the voltage waveforms 'vinp' (input) and 'nout' (output)
.print dc V(ninp)
.print dc V(nout)

* DC output measurment
*.measure dc maxvout max V(nout) at=0
*.options POST=1 brief
""" % (dc_sweep_end_voltage, dc_sweep_start_voltage, step_voltage)
            
            metrics_calculator = NmseOnTargetWaveformCalculator(target_waveform, 1)
            
            #build list of metrics
            dc_metrics = [
                #'maxvout' is just for testing
                #Metric('maxvout', 0, 1.8, True, 0, 3),
                METRIC__nmse
               ]

            #for ninp, nout in .print commands
            output_file_num_vars = {'sw0':2} 

            #set this by: examine .sw0 file, and supply which line the numbers
            # start at.  Start counting at 0.
            # One output var => 4.
            # Two output vars => 5.
            # ...
            output_file_start_line = {'sw0':5}
            
            sim = Simulator(
                {
                #'ms0':['maxvout'],
                'sw0':['nmse'],
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                [],
                output_file_num_vars,
                output_file_start_line,
                WAVEFORM_NUMBER_WIDTH,
                metrics_calculator,
                )
            
            reference_waveforms = numpy.ones((1,len(target_waveform)),
                                               dtype=float)
            reference_waveforms[0,:] = target_waveform
            
            dc_an = CircuitAnalysis(dc_env_points, dc_metrics, sim, AC_COST, reference_waveforms)
            analyses.append(dc_an)
            
        #-----------------------
        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup)
             
        #-------------------------------------------------------
        #finally, build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps
    	
     
    def minimizeNmseOnTargetShape(self, process, shape, extra_bump_measures):
        """
        @description
        
          Return a problem where the shape of a desired waveform is
          specified as a shape, but is parameterizable.
        
        @arguments

          shape -- string -- one of regressor.Pwl.PWL_APPROACHES,
            such as 'hockey', 'bump', ...
          extra_bump_measures -- bool -- if shape is 'bump', add other measures
            besides nmse?
               
        @return

          ps -- ProblemSetup object
    
        @exceptions
    
        @notes
        """
        #preconditions
        if shape == 'bump':
            assert extra_bump_measures in [True, False]
        else:
            extra_bump_measures = False
        
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = SizesLibraryStrategy(devices_setup)
        library = SizesLibrary(lib_ss)

        # part
        # -lots of flexibility here!
        part = library.mos4()
           
        #build part
 
        #connections = part.unityPortMap()
        connections = {'G':'ninp', 'D':'nout',
                       'S':'ndd', 'B':'gnd'}
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        #we'll be building this up
        analyses = []

        #-------------------------------------------------------
        #shared info between analyses
        # (though any of this can be analysis-specific if we'd wanted
        #  to set them there instead)
        max_simulation_time = 1.5 #in seconds
        
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/nmse/'
        
        simulator_options_string = """
.include %ssimulator_options.inc
""" % cir_file_path
        
        #-------------------------------------------------------
        #build dc analysis
        if True:
            d = {'pVdd':devices_setup.vdd(),
		 'pVinac':0 
                 }
            dc_env_points = [EnvPoint(True, d)]

            dc_sweep_start_voltage = 0.0
            dc_sweep_end_voltage = 1.8
            num_steps = 10
            step_voltage = (dc_sweep_end_voltage - dc_sweep_start_voltage) / \
                           float(num_steps)
            
            test_fixture_string = """


* biasing circuitry
Vdd		ndd		gnd	DC=pVdd
Vindc		ninpdc		gnd	DC=5
Vinac		ninpdc		ninp	AC=pVinac

* simulation statements

.dc	Vindc	%f	%f	%f		

* output the voltage waveforms 'vinp' (input) and 'nout' (output)
.print dc V(ninp)
.print dc V(nout)

* DC output measurment
*.measure dc maxvout max V(nout) at=0
*.options POST=1 brief
""" % (dc_sweep_end_voltage, dc_sweep_start_voltage, step_voltage)

            if extra_bump_measures:
                assert shape == 'bump'
                metrics_calculator = TransientWaveformCalculator(
                    0, 1, False)

                #build list of metrics
                dc_metrics = [
                    #"analog" measures
                    Metric('dynamic_range', 1.0, INF, True, 0, 2), #OBJECTIVE
                    Metric('slewrate', 0.5e8, INF, True, 0, 3e7), #OBJECTIVE

                    #measures to help guarantee the shape is good
                    METRIC__nmse,
                    METRIC__correlation,
                    METRIC__ymin_before_ymax,
                    METRIC__ymin_after_ymax,
                   ]
                assert sorted(metrics_calculator.metricNames()) == \
                       sorted([mm.name for mm in dc_metrics])
            else:
                metrics_calculator = NmseOnTargetShapeCalculator(shape, 0, 1) 

                #build list of metrics
                dc_metrics = [
                    #"minimize nmse", with arbitrary feasibility threshold of 1000.0
                    Metric('nmse', -INF, 1000.0, True, 0, 1),	    
                   ]

            #for ninp, nout in .print commands
            output_file_num_vars = {'sw0':2} 

            #set this by: examine .sw0 file, and supply which line the numbers
            # start at.  Start counting at 0.
            # One output var => 4.
            # Two output vars => 5.
            # ...
            output_file_start_line = {'sw0':5}
            
            sim = Simulator(
                {
                'sw0':metrics_calculator.metricNames(),
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                [],
                output_file_num_vars,
                output_file_start_line,
                WAVEFORM_NUMBER_WIDTH,
                metrics_calculator,
                )
            
            reference_waveforms = None
            dc_an = CircuitAnalysis(dc_env_points, dc_metrics, sim, AC_COST, reference_waveforms)
            analyses.append(dc_an)
            
        #-----------------------
        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup)
             
        #-------------------------------------------------------
        #finally, build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps
    
    def OP_currentmirrorEMC_Problem(self, process):
        """
        @description
            current mirror EMC problem with a flexnode3 on the input node.
            the goals are correct mirroring functionality and EMI resistance
            operating point driven
        """
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = OpLibraryStrategy(devices_setup)
        library = OpLibraryEMC(lib_ss)
        
        # main part
        part = library.currentMirrorEMC()
        
        connections = {'nin':'nin','nout':'nout','Vdd':'ndd','Vss':'nss'}
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None
            
        embedded_part = EmbeddedPart(part,connections,functions)
        
        analyses = []
        
        max_simulation_time = 50
        
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/cmEMC/'
        
        simulator_options_string = """\n.include %ssimulator_options.inc\n""" % cir_file_path

        d = {'pRload':2e3,
             'pVdd':devices_setup.vdd(),
             'pIbias':-1e-4,
             'pVemi':0.1,
             'pfemi':1e6,
             'pCemi':6.8e-9,
             'pfstart':1e3,
             'pfstop':1e9,
             'ptstart':20e-6,
             'ptstop':40e-6,
             'pTemp':25}
        env_points = [EnvPoint(True, d)]
        
        # ac analyse
        test_fixture_string = """
* biasing circuitry
Vdd ndd 0 DC=pVdd
Vss nss 0 DC=0

* load
Rload nout ndd pRload

* input sources
Ibias nin ndd DC=pIbias
Vemi nemi 0 AC=pVemi
Cemi nin nemi C=pCemi

* simulation statements
.tf I(Rload) Ibias
.ac dec 3 pfstart pfstop

*measurements
.measure ac damping avg vdb(nout) from=pfemi to=pfstop
Ecur1 curnode gnd volts='-I(Rload)'
EPWR1 pwrnode gnd volts='-pVdd*I(Vdd)'
"""

        #build list of metrics
        cmemc_metrics = METRICS__cmEMC_ac[:]
        setAsObjective(cmemc_metrics, 'pwrnode')

        sim = Simulator({'ma0':['damping'],
                         'ic0':['curnode','pwrnode'],
                         'lis':[DOCS_METRIC_NAME,'i(rload)/ibias']},
                         cir_file_path,
                         max_simulation_time,
                         simulator_options_string,
                         test_fixture_string,
                         TYPICAL_DOC_MEASURES)
        ac_an = CircuitAnalysis(env_points, cmemc_metrics, sim,1)
        analyses.append(ac_an)
        
        #-------------------------------------------------------
        #add function DOCs analysis
        funcDOCs_an = FunctionAnalysis(
            embedded_part.functionDOCsAreFeasible, [EnvPoint(True)], 0.99, INF, False, 0, 1)
        analyses.append(funcDOCs_an)
        
        #-------------------------------------------------------
        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup)
        
        #-------------------------------------------------------
        #build transient analysis
        if not DISABLE_TRANSIENT_GLOBALLY:
            test_fixture_string = """
* biasing circuitry
Vdd ndd 0 DC=pVdd
Vss nss 0 DC=0

* load
Rload nout ndd pRload

* input sources
Ibias nin ndd DC=pIbias
Vemi nemi 0 sin 0 pVemi pfemi
Cemi nin nemi C=pCemi

* simulation statements
.tran 100p ptstop

*measurements
.measure tran startVout avg v(nout) from=0 to=1u
.measure tran settledVout avg v(nout) from=ptstart to=ptstop
.measure tran dvout PARAM='settledVout-startVout'
"""
            #build list of metrics
            cmemc_metrics = METRICS__cmEMC_tran[:]
            setAsObjective(cmemc_metrics, 'dvout')
    
            sim = Simulator({'mt0':['dvout']},
                             cir_file_path,
                             max_simulation_time,
                             simulator_options_string,
                             test_fixture_string,
                             TYPICAL_DOC_MEASURES)
            tran_an = CircuitAnalysis(env_points, cmemc_metrics, sim,1)
            analyses.append(tran_an)
        
        #-------------------------------------------------------
        #build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps
    	
        
    def opampEMC_Problem(self, process):
        """
        @description
            opamp EMC problem
        """
        #build library
        devices_setup = DevicesSetup(process)
        lib_ss = OpLibraryStrategy(devices_setup)
        library = OpLibraryEMC(lib_ss)
        
        #build part
        part = library.dsAmp2()
        
        #build embedded part
        connections = {'Vin1':'ninp','Vin2':'ninn','out':'nout','Vss':'gnd','Vdd':'ndd'}
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
            
        embedded_part = EmbeddedPart(part, connections, functions)

        #we'll be building this up
        analyses = []

        #-------------------------------------------------------
        #shared info between analyses
        # (though any of this can be analysis-specific if we'd wanted
        #  to set them there instead)        
        max_simulation_time = 50 #in seconds
        
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/opampEMC/'
        
        simulator_options_string = """
.include %ssimulator_options.inc
""" % cir_file_path
        
        #-------------------------------------------------------
        #build dc analysis
        if False:
            pass

        #-------------------------------------------------------
        #build ac analysis
        if True:
            d = {'pCload':1e-12,
                 'pVdd':devices_setup.vdd(),
                 'pVdcin':0.9,
                 'pVout':0.9,
                 'pRfb':1.000e+09,
                 'pCfb':1.000e-03,
                 'pVemi':0.1,
                 'pfemi':1e6,
                 'pCemi':6.8e-9,
                 'pTemp':25,
                 }
            ac_env_points = [EnvPoint(True, d)]
            test_fixture_string = """          
Cload    nout    gnd    pCload

* biasing circuitry

Vdd        ndd        gnd    DC=pVdd
Vss        gnd        0    DC=0
Vindc        ninpdc        gnd    DC=pVdcin
Vinac        ninp        ninpdc    AC=1 SIN(0 1 10k)

* feedback loop for dc biasing of output stage

Efb1    nlpfin    gnd    nout    gnd    1
Rfb    nlpfin    nlpfout    pRfb
Cfb    nlpfout    gnd    pCfb

.param pFb=10
Efb2    ninn    gnd    volts='MAX(0,MIN(1.8, (V(nlpfout)-pVout) * pFb + pVdcin ))'

* this measures the amount of feedback biasing there is
EFBM fbmnode gnd volts='ABS((V(nlpfout)-pVout) * pFb)'

* simulation statements

.op
.ac    dec    50    1.0e0    10.0e9

* pole-zero analysis
.pz v(nout) Vinac

* Frequency-domain measurements
.measure ac ampl       max vdb(nout) at=0
.measure ac inampl max vdb(ninp,ninn) at=0
.measure ac gain PARAM='ampl-inampl'
.measure ac phase FIND vp(nout) WHEN vdb(nout)=0 CROSS=1
.measure ac phasemargin PARAM='180+phase'
.measure ac GBW WHEN vdb(nout)=0 CROSS=1
.measure ac phase0 FIND vp(nout) at=1e1

.measure ac pole1 WHEN vp(nout)=135 CROSS=1
.measure ac pole2 WHEN vp(nout)=45 CROSS=1

* power measurement
EPWR1 pwrnode gnd volts='-pVdd*I(Vdd)'


"""
            #build list of metrics
            ac_metrics = METRICS__ac_metrics_big[:]
            setAsObjective(ac_metrics, 'gain')
            setAsObjective(ac_metrics, 'gbw')
            setAsObjective(ac_metrics, 'pwrnode')

            sim = Simulator(
                {
#                 'ma0':['gain','phasemargin','phase0','gbw','pole1','pole2'],
                'ma0':['gain','phasemargin','phase0','gbw'],
                'ic0':['pwrnode','fbmnode'],
                'lis':[DOCS_METRIC_NAME,'pole1fr','pole2fr','pole2_margin']
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                TYPICAL_DOC_MEASURES)
            ac_an = CircuitAnalysis(ac_env_points, ac_metrics, sim, AC_COST)
            analyses.append(ac_an)
            
        #-------------------------------------------------------
        #minimize area (an objective)
        addArea(analyses, embedded_part, devices_setup)

        #note that we will later add 'novelty' as an OBJECTIVE too
        
        #-------------------------------------------------------
        #add function DOCs analysis
        #:NOTE: tlm - turn this off for novelty (hurts op pt driven search)
        funcDOCs_an = FunctionAnalysis(
            embedded_part.functionDOCsAreFeasible, [EnvPoint(True)], 0.99, INF, False, 0, 1)
        analyses.append(funcDOCs_an)
        
        #-------------------------------------------------------
        #build transient analysis
        if True and not DISABLE_TRANSIENT_GLOBALLY:
            tran_env_points = ac_env_points
            test_fixture_string = """
Cload    nout    gnd    pCload

* biasing circuitry

Vdd        ndd        gnd    DC=pVdd
Vss        gnd        0    DC=0
Vindc        ninpdc        gnd    DC=pVdcin
VinTran         ninp            ninpdc  PULSE(-0.2 0.2 0.2e-6 100p 100p 0.5e-6 1e-6)

Enin2            ninn            gnd    volts = 'pVdcin-V(ninp, ninpdc)'

* simulation statements

.tran 100p 1e-6

* output the voltage waveforms 'vinp' (input) and 'nout' (output)
.print tran V(nout)




* input sources
Ibias nin ndd DC=pIbias
Vemi nemi 0 sin 0 pVemi pfemi
Cemi nin nemi C=pCemi

* simulation statements
.tran 100p ptstop

*measurements
.measure tran startVout avg v(nout) from=0 to=1u
.measure tran settledVout avg v(nout) from=ptstart to=ptstop
.measure tran dvout PARAM='settledVout-startVout'
"""
            transient_extension = 'tr0'
            metrics_calculator = TransientWaveformCalculator(
                0, 1, chop_initial=True)

            tran_metrics = METRICS__tran_metrics_new[:]
            setAsObjective(tran_metrics, 'dynamic_range')
            setAsObjective(tran_metrics, 'slewrate')
                            
            #for nout in .print commands ('time' variable comes for free)
            output_file_num_vars = {transient_extension:1} 

            #set this by: examine .tr0 file, and supply which line the numbers
            # start at.  Start counting at 0.
            # One output var => 4.
            # Two output vars => 5.
            # ...
            output_file_start_line = {transient_extension:4}
         
            sim = Simulator(
                {
                'tr0':metrics_calculator.metricNames(),
                #'lis':[DOCS_METRIC_NAME],
                },
                cir_file_path,
                max_simulation_time,
                simulator_options_string,
                test_fixture_string,
                [],
                output_file_num_vars,
                output_file_start_line,
                WAVEFORM_NUMBER_WIDTH,
                metrics_calculator,
                )
            tran_an = CircuitAnalysis(tran_env_points, tran_metrics, sim, TRAN_COST)
                                                  
            analyses.append(tran_an)
## """
##             #FIXME: add more metrics?
##             tran_metrics = METRICS__tran_metrics_old[:]
                                      
##             sim = Simulator(
##                 {
##                 'ma0':['srneg','srpos','outmax','outmin'],
##                 },
##                 cir_file_path,
##                 max_simulation_time,
##                 simulator_options_string,
##                 test_fixture_string,
##                 TYPICAL_DOC_MEASURES)
##             tran_an = CircuitAnalysis(tran_env_points, tran_metrics, sim, TRAN_COST)
                                                  
##             analyses.append(tran_an)

        #-------------------------------------------------------
        #finally, build PS and return
        ps = ProblemSetup(embedded_part, analyses, library, devices_setup)
        return ps

class _simulationFunc:
    """Helper for ten-objective function problem"""
    def __init__(self, var):
        self.var = var

    def __call__(self, point):
        return point[self.var]
    
class _sphereFunc:
    """Helper for sphere function problem"""
    def __init__(self):
        pass
    
    def __call__(self, scaled_point):
        assert scaled_point.is_scaled
        return sum(value**2 for value in scaled_point.itervalues())

