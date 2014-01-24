
import os
import random
import unittest

import numpy

from adts import *
from util.constants import DOCS_METRIC_NAME, INF
from problems.MetricCalculators import NmseOnTargetWaveformCalculator
from problems.SizesLibrary import Point18SizesLibrary

def getSizesLibrary():
    return Point18SizesLibrary()

def some_function(x):
    return x+3

def buildSimulatorStub(metrics_per_outfile):
    lis_measures = ['region','vgs']
    return Simulator(metrics_per_outfile, '/', 0,'', '', lis_measures)

class AnalysisTest(unittest.TestCase):

    def setUp(self):
        pass
        
    def testFunctionAnalysis(self):
        an = FunctionAnalysis(some_function, [EnvPoint(True)], 10, 20, False, 10, 20)
        self.assertEqual(len(an.env_points), 1)
        self.assertEqual(an.function, some_function)
        self.assertEqual(len(an.metrics), 1)
        self.assertTrue(isinstance(an.metric, Metric))
        self.assertEqual(an.metric.name, 'metric_some_function')
        self.assertEqual(an.metric.min_threshold, 10)
        self.assertEqual(an.metric.max_threshold, 20)
        self.assertEqual(an.metric.improve_past_feasible, False)
        self.assertTrue(len(str(an)) > 0)

        self.assertRaises(ValueError, FunctionAnalysis, some_function,
                          [EnvPoint(False)], 10, 20, False, 10, 20)
        
        an2 = FunctionAnalysis(some_function, [EnvPoint(True)], 10, 20, False, 10, 20)
        self.assertNotEqual(an.ID, an2.ID)

    def testCircuitAnalysis(self):
        #These tests are configured for metric: percent DOCs met.  If it
        # changes, then change these tests as well, if needed.
        self.assertEqual(DOCS_METRIC_NAME, 'sim_DOCs_cost')
        
        #build up metrics
        metrics = [Metric('gain', 10, INF, False, 0, 10),
                   Metric('pwrnode', 10, 20, False, 0, 10),
                   Metric(DOCS_METRIC_NAME, -INF, 1.0e-5, False, 0, 10)]

        #test building up the simulator
        # -typical use
        d = {'ma0':['gain'], 'ic0':['pwrnode'], 'lis':[DOCS_METRIC_NAME]}
        sim = buildSimulatorStub(d)
        self.assertEqual(sim.metrics_per_outfile, d)
        self.assertEqual(sorted(sim.metricNames()),
                         sorted(['gain','pwrnode',DOCS_METRIC_NAME]))

        # -complain if the dict's values are not in lists
        self.assertRaises(ValueError, buildSimulatorStub,
                          {'ma0':'gain', 'ic0':'pwrnode',
                           'lis':DOCS_METRIC_NAME})
        
        # -only allow lis to have one metric DOCS_METRIC_NAME
        self.assertRaises(ValueError, buildSimulatorStub,
                          {'lis':['blah']})

        # -allow DOCS_METRIC_NAME to only come from 'lis'
        self.assertRaises(ValueError, buildSimulatorStub,
                          {'ma0':[DOCS_METRIC_NAME]})

        #test building up the analysis
        an = CircuitAnalysis([EnvPoint(True)], metrics, sim, 1.1)
        self.assertEqual(len(an.env_points), 1)
        self.assertEqual(len(an.metrics), 3)
        self.assertEqual(an.relative_cost, 1.1)
        self.assertEqual(an.relativeCost(), an.relative_cost)

        an2 = CircuitAnalysis([EnvPoint(True)], metrics, sim, 2.0) 
        self.assertNotEqual(an.ID, an2.ID)

        # -complain if dict's keys are not in the set of allowed keys
        self.assertRaises(ValueError, buildSimulatorStub, {'bad_value':'gain'})

        # -need env_point objects
        self.assertRaises(ValueError, CircuitAnalysis, ['not_env_point'], metrics, sim, 1.0)

        # -need env_point to be scaled
        self.assertRaises(ValueError, CircuitAnalysis, [EnvPoint(False)], metrics, sim, 1.0)

        # -need metric objects
        self.assertRaises(ValueError, CircuitAnalysis, [EnvPoint(True)], ['not_a_metric'], sim, 1.0)

        # -relative cost must be a number, and > 0.0
        self.assertRaises(AssertionError, CircuitAnalysis, [EnvPoint(True)], metrics, sim, None)
        self.assertRaises(AssertionError, CircuitAnalysis, [EnvPoint(True)], metrics, sim, -1.0)
        self.assertRaises(AssertionError, CircuitAnalysis, [EnvPoint(True)], metrics, sim, 0.0)
        
        #add to this when we support circuit analyses more fully
        
    def testExtractLisResults(self):
        #build an analysis
        d = {'ma0':['gain'], 'lis':[DOCS_METRIC_NAME]}
        sim = buildSimulatorStub(d)

        #test
        # FIXME: the following directory setup is a make-it-work hack
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        if 'adts/test/' not in pwd:
            pwd += 'adts/test/'
        lis_file = os.path.abspath(pwd + 'test_lisfile.lis')
        success, lis_results = sim._extractLisResults(lis_file)
        self.assertTrue(success)
        self.assertEqual(lis_results,
                         {'lis__m0__region': 0, 'lis__m14__region': 0, 'lis__m13__region': 1, 'lis__m10__region': 0, 'lis__m6__vgs': -1.5269999999999999, 'lis__m8__vgs': 12.050000000000001, 'lis__m2__region': 1, 'lis__m2__vgs': 0.40579999999999999, 'lis__m4__vgs': -2.4329999999999998, 'lis__m6__region': 1, 'lis__m4__region': 0, 'lis__m0__vgs': -1.5840000000000001, 'lis__m13__vgs': 1.3740000000000001, 'lis__m8__region': 0, 'lis__m14__vgs': 1.3740000000000001, 'lis__m10__vgs': -2.4329999999999998})

        #test bad -- missing a whole 'models' section
        bad_file = os.path.abspath(pwd + 'test_lisfile_bad.lis')
        success, lis_results = sim._extractLisResults(bad_file)
        self.assertFalse(success)
        
        #test bad2 -- device names don't line up with targeted measures
        bad_file = os.path.abspath(pwd + 'test_lisfile_bad2.lis')
        success, lis_results = sim._extractLisResults(bad_file)
        self.assertFalse(success)

    def testNmseOnTargetWaveformCalculator(self):
        waveforms_array = numpy.array([[1.0, 1.0, 1.0, 1.0, 1.0], [0.2, 0.4, 0.4, 0.5, 0.1]])
        calculator = NmseOnTargetWaveformCalculator([0.2, 0.3, 0.4, 0.5, 0.2], 1)
        wrange = 0.5 - 0.2
        expected_nmse = (0.1/wrange)**2 + (0.1/wrange)**2
        nmse = calculator.compute(waveforms_array)['nmse']
        self.assertAlmostEqual(nmse, expected_nmse, 5)

    def testCreateFullNetlist(self):
        """Test CircuitAnalysis.createFullNetlist.  Test nominal and non-nominal netlist generation."""
        #build CircuitAnalysis.
        # -the construction here is taken from OP_dsViAmp_Problem.OP_dsViAmp_Problem()
        d = {'pCload':1e-12, 'pVdd':1.8, 'pVdcin':0.9, 'pVout':0.9, 'pRfb':1.000e+09, 'pCfb':1.000e-03, 'pTemp':27}
        env_points = [EnvPoint(True, d)]

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

* do ac analysis
.ac	dec	50	1.0e0	10.0e9

* Frequency-domain measurements
.measure ac gain PARAM='ampl-inampl'

"""
        metrics = [Metric('gain', 10, INF, False, 0, 10)]
        
        pwd = os.getenv('PWD')
        if pwd[-1] != '/':
            pwd += '/'
        cir_file_path = pwd + 'problems/miller2/'

        simulator_options_string = "\n.include %ssimulator_options.inc\n" % cir_file_path   
        
        simulator = Simulator(metrics_per_outfile={'ma0':['gain']},
                              cir_file_path=cir_file_path,
                              max_simulation_time=5,
                              simulator_options_string=simulator_options_string,
                              test_fixture_string=test_fixture_string,
                              lis_measures=[])

        an = CircuitAnalysis(env_points, metrics, simulator, relative_cost=1.1)
        
        #build toplevel_embedded_part
        library = getSizesLibrary()
        part = library.dsViAmp1_VddGndPorts()
        connections = {'Vin1':'ninp', 'Vin2':'ninn', 'Iout':'nout','Vdd':'ndd', 'gnd':'gnd'}
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None
        toplevel_embedded_part = EmbeddedPart(part, connections, functions)

        #build scaled_opt_point.  Force a topology that has resistors, and with a certain degen_R.
        scaled_opt_point = part.point_meta.createRandomScaledPoint(False)
        scaled_opt_point.update({'cascode_is_wire' : 0,  'cascode_recurse' : 0,  'chosen_part_index' : 0,
                                 'degen_choice' : 1,  'input_is_pmos' : 0,  'load_chosen_part_index' : 0,
                                 'load_topref_usemos' : 0, 'degen_R': 641739.47363375011})
        expected_nominal_R_string = 'R=%g' % scaled_opt_point['degen_R'] #like in Var.VarMeta.spiceNetlistStr

        #create & test nominal netlist, from always-nominal problem setup
        devices_setup = DevicesSetup('UMC180')
        self.assertTrue(devices_setup.always_nominal)
        rnd_point = devices_setup.all_rnd_points[0]
        self.assertTrue(rnd_point.isNominal())
        variation_data = (rnd_point, env_points[0], devices_setup)
        netlist = an.createFullNetlist(toplevel_embedded_part, scaled_opt_point, variation_data)
        self.assertTrue('*------Design---------' in netlist)             #have design?
        self.assertTrue('gnd Nnom M=' in netlist)                        #design uses nominal NMOS model?
        self.assertTrue('ndd Pnom M=' in netlist)                        #design uses nominal PMOS model?
        self.assertTrue(("param pTemp = %5.3e" % d['pTemp']) in netlist) #have env variables?
        self.assertTrue(test_fixture_string in netlist)                  #have test fixture?
        self.assertTrue(simulator_options_string in netlist)             #have simulator options?
        self.assertTrue(expected_nominal_R_string in netlist)            #degen resistor is nominal?

        #create & test nominal netlist, from non-nominal problem setup
        devices_setup = DevicesSetup('UMC180')
        devices_setup.makeRobust()
        self.assertFalse(devices_setup.always_nominal)
        rnd_point = devices_setup.nominalRndPoint()
        self.assertTrue(rnd_point.isNominal())
        variation_data = (rnd_point, env_points[0], devices_setup)
        netlist = an.createFullNetlist(toplevel_embedded_part, scaled_opt_point, variation_data)
        
        nom_netlist = netlist

        #create & test non-nominal netlist 
        devices_setup = DevicesSetup('UMC180')
        devices_setup.makeRobust()
        rnd_point = devices_setup.all_rnd_points[2]
        self.assertFalse(rnd_point.isNominal())
        variation_data = (rnd_point, env_points[0], devices_setup)
        netlist = an.createFullNetlist(toplevel_embedded_part, scaled_opt_point, variation_data)
        self.assertFalse('Nnom M=' in netlist)                    #design does not use nominal NMOS model?
        self.assertFalse('Pnom M=' in netlist)                    #design does not use nominal PMOS model?
        self.assertTrue('gnd NM' in netlist)                      #design does use non-nominal NMOS model?
        self.assertTrue('ndd PM' in netlist)                      #design does use non-nominal PMOS model?
        self.assertTrue('.model NM' in netlist)                   #does define non-nominal NMOS model?
        self.assertTrue('.model PM' in netlist)                   #does define non-nominal PMOS model?
        self.assertTrue(len(netlist) > len(nom_netlist)*3)               #have non-nominal models? (take up space)
        self.assertFalse(expected_nominal_R_string in netlist)           #degen resistor is non-nominal?
        
    def tearDown(self):
        pass

if __name__ == '__main__':

    import logging
    logging.basicConfig()
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    
    unittest.main()
