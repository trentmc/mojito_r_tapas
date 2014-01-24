import unittest

import random
import string
import os
import time
import logging

from adts import *
from adts.Part import replaceAutoNodesWithXXX

from problems.OpLibrary2 import *
from problems.Library import replaceAfterMRWithBlank

#make this global for testing so we only have one disk access
DEVICES_SETUP = DevicesSetup('UMC180')
_GLOBAL_approx_mos_models = DEVICES_SETUP.approxMosModels()

log = logging.getLogger('problems')

class OpLibrary2Test(unittest.TestCase):

    def setUp(self):
        self.just1 = True #to make True is a HACK

        self.lib = OpLibrary2(OpLibrary2Strategy(DEVICES_SETUP))
        
        self.max_approx_error_relative=0.25

    #=================================================================
    #One Test for each Part
    def testNmos4Sized(self):
        if self.just1: return
        part = self.lib.nmos4_sized()
        self.assertEqual( part.externalPortnames(), ['D','G','S','B'])
        instance = EmbeddedPart(part, {'D':'1', 'G':'2', 'S':'3', 'B':'nblah'},
                                {'W':3*0.18e-6, 'L':5*0.18e-6, 'M':1})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = """M0 1 2 3 nblah Nnom M=1 L=9e-07 W=5.4e-07
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testPmos4Sized(self):
        if self.just1: return
        part = self.lib.pmos4_sized()
        instance = EmbeddedPart(part, {'D':'1', 'G':'2', 'S':'3', 'B':'nblah'},
                                {'W':3*0.18e-6, 'L':5*0.18e-6, 'M':1})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = """M0 1 2 3 nblah Pnom M=1 L=9e-07 W=5.4e-07
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testNmos4_simple(self):
        if self.just1: return
        part = self.lib.nmos4()
        self.assertEqual( part.externalPortnames(), ['D','G','S','B'])
        instance = EmbeddedPart(part, {'D':'1', 'G':'2', 'S':'3', 'B':'nblah'},
                                {'Vds':1.0,'Vgs':1.0,'Vbs':0.0,'L':0.18e-6,'Ids':1e-3})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = """M0 1 2 3 nblah Nnom M=3 L=1.8e-07 W=2.4e-05
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)
        
    def testNmos4_bigL(self):
        if self.just1: return
        part = self.lib.nmos4()
        instance = EmbeddedPart(part, {'D':'1', 'G':'2', 'S':'3', 'B':'nblah'},
                                {'Vds':1.0,'Vgs':1.0,'Vbs':0.0,'L':1e-6,'Ids':1e-3})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = """M0 1 2 3 nblah Nnom M=5 L=1e-06 W=4.83382e-06
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)
        
    def testNmos4_bigL_bigIds(self):
        if self.just1: return
        part = self.lib.nmos4()
        instance = EmbeddedPart(part, {'D':'1', 'G':'2', 'S':'3', 'B':'nblah'},
                                {'Vds':1.0,'Vgs':1.0,'Vbs':0.0,'L':1e-6,'Ids':2e-3})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = """M0 1 2 3 nblah Nnom M=10 L=1e-06 W=4.83382e-06
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)
        
    def testPmos4(self):
        if self.just1: return
        part = self.lib.pmos4()
        instance = EmbeddedPart(part, {'D':'ndrain', 'G':'ngate', 'S':'nsource', 'B':'nbulk'},
                                {'Vds':1.0,'Vgs':1.0,'Vbs':0.0,'L':1e-6,'Ids':1e-3})
        
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = """M0 ndrain ngate nsource nblah Pnom M=22 L=1e-06 W=4.84575e-06
"""
        actual_str = instance.spiceNetlistStr()
        
        self._compareStrings(target_str, actual_str)
        
    def testDcvs(self):
        if self.just1: return
        part = self.lib.dcvs()
        n0 = part.externalPortnames()[0]
        instance = EmbeddedPart(part, {n0:'1'}, {'DC':1.8})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = """V0 1 0  DC 1.8
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testWire(self):
        if self.just1: return
        part = self.lib.wire()
        instance = EmbeddedPart(part, {'1':'a', '2':'b'}, {})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = """Rwire0 a b  R=0
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testOpenCircuit(self):
        if self.just1: return
        part = self.lib.openCircuit()
        instance = EmbeddedPart(part, {'1':'a', '2':'b'}, {})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = "" #yes, target string is _empty_
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testShortOrOpenCircuit(self):
        if self.just1: return
        part = self.lib.shortOrOpenCircuit()

        #instantiate as short circuit
        instance = EmbeddedPart(part, {'1':'a', '2':'b'},
                                {'chosen_part_index':0})

        target_str = """Rwire0 a b  R=0
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

        #instantiate as open circuit
        instance = EmbeddedPart(part, {'1':'a', '2':'b'},
                                {'chosen_part_index':1})

        target_str = ""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testSizedLogscaleResistor(self):
        if self.just1: return

        part = self.lib.sizedLogscaleResistor()
        instance = EmbeddedPart(part, {'1':'a', '2':'b'}, {'R':10.2e3})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = "R0 a b  R=10200\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testSizedLogscaleResistor(self):
        if self.just1: return

        part = self.lib.sizedLinscaleResistor()
        instance = EmbeddedPart(part, {'1':'a', '2':'b'}, {'R':10.2e3})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = "R0 a b  R=10200\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testResistor(self):
        if self.just1: return

        part = self.lib.resistor()

        # warning: current/voltage values can be railed!
        instance = EmbeddedPart(part, {'1':'a', '2':'b'}, {'V':1,'I':0.001})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = "R0 a b  R=1000\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testRailing(self):
        """test to see that it rails to be <= max allowed value.
        We use resistance of resistor for the test."""
        if self.just1: return

        part = self.lib.sizedLogscaleResistor()
        instance = EmbeddedPart(part, {'1':'a', '2':'b'}, {'R':10.2e3})

        R_varmeta = self.lib._ref_varmetas['logscale_R']
        self.assertTrue(R_varmeta.logscale)
        max_R = 10**R_varmeta.max_unscaled_value
        instance = EmbeddedPart(part, {'1':'a', '2':'b'}, {'R':max_R*10.0})
        target_str = "R0 a b  R=%g\n" % max_R
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testCapacitor(self):
        if self.just1: return

        part = self.lib.capacitor()
        instance = EmbeddedPart(part, {'1':'a', '2':'b'}, {'C':1.0e-6})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )

        target_str = "C0 a b  C=1e-06\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str, actual_str)

    def testMos4(self):
        if self.just1: return
        part = self.lib.mos4()
        self.assertTrue( isinstance( part, FlexPart ) )
        self.assertTrue( len(str(part)) > 0 )

        # sizing is assumed to be tested somewhere else

        #instantiate as nmos
        instance0 = EmbeddedPart(part, {'D':'1', 'G':'2', 'S':'3', 'B':'4'},
                                 {'chosen_part_index':0,
                                  'Vds':1.0,'Vgs':1.0,'Vbs':0.0,
                                  'L':1e-6,'Ids':1e-3}
                                 )
        self.assertTrue( len(str(instance0)) > 0 )

        target_str0 = """M0 1 2 3 4 Nnom 
"""
        actual_str0 = instance0.spiceNetlistStr()
        self._compareStrings3(target_str0, actual_str0)

        #instantiate as pmos
        instance0 = EmbeddedPart(part, {'D':'1', 'G':'2', 'S':'3', 'B':'4'},
                                 {'chosen_part_index':1,
                                  'Vds':1.0,'Vgs':1.0,'Vbs':0.0,
                                  'L':1e-6,'Ids':1e-3}
                                 )
        self.assertTrue( len(str(instance0)) > 0 )

        target_str0 = """M0 1 2 3 4 Pnom 
"""
        actual_str0 = instance0.spiceNetlistStr()
        self._compareStrings3(target_str0, actual_str0)

    def testMos3(self):
        if self.just1: return
        part = self.lib.mos3()

        #instantiate as pmos
        instance = EmbeddedPart(part, {'D':'1', 'G':'2', 'S':'3'},
                                {'use_pmos':1,
                                 'Vds':1.0,'Vgs':1.0,
                                 'L':1e-6,'Ids':1e-3}
                               )
        
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0 )
        
        target_str = """M0 1 2 3 3 Pnom 
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str, actual_str)

    def testBiasedMos(self):
        if self.just1: return
        part = self.lib.biasedMos()

        instance = EmbeddedPart(part, {'D':'1', 'S':'2'},
                                {'Vgs':1.0,'Vds':1.0,
                                 'L':1e-6,'Ids':1e-3,
                                 'use_pmos':1,
                                 'Vs':1.8})
        
        target_str = """M0 1 n_auto_2 2 2 Pnom M=22 L=1e-06 W=4.84575e-06
V1 n_auto_2 0  DC 0.8
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str, actual_str)
        
    def testBiasedMosOrWire(self):
        if self.just1: return
        part = self.lib.biasedMosOrWire()

        conns = {'D':'a', 'S':'b'}
        point = {'Vgs':1.0,'Vds':1.0,
                 'L':1e-6,'Ids':1e-3,
                 'use_pmos':1,
                 'Vs':1.8}

        #instantiate as biasedMos
        point['chosen_part_index'] = 0 
        instance = EmbeddedPart(part, conns, point)
        
        target_str = """M0 a n_auto_2 b b Pnom M=22 L=1e-06 W=4.84575e-06
V1 n_auto_2 0  DC 0.8
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str, actual_str)

        #instantiate as wire
        point['chosen_part_index'] = 1
        instance = EmbeddedPart(part, conns, point)
        
        target_str = """Rwire0 a b  R=0
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings2(target_str, actual_str)

    def testRC_series(self):
        if self.just1: return
        part = self.lib.RC_series()

        instance = EmbeddedPart(part,
                                {'N1':'1', 'N2':'2'},
                                {'R':10.0e3, 'C':10.0e-6})
        
        target_str = """R0 1 n_auto_112  R=10000
C1 n_auto_112 2  C=1e-05
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings2(target_str, actual_str)

        #log.info(instance.spiceNetlistStr(True, True))
        
    def testCascodeDevice(self):
        if self.just1: return
        part = self.lib.cascodeDevice()
        self.assertEqual(sorted(part.externalPortnames()),
                         sorted(['D','S','loadrail','opprail']))

        #instantiate as biasedMos -- nmos
        conns = {'D':'1','S':'2', 'loadrail':'3','opprail':'4'}
        point = {'chosen_part_index':0,
                 'loadrail_is_vdd':1,
                 'Vgs':1.0,'Vds':1.0,
                 'L':1e-6,'Ids':1e-3,
                 'Vs':0}
        instance = EmbeddedPart(part, conns, point)
        target_str = """M0 1 n_auto_2 2 2 Nnom M=9 L=1e-06 W=1.98881e-06
V1 n_auto_2 0  DC 1
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str, actual_str)

        #log.info(instance.spiceNetlistStr(True, True))

        #instantiate as biasedMos -- pmos
        point = {'chosen_part_index':0,
                 'loadrail_is_vdd':0,
                 'Vgs':1.0,'Vds':1.0,
                 'L':1e-6,'Ids':1e-3,
                 'Vs':1.8}
        instance = EmbeddedPart(part, conns, point)
        target_str = """M0 1 XXX 2 2 Pnom M=8 L=1e-06 W=1.81982e-06
V1 n_auto_2 0  DC 0.8
"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str, actual_str)

        #log.info(instance.spiceNetlistStr(True, True))
                
    def testCascodeDeviceOrWire(self):
        if self.just1: return
        part = self.lib.cascodeDeviceOrWire()
        self.assertEqual(sorted(part.externalPortnames()),
                         sorted(['D','S','loadrail','opprail']))

        #instantiate as cascodeDevice
        instance0 = EmbeddedPart(part,
                                 {'D':'1', 'S':'2','loadrail':'3','opprail':'4'},
                                 {'chosen_part_index':0,
                                  'cascode_recurse':0,
                                  'loadrail_is_vdd':1,
                                  'Vgs':0.8,'Vds':1.0,
                                  'L':1e-6,'Ids':1e-3,
                                  'Vs':0}
                                 )
                                 
        target_str0 = """M0 1 n_auto_2 2 2 Nnom M=5 L=1e-06 W=4.83382e-06
V1 n_auto_2 0  DC 0.8
"""
        actual_str0 = instance0.spiceNetlistStr()
        self._compareStrings3(target_str0, actual_str0)

        #log.info(instance0.spiceNetlistStr(True, True))
        
        #instantiate as wire
        instance1 = EmbeddedPart(part,
                                 {'D':'1', 'S':'2','loadrail':'3','opprail':'4'},
                                 {'chosen_part_index':1,
                                  'cascode_recurse':0,
                                  'loadrail_is_vdd':1,
                                  'Vgs':0.8,'Vds':1.0,
                                  'L':1e-6,'Ids':1e-3,
                                  'Vs':0}
                                 )
        
        target_str1 = """Rwire0 1 2  R=0
"""
        actual_str1 = instance1.spiceNetlistStr()
        self._compareStrings(target_str1, actual_str1)

        #log.info(instance1.spiceNetlistStr(True, True))
        
    def testInputCascode_Stacked(self):
        if self.just1: return
        part = self.lib.inputCascode_Stacked()
        self.assertEqual(sorted(part.externalPortnames()),
                         sorted(['Vin', 'Iout', 'loadrail', 'opprail']))
        self.assertEqual(len(part.embedded_parts), 2)

        #'nmos version':
        #instantiate with input_is_pmos=False (ie input is nmos)
        # cascode_is_wire=0, degen_choice of biasedMos (1)
        #Remember: input_is_pmos=False means loadrail_is_Vdd=True; and vice-versa
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'Vdd','opprail':'gnd'}
        point = self._stackedPoint(input_is_pmos=False)
        instance = EmbeddedPart(part, conn, point)
        target_str = self._stackedNetlist(input_is_pmos=False)
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str, actual_str)

        #log.info(instance.spiceNetlistStr(True, True))

        #'pmos (upside-down) version':
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'gnd','opprail':'Vdd'}
        point = self._stackedPoint(input_is_pmos=True)
        instance = EmbeddedPart(part, conn, point)
        target_str = self._stackedNetlist(input_is_pmos=True)
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str, actual_str)

        #log.info(instance.spiceNetlistStr(True, True))

    def _stackedPoint(self, input_is_pmos):
        """For testing testInputCascode_Stacked"""
     
        if input_is_pmos==0:
            return { 'input_is_pmos':0,
                 'Ibias':1e-3, 'Vds': 1.0, 'Vs':0, 
                 'cascode_is_wire':0,
                 'cascode_L':1e-6,
                 'cascode_Vgs':0.7,
                 'cascode_recurse':0,
                 
                 'ampmos_Vgs':1.0,'ampmos_L':1e-6,'fracAmp':1.1/1.8,
                                  
                 }        
        else:
            return { 'input_is_pmos':1,
                 'Ibias':1e-3, 'Vds': 1.0, 'Vs':1.8, 
                
                 'cascode_is_wire':0,
                 'cascode_L':1e-6,
                 'cascode_Vgs':0.7,
                 'cascode_recurse':0,
                 
                 'ampmos_Vgs':1.0,'ampmos_L':1e-6,'fracAmp':1.1/1.8,
                                  
                 }    
            

    def _stackedNetlist(self, input_is_pmos):
        """For testing testInputCascode_Stacked"""
        if input_is_pmos:
            #all transistors should be pmos
            return """M0 Iout n_auto_18 n_auto_16 n_auto_16 Pnom M=22 L=1e-06 W=4.84575e-06
V1 n_auto_18 0  DC 0.488889
M2 n_auto_16 Vin Vdd Vdd Pnom M=22 L=1e-06 W=4.84575e-06
"""
        else:
            #all transistors should be nmos
            return """M0 Iout n_auto_15 n_auto_13 n_auto_13 Nnom M=5 L=1e-06 W=4.83382e-06
V1 n_auto_15 0  DC 1.31111
M2 n_auto_13 Vin gnd gnd Nnom M=5 L=1e-06 W=4.83382e-06
"""
        
    def testInputCascode_Folded(self):
        if self.just1: return
        part = self.lib.inputCascode_Folded()
        self.assertEqual(sorted(part.externalPortnames()),
                         sorted(['Vin', 'Iout', 'loadrail', 'opprail']))
        self.assertEqual(len(part.embedded_parts), 3)

        #instantiate with input_is_pmos=True, degen_choice of biasedMos (1)
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'Vdd','opprail':'gnd'}
        point = self._foldedPoint(input_is_pmos=True)
        instance = EmbeddedPart(part, conn, point)
        target_str = self._foldedNetlist(input_is_pmos=True)
        actual_str = instance.spiceNetlistStr()
        #log.info(instance.spiceNetlistStr(True, True))
        self._compareStrings3(target_str, actual_str)

        #instantiate with input_is_pmos=False
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'gnd','opprail':'Vdd'}
        point = self._foldedPoint(input_is_pmos=False)
        instance = EmbeddedPart(part, conn, point)
        target_str = self._foldedNetlist(input_is_pmos=False)
        actual_str = instance.spiceNetlistStr()
        #log.info(instance.spiceNetlistStr(True, True))
        self._compareStrings3(target_str, actual_str)

    def _foldedPoint(self, input_is_pmos):
        """For testing testInputCascode_Folded"""
        point={'input_is_pmos':'fillme', 'Vs':'fillme', 
                 'Ibias':1e-3, 'Ibias2':1e-3, 'Vds': 1.0, 
                 'cascode_is_wire':0,
                 'cascode_L':1e-6,
                 'cascode_Vgs':0.7,
                 'cascode_recurse':0,
                 
                 'ampmos_Vgs':0.8,'ampmos_L':1e-6,'fracAmp':1.1/1.8,
                                  
                 'inputbias_L':1e-6,
                 'inputbias_Vgs':0.6,
                 }   
                 
        if input_is_pmos==0:
            point['input_is_pmos']=0
            point['Vs']=0
            
            return point  
        else:
            point['input_is_pmos']=1
            point['Vs']=1.8
            
            return point     
    
    def _foldedNetlist(self, input_is_pmos):
        """For testing testInputCascode_Folded"""
        if input_is_pmos:
            return """M0 Iout n_auto_15 n_auto_14 n_auto_14 Nnom M=5 L=1e-06 W=4.83382e-06
V1 n_auto_15 0  DC 1.4
M2 n_auto_14 Vin Vdd Vdd Pnom M=22 L=1e-06 W=4.84575e-06
M3 n_auto_14 n_auto_16 gnd gnd Nnom M=10 L=1e-06 W=4.83382e-06
V4 n_auto_16 0  DC 0.6
"""
        else:
            return """M0 Iout n_auto_19 n_auto_18 n_auto_18 Pnom M=22 L=1e-06 W=4.84575e-06
V1 n_auto_19 0  DC 0.4
M2 n_auto_18 Vin gnd gnd Nnom M=5 L=1e-06 W=4.83382e-06
M3 n_auto_18 n_auto_20 Vdd Vdd Pnom M=43 L=1e-06 W=4.95845e-06
V4 n_auto_20 0  DC 1.2
"""
        
    def testInputCascodeFlex(self):
        if self.just1: return
        part = self.lib.inputCascodeFlex()

        assert len(part.point_meta) == \
                (len(self.lib.inputCascode_Folded().point_meta) + 1)

        #case 1: input_is_pmos=F, loadrail_is_Vdd=T
        # (therefore stacked, ie choice=0)
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'Vdd','opprail':'gnd'}
        point = self._inputCascodeFlex_Point(0, 0)
        instance = EmbeddedPart(part, conn, point)
        self._compareStrings3(self._stackedNetlist(0),
                              instance.spiceNetlistStr())

        #case 2: input_is_pmos=T, loadrail_is_Vdd=F
        # (therefore stacked, ie choice=0)
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'gnd','opprail':'Vdd'}
        point = self._inputCascodeFlex_Point(0, 1)
        instance = EmbeddedPart(part, conn, point)
        self._compareStrings3(self._stackedNetlist(1),
                              instance.spiceNetlistStr())

        #case 3: input_is_pmos=T, loadrail_is_Vdd=T
        # (therefore stacked, ie choice=0)
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'Vdd','opprail':'gnd'}
        point = self._inputCascodeFlex_Point(1, 1)
        instance = EmbeddedPart(part, conn, point)
        self._compareStrings3(self._foldedNetlist(1),
                              instance.spiceNetlistStr())

        #case 4: input_is_pmos=F, loadrail_is_Vdd=F
        # (therefore stacked, ie choice=0)
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'gnd','opprail':'Vdd'}
        point = self._inputCascodeFlex_Point(1, 0)
        instance = EmbeddedPart(part, conn, point)
        self._compareStrings3(self._foldedNetlist(0),
                              instance.spiceNetlistStr())

    def _inputCascodeFlex_Point(self, chosen_part_index, input_is_pmos):
        #can easily  generate this, because it only adds two vars
        # above and beyond a foldedPoint. 
        point = self._foldedPoint(input_is_pmos)
        point['chosen_part_index'] = chosen_part_index #add var
        point['cascode_is_wire'] = 0                   #add var
        return point
        
    def testInputCascodeStage(self):
        if self.just1: return
        part = self.lib.inputCascodeStage()
        self._testInputCascodeStage_and_SsViInput(part)
        
    def testSsViInput(self):
        if self.just1: return
        part = self.lib.inputCascodeStage()
        self._testInputCascodeStage_and_SsViInput(part)
        
    def _testInputCascodeStage_and_SsViInput(self, part):
        if self.just1: return

        #case 1: input_is_pmos=F, loadrail_is_Vdd=T
        # (therefore stacked)
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'Vdd','opprail':'gnd'}
        point = self._inputCascodeStage_Point(1, 0)
        instance = EmbeddedPart(part, conn, point)
        self._compareStrings3(self._stackedNetlist(0),
                              instance.spiceNetlistStr())

        #case 2: input_is_pmos=T, loadrail_is_Vdd=F
        # (therefore stacked)
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'gnd','opprail':'Vdd'}
        point = self._inputCascodeStage_Point(0, 1)
        instance = EmbeddedPart(part, conn, point)
        self._compareStrings3(self._stackedNetlist(1),
                              instance.spiceNetlistStr())

        #case 3: input_is_pmos=T, loadrail_is_Vdd=T
        # (therefore stacked)
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'Vdd','opprail':'gnd'}
        point = self._inputCascodeStage_Point(1, 1)
        instance = EmbeddedPart(part, conn, point)
        self._compareStrings3(self._foldedNetlist(1),
                              instance.spiceNetlistStr())

        #case 4: input_is_pmos=F, loadrail_is_Vdd=F
        # (therefore stacked)
        conn = {'Vin':'Vin', 'Iout':'Iout', 'loadrail':'gnd','opprail':'Vdd'}
        point = self._inputCascodeStage_Point(0, 0)
        instance = EmbeddedPart(part, conn, point)
        self._compareStrings3(self._foldedNetlist(0),
                              instance.spiceNetlistStr())

    def _inputCascodeStage_Point(self, loadrail_is_vdd, input_is_pmos):
        #can easily  generate this, because it only adds one var
        # above and beyond a foldedPoint, and removes another var
        dummy = 0
        point = self._inputCascodeFlex_Point(dummy, input_is_pmos)
        point['loadrail_is_vdd'] = loadrail_is_vdd #add var
        del point['chosen_part_index']             #remove var

        return point
                
    def testSsIiLoad(self):
        if self.just1: return
       
        part = self.lib.ssIiLoad()
        self.assertEqual(sorted(part.externalPortnames()),
                         sorted(['Iout','loadrail','opprail']))
        
        # loadrail_is_vdd=0, so parts are all nmos
        conns = {'Iout':'Iout','loadrail':'gnd','opprail':'Vdd'}
        point = {'loadrail_is_vdd':0,
                 'chosen_part_index':1, 
                 'Ibias':1e-3, 'Vds':0.8,'Vs':0,         
                 'L':1e-6,
                 'Vgs':1.0,
                 }

        instance = EmbeddedPart(part, conns, point)
                                 
        target_str = """M0 Iout n_auto_88 gnd gnd Nnom M=5 L=1e-06 W=4.83382e-06
V1 n_auto_88 0  DC 1
"""
        actual_str = instance.spiceNetlistStr()
        
        self._compareStrings3(target_str, actual_str)
        
        #simple-as-possible: 10K load, no cascoding
        conns = {'Iout':'Iout','loadrail':'Vdd','opprail':'gnd'}
        point = {'loadrail_is_vdd':0,
                 'chosen_part_index':0, 
                 'Ibias':1e-3, 'Vds':0.8,'Vs':0,         
                 'L':1e-6,
                 'Vgs':1.0,
                 }
        instance = EmbeddedPart(part, conns, point)
                                 
        target_str = """R0 Vdd Iout  R=800
"""
        actual_str = instance.spiceNetlistStr()
        
        self._compareStrings3(target_str, actual_str)

        #test Ibias railing
        conns = {'Iout':'Iout','loadrail':'Vdd','opprail':'gnd'}
        
        test_bias=self.lib.ss.max_Ires*1.1
        Vds=1.0
        
        point = {'loadrail_is_vdd':0,
                 'chosen_part_index':0, 
                 'Ibias':test_bias, 'Vds':Vds,'Vs':0,         
                 'L':1e-6,
                 'Vgs':1.0,
                 }
        instance = EmbeddedPart(part, conns, point)
                                 
        target_str = """R0 Vdd Iout  R=%g
""" % abs((Vds)/self.lib.ss.max_Ires)
        actual_str = instance.spiceNetlistStr()
        
        self._compareStrings2(target_str, actual_str)     
           
    def testSsViAmp1_simple(self):
        if self.just1: return
        part = self.lib.ssViAmp1()
        self.assertEqual(sorted(part.externalPortnames()),
                         sorted(['Vin','Iout','loadrail','opprail']))

        # Simple as possible: nmos input, 10K resistor load
        #  (no input cascode, no source degen, no load cascode)
        #  (loadrail_is_vdd=1, input_is_pmos=0)
        conns = {'Vin':'Vin','Iout':'Iout','loadrail':'Vdd','opprail':'gnd'}
        point = self._ssViAmp1_Point()
        instance = EmbeddedPart(part, conns, point)
                                 
        target_str = """Rwire0 Iout n_auto_103  R=0
M1 n_auto_103 Vin gnd gnd Nnom M=5 L=1e-06 W=4.83382e-06
R2 Vdd Iout  R=1000
"""
        actual_str = instance.spiceNetlistStr()
        
        self._compareStrings3(target_str, actual_str)
        
    def testSsViAmp1_complex(self):
        if self.just1: return
        part = self.lib.ssViAmp1()
        conns = {'Vin':'Vin','Iout':'Iout','loadrail':'gnd','opprail':'Vdd'}
        point = self._ssViAmp1_Point2(0,0)
        
        instance = EmbeddedPart(part, conns, point)
                                 
        target_str = """M0 Iout n_auto_17 n_auto_16 n_auto_16 Pnom M=13 L=1e-06 W=4.85917e-06
V1 n_auto_17 0  DC 0.7
M2 n_auto_16 Vin n_auto_15 n_auto_15 Nnom M=11 L=1e-06 W=4.98272e-06
R3 n_auto_15 gnd  R=100
M4 n_auto_16 n_auto_18 Vdd Vdd Pnom M=31 L=1e-06 W=4.92096e-06
V5 n_auto_18 0  DC 0.8
M6 n_auto_19 n_auto_20 gnd gnd Nnom M=8 L=1e-06 W=4.44066e-06
V7 n_auto_20 0  DC 0.8
M8 Iout n_auto_21 n_auto_19 n_auto_19 Nnom M=8 L=1e-06 W=4.44066e-06
V9 n_auto_21 0  DC 1.1
"""
        actual_str = instance.spiceNetlistStr()
        
        self._compareStrings3(target_str, actual_str)

    def testSsViAmp1_simpleWithIbiasRailingCheck(self):
        if self.just1: return
        
        part = self.lib.ssViAmp1()
        conns = {'Vin':'Vin','Iout':'Iout','loadrail':'Vdd','opprail':'gnd'}
        point = self._ssViAmp1_Point()
        
        test_bias=self.lib.ss.max_Ibias*10 #will this get railed correctly?
        Vout=1.0
        Vloadrail=1.8
        point['Ibias']=test_bias
        
        point['Vout']=Vout
        point['Vloadrail']=Vloadrail
        point['ampmos_L']=0.18e-6
        
        instance = EmbeddedPart(part, conns, point)
                                 
        target_str = """Rwire0 Iout XXX  R=0
M1 XXX Vin gnd gnd Nnom M=31 L=1.8e-07 W=4.94358e-06
R3 Vdd Iout  R=%g
""" % abs((Vout - Vloadrail)/self.lib.ss.max_Ibias)

        actual_str = instance.spiceNetlistStr()
        
        self._compareStrings2(target_str, actual_str)
          
        
    def _ssViAmp1_Point(self, loadrail_is_vdd=1, input_is_pmos=0):
        return  {'loadrail_is_vdd':loadrail_is_vdd,
                 'input_is_pmos':input_is_pmos,
                 
                 'Ibias':1e-3, 'Ibias2':1e-3,
                 'Vout': 1.0,
                 
                 #for input:            
                 'inputcascode_is_wire':1,
                 'inputcascode_L':1e-6,
                 'inputcascode_Vgs':1.0,
                 'inputcascode_recurse':0,
                 
                 'ampmos_Vgs':1.0,'ampmos_L':1e-6,'ampmos_fracAmp':1.1/1.8,
                                  
                 'inputbias_L':1e-6,
                 'inputbias_Vgs':1.0,             

                 #for load:
                 'load_part_index':0,         
                 'load_L':1e-6,
                 'load_Vgs':1.0,
                 }

    def _ssViAmp1_Point2(self, loadrail_is_vdd=1, input_is_pmos=0):
        return  {'loadrail_is_vdd':loadrail_is_vdd,
                 'input_is_pmos':input_is_pmos,
                 
                 'Ibias':1e-3, 'Ibias2':1e-3,
                 'Vout': 1.0,
                 
                 #for input:            
                 'inputcascode_is_wire':0,
                 'inputcascode_L':1e-6,
                 'inputcascode_Vgs':1.0,
                 'inputcascode_recurse':0,
                 
                 'ampmos_Vgs':1.0,'ampmos_L':1e-6,'ampmos_fracAmp':1.1/1.8,
                                  
                 'inputbias_L':1e-6,
                 'inputbias_Vgs':1.0,             

                 #for load:
                 'load_part_index':2,         
                 'load_L':1e-6,
                 'load_Vgs':1.0,
                 }

    def _ssViAmp1_VddGndPorts_Point(self):
        point = self._ssViAmp1_Point()
        point['chosen_part_index'] = point['loadrail_is_vdd']
        del point['loadrail_is_vdd']
        return point
    
    def testSsViAmp1_VddGndPorts_simple(self):
        if self.just1: return
        part = self.lib.ssViAmp1_VddGndPorts()
        self.assertEqual(sorted(part.externalPortnames()),
                         sorted(['Vin','Iout','Vdd','gnd']))

        conns = {'Vin':'Vin','Iout':'Iout','Vdd':'Vdd','gnd':'gnd'}
        point = self._ssViAmp1_VddGndPorts_Point()
        instance = EmbeddedPart(part, conns, point)
                                 
        target_str = """FIXME
        This should have a summary string at the top, looking like:
==== Summary for: ssViAmp1_VddGndPorts ====
* Ibias = blah
* Ibias2 = blah
* degen_choice (0=wire,1=resistor) = blah
* load type (0=resistor,1=biasedMos,2=ssIiLoad_Cascoded) = blah
==== Done summary ====

<<rest of netlist should go here>>
"""
        actual_str = instance.spiceNetlistStr()
        print actual_str
        
        #self._compareStrings3(target_str, actual_str)
        
        
    #=================================================================
    #Helper functions for these unit tests
    def _compareStrings(self, target_str_in, actual_str):
        """This is complex because it needs to take into account
        that auto-named nodes can have different values"""
        maxn = 30
        cand_xs = range(maxn)
        cand_ys = [-1]
        cand_zs = [-2]
        if 'yyy' in target_str_in: cand_ys = range(maxn)
        if 'zzz' in target_str_in: cand_zs = range(maxn)

        self.assertTrue( self._foundMatch(target_str_in, actual_str,
                                          cand_xs, cand_ys, cand_zs),
                         '\ntarget=\n[%s]\n\nactual=\n[%s]\n' %
                         (target_str_in, actual_str))

    def _foundMatch(self, target_str_in, actual_str, cand_xs, cand_ys, cand_zs):
        assert len(cand_xs)>0 and len(cand_ys)>0 and len(cand_zs)>0
        for nodenum_xxx in cand_xs:
            for nodenum_yyy in cand_ys:
                if nodenum_yyy == nodenum_xxx: continue
                for nodenum_zzz in cand_zs:
                    if nodenum_zzz == nodenum_xxx: continue
                    if nodenum_zzz == nodenum_yyy: continue
                    target_str = target_str_in.replace('xxx', str(nodenum_xxx))
                    target_str = target_str.replace('yyy', str(nodenum_yyy))
                    target_str = target_str.replace('zzz', str(nodenum_zzz))
                    if actual_str == target_str:
                        return True
        return False

    
    def _compareStrings2(self, target_str, actual_str):
        """Compres equality of two strings, but ignores the actual
        values in the auto-created nodenumbers"""
        self._compareStrings(replaceAutoNodesWithXXX(target_str),
                             replaceAutoNodesWithXXX(actual_str))

    def _compareStrings3(self, target_str, actual_str):
        """Like compareStrings2, but also on every line, ignores everything
        after the M=.
        This way, we don't have to tediously rewrite our netlist every
        time we make a change to the way that M, W, and L are calculated
        as a function of Ibias etc in operating point formulation.

        For the simpler circuits we should still be checking the W and L values.
        """
        self._compareStrings2(self.cleanLineWhitespace(replaceAfterMRWithBlank(target_str)),
                              self.cleanLineWhitespace(replaceAfterMRWithBlank(actual_str)))
    

    def cleanLineWhitespace(self, in_str):
        lines = in_str.splitlines()
        new_str = ""

        for l in lines:
            new_str += l.strip() + "\n"

        return new_str
    
    def tearDown(self):
        pass

        
    

if __name__ == '__main__':
    import logging
    logging.basicConfig()
    logging.getLogger('library').setLevel(logging.DEBUG)
    
    unittest.main()
