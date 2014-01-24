import unittest

import random
import string
import os
import time
import logging

from adts import *
from adts.Part import replaceAutoNodesWithXXX

from problems.OpLibrary import *
from problems.OpLibraryEMC import *
from problems.Library import replaceAfterMRWithBlank

#make this global for testing so we only have one disk access
DEVICES_SETUP = DevicesSetup('UMC180')
_GLOBAL_approx_mos_models = DEVICES_SETUP.approxMosModels()

log = logging.getLogger('problems')

class OpLibraryEMCTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

        self.lib = OpLibraryEMC(OpLibraryStrategy(DEVICES_SETUP))
        
        self.max_approx_error_relative = 0.25

    #=================================================================
    #One Test for each Part
    
    def testNode2(self):
        if self.just1: return
        part = self.lib.node2()
        self.assertEqual(part.externalPortnames(),['1','2','Vss'])
        instance = EmbeddedPart(part,{'1':'1','2':'2','Vss':'Vss'},{'C':1e-9})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0)
        
        target_str = """Rwire0 1 2  R=0\nC1 1 Vss  C=1e-09\n"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str,actual_str)
        
    def testNode3(self):
        if self.just1: return
        part = self.lib.node3()
        self.assertEqual(part.externalPortnames(),['1','2','3','Vss'])
        instance = EmbeddedPart(part,{'1':'1','2':'2','3':'3','Vss':'Vss'},{'C':1e-9})
        self.assertTrue( len(str(part)) > 0 )
        self.assertTrue( len(str(instance)) > 0)
        
        target_str = """Rwire0 1 2  R=0\nC1 1 Vss  C=1e-09\nRwire2 1 3  R=0\n"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str,actual_str)
        
    def testBranch(self):
        if self.just1: return
        part = self.lib.branch()
        conns = {'Vin':'Vin', 'Vout':'Vout', 'Drail':'Drail', 'Srail':'Srail'}
        point = {'cascode_do_stack':0,
                 'R':10.2e3,
                 'Vin':1.4,
                 'Vout':0.7,
                 'amp_L':1e-6,
                 'cascode_D_Vgs':1.0,
                 'cascode_D_L':1e-6,
                 'cascode_S_Vgs':1.0,
                 'cascode_S_L':1e-6,
                 'cascode_fracVi':0.5,
                 'Ibias':1e-4}
        
        point['index'] = 0
        instance = EmbeddedPart(part,conns,point)
        target_str = """Rwire0 Vin Vout  R=0\n"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str,actual_str)
        
        point['index'] = 1
        instance = EmbeddedPart(part,conns,point)
        target_str = """R0 Vin Vout  R=10200\n"""
        actual_str = instance.spiceNetlistStr()
        self._compareStrings(target_str,actual_str)
        
        point['index'] = 2
        point['Vin'] = 1.4
        point['Vout'] = 0.7
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Drail Vin Vout Vout Nnom\n"+\
                        "M1 Vout XXX Srail Srail Nnom\n"+\
                        "V2 XXX 0  DC 1\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['index'] = 2
        point['Vin'] = 0.7
        point['Vout'] = 1.4
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Drail Vin Vout Vout Pnom\n"+\
                        "M1 Vout XXX Srail Srail Pnom\n"+\
                        "V2 XXX 0  DC 0.8\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['index'] = 3
        point['Vin'] = 1.4
        point['Vout'] = 0.7
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Vout XXX Vin Vin Pnom\n"+\
                        "V1 XXX 0  DC 0.4\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['index'] = 3
        point['Vin'] = 0.7
        point['Vout'] = 1.4
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Vout XXX Vin Vin Nnom\n"+\
                        "V1 XXX 0  DC 1.7"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testBranchVddVss(self):
        if self.just1: return
        part = self.lib.branchVddVss()
        conns = {'Vin':'Vin', 'Vout':'Vout', 'Vdd':'Vdd', 'Vss':'Vss'}
        point = {'index':2,
                 'cascode_do_stack':0,
                 'R':10.2e3,
                 'Vin':1.4,
                 'Vout':0.7,
                 'amp_L':1e-6,
                 'cascode_D_Vgs':1.0,
                 'cascode_D_L':1e-6,
                 'cascode_S_Vgs':1.0,
                 'cascode_S_L':1e-6,
                 'cascode_fracVi':0.5,
                 'Ibias':1e-4}
        
        point['Drail_is_Vss'] = 0
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Vdd Vin Vout Vout Nnom\n"+\
                        "M1 Vout XXX Vss Vss Nnom\n"+\
                        "V2 XXX 0  DC 1"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['Drail_is_Vss'] = 1
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Vss Vin Vout Vout Nnom\n"+\
                        "M1 Vout XXX Vdd Vdd Nnom\n"+\
                        "V2 XXX 0  DC 1"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testNode2BranchNode3(self):
        if self.just1: return
        part = self.lib.node2BranchNode3()
        conns = {'Vin':'Vin', 'Vout1':'Vout1', 'Vout2':'Vout2', 'Vdd':'Vdd', 'Vss':'Vss'}
        point = {'in_C':1e-9,
                 'out_C':1e-9,
                 'branch_index':2,
                 'branch_cascode_do_stack':0,
                 'branch_R':10.2e3,
                 'branch_Vin':1.4,
                 'branch_Vout':0.7,
                 'branch_amp_L':1e-6,
                 'branch_cascode_D_Vgs':1.0,
                 'branch_cascode_D_L':1e-6,
                 'branch_cascode_S_Vgs':1.0,
                 'branch_cascode_S_L':1e-6,
                 'branch_cascode_fracVi':0.5,
                 'branch_Ibias':1e-4}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "Rwire0 Vin XXX\n"+\
                        "C1 Vin Vss  C=1e-09\n"+\
                        "M2 Vdd XXX XXX XXX Nnom\n"+\
                        "M3 XXX XXX Vss Vss Nnom\n"+\
                        "V4 XXX 0  DC 1\n"+\
                        "Rwire5 XXX Vout1\n"+\
                        "C6 XXX Vss  C=1e-09\n"+\
                        "Rwire7 XXX Vout2\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testNode3BranchNode2(self):
        if self.just1: return
        part = self.lib.node3BranchNode2()
        conns = {'Vin1':'Vin1', 'Vin2':'Vin2', 'Vout':'Vout', 'Vdd':'Vdd', 'Vss':'Vss'}
        point = {'in_C':1e-9,
                 'out_C':1e-9,
                 'branch_index':2,
                 'branch_cascode_do_stack':0,
                 'branch_R':10.2e3,
                 'branch_Vin':1.4,
                 'branch_Vout':0.7,
                 'branch_amp_L':1e-6,
                 'branch_cascode_D_Vgs':1.0,
                 'branch_cascode_D_L':1e-6,
                 'branch_cascode_S_Vgs':1.0,
                 'branch_cascode_S_L':1e-6,
                 'branch_cascode_fracVi':0.5,
                 'branch_Ibias':1e-4}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "Rwire0 Vin1 Vin2\n"+\
                        "C1 Vin1 Vss  C=1e-09\n"+\
                        "Rwire2 Vin1 XXX\n"+\
                        "M3 Vdd XXX XXX XXX Nnom\n"+\
                        "M4 XXX XXX Vss Vss Nnom\n"+\
                        "V5 XXX 0  DC 1\n"+\
                        "Rwire6 XXX Vout\n"+\
                        "C7 XXX Vss  C=1e-09\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testFlexNode3(self):
        if self.just1: return
        part = self.lib.flexNode3()
        conns = {'Vin':'Vin', 'Vio1':'Vio1', 'Vout':'Vout', 'Vdd':'Vdd', 'Vss':'Vss'}
        point = {'in_C':1e-9,
                 'out_C':1e-9,
                 'branch_index':2,
                 'branch_cascode_do_stack':0,
                 'branch_R':10.2e3,
                 'branch_Vin':1.4,
                 'branch_Vout':0.7,
                 'branch_amp_L':1e-6,
                 'branch_cascode_D_Vgs':1.0,
                 'branch_cascode_D_L':1e-6,
                 'branch_cascode_S_Vgs':1.0,
                 'branch_cascode_S_L':1e-6,
                 'branch_cascode_fracVi':0.5,
                 'branch_Ibias':1e-4}
        
        point['chosen_part_index'] = 0
        instance = EmbeddedPart(part,conns,point)
        target_str = "Rwire0 Vin XXX\n"+\
                        "C1 Vin Vss  C=1e-09\n"+\
                        "M2 Vdd XXX XXX XXX Nnom\n"+\
                        "M3 XXX XXX Vss Vss Nnom\n"+\
                        "V4 XXX 0  DC 1\n"+\
                        "Rwire5 XXX Vio1\n"+\
                        "C6 XXX Vss  C=1e-09\n"+\
                        "Rwire7 XXX Vout\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['chosen_part_index'] = 1
        instance = EmbeddedPart(part,conns,point)
        target_str = "Rwire0 Vin Vio1\n"+\
                        "C1 Vin Vss  C=1e-09\n"+\
                        "Rwire2 Vin XXX\n"+\
                        "M3 Vdd XXX XXX XXX Nnom\n"+\
                        "M4 XXX XXX Vss Vss Nnom\n"+\
                        "V5 XXX 0  DC 1\n"+\
                        "Rwire6 XXX Vout\n"+\
                        "C7 XXX Vss  C=1e-09\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
    
    def testCurrentMirrorEMC(self):
        if self.just1: return
        part = self.lib.currentMirrorEMC()
        conns = {'nin':'nin', 'nout':'nout', 'Vdd':'Vdd', 'Vss':'Vss'}
        point = {'Iin':1e-4,
                 'Vin':1.6,
                 'Vout':1.6,
                 'Vg2':0.8,
                 'chosen_part_index':0,
                 'C1':1e-9,
                 'C2':1e-9,
                 'branch_index':2,
                 'branch_amp_L':1e-6,
                 'branch_cascode_D_L':1e-6,
                 'branch_cascode_D_Vgs':0.8,
                 'branch_cascode_S_L':1e-6,
                 'branch_cascode_S_Vgs':0.8,
                 'branch_cascode_fracVi':0.5,
                 'branch_R':10.2e3,
                 'branch_Ibias':1e-4,
                 'ref_L':1e-6,
                 'out_L':1e-6
                 }
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "Rwire0 nin XXX\n"+\
                        "C1 nin Vss  C=1e-09\n"+\
                        "M2 Vdd XXX XXX XXX Nnom\n"+\
                        "M3 XXX XXX Vss Vss Nnom\n"+\
                        "V4 XXX 0  DC 0.8\n"+\
                        "Rwire5 XXX XXX\n"+\
                        "C6 XXX Vss  C=1e-09\n"+\
                        "Rwire7 XXX XXX\n"+\
                        "M8 nin XXX Vss Vss Nnom\n"+\
                        "M9 nout XXX Vss Vss Nnom\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testRail(self):
        if self.just1: return
        part = self.lib.rail()
        conns = part.unityPortMap()
        point = {}
        
        point['isVdd'] = False
        instance = EmbeddedPart(part,conns,point)
        target_str = "Rwire0 rail Vss" #TODO
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['isVdd'] = True
        instance = EmbeddedPart(part,conns,point)
        target_str = "Rwire0 rail Vdd"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    
    def testCascode_stacked(self):
        if self.just1: return
        part = self.lib.cascode_stacked()
        conns = part.unityPortMap()
        point = {'Vin':0.4,
                 'Vout':1.4,
                 'Ibias':1e-4,
                 'casc_Vgs':0.9,
                 'casc_L':0.36e-6,
                 'use_pmos':False
                 }
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 out XXX in in Nnom\n"+\
                        "V1 XXX 0  DC 1.3\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testCascode_folded(self):
        if self.just1: return
        part = self.lib.cascode_folded()
        conns = part.unityPortMap()
        point = {'Vin':0.4,
                 'Vout':1.4,
                 'Ibias':1e-4,
                 'casc_Vgs':0.9,
                 'casc_L':0.36e-6,
                 'bias_Vgs':0.8,
                 'bias_L':0.18e-6,
                 'use_pmos':False
                 }
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 out XXX in in Nnom\n"+\
                        "V1 XXX 0  DC 1.3\n"+\
                        "M2 in XXX XXX XXX Nnom\n"+\
                        "V3 XXX 0  DC 0.8\n"+\
                        "Rwire4 XXX Vss\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testCascode(self):
        if self.just1: return
        part = self.lib.cascode()
        conns = part.unityPortMap()
        point = {'Vin':0.4,
                 'Vout':1.4,
                 'Ibias':1e-4,
                 'casc_Vgs':0.9,
                 'casc_L':0.36e-6,
                 'bias_Vgs':0.8,
                 'bias_L':0.18e-6,
                 'use_pmos':False}
        
        point['fold'] = 0
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 out XXX in in Nnom\n"+\
                        "V1 XXX 0  DC 1.3\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['fold'] = 1
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 out XXX in in Nnom\n"+\
                        "V1 XXX 0  DC 1.3\n"+\
                        "M2 in XXX XXX XXX Nnom\n"+\
                        "V3 XXX 0  DC 0.8\n"+\
                        "Rwire4 XXX Vss\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
    
    def testCascodeOrWire(self):
        if self.just1: return
        part = self.lib.cascodeOrWire()
        conns = part.unityPortMap()
        point = {'Vin':0.4,
                 'Vout':1.4,
                 'Ibias':1e-4,
                 'casc_Vgs':0.9,
                 'casc_L':0.36e-6,
                 'bias_Vgs':0.8,
                 'bias_L':0.18e-6,
                 'use_pmos':False,
                 'fold':False}
        
        point['cascode'] = False
        instance = EmbeddedPart(part,conns,point)
        target_str = "Rwire0 in out"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['cascode'] = True
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 out XXX in in Nnom\n"+\
                        "V1 XXX 0  DC 1.3"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
    
    def testInputPair_classic(self):
        if self.just1: return
        part = self.lib.inputPair_classic()
        conns = part.unityPortMap()
        point = {'Ibias':1e-4,
                 'Vinop':1.3,
                 'Voutop':0.8,
                 'use_pmos':False,
                 'amp_Vgs':0.9,
                 'amp_L':0.36e-6,
                 'bias_Vgs':0.9,
                 'bias_L':0.18e-6}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Iout1 Vin1 XXX XXX Nnom\n"+\
                        "M1 Iout2 Vin2 XXX XXX Nnom\n"+\
                        "M2 XXX XXX XXX XXX Nnom\n"+\
                        "V3 XXX 0  DC 0.9\n"+\
                        "C4 XXX Vdd  C=3e-13\n"+\
                        "Rwire5 XXX Vss"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testInputPair_Redoute(self):
        if self.just1: return
        part = self.lib.inputPair_Redoute()
        conns = part.unityPortMap()
        point = {'Ibias':1e-4,
                 'Vinop':1.3,
                 'Voutop':0.8,
                 'use_pmos':False,
                 'amp_Vgs':0.9,
                 'amp_L':0.36e-6,
                 'bias_Vgs':0.9,
                 'bias_L':0.18e-6}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Iout1 Vin1 XXX XXX Nnom\n"+\
                        "M1 Iout2 Vin2 XXX XXX Nnom\n"+\
                        "M2 XXX Vin1 XXX XXX Nnom\n"+\
                        "M3 XXX Vin2 XXX XXX Nnom\n"+\
                        "M4 XXX XXX XXX XXX Nnom\n"+\
                        "V5 XXX 0  DC 0.9\n"+\
                        "M6 XXX XXX XXX XXX Nnom\n"+\
                        "V7 XXX 0  DC 0.9\n"+\
                        "C8 XXX Vdd  C=6e-13\n"+\
                        "Rwire9 XXX Vdd\n"+\
                        "Rwire10 XXX Vss\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testInputPair_Fiori(self):
        if self.just1: return
        part = self.lib.inputPair_Fiori()
        conns = part.unityPortMap()
        point = {'Ibias':1e-4,
                 'Vinop':1.3,
                 'Voutop':0.8,
                 'use_pmos':False,
                 'amp_Vgs':0.9,
                 'amp_L':0.36e-6,
                 'bias_Vgs':0.9,
                 'bias_L':0.18e-6}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Iout1 Vin1 XXX XXX Nnom\n"+\
                        "M1 Iout2 Vin2 XXX XXX Nnom\n"+\
                        "M2 Iout2 Vin1 XXX XXX Nnom\n"+\
                        "M3 Iout1 Vin2 XXX XXX Nnom\n"+\
                        "M4 XXX XXX XXX XXX Nnom\n"+\
                        "V5 XXX 0  DC 0.9\n"+\
                        "M6 XXX XXX XXX XXX Nnom\n"+\
                        "V7 XXX 0  DC 0.9\n"+\
                        "C8 XXX Vdd  C=3e-13\n"+\
                        "C9 XXX Vdd  C=3e-13\n"+\
                        "Rwire10 XXX Vss\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testInputPair(self):
        if self.just1: return
        part = self.lib.inputPair()
        conns = part.unityPortMap()
        point = {'Ibias':1e-4,
                 'Vinop':1.3,
                 'Voutop':0.8,
                 'use_pmos':False,
                 'amp_Vgs':0.9,
                 'amp_L':0.36e-6,
                 'bias_Vgs':0.9,
                 'bias_L':0.18e-6}
        
        point['input'] = 0
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Iout1 Vin1 XXX XXX Nnom\n"+\
                        "M1 Iout2 Vin2 XXX XXX Nnom\n"+\
                        "M2 XXX XXX XXX XXX Nnom\n"+\
                        "V3 XXX 0  DC 0.9\n"+\
                        "C4 XXX Vdd  C=3e-13\n"+\
                        "Rwire5 XXX Vss\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['input'] = 1
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Iout1 Vin1 XXX XXX Nnom\n"+\
                        "M1 Iout2 Vin2 XXX XXX Nnom\n"+\
                        "M2 XXX Vin1 XXX XXX Nnom\n"+\
                        "M3 XXX Vin2 XXX XXX Nnom\n"+\
                        "M4 XXX XXX XXX XXX Nnom\n"+\
                        "V5 XXX 0  DC 0.9\n"+\
                        "M6 XXX XXX XXX XXX Nnom\n"+\
                        "V7 XXX 0  DC 0.9\n"+\
                        "C8 XXX Vdd  C=6e-13\n"+\
                        "Rwire9 XXX Vdd\n"+\
                        "Rwire10 XXX Vss\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['input'] = 2
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 Iout1 Vin1 XXX XXX Nnom\n"+\
                        "M1 Iout2 Vin2 XXX XXX Nnom\n"+\
                        "M2 Iout2 Vin1 XXX XXX Nnom\n"+\
                        "M3 Iout1 Vin2 XXX XXX Nnom\n"+\
                        "M4 XXX XXX XXX XXX Nnom\n"+\
                        "V5 XXX 0  DC 0.9\n"+\
                        "M6 XXX XXX XXX XXX Nnom\n"+\
                        "V7 XXX 0  DC 0.9\n"+\
                        "C8 XXX Vdd  C=3e-13\n"+\
                        "C9 XXX Vdd  C=3e-13\n"+\
                        "Rwire10 XXX Vss\n"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testCascodedInputPair(self):
        if self.just1: return
        part = self.lib.cascodedInputPair()
        conns = part.unityPortMap()
        point = {'Ibias':1e-4,
                 'Vinop':1.3,
                 'input_Voutop':0.8,
                 'input_is_pmos':False,
                 'amp_Vgs':0.9,
                 'amp_L':0.36e-6,
                 'bias_Vgs':0.9,
                 'bias_L':0.18e-6,
                 'input':0,
                 'Vout':1.1,
                 'casc_Vgs':0.8,
                 'casc_L':0.36e-6,
                 'load_is_Vss':False,
                 'cascode':True,
                 'fold':False}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 XXX Vin1 XXX XXX Nnom\n"+\
                        "M1 XXX Vin2 XXX XXX Nnom\n"+\
                        "M2 XXX XXX XXX XXX Nnom\n"+\
                        "V3 XXX 0  DC 0.9\n"+\
                        "C4 XXX Vdd  C=3e-13\n"+\
                        "Rwire5 XXX Vss\n"+\
                        "M6 Iout1 XXX XXX XXX Nnom\n"+\
                        "V7 XXX 0  DC 1.6\n"+\
                        "M8 Iout2 XXX XXX XXX Nnom\n"+\
                        "V9 XXX 0  DC 1.6"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testCascodedCurrentMirror(self):
        if self.just1: return
        part = self.lib.cascodedCurrentMirror()
        conns = part.unityPortMap()
        point = {'frac':0.5,
                 'use_pmos':False,
                 'Iin':1e-4,
                 'Iout':1e-4,
                 'Vds_in':0.9,
                 'Vds_out':0.4,
                 'L':0.36e-6,
                 'casc_Vgs':0.9,
                 'casc_L':0.36e-6,
                 'cascode':True}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 XXX Iref XXX XXX Nnom\n"+\
                     "M1 miller Iref XXX XXX Nnom\n"+\
                     "M2 Iref XXX XXX XXX Nnom\n"+\
                     "V3 XXX 0  DC 1.35\n"+\
                     "M4 Iout XXX miller miller Nnom\n"+\
                     "V5 XXX 0  DC 1.1\n"+\
                     "Rwire6 XXX Vss"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testDsAmp1(self):
        if self.just1: return
        part = self.lib.dsAmp1()
        conns = part.unityPortMap()
        point = {'Ibias':1e-4,
                 'Vinop':1.3,
                 'input_Voutop':0.8,
                 'input_is_pmos':False,
                 'amp_Vgs':0.9,
                 'amp_L':0.36e-6,
                 'bias_Vgs':0.9,
                 'bias_L':0.18e-6,
                 'input':0,
                 'Vout':1.1,
                 'casc_Vgs':0.8,
                 'casc_L':0.36e-6,
                 'load_is_Vss':False,
                 'cascode':True,
                 'fold':False,
                 'cm_frac':0.5,
                 'cm_L':0.36e-6}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 XXX Vin1 XXX XXX Nnom\n"+\
                     "M1 XXX Vin2 XXX XXX Nnom\n"+\
                     "M2 XXX XXX XXX XXX Nnom\n"+\
                     "V3 XXX 0  DC 0.9\n"+\
                     "C4 XXX Vdd  C=3e-13\n"+\
                     "Rwire5 XXX Vss\n"+\
                     "M6 XXX XXX XXX XXX Nnom\n"+\
                     "V7 XXX 0  DC 1.6\n"+\
                     "M8 out XXX XXX XXX Nnom\n"+\
                     "V9 XXX 0  DC 1.6\n"+\
                     "M10 XXX XXX XXX XXX Pnom\n"+\
                     "M11 miller XXX XXX XXX Pnom\n"+\
                     "M12 XXX XXX XXX XXX Pnom\n"+\
                     "V13 XXX 0  DC 0.65\n"+\
                     "M14 out XXX miller miller Pnom\n"+\
                     "V15 XXX 0  DC 0.65\n"+\
                     "Rwire16 XXX Vdd"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testSsAmp1(self):
        if self.just1: return
        part = self.lib.ssAmp1()
        conns = part.unityPortMap()
        point = {'Ibias':1e-4,
                 'use_pmos':False,
                 'Vout':0.9,
                 'amp_Vgs':0.9,
                 'amp_L':0.36e-6,
                 'bias_Vgs':0.9,
                 'bias_L':0.36e-6,
                 'casc_Vgs':0.9,
                 'casc_L':0.36e-6,
                 'cascode':True,
                 'amp_frac':0.5,
                 'bias_frac':0.5}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 XXX Vin XXX XXX Nnom\n"+\
                     "M1 out XXX XXX XXX Nnom\n"+\
                     "V2 XXX 0  DC 1.35\n"+\
                     "M3 XXX XXX XXX XXX Pnom\n"+\
                     "V4 XXX 0  DC 0.9\n"+\
                     "M5 out XXX XXX XXX Pnom\n"+\
                     "V6 XXX 0  DC 0.45\n"+\
                     "Rwire7 XXX Vss\n"+\
                     "Rwire8 XXX Vdd"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testSsAmp1OrWire(self):
        if self.just1: return
        part = self.lib.ssAmp1OrWire()
        conns = part.unityPortMap()
        point = {'Ibias':1e-4,
                 'use_pmos':False,
                 'Vout':0.9,
                 'amp_Vgs':0.9,
                 'amp_L':0.36e-6,
                 'bias_Vgs':0.9,
                 'bias_L':0.36e-6,
                 'casc_Vgs':0.9,
                 'casc_L':0.36e-6,
                 'cascode':True,
                 'amp_frac':0.5,
                 'bias_frac':0.5}
        
        point['output_stage']=True
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 XXX Vin XXX XXX Nnom\n"+\
                     "M1 out XXX XXX XXX Nnom\n"+\
                     "V2 XXX 0  DC 1.35\n"+\
                     "M3 XXX XXX XXX XXX Pnom\n"+\
                     "V4 XXX 0  DC 0.9\n"+\
                     "M5 out XXX XXX XXX Pnom\n"+\
                     "V6 XXX 0  DC 0.45\n"+\
                     "Rwire7 XXX Vss\n"+\
                     "Rwire8 XXX Vdd"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
        point['output_stage']=False
        instance = EmbeddedPart(part,conns,point)
        target_str = "Rwire0 Vin out"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
        
    def testDsAmp2(self):
        if self.just1: return
        part = self.lib.dsAmp2()
        conns = part.unityPortMap()
        point = {'stage1_Ibias':1e-4,
                 'stage1_Vinop':1.3,
                 'stage1_input_Voutop':0.8,
                 'stage1_input_is_pmos':False,
                 'stage1_amp_Vgs':0.9,
                 'stage1_amp_L':0.36e-6,
                 'stage1_bias_Vgs':0.9,
                 'stage1_bias_L':0.18e-6,
                 'stage1_input':0,
                 'stage1_Vout':1.1,
                 'stage1_casc_Vgs':0.8,
                 'stage1_casc_L':0.36e-6,
                 'stage1_load_is_Vss':False,
                 'stage1_cascode':True,
                 'stage1_fold':False,
                 'stage1_cm_frac':0.5,
                 'stage1_cm_L':0.36e-6,
                 'stage2_output_stage':True,
                 'stage2_Ibias':1e-4,
                 'stage2_use_pmos':False,
                 'stage2_Vout':0.9,
                 'stage2_amp_L':0.36e-6,
                 'stage2_bias_Vgs':0.9,
                 'stage2_bias_L':0.36e-6,
                 'stage2_casc_Vgs':0.9,
                 'stage2_casc_L':0.36e-6,
                 'stage2_cascode':True,
                 'stage2_amp_frac':0.5,
                 'stage2_bias_frac':0.5,
                 'miller_C':1e-9}
        
        instance = EmbeddedPart(part,conns,point)
        target_str = "M0 XXX Vin1 XXX XXX Nnom\n"+\
                     "M1 XXX Vin2 XXX XXX Nnom\n"+\
                     "M2 XXX XXX XXX XXX Nnom\n"+\
                     "V3 XXX 0  DC 0.9\n"+\
                     "C4 XXX Vdd  C=3e-13\n"+\
                     "Rwire5 XXX Vss\n"+\
                     "M6 XXX XXX XXX XXX Nnom\n"+\
                     "V7 XXX 0  DC 1.6\n"+\
                     "M8 XXX XXX XXX XXX Nnom\n"+\
                     "V9 XXX 0  DC 1.6\n"+\
                     "M10 XXX XXX XXX XXX Pnom\n"+\
                     "M11 XXX XXX XXX XXX Pnom\n"+\
                     "M12 XXX XXX XXX XXX Pnom\n"+\
                     "V13 XXX 0  DC 0.65\n"+\
                     "M14 XXX XXX XXX XXX Pnom\n"+\
                     "V15 XXX 0  DC 0.65\n"+\
                     "Rwire16 XXX Vdd\n"+\
                     "M17 XXX XXX XXX XXX Nnom\n"+\
                     "M18 out XXX XXX XXX Nnom\n"+\
                     "V19 XXX 0  DC 1.35\n"+\
                     "M20 XXX XXX XXX XXX Pnom\n"+\
                     "V21 XXX 0  DC 0.9\n"+\
                     "M22 out XXX XXX XXX Pnom\n"+\
                     "V23 XXX 0  DC 0.45\n"+\
                     "Rwire24 XXX Vss\n"+\
                     "Rwire25 XXX Vdd\n"+\
                     "C26 XXX out  C=1e-09"
        actual_str = instance.spiceNetlistStr()
        self._compareStrings3(target_str,actual_str)
    
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
