"""OpLibraryEMC.py

A library of blocks that uses operating-point-driven formulation (as opposed to
device sizes).
"""
import copy
import logging
import math
import os
import pickle
import sys
import types

import numpy

from adts import *
from problems.Library import whoami, Library, saturationSimulationDOCs
from problems.OpLibrary import ApproxMosModels, OpLibrary # for the device sizer
from regressor.Luc import LucStrategy, LucModel, LucFactory
from regressor.PointRegressor import PointRegressor
from util import mathutil
from util.constants import REGION_SATURATION
from util.ascii import asciiRowToStrings, asciiTo2dArray, \
     hdrValFilesToTrainingData

log = logging.getLogger('library')
                
class OpLibraryEMC(OpLibrary):
    """
    @description

      OpLibraryEMC is a subclass of OpLibrary and inherits all parts of that
      library.
      
    @attributes

      ss -- LibraryStrategy object --
    """

    def __init__(self, ss):
        """
        @description

          Constructor.
          
        """
        OpLibrary.__init__(self, ss)
        self.parcap_value = 3e-13
        
        mn, mx = math.log10(float(1e-15)), math.log10(float(1e-9))
        self._ref_varmetas['C_small'] = ContinuousVarMeta(True, mn, mx, 'C')
    
    def node2(self):
        """
        Description: equipotential node with 2 terminals and a capacitance to ground
        Ports: 1,2,Vss
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        wire_part = self.wire()
        capacitor_part = self.smallCapacitor()
        capacitor_varmap = capacitor_part.unityVarMap()
        
        #build the point_meta
        pm = PointMeta({})
        pm = self.updatePointMeta(pm,capacitor_part,capacitor_varmap)
        
        #build the main part
        part = CompoundPart(['1','2','Vss'],pm,name)
        part.addPart(wire_part,{'1':'1','2':'2'},{})
        part.addPart(capacitor_part,{'1':'1','2':'Vss'},capacitor_varmap)

        self._parts[name] = part
        return part
    
    def node3(self):
        """
        Description: equipotential node with 3 terminals and a capacitance to ground
        Ports: 1,2,3,Vss
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        node2_part = self.node2()
        node2_varmap = node2_part.point_meta.unityVarMap()
        wire_part = self.wire()
        
        #build the point_meta
        pm = PointMeta({})
        pm = self.updatePointMeta(pm,node2_part,node2_varmap)
        
        #build the main part
        part = CompoundPart(['1','2','3','Vss'], pm, name=name)
        part.addPart(node2_part,{'1':'1','2':'2','Vss':'Vss'},node2_varmap)
        part.addPart(wire_part,{'1':'1','2':'3'},{})

        self._parts[name] = part
        return part
    
    def branch(self):
        """
        Description: flex part which can select between a wire, a resistor, a level shifter or cascode 
        Ports: Vin, Vout, Drail, Srail
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        wire_part = self.wire()
        resistor_part = self.sizedLogscaleResistor()
        levelShifter_part = self.levelShifter()
        stackedCascodeMos_part = self.stackedCascodeMos()
        
        #build the point_meta
        pm = self.buildPointMeta({'cascode_do_stack':'bool_var',
                                  'R':'logscale_R',
                                  'Vin':'Vds','Vout':'Vds',
                                  'amp_L':'L',
                                  'cascode_D_Vgs':'Vgs', 'cascode_D_L':'L',
                                  'cascode_S_Vgs':'Vgs', 'cascode_S_L':'L',
                                  'cascode_fracVi':'frac',
                                  'Ibias':'Ibias'})
        
        #build functions
        vss, vdd = str(self.ss.vss), str(self.ss.vdd)
        levelShifter_functions = levelShifter_part.point_meta.unityVarMap()
        levelShifter_functions['Drail_is_vdd'] = '(Vout < Vin)'
        
        stackedCascodeMos_functions = {'chosen_part_index':'cascode_do_stack',
                                       'use_pmos':'(Vout<Vin)',
                                       'Vds':"abs(Vout-Vin)",
                                       'Vs':'Vin',
                                       'D_L':'cascode_D_L',
                                       'D_Vgs':'cascode_D_Vgs',
                                       'fracVi':'cascode_fracVi',
                                       'S_L':'cascode_S_L',
                                       'S_Vgs':'cascode_S_Vgs',
                                       'Ids':'Ibias'}
        
        #build the main part
        part = FlexPart(['Vin','Vout','Drail','Srail'], pm, name,'index')
        part.addPartChoice(wire_part, {'1':'Vin','2':'Vout'}, {})
        part.addPartChoice(resistor_part,{'1':'Vin','2':'Vout'},
                           resistor_part.unityVarMap())
        part.addPartChoice(levelShifter_part,levelShifter_part.unityPortMap(),
                           levelShifter_functions)
        part.addPartChoice(stackedCascodeMos_part,{'S':'Vin','D':'Vout'},
                           stackedCascodeMos_functions)
        
        self._parts[name] = part
        return part
    
    def branchVddVss(self):
        """
        Description: a branch with Vdd and Vss ports instead of Drail and Srail
        Ports: Vin, Vout, Vdd, Vss
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        branch_part = self.branch()
        
        #build the point_meta (pm)
        pm = PointMeta({})
        branch_varmap = branch_part.unityVarMap()
        pm = self.updatePointMeta(pm, branch_part, branch_varmap, True)
                                  
        #build the main part
        part = FlexPart(['Vin','Vout','Vdd','Vss'], pm, name,'Drail_is_Vss')
        part.addPartChoice(branch_part,
                           {'Vin':'Vin','Vout':'Vout',
                            'Drail':'Vdd','Srail':'Vss'},
                           branch_varmap)
        part.addPartChoice(branch_part,
                           {'Vin':'Vin','Vout':'Vout',
                            'Drail':'Vss','Srail':'Vdd'},
                            branch_varmap)
                
        self._parts[name] = part
        return part
    
    def node2BranchNode3(self):
        """
        Description: compound part consisting of a branch with a node2 connected to
          Vin and a node3 connected to Vout1 and vout2 
        Ports: Vin, Vout1, Vout2, Vdd, Vss
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        node2_part = self.node2()
        branch_part = self.branchVddVss()
        node3_part = self.node3()
        
        #build the point_meta
        pm = PointMeta({})
        node2_varmap = {'C':'in_C'}
        pm = self.updatePointMeta(pm,node2_part,node2_varmap)
        
        branch_varmap = {}
        for old_name in branch_part.point_meta.keys():
            branch_varmap[old_name] = 'branch_' + old_name
        branch_varmap['Drail_is_Vss'] = 'IGNORE'
        pm = self.updatePointMeta(pm,branch_part,branch_varmap)
        
        node3_varmap = {'C':'out_C'}
        pm = self.updatePointMeta(pm,node3_part,node3_varmap)
        
        #build functions
        branch_functions = branch_varmap
        branch_functions['Drail_is_Vss'] = "switchAndEval(branch_index, {" + \
                                                            "2:'(branch_Vout > branch_Vin)', " + \
                                                            "3:'(branch_Vout < branch_Vin)', " + \
                                                            "'default':'0'})"
        
        #build the main part
        part = CompoundPart(['Vin','Vout1','Vout2','Vdd','Vss'], pm, name)
        
        branch_in = part.addInternalNode()
        branch_out = part.addInternalNode()
        
        part.addPart(node2_part,{'1':'Vin','2':branch_in,'Vss':'Vss'},
                     node2_varmap)
        part.addPart(branch_part,{'Vin':branch_in,'Vout':branch_out,
                                  'Vdd':'Vdd','Vss':'Vss'},branch_functions)
        part.addPart(node3_part,{'1':branch_out,'2':'Vout1','3':'Vout2','Vss':'Vss'},
                     node3_varmap)
        
        self._parts[name] = part
        return part
    
    def node3BranchNode2(self):
        """
        Description: compound part consisting of a branch with a node3 connected to
          Vin1 and Vin2 and a node2 connected to Vout
        Ports: Vin1, Vin2, Vout Vdd, Vss
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        node3_part = self.node3()
        branch_part = self.branchVddVss()
        node2_part = self.node2()
        
        #build the point_meta
        pm = PointMeta({})
        node3_varmap = {'C':'in_C'}
        pm = self.updatePointMeta(pm,node3_part,node3_varmap)
        
        branch_varmap = {}
        for old_name in branch_part.point_meta.keys():
            branch_varmap[old_name] = 'branch_' + old_name
        branch_varmap['Drail_is_Vss'] = 'IGNORE'
        pm = self.updatePointMeta(pm,branch_part,branch_varmap)
        
        node2_varmap = {'C':'out_C'}
        pm = self.updatePointMeta(pm,node2_part,node2_varmap)
        
        #build functions
        branch_functions = branch_varmap
        branch_functions['Drail_is_Vss'] = "switchAndEval(branch_index, {" + \
                                                            "2:'(branch_Vout > branch_Vin)', " + \
                                                            "3:'(branch_Vout < branch_Vin)', " + \
                                                            "'default':'0'})"
        
        #build the main part
        part = CompoundPart(['Vin1', 'Vin2','Vout','Vdd','Vss'], pm, name)
        
        branch_in = part.addInternalNode()
        branch_out = part.addInternalNode()
        
        part.addPart(node3_part,{'1':'Vin1','2':'Vin2','3':branch_in,'Vss':'Vss'},
                     node3_varmap)
        part.addPart(branch_part,{'Vin':branch_in,'Vout':branch_out,
                                  'Vdd':'Vdd','Vss':'Vss'},branch_functions)
        part.addPart(node2_part,{'1':branch_out,'2':'Vout','Vss':'Vss'},
                     node2_varmap)
        
        self._parts[name] = part
        return part
    
    def flexNode3(self):
        """
        Description: a flex component that can be either a node2BranchNode3 or
          a node3BranchNode2. While the ports Vin and Vout have a fixed signal
          type, the port vio1 can be either input or output. 
        Ports: Vin,Vio1,Vout,Vdd,Vss
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        n2bn3_part = self.node2BranchNode3()
        n3bn2_part = self.node3BranchNode2()
        
        #build the point_meta
        emb_varmap = n2bn3_part.unityVarMap()
        pm = PointMeta({})
        pm = self.updatePointMeta(pm,n2bn3_part,emb_varmap)
        
        #build the main part
        part = FlexPart(['Vin','Vio1','Vout','Vdd','Vss'],pm,name)
        n2bn3_portmap = {'Vin':'Vin', 'Vout1':'Vio1', 'Vout2':'Vout', 'Vdd':'Vdd', 'Vss':'Vss'}
        part.addPartChoice(n2bn3_part,n2bn3_portmap,emb_varmap)
        n3bn2_portmap = {'Vin1':'Vin', 'Vin2':'Vio1', 'Vout':'Vout', 'Vdd':'Vdd', 'Vss':'Vss'}
        part.addPartChoice(n3bn2_part,n3bn2_portmap,emb_varmap)
        
        self._parts[name] = part
        return part
    
    
    
    def currentMirrorEMC(self):
        """
        Description: a current mirror with a flexNode3.
        Ports: nin, nout, Vdd, Vss
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        M1_part = self.saturatedMos3()
        M2_part = self.saturatedMos3()
        flexNode3_part = self.flexNode3()
        
        #build the point_meta
        flexNode3_varmap = {'chosen_part_index':'chosen_part_index',
                            'in_C':'C1',
                            'out_C':'C2',
                            'branch_index':'branch_index',
                            'branch_cascode_do_stack':'IGNORE',
                            'branch_R':'branch_R',
                            'branch_Vin':'IGNORE',
                            'branch_Vout':'IGNORE',
                            'branch_amp_L':'branch_amp_L',
                            'branch_cascode_D_Vgs':'branch_cascode_D_Vgs',
                            'branch_cascode_D_L':'branch_cascode_D_L',
                            'branch_cascode_S_Vgs':'branch_cascode_S_Vgs',
                            'branch_cascode_S_L':'branch_cascode_S_L',
                            'branch_cascode_fracVi':'branch_cascode_fracVi',
                            'branch_Ibias':'branch_Ibias'
                            }
        pm = PointMeta({})
        pm = self.updatePointMeta(pm, flexNode3_part, flexNode3_varmap, True)
        pm['Iin'] = self.buildVarMeta('Ibias','Iin')
        pm['Vin'] = self.buildVarMeta('V','Vin')
        pm['Vout'] = self.buildVarMeta('V','Vout')
        pm['ref_L'] = self.buildVarMeta('L','ref_L')
        pm['out_L'] = self.buildVarMeta('L','out_L')
        pm['Vg2'] = self.buildVarMeta('Vgs','Vg2')
        
        #build the functions
        flexNode3_functions = flexNode3_varmap
        flexNode3_functions['branch_cascode_do_stack'] = '0'
        flexNode3_functions['branch_Vin'] = 'Vin'
        flexNode3_functions['branch_Vout'] = 'Vg2'
        M1_functions = {'Ids':'Iin',
                        'Vgs':"switchAndEval(chosen_part_index, {" + \
                                "0:'Vg2', " + \
                                "1:'Vin'})",
                        'Vds':'Vin',
                        'L':'ref_L',
                        'use_pmos':'0'
                        }
        M2_functions = {'Ids':'Iin',
                        'Vgs':'Vg2',
                        'Vds':'Vout',
                        'L':'out_L',
                        'use_pmos':'0'
                        }
        
        #build the main part
        part = CompoundPart(['nin', 'nout', 'Vdd', 'Vss'], pm, name)
        ng1 = part.addInternalNode()
        ng2 = part.addInternalNode()
        flexNode3_portmap = {'Vin':'nin','Vio1':ng1,'Vout':ng2,'Vdd':'Vdd','Vss':'Vss'}
        part.addPart(flexNode3_part,flexNode3_portmap,flexNode3_functions)
        part.addPart(M1_part,{'D':'nin','G':ng1,'S':'Vss'},M1_functions)
        part.addPart(M2_part,{'D':'nout','G':ng2,'S':'Vss'},M2_functions)
        
        self._parts[name] = part
        
        part.addToSummaryStr('Iin','Iin')
        part.addToSummaryStr('Vin','Vin')
        part.addToSummaryStr('Vg2','Vg2')
        part.addToSummaryStr('chosen_part_index','chosen_part_index')
        part.addToSummaryStr('C1','C1')
        part.addToSummaryStr('C2','C2')
        part.addToSummaryStr('branch_index','branch_index')
        part.addToSummaryStr('branch_R','branch_R')
        part.addToSummaryStr('branch_Ibias','branch_Ibias')
        
        return part
    
    def saturatedMos4(self):
        """
        Description: mos4 that has to be in saturated operating region
        Ports: D, G, S, B
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos4_part = self.mos4()

        #build the point_meta (pm)
        
        pm = self.buildPointMeta({'Ids':'Ids', 'Vgs':'Vgs',
                                  'Vds':'Vds', 'Vbs':'Vbs',
                                  'L':'L','use_pmos':'bool_var'})

        #build the functions
        
        mos4_functions={'chosen_part_index':'use_pmos', 
                        'Ids':'Ids' ,
                        'Vgs' :'Vgs'  ,
                        'Vds' :'Vds'  ,
                        'Vbs' :'Vbs'  ,
                        'L'  :'L'    }

        #build the main part
        part = CompoundPart(['D','G','S','B'], pm, name)
        
        part.addPart(mos4_part, mos4_part.unityPortMap(), mos4_functions)

        #add function DOCs
        metric = Metric('MinimumOverdrive', self.ss.vgst_min, self.ss.vgst_max, False,
                        self.ss.vgst_min, self.ss.vgst_max)
        function = 'Vgs-'+str(self.ss.vth_min)
        doc = FunctionDOC(metric, function)
        part.addFunctionDOC(doc)
        
        metric = Metric('SaturationRequirement', 0, self.ss.vdd, False, 0, self.ss.vdd)
        function = '(Vds * ' + str(self.ss.vds_correction_factor) + ')-(Vgs-'+ str(self.ss.vth_max) +')'
        doc = FunctionDOC(metric, function)
        part.addFunctionDOC(doc)

        #add simulation DOCs
        for doc in saturationSimulationDOCs():
            part.addSimulationDOC(doc)

        self._parts[name] = part
        return part
    
    def rail(self):
        """
        Description: (0) Vss or (1) Vdd
        Ports: rail, Vss, Vdd
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        wire_part = self.wire()
        
        #build the point_meta
        pm = PointMeta({})
        
        #build the main part
        part = FlexPart(['rail','Vss','Vdd'],pm,name,'isVdd')
        part.addPartChoice(wire_part,{'1':'rail','2':'Vss'},{})
        part.addPartChoice(wire_part,{'1':'rail','2':'Vdd'},{})
        
        self._parts[name] = part
        return part
    
    def smallCapacitor(self):
        """
        Description: capacitor with a lower capacitance range
        Ports: 1,2
        Variables: C
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        part = AtomicPart('C', ['1','2'], self.buildPointMeta(['C_small']), name=name)
        self._parts[name] = part
        return part
    
    def cascode_stacked(self):
        """
        Description: stacked cascode structure
        Ports: in, out
        Variables: Vin, Vout, Ibias, casc_Vgs, casc_L, use_pmos
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        cascode_part = self.biasedMos()
        
        #build the point_meta
        pm = PointMeta({})
        pm['Vin'] = self.buildVarMeta('V','Vin')
        pm['Vout'] = self.buildVarMeta('V','Vout')
        pm['Ibias'] = self.buildVarMeta('Ibias','Ibias')
        pm['casc_Vgs'] = self.buildVarMeta('Vgs','casc_Vgs')
        pm['casc_L'] = self.buildVarMeta('L','casc_L')
        pm['use_pmos'] = self.buildVarMeta('bool_var','use_pmos')
        
        #build the main part
        part = CompoundPart(['in','out'],pm,name)
        
        cascode_functions = {'Vds':'abs(Vout-Vin)',
                             'Vgs':'casc_Vgs',
                             'Ids':'Ibias',
                             'L':'casc_L',
                             'use_pmos':'use_pmos',
                             'Vs':'Vin'
                             }
        part.addPart(cascode_part,{'D':'out','S':'in'},cascode_functions)
        
        self._parts[name] = part
        return part
    
    def cascode_folded(self):
        """
        Description: folded cascode structure
        Ports: in, out, Vss, Vdd
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        biasedmos_part = self.biasedMos()
        opp_part = self.rail()
        
        #build the point_meta
        pm = PointMeta({})
        pm['Vin'] = self.buildVarMeta('V','Vin')
        pm['Vout'] = self.buildVarMeta('V','Vout')
        pm['Ibias'] = self.buildVarMeta('Ibias','Ibias')
        pm['casc_Vgs'] = self.buildVarMeta('Vgs','casc_Vgs')
        pm['casc_L'] = self.buildVarMeta('L','casc_L')
        pm['bias_Vgs'] = self.buildVarMeta('Vgs','bias_Vgs')
        pm['bias_L'] = self.buildVarMeta('L','bias_L')
        pm['use_pmos'] = self.buildVarMeta('bool_var','use_pmos')
        
        #build the main part
        part = CompoundPart(['in','out','Vss','Vdd'],pm,name)
        opp_node = part.addInternalNode()
        
        cascode_functions = {'Vds':'abs(Vout-Vin)',
                             'Vgs':'casc_Vgs',
                             'Ids':'Ibias',
                             'L':'casc_L',
                             'use_pmos':'use_pmos',
                             'Vs':'Vin'
                             }
        bias_functions = {'Vds':"switchAndEval(use_pmos, {" + \
                                "0:'Vin-"+str(self.ss.vss)+"', "+ \
                                "1:'"+str(self.ss.vdd)+"-Vin'})",
                          'Vgs':'bias_Vgs',
                          'Ids':'2*Ibias',
                          'L':'bias_L',
                          'use_pmos':'use_pmos',
                          'Vs':"switchAndEval(use_pmos, {" + \
                               "0:'"+str(self.ss.vss)+"', "+ \
                               "1:'"+str(self.ss.vdd)+"'})"
                          }
        part.addPart(biasedmos_part,{'D':'out','S':'in'},cascode_functions)
        part.addPart(biasedmos_part,{'D':'in','S':opp_node},bias_functions)
        part.addPart(opp_part,{'rail':opp_node,'Vss':'Vss','Vdd':'Vdd'},
                     {'isVdd':'use_pmos'})
        
        self._parts[name] = part
        return part
    
    def cascode(self):
        """
        Description: A flex component that can be a (0) cascode_stacked,
                        or a (1) cascode_folded
        Ports: in, out, Vss, Vdd
        Variables: same as cascode_folded + fold
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        stacked_part = self.cascode_stacked()
        folded_part = self.cascode_folded()
        
        #build the point_meta
        pm = PointMeta({})
        folded_varmap = folded_part.unityVarMap()
        pm = self.updatePointMeta(pm,folded_part,folded_varmap)
        
        #build the main part
        part = FlexPart(['in','out','Vss','Vdd'],pm,name,'fold')
        part.addPartChoice(stacked_part,stacked_part.unityPortMap(),
                           stacked_part.unityVarMap())
        part.addPartChoice(folded_part,folded_part.unityPortMap(),folded_varmap)
        
        self._parts[name] = part
        return part
    
    def cascodeOrWire(self):
        """
        Description: A flex component that can be a (0) wire,
                        or a (1) cascode
        Ports: in, out, Vss, Vdd
        Variables: same as cascode + cascode
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        casc_part = self.cascode()
        wire_part = self.wire()
        
        #build the point_meta
        pm = PointMeta({})
        casc_varmap = casc_part.unityVarMap()
        pm = self.updatePointMeta(pm,casc_part,casc_varmap)
        
        #build the main part
        part = FlexPart(['in','out','Vss','Vdd'],pm,name,'cascode')
        part.addPartChoice(wire_part,{'1':'in','2':'out'},wire_part.unityVarMap())
        part.addPartChoice(casc_part,casc_part.unityPortMap(),casc_varmap)
        
        self._parts[name] = part
        return part
    
    def inputPair_classic(self):
        """
        Description: classic input stage
        Ports: Vin1, Vin2, Iout1, Iout2, Vss, Vdd
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        ampmos_part = self.saturatedMos3()
        bias_part = self.biasedMos()
        parcap_part = self.smallCapacitor()
        rail_part = self.rail()
        
        #build the point_meta
        pm = PointMeta({})
        pm['Ibias'] = self.buildVarMeta('Ibias','Ibias')
        pm['Vinop'] = self.buildVarMeta('V','Vinop')
        pm['Voutop'] = self.buildVarMeta('V','Voutop')
        pm['use_pmos'] = self.buildVarMeta('bool_var','use_pmos')
        pm['amp_Vgs'] = self.buildVarMeta('Vgs','amp_Vgs')
        pm['amp_L'] = self.buildVarMeta('L','amp_L')
        pm['bias_Vgs'] = self.buildVarMeta('Vgs','bias_Vgs')
        pm['bias_L'] = self.buildVarMeta('L','bias_L')
        
        #build the main part
        part = CompoundPart(['Vin1','Vin2','Iout1','Iout2','Vss','Vdd'],pm,name)
        vgnd_node = part.addInternalNode()
        opp_node = part.addInternalNode()
        
        #build the functions
        Vvgndop = "( Vinop - amp_Vgs )"
        ampmos_functions = {'Ids':'Ibias / 2',
                            'Vgs':'amp_Vgs',
                            'Vds':"Voutop - "+Vvgndop,
                            'L':'amp_L',
                            'use_pmos':'use_pmos'
                            }
        part.addPart(ampmos_part,
                     {'D':'Iout1','G':'Vin1','S':vgnd_node},
                     ampmos_functions
                     )
        part.addPart(ampmos_part,
                     {'D':'Iout2','G':'Vin2','S':vgnd_node},
                     ampmos_functions
                     )
        bias_functions = {'Vds':Vvgndop,
                          'Vgs':'bias_Vgs',
                          'Ids':'Ibias',
                          'L':'bias_L',
                          'use_pmos':'use_pmos',
                          'Vs':"switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vss)+"', "+ \
                                    "1:'"+str(self.ss.vdd)+"'})"
                          }
        part.addPart(bias_part,
                     {'D':vgnd_node,'S':opp_node},
                     bias_functions
                     )
        part.addPart(parcap_part,{'1':vgnd_node,'2':'Vdd'},
                     {'C':str(self.parcap_value)})
        part.addPart(rail_part,{'rail':opp_node,'Vss':'Vss','Vdd':'Vdd'},
                     {'isVdd':'use_pmos'})
        
        self._parts[name] = part
        return part
    
    def inputPair_Redoute(self):
        """
        Description: EMI resistant differential input stage by J.M. Redoute
        Ports: Vin1, Vin2, Iout1, Iout2, Vss, Vdd
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        ampmos_part = self.saturatedMos4()
        replica_part = self.saturatedMos3()
        bias_part = self.biasedMos()
        parcap_part = self.smallCapacitor()
        rail_part = self.rail()
        
        #build the point_meta
        pm = PointMeta({})
        pm['Ibias'] = self.buildVarMeta('Ibias','Ibias')
        pm['Vinop'] = self.buildVarMeta('V','Vinop')
        pm['Voutop'] = self.buildVarMeta('V','Voutop')
        pm['use_pmos'] = self.buildVarMeta('bool_var','use_pmos')
        pm['amp_Vgs'] = self.buildVarMeta('Vgs','amp_Vgs')
        pm['amp_L'] = self.buildVarMeta('L','amp_L')
        pm['bias_Vgs'] = self.buildVarMeta('Vgs','bias_Vgs')
        pm['bias_L'] = self.buildVarMeta('L','bias_L')
        
        #build the main part
        part = CompoundPart(['Vin1','Vin2','Iout1','Iout2','Vss','Vdd'],pm,name)
        vgnd_node = part.addInternalNode()
        bulk_node = part.addInternalNode()
        load_node = part.addInternalNode()
        opp_node = part.addInternalNode()
        
        #build the functions
        Vvgndop = "( Vinop - amp_Vgs )"
        ampmos_functions = {'Ids':'Ibias / 2',
                            'Vgs':'amp_Vgs',
                            'Vds':"Voutop - "+Vvgndop,
                            'Vbs':'0',
                            'L':'amp_L',
                            'use_pmos':'use_pmos'
                            }
        replica_functions = {'Ids':'Ibias / 2',
                            'Vgs':'amp_Vgs',
                            'Vds':str(self.ss.vdd-self.ss.vss)+" - "+Vvgndop,
                            'L':'amp_L',
                            'use_pmos':'use_pmos'
                            }
        bias_functions = {'Vds':Vvgndop,
                          'Vgs':'bias_Vgs',
                          'Ids':'Ibias',
                          'L':'bias_L',
                          'use_pmos':'use_pmos',
                          'Vs':"switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vss)+"', "+ \
                                    "1:'"+str(self.ss.vdd)+"'})"
                          }
        part.addPart(ampmos_part,
                     {'D':'Iout1','G':'Vin1','S':vgnd_node,'B':bulk_node},
                     ampmos_functions
                     )
        part.addPart(ampmos_part,
                     {'D':'Iout2','G':'Vin2','S':vgnd_node,'B':bulk_node},
                     ampmos_functions
                     )
        part.addPart(replica_part,
                     {'D':load_node,'G':'Vin1','S':bulk_node},
                     replica_functions
                     )
        part.addPart(replica_part,
                     {'D':load_node,'G':'Vin2','S':bulk_node},
                     replica_functions
                     )
        part.addPart(bias_part,
                     {'D':vgnd_node,'S':opp_node},
                     bias_functions
                     )
        part.addPart(bias_part,
                     {'D':bulk_node,'S':opp_node},
                     bias_functions
                     )
        part.addPart(parcap_part,{'1':bulk_node,'2':'Vdd'},{'C':str(2*self.parcap_value)})
        part.addPart(rail_part,{'rail':load_node,'Vss':'Vss','Vdd':'Vdd'},
                     {'isVdd':'1-use_pmos'})
        part.addPart(rail_part,{'rail':opp_node,'Vss':'Vss','Vdd':'Vdd'},
                     {'isVdd':'use_pmos'})
                
        self._parts[name] = part
        return part
    
    def inputPair_Fiori(self):
        """
        Description: EMI resistant differential input stage by F. Fiori
        Ports: Vin1, Vin2, Iout1, Iout2, Vss, Vdd
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        ampmos_part = self.saturatedMos3()
        bias_part = self.biasedMos()
        parcap_part = self.smallCapacitor()
        rail_part = self.rail()
        
        #build the point_meta
        pm = PointMeta({})
        pm['Ibias'] = self.buildVarMeta('Ibias','Ibias')
        pm['Vinop'] = self.buildVarMeta('V','Vinop')
        pm['Voutop'] = self.buildVarMeta('V','Voutop')
        pm['use_pmos'] = self.buildVarMeta('bool_var','use_pmos')
        pm['amp_Vgs'] = self.buildVarMeta('Vgs','amp_Vgs')
        pm['amp_L'] = self.buildVarMeta('L','amp_L')
        pm['bias_Vgs'] = self.buildVarMeta('Vgs','bias_Vgs')
        pm['bias_L'] = self.buildVarMeta('L','bias_L')
        
        #build the main part
        part = CompoundPart(['Vin1','Vin2','Iout1','Iout2','Vss','Vdd'],pm,name)
        vgnd1_node = part.addInternalNode()
        vgnd2_node = part.addInternalNode()
        opp_node = part.addInternalNode()
        
        #build the functions
        Vvgndop = "( Vinop - amp_Vgs )"
        ampmos_functions = {'Ids':'Ibias / 2',
                            'Vgs':'amp_Vgs',
                            'Vds':"Voutop - "+Vvgndop,
                            'L':'amp_L',
                            'use_pmos':'use_pmos'
                            }
        bias_functions = {'Vds':Vvgndop,
                          'Vgs':'bias_Vgs',
                          'Ids':'Ibias',
                          'L':'bias_L',
                          'use_pmos':'use_pmos',
                          'Vs':"switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vss)+"', "+ \
                                    "1:'"+str(self.ss.vdd)+"'})"
                          }
        part.addPart(ampmos_part,
                     {'D':'Iout1','G':'Vin1','S':vgnd1_node},
                     ampmos_functions
                     )
        part.addPart(ampmos_part,
                     {'D':'Iout2','G':'Vin2','S':vgnd1_node,},
                     ampmos_functions
                     )
        part.addPart(ampmos_part,
                     {'D':'Iout2','G':'Vin1','S':vgnd2_node},
                     ampmos_functions
                     )
        part.addPart(ampmos_part,
                     {'D':'Iout1','G':'Vin2','S':vgnd2_node,},
                     ampmos_functions
                     )
        part.addPart(bias_part,
                     {'D':vgnd1_node,'S':opp_node},
                     bias_functions
                     )
        part.addPart(bias_part,
                     {'D':vgnd2_node,'S':opp_node},
                     bias_functions
                     )
        part.addPart(parcap_part,{'1':vgnd1_node,'2':'Vdd'},{'C':str(self.parcap_value)})
        part.addPart(parcap_part,{'1':vgnd2_node,'2':'Vdd'},{'C':str(self.parcap_value)})
        part.addPart(rail_part,{'rail':opp_node,'Vss':'Vss','Vdd':'Vdd'},
                     {'isVdd':'use_pmos'})
        
        self._parts[name] = part
        return part
    
    def inputPair(self):
        """
        Description: A flex component that can be (0) inputPair_classic,
                        (1) inputPair_Redoute or (2) inputPair_Fiori
        Ports: Vin1, Vin2, Iout1, Iout2, Vss, Vdd
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        classic_part = self.inputPair_classic()
        redoute_part = self.inputPair_Redoute()
        fiori_part = self.inputPair_Fiori()
        
        #build the point_meta
        varmap = classic_part.unityVarMap()
        pm = PointMeta({})
        pm = self.updatePointMeta(pm,classic_part,varmap)
        
        #build the main part
        part = FlexPart(['Vin1','Vin2','Iout1','Iout2','Vss','Vdd'],pm,name,'input')
        part.addPartChoice(classic_part,classic_part.unityPortMap(),varmap)
        part.addPartChoice(redoute_part,redoute_part.unityPortMap(),varmap)
        part.addPartChoice(fiori_part,fiori_part.unityPortMap(),varmap)
        
        self._parts[name] = part
        return part
    
    def cascodedInputPair(self):
        """
        Description: an inputpair that is cascoded
        Ports: Vin1, Vin2, Iout1, Iout2, Vss, Vdd
        Variables:
        """
        
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        input_part = self.inputPair()
        casc_part = self.cascodeOrWire()
        
        pm = PointMeta({})
        input_varmap = {'Ibias':'Ibias',
                        'Vinop':'Vinop',
                        'Voutop':'input_Voutop',
                        'use_pmos':'input_is_pmos',
                        'amp_Vgs':'amp_Vgs',
                        'amp_L':'amp_L',
                        'bias_Vgs':'bias_Vgs',
                        'bias_L':'bias_L',
                        'input':'input'}
        pm = self.updatePointMeta(pm, input_part, input_varmap, True)
        casc_varmap = {'Vin':'IGNORE',
                       'Vout':'Vout',
                       'Ibias':'IGNORE',
                       'casc_Vgs':'casc_Vgs',
                       'casc_L':'casc_L',
                       'bias_Vgs':'bias_Vgs',
                       'bias_L':'bias_L',
                       'use_pmos':'load_is_Vss',
                       'cascode':'cascode',
                       'fold':'fold'}
        pm = self.updatePointMeta(pm, casc_part, casc_varmap, True)
        
        input_varmap['use_pmos'] = "switchAndEval(fold, {" + \
                                "0:'load_is_Vss', " + \
                                "1:'1-load_is_Vss'})"
        casc_varmap['Vin'] = "switchAndEval(input_is_pmos, {" + \
                                "0:'"+str(self.ss.vss)+"+input_Voutop', " + \
                                "1:'"+str(self.ss.vdd)+"-input_Voutop'})"
        casc_varmap['Ibias'] = "Ibias/2"
        
        part = CompoundPart(['Vin1','Vin2','Iout1','Iout2','Vss','Vdd'],pm,name)
        a_node = part.addInternalNode()
        b_node = part.addInternalNode()
        part.addPart(input_part,
                     {'Vin1':'Vin1','Vin2':'Vin2','Iout1':a_node,'Iout2':b_node,'Vss':'Vss','Vdd':'Vdd'},
                     input_varmap)
        part.addPart(casc_part,
                     {'in':a_node,'out':'Iout1','Vss':'Vss','Vdd':'Vdd'},
                     casc_varmap)
        part.addPart(casc_part,
                     {'in':b_node,'out':'Iout2','Vss':'Vss','Vdd':'Vdd'},
                     casc_varmap)
        
        self._parts[name] = part
        return part
    
    def cascodedCurrentMirror(self):
        """
        Description: simple 2-transistor current mirror
        Ports: Iref, Iout, miller, Vss, Vdd
        Variables:
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos_part = self.saturatedMos3()
        casc_part = self.cascodeOrWire()
        opp_part = self.rail()

        #build the point_meta (pm)
        pm = self.buildPointMeta({'frac':'frac',
                                  'use_pmos':'bool_var',
                                  'Iin':'Ids','Iout':'Ids',
                                  'Vds_in':'Vds','Vds_out':'Vds',
                                  'L':'L'})
        casc_varmap = {'Vin':'IGNORE',
                       'Vout':'IGNORE',
                       'Ibias':'IGNORE',
                       'casc_Vgs':'casc_Vgs',
                       'casc_L':'casc_L',
                       'bias_Vgs':'casc_Vgs',
                       'bias_L':'casc_L',
                       'use_pmos':'use_pmos',
                       'cascode':'cascode',
                       'fold':'IGNORE'}
        pm = self.updatePointMeta(pm, casc_part, casc_varmap, True)
        
        #build the main part
        part = CompoundPart(['Iref','Iout','miller','Vss','Vdd'], pm, name)
        opp_node = part.addInternalNode()
        ref_node = part.addInternalNode()

        ref_functions = {'use_pmos':'use_pmos',
                         'Vgs':'Vds_in',
                         'Vds':"switchAndEval(cascode, {" + \
                                "0:'Vds_in', " + \
                                "1:'Vds_in*frac'})",
                         'Ids':'Iin',
                         'L':'L'}
        part.addPart( mos_part, {'D':ref_node,'G':'Iref','S':opp_node},
                      ref_functions)
        out_functions = {'use_pmos':'use_pmos',
                         'Vgs':'Vds_in',
                         'Vds':"switchAndEval(cascode, {" + \
                                "0:'Vds_out', " + \
                                "1:'Vds_out*frac'})",
                         'Ids':'Iout',
                         'L':'L'}
        part.addPart( mos_part, {'D':'miller','G':'Iref','S':opp_node},
                      out_functions)
        cascref_functions = casc_varmap
        cascref_functions['fold'] = '0'
        cascref_functions['Vin'] = "switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vss)+"+Vds_in*frac', " + \
                                    "1:'"+str(self.ss.vdd)+"-Vds_in*frac'})"
        cascref_functions['Vout'] = "switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vss)+"+Vds_in', " + \
                                    "1:'"+str(self.ss.vdd)+"-Vds_in'})"
        cascref_functions['Ibias'] = 'Iin'
        part.addPart(casc_part,
                     {'in':ref_node,'out':'Iref','Vss':'Vss','Vdd':'Vdd'},
                     cascref_functions)
        cascout_functions = copy.deepcopy(cascref_functions)
        cascout_functions['Vin'] = "switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vss)+"+Vds_out*frac', " + \
                                    "1:'"+str(self.ss.vdd)+"-Vds_out*frac'})"
        cascout_functions['Vout'] = "switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vss)+"+Vds_out', " + \
                                    "1:'"+str(self.ss.vdd)+"-Vds_out'})"
        cascout_functions['Ibias'] = 'Iout'
        part.addPart(casc_part,
                     {'in':'miller','out':'Iout','Vss':'Vss','Vdd':'Vdd'},
                     cascout_functions)
        part.addPart(opp_part,{'rail':opp_node,'Vss':'Vss','Vdd':'Vdd'},
                     {'isVdd':'use_pmos'})
        
        self._parts[name] = part
        return part
    
    def dsAmp1(self):
        """
        Description: Differential-in voltage, single-ended out amplifier.
        Ports: Vin1, Vin2, out, miller, Vss, Vdd
        Variables: 
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        input_part = self.cascodedInputPair()
        cm_part = self.cascodedCurrentMirror()
        
        #build the point_meta
        pm = PointMeta({})
        input_varmap = input_part.unityVarMap()
        pm = self.updatePointMeta(pm, input_part, input_varmap, True)
        cm_varmap = cm_part.unityVarMap()
        cm_varmap['frac'] = 'cm_frac'
        cm_varmap['L'] = 'cm_L'
        cm_varmap['use_pmos'] = 'IGNORE'
        cm_varmap['Iin'] = 'IGNORE'
        cm_varmap['Iout'] = 'IGNORE'
        cm_varmap['Vds_in'] = 'IGNORE'
        cm_varmap['Vds_out'] = 'IGNORE'
        pm = self.updatePointMeta(pm, cm_part, cm_varmap, True)
        
        cm_varmap['use_pmos'] = '1-load_is_Vss'
        cm_varmap['Iin'] = 'Ibias/2'
        cm_varmap['Iout'] = 'Ibias/2'
        cm_varmap['Vds_in'] = "switchAndEval(load_is_Vss, {" + \
                                    "0:'"+str(self.ss.vdd)+"-Vout', " + \
                                    "1:'"+str(self.ss.vss)+"+Vout'})"
        cm_varmap['Vds_out'] = "switchAndEval(load_is_Vss, {" + \
                                    "0:'"+str(self.ss.vdd)+"-Vout', " + \
                                    "1:'"+str(self.ss.vss)+"+Vout'})"
        
        #build the main part
        part = CompoundPart(['Vin1','Vin2','out','miller','Vss','Vdd'],pm,name)
        node = part.addInternalNode()
        part.addPart(input_part,
                     {'Vin1':'Vin1','Vin2':'Vin2','Iout1':node,'Iout2':'out','Vss':'Vss','Vdd':'Vdd'},
                     input_varmap)
        part.addPart(cm_part,
                     {'Iref':node,'Iout':'out','miller':'miller','Vss':'Vss','Vdd':'Vdd'},
                     cm_varmap)
        
        self._parts[name] = part
        return part
    
    def ssAmp1(self):
        """
        Description: single ended amplifier 
        Ports: Vin, out, Vss, Vdd
        Variables: 
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        amp_part = self.saturatedMos3()
        casc_part = self.cascodeOrWire()
        bias_part = self.biasedMos()
        rail_part = self.rail()
        
        #build the point_meta
        pm = self.buildPointMeta({'Ibias':'Ibias','use_pmos':'bool_var',
                                  'Vout':'V',
                                  'amp_Vgs':'Vgs','amp_L':'L',
                                  'bias_Vgs':'Vgs','bias_L':'L',
                                  'casc_Vgs':'Vgs','casc_L':'L',
                                  'cascode':'bool_var',
                                  'amp_frac':'frac','bias_frac':'frac'})
        
        amp_Vds = "(use_pmos*("+str(self.ss.vdd)+"-Vout) + (1-use_pmos)*(Vout-"+str(self.ss.vss)+"))"
        bias_Vds = "(use_pmos*(Vout-"+str(self.ss.vss)+") + (1-use_pmos)*("+str(self.ss.vdd)+"-Vout))"
        amp_functions = {'Ids':'Ibias',
                         'Vgs':'amp_Vgs',
                         'Vds':"switchAndEval(cascode, {" + \
                                    "0:'"+amp_Vds+"', " + \
                                    "1:'"+amp_Vds+"*amp_frac'})",
                         'L':'amp_L',
                         'use_pmos':'use_pmos'}
        amp_casc_functions = {'Vin':"switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vss)+"+"+amp_Vds+"*amp_frac', " + \
                                    "1:'"+str(self.ss.vdd)+"-"+amp_Vds+"*amp_frac'})",
                              'Vout':'Vout',
                              'Ibias':'Ibias',
                              'casc_Vgs':'casc_Vgs',
                              'casc_L':'casc_L',
                              'bias_Vgs':'bias_Vgs',
                              'bias_L':'bias_L',
                              'use_pmos':'use_pmos',
                              'fold':'0',
                              'cascode':'cascode'}
        bias_functions = {'Vs':"switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vdd)+"', " + \
                                    "1:'"+str(self.ss.vss)+"'})",
                           'Ids':'Ibias',
                           'Vgs':'bias_Vgs',
                           'Vds':"switchAndEval(cascode, {" + \
                                    "0:'"+amp_Vds+"', " + \
                                    "1:'"+amp_Vds+"*bias_frac'})",
                           'L':'bias_L',
                           'use_pmos':'1-use_pmos'}
        bias_casc_functions = {'Vin':"switchAndEval(use_pmos, {" + \
                                    "0:'"+str(self.ss.vdd)+"-"+bias_Vds+"*amp_frac', " + \
                                    "1:'"+str(self.ss.vss)+"+"+bias_Vds+"*amp_frac'})",
                              'Vout':'Vout',
                              'Ibias':'Ibias',
                              'casc_Vgs':'casc_Vgs',
                              'casc_L':'casc_L',
                              'bias_Vgs':'bias_Vgs',
                              'bias_L':'bias_L',
                              'use_pmos':'1-use_pmos',
                              'fold':'0',
                              'cascode':'cascode'}
        
        #build the main part
        part = CompoundPart(['Vin','out','Vss','Vdd'],pm,name)
        amp_node = part.addInternalNode()
        bias_node = part.addInternalNode()
        opp_node = part.addInternalNode()
        load_node = part.addInternalNode()
        part.addPart(amp_part,
                     {'D':amp_node,'G':'Vin','S':opp_node},
                     amp_functions)
        part.addPart(casc_part,
                     {'in':amp_node,'out':'out','Vss':'Vss','Vdd':'Vdd'},
                     amp_casc_functions)
        part.addPart(bias_part,
                     {'D':bias_node,'S':load_node},
                     bias_functions)
        part.addPart(casc_part,
                     {'in':bias_node,'out':'out','Vss':'Vss','Vdd':'Vdd'},
                     bias_casc_functions)
        part.addPart(rail_part,{'rail':opp_node,'Vss':'Vss','Vdd':'Vdd'},
                     {'isVdd':'use_pmos'})
        part.addPart(rail_part,{'rail':load_node,'Vss':'Vss','Vdd':'Vdd'},
                     {'isVdd':'1-use_pmos'})
        
        self._parts[name] = part
        return part
    
    def ssAmp1OrWire(self):
        """
        Description: A flex component that can be a (0) wire,
                        or a (1) ssAmp1
        Ports: Vin, out, Vss, Vdd
        Variables: same as ssAmp1 + output_stage
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        ssAmp1_part = self.ssAmp1()
        wire_part = self.wire()
        
        #build the point_meta
        pm = PointMeta({})
        ssAmp1_varmap = ssAmp1_part.unityVarMap()
        pm = self.updatePointMeta(pm,ssAmp1_part,ssAmp1_varmap)
        
        #build the main part
        part = FlexPart(['Vin','out','Vss','Vdd'],pm,name,'output_stage')
        part.addPartChoice(wire_part,{'1':'Vin','2':'out'},wire_part.unityVarMap())
        part.addPartChoice(ssAmp1_part,ssAmp1_part.unityPortMap(),ssAmp1_varmap)
        
        self._parts[name] = part
        return part
    
    def dsAmp2(self):
        """
        Description: dual stage amplifier
        Ports: Vin1, Vin2, out, miller, Vss, Vdd
        Variables: 
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        stage1_part = self.dsAmp1()
        stage2_part = self.ssAmp1OrWire()
        cap_part = self.smallCapacitor()
        
        #build the point_meta
        pm = PointMeta({})
        stage1_functions = {}
        for old_name in stage1_part.point_meta.keys():
            stage1_functions[old_name] = 'stage1_' + old_name
        pm = self.updatePointMeta(pm,stage1_part,stage1_functions)
        
        stage2_functions = {}
        for old_name in stage2_part.point_meta.keys():
            stage2_functions[old_name] = 'stage2_' + old_name
        stage2_functions['amp_Vgs'] = 'IGNORE'
        pm = self.updatePointMeta(pm,stage2_part,stage2_functions)
        
        pm['miller_C'] = self.buildVarMeta('C_small','miller_C')
        
        stage2_functions['amp_Vgs'] = "switchAndEval(stage2_use_pmos, {" + \
                                        "0:'stage1_Vout-"+str(self.ss.vss)+"', " + \
                                        "1:'"+str(self.ss.vdd)+"-stage1_Vout'})"
        
        #build the main part
        part = CompoundPart(['Vin1','Vin2','out','Vss','Vdd'],pm,name)
        stage1out_node = part.addInternalNode()
        miller_node = part.addInternalNode()
        
        part.addPart(stage1_part,
                     {'Vin1':'Vin1','Vin2':'Vin2','out':stage1out_node,
                      'miller':miller_node,'Vss':'Vss','Vdd':'Vdd'},
                     stage1_functions)
        part.addPart(stage2_part,
                     {'Vin':stage1out_node,'out':'out','Vss':'Vss','Vdd':'Vdd'},
                     stage2_functions)
        part.addPart(cap_part,{'1':miller_node,'2':'out'},{'C':'miller_C'})
        
        self._parts[name] = part
        return part

#    def template(self):
#        """
#        Description: 
#        Ports: 
#        Variables: 
#        """
#        name = whoami()
#        if self._parts.has_key(name):  return self._parts[name]
#        
#        #parts to embed
#        
#        
#        #build the point_meta
#        
#        
#        #build the main part
#        
#        
#        self._parts[name] = part
#        return part
