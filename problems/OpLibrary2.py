"""OpLibrary2.py

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
from problems.Library import whoami, Library
from problems.OpLibrary import ApproxMosModels, OpLibrary, OpLibraryStrategy # for the device sizer
from regressor.Luc import LucStrategy, LucModel, LucFactory
from regressor.PointRegressor import PointRegressor
from util import mathutil
from util.constants import REGION_SATURATION
from util.ascii import asciiRowToStrings, asciiTo2dArray, hdrValFilesToTrainingData

log = logging.getLogger('library')
                
class OpLibrary2Strategy(OpLibraryStrategy):
    pass

class OpLibrary2(OpLibrary):
    """
    @description

      OpLibrary is a subclass of OpLibrary and inherits all parts of that
      library. However, it overrides some parts to allow for a different
      search space.

      By using this inheritance technique, the search space can be modified without
      breaking support for past databases.
      
    @attributes

      ss -- LibraryStrategy object --
      wire_factory -- WireFactory object -- builds wire Part
      _ref_varmetas -- dict of generic_var_name : varmeta.  
      _parts -- dict of part_name : Part object
      
    @notes
    
      Generic var names in ref_varmetas are: W, L, K, R, C, GM, DC_I, Ibias,
        DC_V, discrete_Vbias, cont_Vbias
    """

    def __init__(self, ss):
        """
        @description

          Constructor.
        
        @arguments

          ss -- OpLibraryStrategy object --
        
        @return

          new_library -- OpLibrary2 object
    
        @exceptions
    
        @notes

          just calls the parent's constructor
          
        """
        OpLibrary.__init__(self, ss)
        
    def crossCoupledMirror(self):
        """
           Cross Coupled mirrors as in SANSEN/P105/0331
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cm1_part = self.currentMirror()
        cm2_part = self.currentMirror()

        #build the point_meta (pm)
        pm = PointMeta({})
        cm1_varmeta_map = cm1_part.unityVarMap()
        cm1_varmeta_map['chosen_part_index'] = 'chosen_cm_index'
        cm2_varmeta_map = cm2_part.unityVarMap()
        cm2_varmeta_map['chosen_part_index'] = 'chosen_cm_index'
        pm = self.updatePointMeta(pm, cm1_part, cm1_varmeta_map)
        pm = self.updatePointMeta(pm, cm2_part, cm2_varmeta_map, True)

        # add a variable to control the relative size of the diode vs the
        # cross coupling
        pm.addVarMeta( self.buildVarMeta('frac', 'fracMirror') )

        #build functions
        cm1_functions = cm1_varmeta_map
        cm2_functions = cm2_varmeta_map

        cm1_functions['Iin']  = '(Iin * (fracMirror))'
        cm1_functions['Iout'] = '(Iout * (1-fracMirror))'
        cm2_functions['Iin']  = '(Iout * (fracMirror))'
        cm2_functions['Iout'] = '(Iin * (1-fracMirror))'
        
        #build the main part
        part = CompoundPart(['Irefnode','Ioutnode', 'oprail'], pm, name)

        part.addPart( cm1_part,
                      {'Irefnode':'Irefnode', 'Ioutnode':'Ioutnode',
                       'oprail':'oprail'},
                      cm1_functions )
        part.addPart( cm2_part,
                      {'Irefnode':'Ioutnode', 'Ioutnode':'Irefnode',
                       'oprail':'oprail'},
                      cm2_functions )
        
        self._parts[name] = part
        return part
    
    def currentMirrorExtended(self):
        """
        Description: current mirror (selects one of several possible
          implementations)
          
        Ports: Irefnode, Ioutnode, oprail
        
        Variables:
            chosen_part_index,
            use_pmos, Iin, Iout, Vds_in,
            Vds_out, L, Vs, fracIn, fracOut,
            cascode_L, cascode_Vgs
          
        Variable breakdown:
          For overall part: chosen_part_index
            0: use currentMirror_Simple
            1: use currentMirror_Cascode
            2: use currentMirror_LowVoltageA
            3: use crossCoupledMirror
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cm_Simple = self.currentMirror_Simple()
        cm_Cascode = self.currentMirror_Cascode()
        cm_LowVoltageA = self.currentMirror_LowVoltageA()
        cross_coupled = self.crossCoupledMirror()

        cross_coupled_varmap = cross_coupled.unityVarMap()
        cross_coupled_varmap['chosen_cm_index'] = 'cc_cm_index'

        #build the point_meta (pm)
        pm = self.buildPointMeta({})
        pm = self.updatePointMeta(pm, cm_Simple, cm_Simple.unityVarMap())
        pm = self.updatePointMeta(pm, cm_Cascode, cm_Cascode.unityVarMap(), True)
        pm = self.updatePointMeta(pm, cm_LowVoltageA, cm_LowVoltageA.unityVarMap(), True)
        pm = self.updatePointMeta(pm, cross_coupled, cross_coupled_varmap, True)
        
        #build the main part
        part = FlexPart(['Irefnode','Ioutnode', 'oprail'], pm, name)
        portmap = cm_Simple.unityPortMap()
        part.addPartChoice(cm_Simple, portmap, cm_Simple.unityVarMap())
        part.addPartChoice(cm_Cascode, portmap, cm_Cascode.unityVarMap())
        part.addPartChoice(cm_LowVoltageA, portmap, cm_LowVoltageA.unityVarMap())
        part.addPartChoice(cross_coupled, portmap, cross_coupled_varmap)
        
        self._parts[name] = part
        return part
    
    def dsIiLoad(self):
        """
           Same as OpLibrary but using the extended current mirror
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cm_part = self.currentMirrorExtended()
        wire_part = self.wire()

        #build the point_meta (pm)
        pm = PointMeta({})
        cm_varmeta_map = cm_part.unityVarMap()
        wire_varmeta_map = wire_part.unityVarMap()

        cm_varmeta_map['use_pmos'] = 'loadrail_is_vdd'
        pm = self.updatePointMeta(pm, cm_part, cm_varmeta_map)
        pm = self.updatePointMeta(pm, wire_part, wire_varmeta_map)

        #build functions
        cm_functions = cm_varmeta_map
        wire_functions = wire_varmeta_map

        #build the main part
        part = CompoundPart(['Iin1', 'Iin2', 'Iout', 'loadrail'], pm, name)

        part.addPart( cm_part,
                      {'Irefnode':'Iin1', 'Ioutnode':'Iin2',
                       'oprail':'loadrail'},
                      cm_functions )
        part.addPart( wire_part, {'1':'Iin2','2':'Iout'},
                      wire_functions )
        
        self._parts[name] = part
        return part


    def dsViAmp1(self):
        """
           see OpLibrary
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        input_part = self.ddViInput_Flex()
        load_part = self.dsIiLoad()

        #build the point_meta (pm)
        pm = PointMeta({})
        #pm['Vds_internal'] = self.buildVarMeta('Vds','Vds_internal')
        
        input_varmeta_map = {        
            'loadrail_is_vdd':'loadrail_is_vdd',
            'input_is_pmos':'input_is_pmos',
            'chosen_part_index':'IGNORE', # has to be derived from the input/load config
            'Ibias':'Ibias',
            'Ibias2':'Ibias2',
            
            'Vds1':'Vds_internal', 'Vds2':'IGNORE', 'Vs':'IGNORE',
            'Vin_cmm':'Vin_cmm',
                
            'cascode_L':'inputcascode_L',
            'cascode_Vgs':'inputcascode_Vgs',
            'cascode_recurse':'inputcascode_recurse',
            'cascode_is_wire':'inputcascode_is_wire',
                 
            'ampmos_L':'ampmos_L', 'fracAmp':'fracAmp', 
                                  
            'degen_choice':'degen_choice','fracDeg':'degen_fracDeg',
                
            'inputbias_L':'inputbias_L',
            'inputbias_Vgs':'inputbias_Vgs',       
            'fracVgnd':'fracVgnd',
            
            'folder_L':'folder_L','folder_Vgs':'folder_Vgs',

            }
              
        load_varmeta_map = {
            'chosen_part_index':'load_chosen_part_index',
            'loadrail_is_vdd':'loadrail_is_vdd',
            'Iin':'Ibias','Iout':'Ibias', ## NOTE: this will require a hack because if it is folded, we need another bias current
            'Vds_in':'Vds_internal','Vds_out':'Vds_internal',
            'fracIn':'load_fracIn','fracOut':'load_fracOut',
            'Vs':'Vout',
            'L':'load_L','cascode_L':'load_cascode_L',
            'cascode_Vgs':'load_cascode_Vgs',
            'fracMirror':'fracMirror','cc_cm_index':'cc_cm_index'
            }

        pm = self.updatePointMeta(pm, input_part, input_varmeta_map)
        pm = self.updatePointMeta(pm, load_part, load_varmeta_map, True)

        #build functions
        input_functions = input_varmeta_map
        load_functions = load_varmeta_map

        input_functions['chosen_part_index'] = '(input_is_pmos == loadrail_is_vdd)'

        # we have to figure out if the input stage is folded or not
        # because it influences the rail voltages and the bias currents
        
        is_folded = '(input_is_pmos==loadrail_is_vdd)'
                                 
        load_functions['Iin']="switchAndEval("+is_folded+", {" + \
                                 "True:'Ibias2', " + \
                                 "False:'Ibias'  })"  
        load_functions['Iout']="switchAndEval("+is_folded+", {" + \
                                 "True:'Ibias2', " + \
                                 "False:'Ibias'  })"  
                                                              
        load_functions['Vds_out']="switchAndEval(loadrail_is_vdd, {" + \
                                 "0:'"+str(self.ss.vdd)+"-Vout', " + \
                                 "1:'Vout-"+str(self.ss.vss)+"'  })"
        load_functions['Vds_in']="(" + str(self.ss.vdd-self.ss.vss) + " - Vds_internal )"
                                 
        load_functions['Vs']="switchAndEval(loadrail_is_vdd, {" + \
                                 "1:'"+str(self.ss.vdd)+"', " + \
                                 "0:'"+str(self.ss.vss)+"'  })"        

        input_functions['Vds1']="Vds_internal"
        input_functions['Vds2']="switchAndEval(input_is_pmos, {" + \
                                 "1:'"+str(self.ss.vdd)+"-Vout', " + \
                                 "0:'Vout-"+str(self.ss.vss)+"'  })"
        input_functions['Vs']="switchAndEval(input_is_pmos, {" + \
                                 "1:'"+str(self.ss.vdd)+"', " + \
                                 "0:'"+str(self.ss.vss)+"'  })"          
        
        #build the main part
        part = CompoundPart(['Vin1', 'Vin2', 'Iout', 'loadrail','opprail'],
                            pm, name)

        n1 = part.addInternalNode()
        n2 = part.addInternalNode()
        
        part.addPart( input_part,
                      {'Vin1':'Vin1', 'Vin2':'Vin2', 'Iout1':n1, 'Iout2':n2,
                       'loadrail':'loadrail', 'opprail':'opprail'},
                      input_functions )
        part.addPart( load_part,
                      {'Iin1':n1, 'Iin2':n2, 'Iout':'Iout',
                       'loadrail':'loadrail'},
                      load_functions )
        
        self._parts[name] = part
        return part
  
