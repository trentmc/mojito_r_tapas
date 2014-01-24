"""OpLibrary.py

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
from regressor.Luc import LucStrategy, LucModel, LucFactory
from regressor.PointRegressor import PointRegressor
from util import mathutil
import util.constants as constants
from util.FileHash import hashfile

log = logging.getLogger('library')

class ApproxMosModels:
    """
    @description

      Holds information about the MOS lookup tables which the user
      needs to specify.
      
    @attributes
      
    @notes
    """

    def __init__(self, nmos_filebase, pmos_filebase):
        """
        @description
        
          nmos_filebase -- string -- To crate nmos data.  Expect ascii files:
            -nmos_filebase.hdr which has one line holding each var name
            -nmos_filebase.val where each column corresponds to one variable,
             and each row is a different sample.
          pmos_filebase -- string -- same as nmos_filebase

        @arguments

        @return

          mosdata -- ApproxMosModels  object --

        @exceptions

        @notes
        """
        self._nmos_model = self._buildModel(nmos_filebase, 'W')
        self._pmos_model = self._buildModel(pmos_filebase, 'W')

    def estimateMosWidth(self, regressor_id , input_point, mult=1):
        """
        @description

          Estimates mos width, by using its internal regressor-style
          mos model.

        @arguments
        
          regressor_id -- (0: use nmos regressor, 1: use pmos regressor)
          input_point -- dict of var_name : var_value.  Typically
            includes the vars: Vbs, Vds, Ids, L, Vgs
          mult -- the device multiplier used
          
        @return

          width -- float -- 

        @exceptions

        @notes

          Helper function

        """
                
        # correct for the device multiplier
        new_point = copy.copy(input_point)
        new_point['Ids'] = new_point['Ids'] / mult
        
        if regressor_id == 0:
            model = self._nmos_model
        elif regressor_id == 1:
            model = self._pmos_model
        else:
            raise ValueError("Invalid model id")

        return model.simulatePoint(new_point)

    def estimateNmosWidth(self, input_point, mult=1):
        """
        @description

          Estimates nmos width, by using its internal regressor-style
          nmos model.

        @arguments
        
          input_point -- dict of var_name : var_value.  Typically
            includes the vars: Vbs, Vds, Ids, L, Vgs
          mult -- the device multiplier used
          
        @return

          width -- float -- 

        @exceptions

        @notes
        """
        return self.estimateMosWidth(0, input_point, mult)

    def estimatePmosWidth(self, input_point, mult=1):
        """Like estimateNmosWidth(), except pmos.
        """
        return self.estimateMosWidth(1, input_point, mult)
    
    def isMosOpPointValid(self, regressor_id , input_point, mult=1):
        """
        @description

          Estimates mos whether a mos point is valid for the internal regressor-style
          mos model.

        @arguments
        
          regressor_id -- (0: use nmos regressor, 1: use pmos regressor)
          input_point -- dict of var_name : var_value.  Typically
            includes the vars: Vbs, Vds, Ids, L, Vgs
          mult -- the device multiplier used
          
        @return

          width -- float -- 

        @exceptions

        @notes

          Helper function

        """
                
        # correct for the device multiplier
        new_point = copy.copy(input_point)
        new_point['Ids'] = new_point['Ids'] / mult
        
        if regressor_id == 0:
                model = self._nmos_model
        elif regressor_id == 1:
                model = self._pmos_model
        else:
            raise ValueError("Invalid model id")
        
        return model.isValidForPoint(new_point)

    def isNmosOpPointValid(self, input_point, mult=1):
        return self.isMosOpPointValid(0, input_point, mult)

    def isPmosOpPointValid(self, input_point, mult=1):
        return self.isMosOpPointValid(1, input_point, mult)

    def getInputUpperBound(self, regressor_id, varname):
        """
        @description
            Returns the validity upper bound for a certain input variable.
            
        @arguments
            regressor_id -- (0: use nmos regressor, 1: use pmos regressor)
            idx -- the index of the input variable
            
        @return

          the upper bound for this input variable
    
        @exceptions
    
        @notes                
        """
        if regressor_id == 0:
                model = self._nmos_model
        elif regressor_id == 1:
                model = self._pmos_model
        else:
            raise ValueError("Invalid model id")

        return model.getInputUpperBound(varname)

    def getNmosInputUpperBound(self, varname):
        return self.getInputUpperBound(0, varname)
    
    def getPmosInputUpperBound(self, varname):
        return self.getInputUpperBound(1, varname)
    
    def getInputLowerBound(self, regressor_id, varname):
        """
        @description
            Returns the validity lower bound for a certain input variable.
            
        @arguments
            regressor_id -- (0: use nmos regressor, 1: use pmos regressor)
            idx -- the index of the input variable
            
        @return

          the lower bound for this input variable
    
        @exceptions
    
        @notes                
        """
        if regressor_id == 0:
            model = self._nmos_model
        elif regressor_id == 1:
            model = self._pmos_model
        else:
            raise ValueError("Invalid model id")

        return model.getInputLowerBound(varname)

    def getNmosInputLowerBound(self, varname):
        return self.getInputLowerBound(0, varname)
    
    def getPmosInputLowerBound(self, varname):
        return self.getInputLowerBound(1, varname)

    def loadPickledTrainingData(self, filename):
        """
        @description

          Loads regressor training data

        @arguments
        
          filename -- string -- expect to find a loadable file at filename

        @return

          input_varnames -- list of string
          input_data -- 2d array
          output_varnames -- list of string
          output_data -- 2dy array
                
        @exceptions

        @notes

        """
        # load input data
        fid = open(filename,'r')
        data = pickle.load(fid)
        fid.close()

        input_varnames = data[0]
        input_data = data[1]
        output_varnames = data[2]
        output_data = data[3]

        return (input_varnames, numpy.array(input_data), output_varnames, numpy.array(output_data))
    
    def _buildModel(self, filebase, target_varname):
        """
        @description

          Builds a regressor that maps other variables to target_varname

        @arguments
        
          filename -- string -- expect to find a loadable file at filename

        @return

          model -- PointRegressor object -- 

        @exceptions

        @notes

          Currently the regressor is a Luc.
        """
        filename = filebase + '.pickle'
        log.info("Build LUT model using file %s: begin" % filename)

        #build Luc regressor
        lut_ss = LucStrategy()
        
        need_new_model = True

        #hash the data file such that we can detect and update dataset
        source_hash = hashfile(filename)
        log.info("Source data hash: %s" % source_hash)

        model = PointRegressor()

        try:
            if model.loadFromFile(filebase):
                log.info("Loaded PointRegressor from cache")
                if model.source_hash != source_hash:
                    log.info("Source data hash different from cache hash")
                    log.info("  source hash: %s" % source_hash)
                    log.info("  model hash : %s" % model.source_hash)
                else:
                    log.info("reusing cached model")
                    need_new_model = False
        except:
            import pdb;pdb.set_trace()
            log.info("cache load failed, rebuilding model")

        if need_new_model:
            log.info("Build new model...")

            #get training data
            log.info(" Load training data from %s..." % filename)

            (input_varnames, input_data, output_varnames, output_data) = \
                             self.loadPickledTrainingData(filename)

            log.info("  Loaded %d training points" % input_data.shape[0])
            log.info("   input vars : %s" % str(input_varnames) )
            log.info("   output vars: %s" % str(output_varnames) )

            #build regressor
            log.info(" Building regressor...")
            regressor = LucFactory().build(input_data, output_data, lut_ss)

            #build PointRegressor
            model = PointRegressor(regressor, input_varnames, False)

            model.source_hash = source_hash

            log.info(" Saving cached model...")
            model.saveToFile(filebase)

            
        log.info("Build LUT model using filebase=%s: done" % filebase)
        return model
        
class OpLibraryStrategy:
    """
    @description

      Strategy to build an OpLibrary object.
      
    @attributes

      Holds things like:
      -feature_size
      -vth, vdd
      -min/max for R's, C's, L's, W's, Ibias's, Vbias's, Gm's, etc
      
    @notes
    """
    
    def __init__(self, devices_setup):
        """
        @description

          Constructor.

        @arguments

          devices_setup -- DevicesSetup

        @return

          library_ss -- OpLibraryStrategy object --

        @exceptions

        @notes

          A _lot_ of defaults get set here!
       """
        #preconditions
        assert isinstance(devices_setup, DevicesSetup)
        
        #extract info from devices_setup
        feature_size = devices_setup.featureSize() #float -- feature size (in m) (e.g. 0.18e-6)
        vdd = devices_setup.vdd()                  #float -- power supply voltage (in V) (e.g. 5.0)
        approx_mos_models = devices_setup.approxMosModels() #ApproxMosModels -- for lookup tables
        
        #we can maybe get rid of many of what's below
        self.feature_size = feature_size
        self.vdd = vdd
        self.approx_mos_models = approx_mos_models
        self.vss = 0
        
        # these values assist the DOC's to skip simulation if an operating
        # point won't be found
        base_vth = constants.NMOS_BASE_VTH0[devices_setup.process]
        self.vth_min    = base_vth - 0.1  #minimal vth, used to see if vgs-vth > 0; e.g. 0.3 V
        self.vth_max    = base_vth + 0.2  #maximal vth, used to see if vds > vgs-vth; e.g. 0.5 V

        self.vgst_min   = 0  # minimal gate overdrive
        self.vgst_max   = self.vdd - self.vth_min  # maximal gate overdrive

        # relaxes the saturation prediction by using:
        # vds * (self.vds_correction_factor) > vgs-vth 
        self.vds_correction_factor = 1.2 
                
        femto, pico, nano, micro, milli = 1e-15, 1e-12, 1e-9, 1e-6, 1e-3
        kilo, mega, giga = 1e3, 1e6, 1e9

        #for operating point driven formulation
        self.KPn=565 * micro
        self.KPp=74 * micro
        self.an=2.43
        self.ap=0.7
        
        self.finger_overhead_L = (0.4 + 0.07*2) * micro
        
        self.Vtn = base_vth + 0.1 #NMOS threshold voltage; e.g. 0.4 V
        self.Vtp = base_vth + 0.1 #PMOS threshold voltage; e.g. 0.4 V
        
        #default values (user can change these after initialization)

        #mos width
        if devices_setup.process == 'UMC180':  self.min_W = .24 * micro
        elif devices_setup.process == 'UMC90': self.min_W = .12 * micro
        else: raise "need to handle process %s" % devices_setup.process
        
        self.max_W = 100 * micro

        #mos length
#         self.min_L = self.feature_size
#         self.max_L = 2 * micro
        #self.min_L = 0.4 * micro
        #self.max_L = 0.4 * micro
        
        # get the limits from the sizing model validity range
        min_L_nmos = self.approx_mos_models.getNmosInputLowerBound('L')
        min_L_pmos = self.approx_mos_models.getPmosInputLowerBound('L')
        self.min_L = max([min_L_nmos, min_L_pmos])
        max_L_nmos = self.approx_mos_models.getNmosInputUpperBound('L')
        max_L_pmos = self.approx_mos_models.getPmosInputUpperBound('L')
        self.max_L = min([max_L_nmos, max_L_pmos])
        log.info("Bounds on device length: %e ... %e" % (self.min_L, self.max_L) )

        #transistor device multiplier (number of fingers)
        self.min_M = 1 
        self.max_M = 100
        
        #multiplier (eg in current mirrors)
        self.min_K = 1
        self.max_K = 10

        #resistance
        self.min_R = constants.DEFAULT_MIN_R
        self.max_R = constants.DEFAULT_MAX_R

        #capacitance
        self.min_C = constants.DEFAULT_MIN_C
        self.max_C = constants.DEFAULT_MAX_C

        #Bounds on the terminal voltages of the MOSFET's

        # get the limits from the sizing model validity range
        min_Vgs_nmos = self.approx_mos_models.getNmosInputLowerBound('Vgs')
        min_Vgs_pmos = self.approx_mos_models.getPmosInputLowerBound('Vgs')
        self.min_Vgs = max([min_Vgs_nmos, min_Vgs_pmos, self.vth_min])
        max_Vgs_nmos = self.approx_mos_models.getNmosInputUpperBound('Vgs')
        max_Vgs_pmos = self.approx_mos_models.getPmosInputUpperBound('Vgs')
        self.max_Vgs = min([max_Vgs_nmos, max_Vgs_pmos, self.vdd - self.vth_min])
        log.info("Bounds on device Vgs: %e ... %e" % (self.min_Vgs, self.max_Vgs) )

        # get the limits from the sizing model validity range
        min_Vds_nmos = self.approx_mos_models.getNmosInputLowerBound('Vds')
        min_Vds_pmos = self.approx_mos_models.getPmosInputLowerBound('Vds')
        self.min_Vds = max([min_Vds_nmos, min_Vds_pmos])
        max_Vds_nmos = self.approx_mos_models.getNmosInputUpperBound('Vds')
        max_Vds_pmos = self.approx_mos_models.getPmosInputUpperBound('Vds')
        self.max_Vds = min([max_Vds_nmos, max_Vds_pmos])

#        self.min_Vds = 0.1
#        self.min_Vds = -self.vdd # HACK to detect library problems
#        self.max_Vds = self.vdd
        log.info("Bounds on device Vds: %e ... %e" % (self.min_Vds, self.max_Vds) )

        # get the limits from the sizing model validity range
        #min_Vbs_nmos = self.approx_mos_models.getNmosInputLowerBound('Vbs')
        #min_Vbs_pmos = self.approx_mos_models.getPmosInputLowerBound('Vbs')
        #self.min_Vbs = max([min_Vbs_nmos, min_Vbs_pmos])
        #max_Vbs_nmos = self.approx_mos_models.getNmosInputUpperBound('Vbs')
        #max_Vbs_pmos = self.approx_mos_models.getPmosInputUpperBound('Vbs')
        #self.max_Vbs = min([max_Vbs_nmos, max_Vbs_pmos])

        self.min_Vbs = 0.0
        self.max_Vbs = 0.0

        log.info("Bounds on device Vbs: %e ... %e" % (self.min_Vbs, self.max_Vbs) )

        self.min_Vs = 0.0
#        self.min_Vs = -self.vdd # HACK to detect library problems
        self.max_Vs = self.vdd

        # Bounds on the current in a MOSFET

        min_Ids_nmos = self.approx_mos_models.getNmosInputLowerBound('Ids')
        min_Ids_pmos = self.approx_mos_models.getPmosInputLowerBound('Ids')
        self.min_Ids = max([min_Ids_nmos, min_Ids_pmos, 10 * nano])
        max_Ids_nmos = self.approx_mos_models.getNmosInputUpperBound('Ids')
        max_Ids_pmos = self.approx_mos_models.getPmosInputUpperBound('Ids')
        self.max_Ids = min([max_Ids_nmos, max_Ids_pmos, 10*milli])
        
        #self.min_Ids = 100 * nano
        #self.max_Ids = 10 * milli
        log.info("Bounds on device Ids: %e ... %e" % (self.min_Ids, self.max_Ids) )
                
        #Bounds on the voltages & current of a resistor
        self.min_Vres = 0.0
        self.max_Vres = self.vdd
        self.min_Ires = 1 * pico
        self.max_Ires = self.max_Ids * 2
                
        #bias current
        self.min_Ibias = self.min_Ids
        self.max_Ibias = self.max_Ids * 2

        #DC voltage source's 'DC' value
        self.min_DC_V = 0.0
        self.max_DC_V = self.vdd
        
        #discrete bias voltage
        # Note that the Vbiases for cascode circuits are separate from this
        # Here, it's probably good to provide a good number of Vbiases,
        # then perhaps as part of the objective function, try to minimize
        # the total number of Vbiases (or at least penalize if > say 4 Vbiases)
        num_Vbiases = 10
        min_Vbias = 0
        max_Vbias = self.vdd
        step = (max_Vbias - min_Vbias) / float(num_Vbiases - 1)
        self.discrete_Vbiases = numpy.arange(min_Vbias, max_Vbias, step)

        #continuous bias voltage (less constraining than discrete)
        self.min_cont_Vbias = self.min_DC_V
        self.max_cont_Vbias = self.max_DC_V
                

class OpLibrary(Library):
    """
    @description

      An 'OpLibrary' holds a set of Parts, e.g. resistors and caps all the
      way up to current mirrors and amplifiers.
      
    @attributes

      ss -- LibraryStrategy object --
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

          new_library -- OpLibrary object
    
        @exceptions
    
        @notes

          This constructor method doesn't bother building each possible Part
          right now; rather, it defers that to a 'just in time' basis
          for the first request for a given part.  Once it builds the
          requested part, it _does_ store it in self._parts, such that
          subsequent calls for the same part do not need to rebuild the part.
        """
        Library.__init__(self)
        
        self.ss = ss

        #'rvm' will be stored as self._ref_varmetas
        rvm = {}
        
        # fraction
        rvm['frac'] = ContinuousVarMeta(False, 0.0, 1.0, 'frac')
        
        # restrict the fraction for the virtual ground a little more strict
        rvm['fracVgnd'] = ContinuousVarMeta(False, 0.0, 0.5, 'fracVgnd')
        
        rvm['W'] = ContinuousVarMeta(False, ss.min_W, ss.max_W, 'W')
        rvm['L'] = ContinuousVarMeta(False, ss.min_L, ss.max_L, 'L')
        rvm['M'] = DiscreteVarMeta(range(ss.min_M, ss.max_M+1), 'M')
        rvm['K'] = DiscreteVarMeta(range(ss.min_K, ss.max_K+1), 'K')

        min_R, max_R = float(ss.min_R), float(ss.max_R)
        rvm['logscale_R'] = ContinuousVarMeta(True, math.log10(min_R), math.log10(max_R), 'logscale_R')
        rvm['linscale_R'] = ContinuousVarMeta(False, min_R, max_R, 'linscale_R')
        
        mn, mx = math.log10(float(ss.min_C)), math.log10(float(ss.max_C))
        rvm['C'] = ContinuousVarMeta(True, mn, mx, 'C')

        rvm['Ids'] = ContinuousVarMeta(False, self.ss.min_Ids, self.ss.max_Ids, 'Ids', False)
        rvm['Vgs'] = ContinuousVarMeta(False, self.ss.min_Vgs, self.ss.max_Vgs, 'Vgs', False)
        rvm['Vds'] = ContinuousVarMeta(False, self.ss.min_Vds, self.ss.max_Vds, 'Vds', False)
        rvm['Vbs'] = ContinuousVarMeta(False, self.ss.min_Vbs, self.ss.max_Vbs, 'Vbs', False)
        rvm['Vs'] = ContinuousVarMeta(False, self.ss.min_Vs, self.ss.max_Vs, 'Vs', False)
                                        
        rvm['V1'] = ContinuousVarMeta(False, self.ss.min_Vres, self.ss.max_Vres, 'V1', False)
        rvm['V2'] = ContinuousVarMeta(False, self.ss.min_Vres, self.ss.max_Vres, 'V2', False) 
        rvm['I'] = ContinuousVarMeta(False, self.ss.min_Ires, self.ss.max_Ires, 'I', False)
        rvm['V'] = ContinuousVarMeta(False,0.0,self.ss.vdd, 'V', False)
                                        
        rvm['Ibias'] = ContinuousVarMeta(False, self.ss.min_Ibias, self.ss.max_Ibias, 'Ibias', False)
        
        rvm['DC_V'] = ContinuousVarMeta(False, self.ss.min_DC_V, self.ss.max_DC_V, 'DC_V', False)
        rvm['cont_Vbias'] = ContinuousVarMeta(
            False, self.ss.min_cont_Vbias, self.ss.max_cont_Vbias, 'cont_Vbias')
        rvm['bool_var']=DiscreteVarMeta([0,1], 'bool_var')
        for keyname, varmeta in rvm.items():
            assert keyname == varmeta.name, (keyname, varmeta.name)
        self._ref_varmetas = rvm

        self._parts = {}
            
    #====================================================================
    #====================================================================
    #Start actual definition of library parts here.  Each part's name
    # is identical to the function name.
    def nmos4_sized(self):
        """
        Description: 4-terminal nmos
        Ports: D,G,S,B
        Variables: W,L,M
        
        Note: this is the physical nmos part described by W, L and M
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        part = AtomicPart('M', ['D', 'G', 'S', 'B'], self.buildPointMeta(['W','L','M']), name)
        
        #add DOCs
        # first the DOC's that can be measured pre-simulation (FunctionDOC's)
        max_thr = self.ss.max_W * self.ss.max_M * 0.99
        metric = Metric('NearMaxWidth', 0,  max_thr, False, 0, max_thr)
        function = 'W*M'
        
        doc = FunctionDOC(metric, function)
        
        part.addFunctionDOC(doc)
                                  
        self._parts[name] = part
        return part

    def pmos4_sized(self):
        """
        Description: 4-terminal pmos
        Ports: D,G,S,B
        Variables: W,L,M
        
        Note: this is the physical nmos part described by W, L and M
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name] 
        part = AtomicPart('M', ['D', 'G', 'S', 'B'], self.buildPointMeta(['W','L', 'M']), name)
        #add DOCs
        # first the DOC's that can be measured pre-simulation (FunctionDOC's)
        max_thr = self.ss.max_W * self.ss.max_M * 0.99 
        metric = Metric('NearMaxWidth', 0 , max_thr, False, 0, max_thr)
        function = 'W*M'
        
        doc = FunctionDOC(metric, function)
                                  
        part.addFunctionDOC(doc)
        
        self._parts[name] = part
        return self._parts[name]
        
    def nmos4(self):
        """
        Description: 4-terminal nmos
        Ports: D,G,S,B
        Variables: Ids, Vd, Vg, Vs, Vb, L

        Note: This is the nmos part that is OP driven
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]

        #parts to embed
        nmos4_sized_part = self.nmos4_sized()

        #build the point_meta (pm)
        pm = self.buildPointMeta(['Ids', 'Vgs', 'Vds', 'Vbs', 'L'])
        
        #build functions

        #for reference, here's the expression without substitution:
        #nmos4_sized_varmeta_map['W'] = \
        # '(L*( (2*Ids*( 1+an*(Vg-Vs-Vtn) ) )/( KPn*( (Vg-Vs-Vtn)**2) )))'
        
        #with substitution:
        an = str(self.ss.an)
        Vtn = str(self.ss.Vtn)
        KPn = str(self.ss.KPn)

        nmos4_sized_functions={'L' : 'L'}
        
        Vgst = '(Vgs-'+Vtn+')'
        a='('+an+'*'+Vgst+')'
        W_tot='L*((2*Ids*(1+'+a+'))/('+KPn+'*('+Vgst+'**2)))'
        
        Loverhead=str(self.ss.finger_overhead_L)
        Mult = 'int(( ( (' + W_tot + ')/(' + Loverhead + '+L)))**0.5 +1)'

        #old W
        #W = '(' +W_tot + ')/(' + Mult + ')'

        #new W: uses a lookup table
        # -this is what will be seen inside evalFunction() of Part.py;
        #  note that evalFunction has the local variable 'part' which
        #  makes this possible.
        W = 'part.approx_mos_models.estimateNmosWidth(point,('+ Mult +'))'
 
        nmos4_sized_functions['W'] = W
        nmos4_sized_functions['M'] = Mult

        #build the main part
        part = CompoundPart(['D', 'G', 'S', 'B'], pm, name)
        
        part.addPart(nmos4_sized_part, {'D':'D','G':'G','S':'S','B':'B'},
                     nmos4_sized_functions)

        #evalFunction() has access to this part but this part still
        # needs to provide access to the approx_mos_models:
        part.approx_mos_models = self.ss.approx_mos_models
        
        # ensure that the approx_mos_models attribute remains
        # reference such that it does not get pickled with every ind
        part.attr_to_keep_as_ref = ['approx_mos_models']

        #add DOCs
        # first the DOC's that can be measured pre-simulation (FunctionDOC's)
        metric = Metric('ValidMosModel', 0.99 , 1.1 , False, 0, 1)
        function = 'part.approx_mos_models.isNmosOpPointValid(point,('+ Mult +'))'
        doc = FunctionDOC(metric, function)
        
        part.addFunctionDOC(doc)
        
        self._parts[name] = part
        return part

    def pmos4(self):
        """
        Description: 4-terminal pmos
        Ports: D,G,S,B
        Variables: Ids, Vd, Vg, Vs, Vb, L

        Note: This is the pmos part that is OP driven
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name] 

        #parts to embed
        pmos4_sized_part = self.pmos4_sized()

        #build the point_meta (pm)
        pm = self.buildPointMeta(['Ids', 'Vgs', 'Vds', 'Vbs', 'L'])
        
        #build functions
        
        #for reference, here's the expression without substitution:
        #pmos4_sized_varmeta_map['W'] = \
        # '(L*((2*Ids*(1+an*(Vg-Vs-Vtn)))/(KPn*((Vg-Vs-Vtn)**2))))'
        
        #with substitution:
        ap = str(self.ss.ap)
        Vtp = str(self.ss.Vtp)
        KPp = str(self.ss.KPp)

        pmos4_sized_functions={'L' : 'L'}
        
        Vgst = '(Vgs-'+Vtp+')'
        a='('+ap+'*'+Vgst+')'
        W_tot='L*((2*Ids*(1+'+a+'))/('+KPp+'*('+Vgst+'**2)))'
        
        Loverhead=str(self.ss.finger_overhead_L)
        Mult = 'int(( ( (' + W_tot + ')/(' + Loverhead + '+L)))**0.5 +1)'
        
        #old W
        #W = '(' +W_tot + ')/(' + Mult + ')'

        #new W: uses a lookup table
        # -this is what will be seen inside evalFunction() of Part.py;
        #  note that evalFunction has the local variable 'part' which
        #  makes this possible.
        W = 'part.approx_mos_models.estimatePmosWidth(point,('+ Mult +'))'

        pmos4_sized_functions['W'] = W
        pmos4_sized_functions['M'] = Mult
        
        #build the main part
        part = CompoundPart(['D', 'G', 'S', 'B'], pm, name)
        
        part.addPart(pmos4_sized_part, {'D':'D','G':'G','S':'S','B':'B'}, pmos4_sized_functions)        

        #evalFunction() has access to this part but this part still
        # needs to provide access to the approx_mos_models:
        part.approx_mos_models = self.ss.approx_mos_models
        # ensure that the approx_mos_models attribute remains
        # reference such that it does not get pickled with every ind
        part.attr_to_keep_as_ref = ['approx_mos_models']

        #add DOCs
        # first the DOC's that can be measured pre-simulation (FunctionDOC's)
        metric = Metric('ValidMosModel', 0.99 , 1.1 , False, 0, 1)
        function = 'part.approx_mos_models.isPmosOpPointValid(point,('+ Mult +'))'
        doc = FunctionDOC(metric, function)
        
        part.addFunctionDOC(doc)

        self._parts[name] = part
        return self._parts[name]

    def dcvs(self):
        """
        Description: DC voltage source, with reference to 0.
        Ports: NPOS  
        Variables: DC (V)

        Note: this does not need 'GND' as a port because there is a special
        check in Part.py for parts of name 'dcvs' which adds a second port
        of '0' there if seen.  This way, we get to keep all Vbias nodes
        internal, which would have otherwise been an issue esp. for cascodes.
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        pm = self.buildPointMeta({'DC':'DC_V'})
        part = AtomicPart('V', ['NPOS'], pm, name=name)
        self._parts[name] = part
        return part
        
    def wire(self):
        """
        Description: wire as a Part.  (short circuit).
        Ports: 1,2
        Variables: (none)

        Implemented as a resistor with resistance 0.
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        part = AtomicPart('R', ['1','2'], PointMeta({}), name=name)

        self._parts[name] = part
        return part
    
    def openCircuit(self):
        """
        Description: open circuit
          Implemented as a 2-port compound part with no embedded parts.
        Ports: 1,2
        Variables: 
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        part = CompoundPart(['1','2'], PointMeta({}), name=name)
        self._parts[name] = part
        return part
    
    def sizedLogscaleResistor(self):
        """
        Description: resistor, where R's search space is log-scaled
        Ports: 1,2
        Variables: R
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #build the point_meta (pm)
        pm = PointMeta({})
        pm['R'] = self.buildVarMeta('logscale_R', 'R')

        #build the main part
        part = AtomicPart('R', ['1','2'], pm, name=name)
        
        self._parts[name] = part
        return part
    
    def sizedLinscaleResistor(self):
        """
        Description: resistor, where R's search space is lin-scaled
        Ports: 1,2
        Variables: R
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #build the point_meta (pm)
        pm = PointMeta({})
        pm['R'] = self.buildVarMeta('linscale_R', 'R')

        #build the main part
        part = AtomicPart('R', ['1','2'], pm, name=name)
        self._parts[name] = part
        return part
        
    def resistor(self):
        """
        Description: resistor
        Ports: 1,2
        Variables: V, I
        Note: OP driven variant
        """
        
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        
        #parts to embed
        sizedres_part = self.sizedLinscaleResistor()

        #build the point_meta (pm)
        pm = self.buildPointMeta(['V','I'])      
        
        #build functions
        # -we have a check to make sure that the (scaled) R doesn't go to zero
        min_R = str(float(self.ss.min_R))
        sizedres_functions={'R':'max(' + min_R + ', abs( V ) / ( I ))'}
        
        #build the main part
        part = CompoundPart(['1', '2'], pm, name)
        
        part.addPart(sizedres_part, {'1':'1','2':'2'}, sizedres_functions)

        self._parts[name] = part
        return part
        
    def capacitor(self):
        """
        Description: resistor
        Ports: 1,2
        Variables: C
        """
        name = whoami()
        if self._parts.has_key(name):  return self._parts[name]
        part = AtomicPart('C', ['1','2'], self.buildPointMeta(['C']), name=name)
        self._parts[name] = part
        return part        

    def mos4(self):
        """
        Description: 4-terminal mos that can be pmos or nmos, depending on
          the input point's value of chosen_part_index.
          
        Ports: D,G,S,B
          
        Variables: chosen_part_index, Ids, Vd, Vg, Vs, Vb, L

        Variable breakdown:
          For overall part: chosen_part_index (=='use_pmos')
            0: use nmos4
            1: use pmos4
          For nmos4: unity
          For pmos4: unity

          # does this make sense !!yes!!
          For pmos4: Ids=-Ids, Vd=vdd-Vd, Vg=vdd-Vg, Vs=vdd-Vs, Vb=vdd-Vb, L=L
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]
        
        #parts to embed
        nmos4_part = self.nmos4()
        pmos4_part = self.pmos4()
        
        #build the point_meta (pm)
        nmos4_varmeta_map = nmos4_part.unityVarMap()
        pmos4_varmeta_map = pmos4_part.unityVarMap()
        pm = self.buildPointMeta(['Ids', 'Vgs', 'Vds', 'Vbs', 'L'])
        
        #build functions
        nmos4_functions = nmos4_varmeta_map
        pmos4_functions = pmos4_varmeta_map
        
        #build the main part
        part = FlexPart(['D', 'G', 'S', 'B'], pm, name)

        part.addPartChoice(nmos4_part, nmos4_part.unityPortMap(),
                           nmos4_functions)
        part.addPartChoice(pmos4_part, pmos4_part.unityPortMap(),
                           pmos4_functions)

        self._parts[name] = part
        return part

    def mos3(self):
        """
        Description: 3-terminal mos that can be pmos or nmos, depending on
          the input point's value of use_pmos.
          
        Ports: D, G, S
        
        Variables: Ids, Vgs, Vds, L, use_pmos
        
        Variable breakdown:
          For mos4: unity, except Vbs=0
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos4_part = self.mos4()

        #build the point_meta (pm)
        pm = self.buildPointMeta({'Ids':'Ids', 
                                  'Vgs':'Vgs', 'Vds':'Vds',
                                  'L':'L','use_pmos':'bool_var'})
        #build the functions
        mos4_functions={'chosen_part_index':'use_pmos', 
                        'Ids':'Ids' ,
                        'Vgs' :'Vgs'  ,
                        'Vds' :'Vds'  ,
                        'Vbs' :'0'  ,
                        'L'  :'L'    }
        
        #build the main part
        part = CompoundPart(['D','G','S'], pm, name)
        
        part.addPart(mos4_part, {'D':'D','G':'G','S':'S','B':'S'},
                     mos4_functions)
                
        self._parts[name] = part
        return part
        
    def saturatedMos3(self):
        """
        Description: mos3 that has to be in saturated operating region
          
        Ports: D, G, S
        
        Variables: Ids, Vgs, Vds, L, use_pmos
        
        Variable breakdown:
          For mos3: unity
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos3_part = self.mos3()

        #build the point_meta (pm)
        pm = PointMeta({})
        mos3_varmeta_map = mos3_part.point_meta.unityVarMap()
        pm = self.updatePointMeta(pm, mos3_part, mos3_varmeta_map)

        #build the functions
        mos3_functions = mos3_varmeta_map

        #build the main part
        part = CompoundPart(['D','G','S'], pm, name)
        
        part.addPart(mos3_part, mos3_part.unityPortMap(), mos3_functions)

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

    def saturatedMos3Cascoded(self):
        """
        Description: two mos's, one free and one biased
          
        Ports: G,D,S
        
        Variables: use_pmos, Vds, Vs, Ids, D_Vgs, S_Vgs, fracVi
          fracVi determines the voltage at the internal node (as a fraction of Vds, scaled between 0 and 1): 
            Vi = Vs + Vds * fracVi * (1 - 2 * (use_pmos==1))

        Variable breakdown:
          biasedMos closest to D: use_pmos=use_pmos, Vgs=D_Vgs, Vds=Vds * (1-fracVi), Vs=Vi, Ids=Ids, L=D_L
          saturatedMos closest to S: use_pmos=use_pmos, Vgs=S_Vgs, Vds=Vds * fracVi,     Vs=Vs, Ids=Ids, L=S_L
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos_part = self.saturatedMos3()
        casc_part = self.biasedMos()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['fracVi']= self.buildVarMeta('frac','fracVi')
        
        casc_varmap = {'use_pmos':'use_pmos',
                       'Vgs':'cascode_Vgs',
                       'Vds':'Vds', # to calc
                       'L':'cascode_L',
                       'Vs':'Vs', # to calc
                       'Ids':'Ids'
                      }
        mos_varmap = {'use_pmos':'use_pmos',
                      'Vgs':'Vgs',
                      'Vds':'Vds', # to calc
                      'L':'L',
                      'Vs':'Vs',
                      'Ids':'Ids'
                      }
        
        pm = self.updatePointMeta(pm, casc_part, casc_varmap)
        pm = self.updatePointMeta(pm, mos_part, mos_varmap, True)
        
        #build the functions
        casc_functions = casc_varmeta_map
        mos_functions = mos_varmeta_map
        
        Vi = '(Vs + Vds * fracVi * (1 - 2 * (use_pmos==1)))'
        casc_functions['Vds'] = 'Vds * (1-fracVi)'
        casc_functions['Vs'] = Vi
        
        mos_functions['Vds'] = 'Vds * (fracVi)'
        mos_functions['Vs'] = 'Vs'
        
        #build the main part
        part = CompoundPart(['G','D','S'], pm, name)
        
        n1 = part.addInternalNode()
        
        part.addPart(casc_part, {'D':'D','S':n1}, casc_functions)
        part.addPart(mos_part, {'G':'G','D':n1,'S':'S'}, mos_functions)
                
        self._parts[name] = part
        return part

    def stackedSaturatedMos3(self):
        """
        Description: one or two biased mos's, stacked
          
        Ports: G, D,S
        
        Variables:
         
         fracVi determines the voltage at the internal node (as a fraction of Vds, scaled between 0 and 1): 
            Vi = Vs + Vds * fracVi * (1 - 2 * (use_pmos==1))

        Variable breakdown:
          For overal part: chosen_part_index (==do_stack)
            0 : use biasedMos
            1 : use twoBiasedMoses
          biasedMos: use_pmos=use_pmos, Vds=Vds, Vgs=Vgs, L=S_L, Vs=Vs, Ids=Ids
          twoBiasedMoses: 1:1 mapping (except chosen_part_index)
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        onemos_part = self.saturatedMos3()
        twomos_part = self.saturatedMos3Cascoded()

        #build the point_meta (pm)
        pm = PointMeta({})
        onemos_varmeta_map = {'use_pmos':'use_pmos',
                              'Vgs':'Vgs','Vds':'Vds','L':'L',
                              'Vs':'Vs','Ids':'Ids'}
        twomos_varmeta_map = twomos_part.unityVarMap()
        
        pm = self.updatePointMeta(pm, twomos_part, twomos_varmeta_map)

        #build the functions
        onemos_functions = onemos_varmeta_map
        twomos_functions = twomos_varmeta_map

        #build the main part
        part = FlexPart(['G','D', 'S'], pm, name)
        
        part.addPartChoice( onemos_part, {'G':'G', 'D':'D','S':'S'}, onemos_functions )
        part.addPartChoice( twomos_part, {'G':'G', 'D':'D','S':'S'}, twomos_functions )
                
        self._parts[name] = part
        return part

    def mosDiode(self):
        """
        Description: MOS diode.  G is tied to D.
          
        Ports: D,S
        
        Variables: Ids, Vds, L, use_pmos
        
        Variable breakdown:
          For mos3: unity, except for Vgs = Vds
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos3_part = self.saturatedMos3()

        #build the point_meta (pm)
        pm = self.buildPointMeta({'Ids':'Ids', 
                                  'Vds':'Vds',
                                  'L':'L','use_pmos':'bool_var'})

        #build the functions
        mos3_functions={'use_pmos':'use_pmos', 
                        'Ids':'Ids' ,
                        'Vgs' :'Vds'  ,
                        'Vds' :'Vds'  ,
                        'L'  :'L'    }
        
        #build the main part
        part = CompoundPart(['D','S'], pm, name)
        
        part.addPart(mos3_part, {'D':'D','G':'D','S':'S'}, mos3_functions)
                                
        self._parts[name] = part
        return part

    def shortOrOpenCircuit(self):
        """
        Description: short circuit OR open circuit.  This can be
          useful for making flexible topologies.
          
        Ports: 1, 2
        
        Variables: chosen_part_index

        Variable breakdown:
          For overall part: chosen_part_index (=='open_circuit')
            0: use wire
            1: use openCircuit

        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        wire_part = self.wire()
        oc_part = self.openCircuit()
        
        #build the point_meta (pm)
        pm = PointMeta({})
        
        #build functions
        wire_functions = {}
        oc_functions = {}
        
        #build the main part
        part = FlexPart(['1', '2'], pm, name)
        part.addPartChoice( wire_part, {'1':'1','2':'2'}, wire_functions )
        part.addPartChoice( oc_part, {'1':'1','2':'2'}, oc_functions )
                
        self._parts[name] = part
        return part

    def resistorOrMosDiode(self):
        """
        Description: resistor OR mosDiode
          
        Ports: D, S
        
        Variables: chosen_part_index, Vds, Ids, L, use_pmos

        Variable breakdown:
          For overall part: chosen_part_index (=='use_mosDiode')
            0: use resistor
            1: use mosDiode
          For resistor: V=Vds, I=Ids
          For mosDiode: Vds=Vds, Ids=Ids, L=L, use_pmos=use_pmos

        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        res_part = self.resistor()
        diode_part = self.mosDiode()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Ids']= self.buildVarMeta('Ids','Ids')

        res_varmeta_map={'V':'Vds','I':'Ids'};
        
        diode_varmeta_map = diode_part.unityVarMap()
        
        pm = self.updatePointMeta(pm, res_part, res_varmeta_map)
        pm = self.updatePointMeta(pm, diode_part, diode_varmeta_map, True)

        #build the functions
        res_functions = res_varmeta_map
        res_functions['I'] = 'Ids'
        
        diode_functions = diode_varmeta_map
        
        #build the main part
        part = FlexPart(['D', 'S'], pm, name)
        
        part.addPartChoice( res_part, {'1':'D','2':'S'}, res_functions)
        part.addPartChoice( diode_part, diode_part.unityPortMap(),
                            diode_functions)
                
        self._parts[name] = part
        return part

    def biasedMos(self):
        """
        Description: mos3 with a DC voltage bias on its base port
          
        Ports: D,S
        
        Variables: Vds, Vgs, Ids, L, use_pmos, Vs

        Variable breakdown:
          For mos3: Vds=Vds, Vgs=Vgs, L=L, use_pmos=use_pmos
          For dcvs: DC=Vgs + Vs
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos3_part = self.saturatedMos3()
        dcvs_part = self.dcvs()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Vs']= self.buildVarMeta('Vs','Vs')
        
        mos3_varmeta_map = mos3_part.unityVarMap()
         
        pm = self.updatePointMeta(pm, mos3_part, mos3_varmeta_map)

        #build the functions
        mos3_functions = mos3_varmeta_map
        dcvs_functions = {}

        dcvs_functions['DC'] = 'Vs+Vgs*(1-2*(use_pmos==1))'
        
        #build the main part
        part = CompoundPart(['D','S'], pm, name)
        
        n1 = part.addInternalNode()
        
        part.addPart(mos3_part, {'D':'D','G':n1,'S':'S'}, mos3_functions)
        part.addPart(dcvs_part, {'NPOS':n1}, dcvs_functions)
                
        self._parts[name] = part
        return part

    def biasedMosTwoDrain(self):
        """
        Description: mos3 with a DC voltage bias on its base port
          provides two drain connections for simplicity
          
        Ports: D,D2,S  (Note: D2 == D)
        
        Variables: Vds, Vgs, Ids, L, use_pmos, Vs

        Variable breakdown:
          For mos3: Vds=Vds, Vgs=Vgs, L=L, use_pmos=use_pmos
          For dcvs: DC=Vgs + Vs
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos3_part = self.saturatedMos3()
        dcvs_part = self.dcvs()
        # to connect the D1 and D2
        wire_part = self.wire()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Vs']= self.buildVarMeta('Vs','Vs')
        
        mos3_varmeta_map = mos3_part.unityVarMap()
         
        pm = self.updatePointMeta(pm, mos3_part, mos3_varmeta_map)

        #build the functions
        mos3_functions = mos3_varmeta_map
        dcvs_functions = {}

        dcvs_functions['DC'] = 'Vs+Vgs*(1-2*(use_pmos==1))'
        
        #build the main part
        part = CompoundPart(['D','D2','S'], pm, name)
        
        n1 = part.addInternalNode()
        
        part.addPart(mos3_part, {'D':'D','G':n1,'S':'S'}, mos3_functions)
        part.addPart(dcvs_part, {'NPOS':n1}, dcvs_functions)
        part.addPart(wire_part, {'1':'D', '2':'D2'}, {})

        self._parts[name] = part
        return part

    def biasedMosOrWire(self):
        """
        Description: either a biasedMos or a wire (short circuit)
          
        Ports: D,S
        
        Variables: chosen_part_index, Vds, Vgs, Ids, L, use_pmos, Vs

        Variable breakdown:
          For overal part: chosen_part_index (==do_short_circuit)
            0 : use biasedMos
            1 : use wire
          For biasedMos: 1:1 mapping of all vars except chosen_part_index
          For wire: <<none>>
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        bias_part = self.biasedMos()
        wire_part = self.wire()

        #build the point_meta (pm)
        pm = PointMeta({})
        
        bias_varmeta_map = bias_part.unityVarMap()
        pm = self.updatePointMeta(pm, bias_part, bias_varmeta_map)

        #build the functions
        bias_functions = bias_varmeta_map
        wire_functions = {}

        #build the main part
        part = FlexPart(['D','S'], pm, name)
        part.addPartChoice(bias_part, bias_part.unityPortMap(), bias_functions)
        part.addPartChoice(wire_part, {'1':'D','2':'S'}, wire_functions)
                
        self._parts[name] = part
        return part

    def RC_series(self):
        """
        Description: resistor and capacitor in series
          
        Ports: N1, N2
        
        Variables: R, C

        Variable breakdown:
          For R: R=R
          For C: C=C
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        # use a sized resistor here, because the series capacitor makes OP
        # formulation not relevant
        res_part = self.sizedLogscaleResistor()
        cap_part = self.capacitor()

        #build the point_meta (pm)
        pm = PointMeta({})
        res_varmeta_map = res_part.unityVarMap()
        cap_varmeta_map = cap_part.unityVarMap()
        
        pm = self.updatePointMeta(pm, res_part, res_varmeta_map)
        pm = self.updatePointMeta(pm, cap_part, cap_varmeta_map)

        #build functions
        res_functions = res_varmeta_map
        cap_functions = cap_varmeta_map

        #build the main part
        part = CompoundPart(['N1','N2'], pm, name)
        
        n1 = part.addInternalNode()
        
        part.addPart(res_part, {'1':'N1','2':n1}, res_functions)
        part.addPart(cap_part, {'1':n1,'2':'N2'}, cap_functions)
                
        self._parts[name] = part
        return part

    def twoBiasedMoses(self):
        """
        Description: two mos's, stacked
          
        Ports: D,S
        
        Variables: use_pmos, Vds, Vs, Ids, D_Vgs, S_Vgs, fracVi
          fracVi determines the voltage at the internal node (as a fraction of Vds, scaled between 0 and 1): 
            Vi = Vs + Vds * fracVi * (1 - 2 * (use_pmos==1))

        Variable breakdown:
          biasedMos closest to D: use_pmos=use_pmos, Vgs=D_Vgs, Vds=Vds * (1-fracVi), Vs=Vi, Ids=Ids, L=D_L
          biasedMos closest to S: use_pmos=use_pmos, Vgs=S_Vgs, Vds=Vds * fracVi,     Vs=Vs, Ids=Ids, L=S_L
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos_part = self.biasedMos()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['fracVi']= self.buildVarMeta('frac','fracVi')
        
        D_varmeta_map = {'use_pmos':'use_pmos',
                         'Vgs':'D_Vgs',
                         'Vds':'Vds',
                         'L':'D_L',
                         'Vs':'Vs',
                         'Ids':'Ids'
                         }
        S_varmeta_map = {'use_pmos':'use_pmos',
                         'Vgs':'S_Vgs',
                         'Vds':'Vds',
                         'L':'S_L',
                         'Vs':'Vs',
                         'Ids':'Ids'
                         }
        
        pm = self.updatePointMeta(pm, mos_part, D_varmeta_map)
        pm = self.updatePointMeta(pm, mos_part, S_varmeta_map, True)
        
        #build the functions
        D_functions = D_varmeta_map
        S_functions = S_varmeta_map
        
        Vi = '(Vs + Vds * fracVi * (1 - 2 * (use_pmos==1)))'
        D_functions['Vds'] = 'Vds * (1-fracVi)'
        D_functions['Vs'] = Vi
        
        S_functions['Vds'] = 'Vds * (fracVi)'
        S_functions['Vs'] = 'Vs'
        
        #build the main part
        part = CompoundPart(['D','S'], pm, name)
        
        n1 = part.addInternalNode()
        
        part.addPart(mos_part, {'D':'D','S':n1}, D_functions)
        part.addPart(mos_part, {'D':n1,'S':'S'}, S_functions)
                
        self._parts[name] = part
        return part

    def twoBiasedMosesTwoDrain(self):
        """
        Description: two mos's, stacked
          
        Ports: D,S
        
        Variables: use_pmos, Vds, Vs, Ids, D_Vgs, S_Vgs, fracVi
          fracVi determines the voltage at the internal node (as a fraction of Vds, scaled between 0 and 1): 
            Vi = Vs + Vds * fracVi * (1 - 2 * (use_pmos==1))

        Variable breakdown:
          biasedMos closest to D: use_pmos=use_pmos, Vgs=D_Vgs, Vds=Vds * (1-fracVi), Vs=Vi, Ids=Ids, L=D_L
          biasedMos closest to S: use_pmos=use_pmos, Vgs=S_Vgs, Vds=Vds * fracVi,     Vs=Vs, Ids=Ids, L=S_L
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mos_part = self.biasedMos()
        mos_part_2d = self.biasedMosTwoDrain()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['fracVi']= self.buildVarMeta('frac','fracVi')
        
        D_varmeta_map = {'use_pmos':'use_pmos',
                         'Vgs':'D_Vgs',
                         'Vds':'Vds',
                         'L':'D_L',
                         'Vs':'Vs',
                         'Ids':'Ids'
                         }
        S_varmeta_map = {'use_pmos':'use_pmos',
                         'Vgs':'S_Vgs',
                         'Vds':'Vds',
                         'L':'S_L',
                         'Vs':'Vs',
                         'Ids':'Ids'
                         }
        
        pm = self.updatePointMeta(pm, mos_part_2d, D_varmeta_map)
        pm = self.updatePointMeta(pm, mos_part, S_varmeta_map, True)
        
        #build the functions
        D_functions = D_varmeta_map
        S_functions = S_varmeta_map
        
        Vi = '(Vs + Vds * fracVi * (1 - 2 * (use_pmos==1)))'
        D_functions['Vds'] = 'Vds * (1-fracVi)'
        D_functions['Vs'] = Vi
        
        S_functions['Vds'] = 'Vds * (fracVi)'
        S_functions['Vs'] = 'Vs'
        
        #build the main part
        part = CompoundPart(['D','D2','S'], pm, name)
        
        n1 = part.addInternalNode()
        
        part.addPart(mos_part_2d, {'D':'D','D2':'D2','S':n1}, D_functions)
        part.addPart(mos_part, {'D':n1,'S':'S'}, S_functions)
                
        self._parts[name] = part
        return part

    def stackedCascodeMos(self):
        """
        Description: one or two biased mos's, stacked
          
        Ports: D,S
        
        Variables: chosen_part_index,
           use_pmos, Vds, Vs, Ids, D_Vgs, D_l, S_Vgs, S_L, fracVi
         
         fracVi determines the voltage at the internal node (as a fraction of Vds, scaled between 0 and 1): 
            Vi = Vs + Vds * fracVi * (1 - 2 * (use_pmos==1))           

        Variable breakdown:
          For overal part: chosen_part_index (==do_stack)
            0 : use biasedMos
            1 : use twoBiasedMoses
          biasedMos: use_pmos=use_pmos, Vds=Vds, Vgs=Vgs, L=S_L, Vs=Vs, Ids=Ids
          twoBiasedMoses: 1:1 mapping (except chosen_part_index)
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        onemos_part = self.biasedMos()
        twomos_part = self.twoBiasedMoses()

        #build the point_meta (pm)
        pm = PointMeta({})
        onemos_varmeta_map = {'use_pmos':'use_pmos',
                              'Vgs':'S_Vgs','Vds':'Vds','L':'S_L',
                              'Vs':'Vs','Ids':'Ids'}
        twomos_varmeta_map = twomos_part.unityVarMap()
        
        pm = self.updatePointMeta(pm, twomos_part, twomos_varmeta_map)

        #build the functions
        onemos_functions = onemos_varmeta_map
        twomos_functions = twomos_varmeta_map

        #build the main part
        part = FlexPart(['D', 'S'], pm, name)
        
        part.addPartChoice( onemos_part, {'D':'D','S':'S'}, onemos_functions )
        part.addPartChoice( twomos_part, {'D':'D','S':'S'}, twomos_functions )
                
        self._parts[name] = part
        return part

    def levelShifter(self):
        """
        Description: 'amplifier' that shifts voltage down
          
        Ports: Vin, Vout, loadrail, opprail
        
        Variables: use_pmos,
          Vin, Vout
          use_pmos,
          amp_L,
          cascode_do_stack,
          cascode_D_Vgs, cascode_D_L, cascode_S_Vgs, cascode_S_L, fracVi
          Ibias
          
          ##note: using mos3 for a source follower requires triple well if implemented with nmos


        Variable breakdown:
          amp (mos3):
            use_pmos=use_pmos, L=amp_L, Ids=Ibias
            if use_pmos
                Vgs=Vout-Vin
                Vds=Vout-Vss
            else
                Vgs=Vin-Vout
                Vds=Vdd-Vout
            
          cascode (stackedCascodeMos):
            chosen_part_index=cascode_do_stack,
            use_pmos=use_pmos, Ids=Ibias
            D_Vgs=cascode_D_Vgs, D_L=cascode_D_L,
            S_Vgs=cascode_S_Vgs, S_L=cascode_S_L,
            fracVi=cascode_fracVi
            
            if use_pmos
                Vs=Vdd
                Vds=Vdd-Vout
            else
                Vs=Vss
                Vds=Vout   
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        amp_part = self.saturatedMos3()
        cascode_part = self.stackedCascodeMos()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Ibias']= self.buildVarMeta('Ibias')
        pm['Drail_is_vdd'] = self.buildVarMeta('bool_var','Drail_is_vdd')

        amp_varmeta_map = {'use_pmos':'IGNORE', 
                           'Vgs':'Vin', 'Vds':'Vout', 
                           'L':'amp_L',
                           'Ids':'IGNORE',
                           }
        casc_varmeta_map = {'chosen_part_index':'cascode_do_stack',
                            'use_pmos':'IGNORE',
                            'Vds':'Vout', 'D_L':'cascode_D_L',
                            'D_Vgs':'cascode_D_Vgs',
                            'fracVi':'cascode_fracVi', 'S_L':'cascode_S_L',
                            'S_Vgs':'cascode_S_Vgs',
                            'Vs':'Vout',
                            'Ids':'IGNORE',
                            }
        pm = self.updatePointMeta(pm, amp_part, amp_varmeta_map, True)
        pm = self.updatePointMeta(pm, cascode_part, casc_varmeta_map, True)
        
        #build functions
        vss, vdd = str(self.ss.vss), str(self.ss.vdd)
        
        amp_functions = amp_varmeta_map
        amp_functions['use_pmos'] = '(1-Drail_is_vdd)'
        amp_functions['Ids'] = 'Ibias'
        amp_functions['Vgs'] = "(Vin-Vout)*(1-2*((1-Drail_is_vdd)==1))"
        amp_functions['Vds'] = "switchAndEval((1-Drail_is_vdd), {" + \
                               "1:'Vout-"+ vss + "', " + \
                               "0:'"+ vdd + "-Vout'})"       
              
        casc_functions = casc_varmeta_map
        casc_functions['use_pmos'] = '(1-Drail_is_vdd)'
        casc_functions['Ids'] = 'Ibias'
        casc_functions['Vs'] = "switchAndEval((1-Drail_is_vdd), {" + \
                               "1:'"+ vdd + "', " + \
                               "0:'"+ vss + "'})"   
        casc_functions['Vds'] = "switchAndEval((1-Drail_is_vdd), {" + \
                               "1:'"+ vdd + "-Vout', " + \
                               "0:'Vout'})"            
        #build the main part
        
        part = CompoundPart(['Drail','Srail','Vin','Vout'], pm, name)
        
        part.addPart(amp_part, {'D':'Drail','G':'Vin','S':'Vout'},
                     amp_functions)
        part.addPart(cascode_part, {'D':'Vout','S':'Srail'}, casc_functions)
                
        self._parts[name] = part
        return part

    def levelShifterOrWire(self):
        """
        Description: this is merely a FlexPart which can select
          a levelShifter or wire part
          
        Ports: Vin, Iout, Drail, Srail
        
        Variables: chosen_part_index (=='use_wire'),
          Drail_is_vdd,
          amp_W, amp_L,
          cascode_do_stack,
          cascode_D_W, cascode_D_L, cascode_D_Vbias,
          cascode_S_W, cascode_S_L, cascode_S_Vbias

        Variable breakdown:
          For overall part: chosen_part_index (=='use_wire')
            0: use levelShifter
            1: use wire
          For levelShifter: 1:1 mapping with levelShifter vars
          For wire:  <<none>>
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        shifter_part = self.levelShifter()
        wire_part = self.wire()

        #build the point_meta (pm)
        pm = PointMeta({})
        shifter_varmap = shifter_part.unityVarMap()
        pm = self.updatePointMeta(pm, shifter_part, shifter_varmap)
                                  
        #build the main part
        part = FlexPart(['Drail','Srail','Vin','Vout'], pm, name)
        
        part.addPartChoice(shifter_part, shifter_part.unityPortMap(),
                           shifter_varmap)
        part.addPartChoice(wire_part, {'1':'Vin','2':'Vout'}, {})
                
        self._parts[name] = part
        return part

    def levelShifterOrWire_VddGndPorts(self):
        """
        Description: Vdd and gnd are ports rather than Drail and Srail.
          The choice of how to allocate Vdd and gnd is
          done by 'chosen_part_index'
          
        Ports: Vin, Iout, Vdd, gnd
        
        Variables: chosen_part_index (==Drail_is_vdd),
          Drail_is_vdd,
          use_wire,
          amp_W, amp_L,
          cascode_do_stack,
          cascode_D_W, cascode_D_L, cascode_D_Vbias,
          cascode_S_W, cascode_S_L, cascode_S_Vbias

        Variable breakdown:
          For overall part: chosen_part_index (=='Drail_is_vdd')
            0: set Drail to gnd, and Srail to vdd
            1: set Drail to vdd, and Srail to gnd
          For levelShifterOrWire: 1:1 mapping, except chosen_part_index=use_wire
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        emb_part = self.levelShifterOrWire()

        #build the point_meta (pm)
        pm = PointMeta({})
        emb_varmap = emb_part.unityVarMap()
        pm = self.updatePointMeta(pm, emb_part, emb_varmap)
        del pm['chosen_part_index']
        pm['use_wire'] = self.buildVarMeta('bool_var','use_wire')

        #build the functions
        emb_functions = emb_varmap
        emb_functions['chosen_part_index'] = 'use_wire'
                                  
        #build the main part
        part = FlexPart(['Vin','Vout','Vdd','gnd'], pm, name)
        
        part.addPartChoice(emb_part,
                           {'Vin':'Vin','Vout':'Vout',
                            'Drail':'gnd','Srail':'Vdd'},
                           emb_varmap)
        part.addPartChoice(emb_part,
                           {'Vin':'Vin','Vout':'Vout',
                            'Drail':'Vdd','Srail':'gnd'},
                           emb_varmap)
                
        self._parts[name] = part
        return part


    def viFeedback_levelShifter(self):
        """
        Description: voltage-current feedback implemented
         as a source follower
          
        Ports: Ifpos, Ifneg, VsensePos, VsenseNeg, loadrail, opprail
        
        Variables: C,
          use_pmos,
          amp_W, amp_L,
          cascode_do_stack,
          cascode_D_W, cascode_D_L, cascode_D_Vbias,
          cascode_S_W, cascode_S_L, cascode_S_Vbias

        Variable breakdown:
          For levelShifter: 1:1 mapping of all vars except C
          For capacitor: C=C
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        sf_part = self.levelShifter()
        cap_part = self.capacitor()

        #build the point_meta (pm)
        pm = PointMeta({})
        sf_varmap = sf_part.unityVarMap()
        cap_varmap = cap_part.unityVarMap()
        
        pm = self.updatePointMeta(pm, sf_part, sf_varmap)
        del pm['Drail_is_vdd']
        pm = self.updatePointMeta(pm, cap_part, cap_varmap)
                                  
        #build the main part
        part = CompoundPart(['loadrail','opprail',
                             'Ifpos','Ifneg',
                             'VsensePos','VsenseNeg'],
                            pm, name)
        
        n1 = part.addInternalNode()

        sf_varmap['Drail_is_vdd'] = '(1-use_pmos)'
        part.addPart(sf_part, {'Drail':'loadrail','Srail':'opprail',
                               'Vin':'VsensePos','Iout':n1}, sf_varmap)
        part.addPart(cap_part, {'1':'Ifpos','2':n1}, cap_varmap)

                
        self._parts[name] = part
        return part

    def viFeedback(self):
        """
        Description: voltage-current feedback, with many options
          for implementation: capacitor, RC_series, and
          viFeedback_levelShifter
          
        Ports: Ifpos, Ifneg, VsensePos, VsenseNeg, loadrail, opprail
        
        Variables: chosen_part_index, R, C,
          use_pmos,
          amp_W, amp_L,
          cascode_do_stack,
          cascode_D_W, cascode_D_L, cascode_D_Vbias,
          cascode_S_W, cascode_S_L, cascode_S_Vbias

        Variable breakdown:
          For overall part: chosen_part_index
            0: use capacitor
            1: use RC_series
            2: use viFeedback_levelShifter
          For capacitor: C=C
          For RC_series: R=R, C=C
          For viFeedback_levelShifter: 1:1 mapping of all vars except R
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cap_part = self.capacitor()
        rc_part = self.RC_series()
        visf_part = self.viFeedback_levelShifter()
        
        #build the point_meta (pm)
        pm = PointMeta({})
        cap_varmap = cap_part.unityVarMap()
        rc_varmap = rc_part.unityVarMap()
        visf_varmap = visf_part.unityVarMap()

        pm = self.updatePointMeta(pm, cap_part, cap_varmap)
        pm = self.updatePointMeta(pm, rc_part, rc_varmap, True)
        pm = self.updatePointMeta(pm, visf_part, visf_varmap, True)

        #build the main part
        part = FlexPart(['loadrail','opprail',
                         'Ifpos','Ifneg',
                         'VsensePos','VsenseNeg'],
                        pm, name)
        
        part.addPartChoice( cap_part, {'1':'Ifpos','2':'VsensePos'}, cap_varmap )
        part.addPartChoice( rc_part, {'N1':'Ifpos','N2':'VsensePos'}, rc_varmap )
        part.addPartChoice( visf_part, visf_part.unityPortMap(), visf_varmap )
                
        self._parts[name] = part
        return part

    def sourceDegen(self):
        """
        Description: Source degeneration of an amplifier.
          Instantiate as a wire, OR resistor.
          
        Ports: D, S
        
        Variables: chosen_part_index, V, I
        
        Variable breakdown:
          For overall part: chosen_part_index
            0: use wire
            1: use resistor
          For wire: <<none>>
          For resistor: V, I
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        wire_part = self.wire()
        resistor_part = self.resistor()

        #build the point_meta (pm)
        pm = PointMeta({})
        wire_varmeta_map = wire_part.unityVarMap()
        resistor_varmeta_map = resistor_part.unityVarMap()
        
        pm = self.updatePointMeta(pm, wire_part, wire_varmeta_map)
        pm = self.updatePointMeta(pm, resistor_part, resistor_varmeta_map)
        
        #build functions
        wire_functions = wire_varmeta_map
        resistor_functions = resistor_varmeta_map

        #build the main part
        part = FlexPart(['D', 'S'], pm, name)
        
        part.addPartChoice( wire_part, {'1':'D','2':'S'}, wire_functions)
        part.addPartChoice( resistor_part, {'1':'D','2':'S'}, resistor_functions)
                
        self._parts[name] = part
        return part


    def cascodeDevice(self):
        """
        Description: biasedMos OR gainBoostedMos.  Used in an inputCascodeStage.
        
        NOTE: currently just biasedMos because gainBoostedMos not implemented!!
          
        Ports: D, S, loadrail, opprail
        
        Variables: chosen_part_index, loadrail_is_vdd, Vgs, Vds, Vs, L, Ids

        Variable breakdown:
          For overall part: chosen_part_index (=='cascode_recurse')
            0: use biasedMos
            1: use gainBoostedMos (FIXME -- not implemented yet)
          For biasedMos: use_pmos=1-loadrail_is_vdd, unity
          For gainBoostedMos: <<none>> (FIXME)
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        biasedMos_part = self.biasedMos()

        #build the point_meta (pm)
        pm = PointMeta({})
        biasedMos_varmeta_map = biasedMos_part.unityVarMap()
        biasedMos_varmeta_map['use_pmos'] = 'loadrail_is_vdd'
        
        pm = self.updatePointMeta(pm, biasedMos_part, biasedMos_varmeta_map)
        
        #build functions
        biasedMos_functions = biasedMos_varmeta_map

        #build the main part
        part = FlexPart(['D', 'S', 'loadrail', 'opprail'], pm, name)

        biasedMos_funcs = biasedMos_varmeta_map
        biasedMos_funcs['use_pmos'] = '1-loadrail_is_vdd'
        part.addPartChoice( biasedMos_part, {'D':'D','S':'S'}, biasedMos_functions)
                
        self._parts[name] = part
        return part

    
    def cascodeDeviceOrWire(self):
        """
        Description: cascodeDevice OR wire
          
        Ports: D, S, loadrail, opprail
        
        Variables: chosen_part_index,
          cascode_recurse, loadrail_is_vdd, Vgs, Vds, L, Vs, Ids

        Variable breakdown:
          For overall part: chosen_part_index (=='cascode_is_wire')
            0: use cascodeDevice
            1: use wire
          For cascodeDevice: chosen_part_index=cascode_recurse,
            <<others are 1-1 mapping>>
          For wire: <<none>>

        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cascodeDevice_part = self.cascodeDevice()
        wire_part = self.wire()

        #build the point_meta (pm)
        pm = PointMeta({})
        cascodeDevice_varmeta_map = cascodeDevice_part.unityVarMap()
        cascodeDevice_varmeta_map['chosen_part_index'] = 'cascode_recurse'
        wire_varmeta_map = wire_part.unityVarMap()
        
        pm = self.updatePointMeta(pm, cascodeDevice_part, cascodeDevice_varmeta_map)
        pm = self.updatePointMeta(pm, wire_part, wire_varmeta_map)
        
        #build functions
        cascodeDevice_functions = cascodeDevice_varmeta_map
        wire_functions = wire_varmeta_map

        #build the main part
        part = FlexPart(['D', 'S', 'loadrail', 'opprail'], pm, name)
        
        part.addPartChoice( cascodeDevice_part,
                            cascodeDevice_part.unityPortMap(),
                            cascodeDevice_functions)
        part.addPartChoice( wire_part, {'1':'D','2':'S'}, wire_functions)
                
        self._parts[name] = part
        return part


    def bleederPart(self):
        """
        Description: provides a choice between several "bleeder" parts. Bleeders
        are elements that divert ('bleed') some of the current into a separate
        branch. This is used e.g. to increase the transconductance of an input
        transistor.
          
        Ports: Iin, opprail
        
        Variables: chosen_part_index: [0 = no bleeder; 1=biased mos; 2=cascoded biased mos]
                   Ibias,
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        opencircuit_part = self.openCircuit()
        onemos_part = self.biasedMos()
        twomos_part = self.twoBiasedMoses()

        #build the point_meta (pm)
        pm = PointMeta({})
        opencircuit_varmeta_map = opencircuit_part.unityVarMap()
        onemos_varmeta_map = {'use_pmos':'use_pmos',
                              'Vgs':'bleeder_Vgs','L':'bleeder_L',
                              'Vds':'Vds','Vs':'Vs',
                              'Ids':'Ibias'}
        twomos_varmeta_map = {'use_pmos':'use_pmos', 'fracVi':'fracBleeder',
                              'S_Vgs':'bleeder_Vgs','S_L':'bleeder_L',
                              'D_Vgs':'bleeder_cascode_Vgs','D_L':'bleeder_cascode_L',
                              'Vds':'Vds','Vs':'Vs',
                              'Ids':'Ibias'}

        pm = self.updatePointMeta(pm, opencircuit_part, opencircuit_varmeta_map)
        pm = self.updatePointMeta(pm, onemos_part, onemos_varmeta_map, True)
        pm = self.updatePointMeta(pm, twomos_part, twomos_varmeta_map, True)

        #build the functions
        opencircuit_functions = opencircuit_varmeta_map
        onemos_functions = onemos_varmeta_map
        twomos_functions = twomos_varmeta_map

        #build the main part
        part = FlexPart(['D', 'S'], pm, name)

        part.addPartChoice( open_part, {'1':'D','2':'S'}, opencircuit_functions )
        part.addPartChoice( onemos_part, {'D':'D','S':'S'}, onemos_functions )
        part.addPartChoice( twomos_part, {'D':'D','S':'S'}, twomos_functions )

        self._parts[name] = part
        return part

    def inputCascode_Stacked(self):
        """
        Description: inputCascode stage in a 'stacked' (as opposed to folded)
          configuration.  Has a cascodeDeviceOrWire, mos3, and sourceDegen.
          
        Ports: Vin, Iout, loadrail, opprail
        
        Variables: input_is_pmos, 
          Ibias, Vds, Vs
          ampmos_Vgs, ampmos_L, fracAmp
          cascode_is_wire, cascode_Vgs, cascode_L, cascode_recurse,
          degen_choice, fracDeg
          
        Variable breakdown:
            ...
            
        Remember: if input_is_pmos=True, then loadrail_is_Vdd=False; OR
                  if input_is_pmos=False, then loadrail_is_Vdd=True; OR
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cascode_part = self.cascodeDeviceOrWire()
        ampmos_part = self.saturatedMos3()
        degen_part = self.sourceDegen()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Ibias']= self.buildVarMeta('Ibias')
        pm['fracAmp']= self.buildVarMeta('frac','fracAmp')
        pm['fracDeg']= self.buildVarMeta('frac','fracDeg')

        cascode_varmeta_map = {
            'chosen_part_index':'cascode_is_wire',
            'cascode_recurse':'cascode_recurse',
            'loadrail_is_vdd':'input_is_pmos',
            'Ids':'IGNORE',
            'Vds':'Vds', # to be calculated
            'Vgs':'cascode_Vgs',
            'L':'cascode_L',
            'Vs':'Vs'
            }

        ampmos_varmeta_map = {
            'Ids':'IGNORE',
            'Vgs':'ampmos_Vgs',
            'Vds':'Vds',
            'L':'ampmos_L',
            'use_pmos':'input_is_pmos'
            }
        degen_varmeta_map = {
            'I':'IGNORE',
            'V':'Vs',
            'chosen_part_index':'degen_choice'
            }
        pm = self.updatePointMeta(pm, cascode_part, cascode_varmeta_map)
        pm = self.updatePointMeta(pm, ampmos_part, ampmos_varmeta_map, True)
        pm = self.updatePointMeta(pm, degen_part, degen_varmeta_map, True)

        use_cascode = '(cascode_is_wire != 1)'
        use_degeneration = '(degen_choice != 0)'

        # these are the equations that matter
        Vds_casc = '( Vds * ( 1.0 - fracAmp) )'
        
        # if cascoded:
        #     Vds_amp_and_deg = Vds * fracAmp
        # else:
        #     Vds_amp_and_deg = Vds
        # resulting in:
        Vds_amp_and_deg = 'Vds * ( 1 + ('+use_cascode+') * ( fracAmp - 1 ))'
        
        # if degenerated:
        #    Vds_amp = Vds_amp_and_deg * ( 1 - fracDeg )
        # else:
        #    Vds_amp = Vds_amp_and_deg
        # resulting in:
        Vds_amp = '(' + Vds_amp_and_deg + ') * ( 1 + (' + use_degeneration + ') * (1 - fracDeg - 1))'
        
        Vds_deg = '(' + Vds_amp_and_deg + ') * fracDeg'

        # this source node voltage we have to calculate
        Vs_casc = 'Vs + Vds * fracAmp * ( 1 - 2 * (input_is_pmos == 1))'
        
        #build functions
        cascode_functions = cascode_varmeta_map
        cascode_functions['Ids'] = 'Ibias'
        cascode_functions['loadrail_is_vdd'] = '1 - input_is_pmos' ## FIXME: should be simply use_pmos
        cascode_functions['Vds'] = Vds_casc
        cascode_functions['Vs'] = Vs_casc
        
        ampmos_functions = ampmos_varmeta_map
        ampmos_functions['Ids'] = 'Ibias'
        ampmos_functions['Vds'] = Vds_amp

        degen_functions = degen_varmeta_map
        degen_functions['I'] = 'Ibias'
        degen_functions['V'] = Vds_deg

        #build the main part
        part = CompoundPart(['Vin', 'Iout', 'loadrail', 'opprail'], pm, name)

        n_cascode_ampmos = part.addInternalNode()
        n_ampmos_degen = part.addInternalNode()

        part.addPart(cascode_part,
                     {'D':'Iout', 'S':n_cascode_ampmos,
                      'loadrail':'loadrail','opprail':'opprail'},
                     cascode_functions)
        part.addPart(ampmos_part,
                     {'D':n_cascode_ampmos,'G':'Vin','S':n_ampmos_degen},
                     ampmos_functions)
        part.addPart(degen_part, {'D':n_ampmos_degen,'S':'opprail'},
                     degen_functions)

        self._parts[name] = part
        return part

    def simpleMosFolder(self):
        """
        Description: one or two biased mos's, stacked. used as a folder.
          
        Ports: D,D2, S (D == D2)
        
        Variables: chosen_part_index,
           use_pmos, Vds, Vs, Ids, D_Vgs, D_l, S_Vgs, S_L, fracVi
         
         fracVi determines the voltage at the internal node (as a fraction of Vds, scaled between 0 and 1): 
            Vi = Vs + Vds * fracVi * (1 - 2 * (use_pmos==1))           

        Variable breakdown:
          For overal part: chosen_part_index (==do_stack)
            0 : use biasedMos
            1 : use twoBiasedMoses
          biasedMos: use_pmos=use_pmos, Vds=Vds, Vgs=Vgs, L=S_L, Vs=Vs, Ids=Ids
          twoBiasedMoses: 1:1 mapping (except chosen_part_index)
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        onemos_part = self.biasedMosTwoDrain()
        twomos_part = self.twoBiasedMosesTwoDrain()

        #build the point_meta (pm)
        pm = PointMeta({})
        onemos_varmeta_map = {'use_pmos':'use_pmos',
                              'Vgs':'S_Vgs','Vds':'Vds','L':'S_L',
                              'Vs':'Vs','Ids':'Ids'}
        twomos_varmeta_map = twomos_part.unityVarMap()

        pm = self.updatePointMeta(pm, twomos_part, twomos_varmeta_map)

        #build the functions
        onemos_functions = onemos_varmeta_map
        twomos_functions = twomos_varmeta_map

        #build the main part
        part = FlexPart(['D','D2', 'S'], pm, name)

        part.addPartChoice( onemos_part, {'D':'D','D2':'D2','S':'S'}, onemos_functions )
        part.addPartChoice( twomos_part, {'D':'D','D2':'D2','S':'S'}, twomos_functions )
        part.addPartChoice( twomos_part, {'D':'D','D2':'D2','S':'S'}, twomos_functions ) # to match up with the current mirror

        self._parts[name] = part
        return part
     
    def folderPart(self):
        """
        Description: provides a choice between several "folder" parts. The main thing
          they all have in common is that they 'bounce' the AC current back to the
          loadrail. Note that the polarity of the 'bounce' is important: we can't use
          a current mirror here since it doesn't fold.
          
        Ports: Iin, Iout, opprail
        
        Variables: chosen_part_index: [0 = mos3]
                   Ibias_in, Ibias_out
        
        
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        simplefolder_part = self.simpleMosFolder()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Ibias_in'] = self.buildVarMeta('Ibias','Ibias_in')
        pm['Ibias_out'] = self.buildVarMeta('Ibias','Ibias_out')

        simplefolder_varmeta_map = {
            'fracVi':'fracFolder',
            'Vds':'Vds',
            'S_Vgs':'folderbias_Vgs',
            'S_L':'folderbias_L',
            'D_Vgs':'foldercascode_Vgs',
            'D_L':'foldercascode_L',
            'use_pmos':'use_pmos',
            'Vs':'Vs',
            'Ids':'IGNORE',
            'chosen_part_index':'folder_use_cascode',
        }

        pm = self.updatePointMeta(pm, simplefolder_part, simplefolder_varmeta_map)

        #build functions
        simplefolder_functions = simplefolder_varmeta_map
        simplefolder_functions['Ids'] = 'Ibias_in+Ibias_out'
        
        #build the main part
        part = FlexPart(['Iin', 'Iout', 'common'], pm, name)

        part.addPartChoice( simplefolder_part,
                            {'D':'Iin', 'D2': 'Iout', 'S': 'common'},
                            simplefolder_functions )
 
        self._parts[name] = part
        return part

    def inputCascode_Folded(self):
        """
        Description: inputCascode stage in a 'folded' (as opposed to stacked)
          configuration.  Has a cascodeDeviceOrWire, mos3, sourceDegen,
          and biasedMos.
          
        Ports: Vin, Iout, loadrail, opprail
        
        Variables: input_is_pmos, 
          Ibias, Ibias2, Vds, Vs
          ampmos_Vgs, ampmos_L, fracAmp
          cascode_is_wire, cascode_Vgs, cascode_L, cascode_recurse,
          degen_choice, fracDeg
          inputbias_L, inputbias_Vgs
          
        Variable breakdown:
          ...

        Remember: if input_is_pmos=True, then loadrail_is_Vdd=True; OR
                  if input_is_pmos=False, then loadrail_is_Vdd=False
                  
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cascode_part = self.cascodeDeviceOrWire()
        ampmos_part = self.saturatedMos3()
        degen_part = self.sourceDegen()
        folder_part = self.folderPart()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Ibias'] = self.buildVarMeta('Ibias')
        pm['Ibias2'] = self.buildVarMeta('Ibias','Ibias2')
        pm['fracAmp'] = self.buildVarMeta('frac','fracAmp')
        pm['fracDeg'] = self.buildVarMeta('frac','fracDeg')
        
        cascode_varmeta_map = {
            'chosen_part_index':'cascode_is_wire',
            'cascode_recurse':'cascode_recurse',
            'loadrail_is_vdd':'input_is_pmos',
            'Ids':'IGNORE',
            'Vds':'Vds',
            'Vgs':'cascode_Vgs',
            'L':'cascode_L',
            'Vs':'Vs'
            }
        
        ampmos_varmeta_map = {
            'Ids':'IGNORE',
            'Vgs':'ampmos_Vgs',
            'Vds':'Vds',
            'L':'ampmos_L',
            'use_pmos':'input_is_pmos'
            }
        degen_varmeta_map = {
            'I':'IGNORE',
            'V':'Vs',
            'chosen_part_index':'degen_choice'
            }

        folder_varmeta_map = {
            'Vds':'Vds', 'folderbias_Vgs':'inputbias_Vgs',
            'folderbias_L':'inputbias_L',
            'use_pmos':'input_is_pmos',
            'Vs':'Vs',
            'Ibias_in':'Ibias','Ibias_out':'Ibias2',
            'chosen_part_index':'chosen_folder_type',
            'folder_use_cascode':'folder_use_cascode',
            'foldercascode_L':'foldercascode_L',
            'foldercascode_Vgs':'foldercascode_Vgs',
            'fracFolder':'fracFolder',
            }

        pm = self.updatePointMeta(pm, cascode_part, cascode_varmeta_map)
        pm = self.updatePointMeta(pm, ampmos_part, ampmos_varmeta_map, True)
        pm = self.updatePointMeta(pm, degen_part, degen_varmeta_map, True)
        pm = self.updatePointMeta(pm, folder_part, folder_varmeta_map, True)

        use_cascode = '(cascode_is_wire != 1)'
        dont_use_cascode = '(cascode_is_wire != 1)'
        use_degeneration = '(degen_choice != 0)'

        # these are the equations that matter
        # the voltage of the opposite rail
        Voprail = '(' + str(self.ss.vdd) + ' * ( input_is_pmos == 0 ) + ' + str(self.ss.vss) + ' * ( input_is_pmos == 1 ) )'
        
        # the DS voltage over the cascode is kept the same when changing from
        # stacked to folded. ensures the smooth transition
        # NOTE: this makes that the 'definition' of fracAmp is different for stacked and folded
        # but that there are points where changing only the folding vs stacked state will result
        # in a working setup
        Vds_casc = 'Vds * (1 - fracAmp)'
        
        # the voltage over the ampmos and degeneration
        # this is Vds + Vds_casc
        Vds_amp_and_deg = '( Vds * (2 - fracAmp) )'

        # so the voltage over the folder is:
        # abs(Voprail - Vs) - Vds_amp_and_deg
        # limited to 0
        Vds_folder = 'max(0, ( abs('+ Voprail +' - Vs) - ( Vds * (2 - fracAmp) ) ) )'

        # if degenerated:
        #    Vds_amp = Vds_amp_and_deg * ( 1 - fracDeg )
        # else:
        #    Vds_amp = Vds_amp_and_deg
        # resulting in:
        Vds_amp = '(' + Vds_amp_and_deg + ') * ( 1 + (' + use_degeneration + ') * (1 - fracDeg - 1))'
        
        Vds_deg = '(' + Vds_amp_and_deg + ') * fracDeg'

        # this source node voltage we have to calculate
        Vs_casc = 'Vs + ( 1 - 2*(input_is_pmos == 1) ) * (' + Vds_amp_and_deg + ')'

        #build functions
        cascode_functions = cascode_varmeta_map
        cascode_functions['Ids'] = 'Ibias2'
        cascode_functions['loadrail_is_vdd'] = 'input_is_pmos'
        cascode_functions['Vds'] = Vds_casc
        cascode_functions['Vs'] = Vs_casc
        
        ampmos_functions = ampmos_varmeta_map
        ampmos_functions['Ids'] = 'Ibias'
        ampmos_functions['Vds'] = Vds_amp
        
        degen_functions = degen_varmeta_map
        degen_functions['I'] = 'Ibias'
        degen_functions['V'] = Vds_deg
       
        folder_functions = folder_varmeta_map
        folder_functions['use_pmos'] = '1 - input_is_pmos'
        folder_functions['Vs'] = Voprail
        folder_functions['Vds'] = Vds_folder

        #build the main part
        part = CompoundPart(['Vin', 'Iout', 'loadrail', 'opprail'], pm, name)
        
        n_ampmos_degen = part.addInternalNode()
        n_folder_ampmos_inputbias = part.addInternalNode()
        n_cascode_folder_inputbias = part.addInternalNode()

        part.addPart(cascode_part,
                     {'D':'Iout', 'S':n_cascode_folder_inputbias,
                      'loadrail':'loadrail','opprail':'opprail'},
                     cascode_functions)
        part.addPart(ampmos_part,
                     {'D':n_folder_ampmos_inputbias,'G':'Vin',
                      'S':n_ampmos_degen},
                     ampmos_functions)
        part.addPart(degen_part, {'D':n_ampmos_degen,'S':'loadrail'},
                     degen_functions)

        part.addPart(folder_part,
                     {'Iin':n_folder_ampmos_inputbias, 'Iout': n_cascode_folder_inputbias, 'common':'opprail'},
                     folder_functions)

        self._parts[name] = part
        return part

    def inputCascodeFlex(self):
        """
        Description: choose between folded or stacked input cascode stage.
          
        Ports: Vin, Iout, loadrail, opprail
        
        Variables: chosen_part_index, input_is_pmos, 
          Ibias, Ibias2, Vds, Vs
          ampmos_Vgs, ampmos_L, fracAmp
          cascode_is_wire, cascode_Vgs, cascode_L, cascode_recurse,
          degen_choice, fracDeg
          inputbias_L, inputbias_Vgs
        
        Variable breakdown:
          For overall part: chosen_part_index
            0: use inputCascode_Stacked
            1: use inputCascode_Folded
          For inputCascode_Stacked:
            All the input variables, as is, except chosen_part_index and
            the inputbias_XXX vars.
          For inputCascode_Folded:
            All the input variables, as is, except chosen_part_index and
            cascode_is_wire.
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        stacked_part = self.inputCascode_Stacked()
        folded_part = self.inputCascode_Folded()

        #build the point_meta (pm)
        pm = PointMeta({})
        stacked_varmeta_map = stacked_part.unityVarMap()
        folded_varmeta_map = folded_part.unityVarMap()

        pm = self.updatePointMeta(pm, stacked_part, stacked_varmeta_map)
        pm = self.updatePointMeta(pm, folded_part, folded_varmeta_map,
                                  True)
        
        #build functions
        stacked_functions = stacked_varmeta_map
        folded_functions = folded_varmeta_map
        
        #build the main part
        part = FlexPart(['Vin', 'Iout', 'loadrail', 'opprail'], pm, name)

        part.addPartChoice( stacked_part,
                            stacked_part.unityPortMap(),
                            stacked_functions )
        part.addPartChoice( folded_part,
                            folded_part.unityPortMap(),
                            folded_functions )
        
        self._parts[name] = part
        return part


    def inputCascodeStage(self):
        """
        Description: Wraps a single embedded part, an inputCascodeFlex.
          AUTOMATICALLY chooses between folded vs. stacked 
          based on loadrail_is_vdd and input_is_pmos.
          
        Ports: Vin, Iout, loadrail, opprail
        
        Variables:
          loadrail_is_vdd, input_is_pmos,
          Ibias, Ibias2, Vds, Vs
          ampmos_Vgs, ampmos_L, fracAmp
          cascode_is_wire, cascode_Vgs, cascode_L, cascode_recurse,
          degen_choice, fracDeg
          inputbias_L, inputbias_Vgs

        
        Variable breakdown:
          For overall part: chosen_part_index (=='use_wire')
            0: use inputCascode_Stacked
            1: use inputCascode_Folded
          For inputCascode_Stacked:
            All the input variables, as is, except chosen_part_index and
            the inputbias_XXX vars.
          For inputCascode_Folded:
            All the input variables, as is, except chosen_part_index and
            cascode_is_wire.
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        emb_part = self.inputCascodeFlex()

        #build the point_meta (pm)
        # -nearly identical to inputCascodeFlex, except it adds one var
        #  and removes another
        pm = PointMeta({})
        emb_varmeta_map = emb_part.unityVarMap()
        pm = self.updatePointMeta(pm, emb_part, emb_varmeta_map)
        pm.addVarMeta( self.buildVarMeta('bool_var', 'loadrail_is_vdd') )#add var
        del pm['chosen_part_index']                                      #remove
        
        
        #build functions
        # -here is where we auto-choose between stacked and folded: note the '=='
        emb_functions = emb_varmeta_map
        emb_functions['chosen_part_index'] = '(input_is_pmos == loadrail_is_vdd)'
        
        #build the main part
        part = CompoundPart(['Vin', 'Iout', 'loadrail', 'opprail'], pm, name)


        part.addPart( emb_part, emb_part.unityPortMap(), emb_functions )
        
        self._parts[name] = part
        return part

    def ssViInput(self):
        """
        Description: Single-ended input voltage in, single-ended current out
          stage.  Turns out that this is merely a wrapper for inputCascodeStage
          (but with a more appropriate name for higher-level use.)
          
        Ports: Vin, Iout, loadrail, opprail
        
        Variables:
          loadrail_is_vdd, input_is_pmos,
          Ibias, Ibias2, Vds, Vs
          ampmos_Vgs, ampmos_L, fracAmp
          cascode_is_wire, cascode_Vgs, cascode_L, cascode_recurse,
         degen_choice, fracDeg
          inputbias_L, inputbias_Vgs

        
        Variable breakdown:
          For inputCascodeStage:
            All, a 1:1 mapping.
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        emb_part = self.inputCascodeStage()

        #build the point_meta (pm)
        pm = PointMeta({})
        emb_varmeta_map = emb_part.unityVarMap()
        pm = self.updatePointMeta(pm, emb_part, emb_varmeta_map)
        
        #build functions
        emb_functions = emb_varmeta_map
        
        #build the main part
        part = CompoundPart(['Vin', 'Iout', 'loadrail', 'opprail'], pm, name)

        part.addPart( emb_part, emb_part.unityPortMap(), emb_functions)
        
        self._parts[name] = part
        return part

    def ssIiLoad_Cascoded(self):
        """
        Description: This is has a biasedMos transistor as the main load device,
           plus a cascodeDevice which may amplify the effect of the load.
          
        Ports: Iout, loadrail, opprail
            
        Variables: loadrail_is_vdd, loadcascode_recurse,
            Ibias, Vds, Vs,
            mainload_L, mainload_Vgs, fracLoad,
            loadcascode_Vgs, loadcascode_L
            
        Variable breakdown:

          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        mainload_part = self.biasedMos()
        loadcascode_part = self.cascodeDevice()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Ibias']= self.buildVarMeta('Ibias')
        pm['fracLoad'] = self.buildVarMeta('frac','fracLoad')
        
        mainload_varmeta_map = {
            'Vds':'Vds', 'Vgs':'mainload_Vgs',
            'L':'mainload_L',
            'use_pmos':'loadrail_is_vdd',
            'Vs':'Vs',
            'Ids':'IGNORE',
            }                      
        loadcascode_varmeta_map = {
            'chosen_part_index':'loadcascode_recurse',
            'loadrail_is_vdd':'loadrail_is_vdd',
            'Ids':'IGNORE',
            'Vds':'Vds',
            'Vgs':'loadcascode_Vgs',
            'L':'loadcascode_L',
            'Vs':'Vs'
            }
            
        pm = self.updatePointMeta(pm, mainload_part, mainload_varmeta_map)
        pm = self.updatePointMeta(pm, loadcascode_part,
                                  loadcascode_varmeta_map, True)
        
        #build functions
        mainload_functions = mainload_varmeta_map
        mainload_functions['Ids'] = 'Ibias'
        mainload_functions['Vs'] =  'Vs'
        # if a cascoded load is used, then the fracLoad should be accounted for
        # else the VDS over the mainload is the VDS over this block
        # this formulation avoids a switchAndEval
        mainload_functions['Vds'] =  "Vds * (1 + (loadcascode_recurse!=0) * (fracLoad-1))"
        
        loadcascode_functions = loadcascode_varmeta_map
        loadcascode_functions['loadrail_is_vdd'] = '1-loadrail_is_vdd'
        loadcascode_functions['Ids'] = 'Ibias'
        # these functions are only relevant when cascoding is used, so no
        # need to check for that.
        # The Vds over the cascode is merely the Vds over the block minus the Vds
        # over the mainload
        loadcascode_functions['Vds'] =  "Vds * (1 - fracLoad)"
        # the reference node of the cascode depends on where the load is attached to
        loadcascode_functions['Vs'] = "(Vs - Vds * fracLoad * (1 - 2 * (loadrail_is_vdd==0)))"

        #build the main part
        part = CompoundPart(['Iout', 'loadrail', 'opprail'], pm, name)

        n_mos_to_cascode = part.addInternalNode()
        
        part.addPart( mainload_part,
                      {'S':'loadrail','D':n_mos_to_cascode},
                      mainload_functions)
        part.addPart( loadcascode_part,
                      {'S':n_mos_to_cascode,'D':'Iout',
                       'loadrail':'loadrail', 'opprail':'opprail'},
                      loadcascode_functions )
        
        self._parts[name] = part
        return part


    def ssIiLoad(self):
        """
        Description: single-ended I in, single-ended I out load.
          Is a Flex part which may be: resistor, biasedMos, or ssIiLoad_Cascoded.
          
        Ports: Iout, loadrail, opprail
        
        Variables:

        Variable breakdown:

          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        res_part = self.resistor()
        biasedmos_part = self.biasedMos()
        cascodemos_part = self.ssIiLoad_Cascoded()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Ibias']= self.buildVarMeta('Ibias')

        res_varmeta_map = {'V':'Vds',
                           'I':'IGNORE',
                           }
        biasedmos_varmeta_map = {
                'Vds':'Vds', 'Vgs':'Vgs',
                'L':'L',
                'use_pmos':'loadrail_is_vdd',
                'Vs':'Vs',
                'Ids':'IGNORE',
                      }
        cascodemos_varmeta_map = {
                'loadrail_is_vdd':'loadrail_is_vdd',
                'Vds':'Vds','Vs':'Vs',
                'mainload_L':'L',
                'mainload_Vgs':'Vgs',
                'loadcascode_recurse':'loadcascode_recurse',
                'loadcascode_L':'loadcascode_L',
                'loadcascode_Vgs':'loadcascode_Vgs',
                'Ibias':'Ibias',
                'fracLoad':'fracLoad',
            }
    
        #pm = self.updatePointMeta(pm, res_part, res_varmeta_map)
        pm = self.updatePointMeta(pm, biasedmos_part, biasedmos_varmeta_map,
                                  True)
        pm = self.updatePointMeta(pm, cascodemos_part, cascodemos_varmeta_map,
                                  True)
        
        #build functions
        res_functions = res_varmeta_map
        res_functions['I'] = 'Ibias'
        
        biasedmos_functions = biasedmos_varmeta_map
        biasedmos_functions['Ids'] = 'Ibias'
        
        cascodemos_functions = cascodemos_varmeta_map
        
        #build the main part
        part = FlexPart(['Iout', 'loadrail', 'opprail'], pm, name)

        part.addPartChoice( res_part, {'1':'loadrail','2':'Iout'},
                            res_functions)
        part.addPartChoice( biasedmos_part,  {'D':'Iout','S':'loadrail'},
                            biasedmos_functions)
        part.addPartChoice( cascodemos_part, cascodemos_part.unityPortMap(),
                            cascodemos_functions)
        
        self._parts[name] = part
        return part


    def ssViAmp1(self):
        """
        Description: single-ended V in, single-ended I out, 1-stage
         (common-source) amplifier.
          
        Ports: Vin, Iout, loadrail, opprail
        
        Variables:

          
        Variable breakdown:
          For ssViInput:

            
          For ssIiLoad:

          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        input_part = self.ssViInput()
        load_part = self.ssIiLoad()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['Ibias']= self.buildVarMeta('Ibias')
        pm['Ibias2']= self.buildVarMeta('Ibias','Ibias2')
        pm['Vout']= self.buildVarMeta('V','Vout')

        input_varmeta_map = {
            'loadrail_is_vdd':'loadrail_is_vdd',
            'input_is_pmos':'input_is_pmos',
            'Ibias':'Ibias',
            'Ibias2':'Ibias2',
            
            'Vds':'IGNORE', 'Vs':'IGNORE',
                
            'cascode_L':'inputcascode_L',
            'cascode_Vgs':'inputcascode_Vgs',
            'cascode_recurse':'inputcascode_recurse',
            'cascode_is_wire':'inputcascode_is_wire',
                 
            'ampmos_Vgs':'ampmos_Vgs', 'ampmos_L':'ampmos_L', 'fracAmp':'ampmos_fracAmp', 
                                  
            'degen_choice':'degen_choice','fracDeg':'degen_fracDeg',
                
            'inputbias_L':'inputbias_L',
            'inputbias_Vgs':'inputbias_Vgs',
            'chosen_folder_type':'chosen_folder_type',
            'folder_use_cascode':'folder_use_cascode',
            'foldercascode_L':'foldercascode_L',
            'foldercascode_Vgs':'foldercascode_Vgs',
            'fracFolder':'fracFolder',
            }
                             
        load_varmeta_map = {
            'chosen_part_index':'load_part_index',
            'loadrail_is_vdd':'loadrail_is_vdd',
            'Ibias':'Ibias', ## NOTE: this will require a hack because if it is folded, we need another bias current
            'Vds':'IGNORE','Vs':'IGNORE',
            'Vgs':'load_Vgs',
            'L':'load_L','fracLoad':'load_fracLoad',
            'loadcascode_recurse':'loadcascode_recurse',
            'loadcascode_L':'loadcascode_L',
            'loadcascode_Vgs':'loadcascode_Vgs',         
            }
    
        pm = self.updatePointMeta(pm, input_part, input_varmeta_map, True)
        pm = self.updatePointMeta(pm, load_part, load_varmeta_map, True)
        

#         del pm['load_part_index']
#         load_varmeta_map['chosen_part_index']=1

        #build functions
        input_functions = input_varmeta_map
        load_functions = load_varmeta_map
        
        # we have to figure out if the input stage is folded or not
        # because it influences the rail voltages and the bias currents
        
        is_folded = '(input_is_pmos==loadrail_is_vdd)'
                                 
        load_functions['Ibias']='((' + is_folded + ' == 1 ) * Ibias2 ) + ' \
                                '((' + is_folded + ' == 0 ) * Ibias )'                                                              
        load_functions['Vds']='(( loadrail_is_vdd == 0 ) * ( Vout - ' + str(self.ss.vss) + ')) +' \
                              '(( loadrail_is_vdd == 1 ) * (' + str(self.ss.vdd) + ' - Vout ))' 
        load_functions['Vs']='(( loadrail_is_vdd == 0 ) * (' + str(self.ss.vss) + ')) +' \
                             '(( loadrail_is_vdd == 1 ) * (' + str(self.ss.vdd) + '))' 


        input_functions['Vds']='(( input_is_pmos == 0 ) * ( Vout - ' + str(self.ss.vss) + ')) +' \
                               '(( input_is_pmos == 1 ) * (' + str(self.ss.vdd) + ' - Vout ))' 
        input_functions['Vs']='(( input_is_pmos == 0 ) * (' + str(self.ss.vss) + ')) +' \
                              '(( input_is_pmos == 1 ) * (' + str(self.ss.vdd) + '))' 

        #build the main part
        part = CompoundPart(['Vin','Iout', 'loadrail', 'opprail'], pm, name)

        part.addPart( input_part, input_part.unityPortMap(), input_functions)

        # -note that the GRAIL pdf wrongly labels the 'Iout' to be 'Iin' (Fig 17)
        part.addPart( load_part, load_part.unityPortMap(), load_functions)
        
        self._parts[name] = part
        return part

        
    def ssViAmp1_VddGndPorts(self):
        """
        Description: Just like ssViAmp1, except it has 'Vdd' and 'gnd'
          as external ports, which are less flexible than ssViAmp1's
          ports of 'loadrail' and 'opprail'.  But it makes it directly
          interfaceable to the outside world (unlike ssViAmp1 on its own).
          
        Ports: Vin, Iout, Vdd, gnd
        
        Variables:
          -like dsViAmp, except replace its 'loadrail_is_vdd'
           with 'chosen_part_index'
        
        Variable breakdown:
          For overall part: chosen_part_index (==loadrail_is_vdd)
            0: set 'loadrail' of ssViAmp1 to 'gnd', and 'opprail' to 'Vdd'
            1: set 'loadrail' of ssViAmp1 to 'Vdd', and 'opprail' to 'gnd'
          For ssViamp1:
            All, a 1:1 mapping, except: loadrail_is_vdd=chosen_part_index
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        amp_part = self.ssViAmp1()

        #build the point_meta (pm)
        # -remember: don't need to add 'chosen_part_index' to pm
        #  because FlexParts do that automatically
        pm = PointMeta({})
        varmeta_map = amp_part.unityVarMap()
        pm = self.updatePointMeta(pm, amp_part, varmeta_map)
        del pm['loadrail_is_vdd']
        del pm['Vout']
        
        
        #build functions
        var_functions = varmeta_map
        var_functions['loadrail_is_vdd'] = 'chosen_part_index'
        var_functions['Vout']='0.9'
        
        #build the main part
        part = FlexPart(['Vin', 'Iout', 'Vdd','gnd'], pm, name)

        part.addPartChoice( amp_part,
                            {'Vin':'Vin', 'Iout':'Iout',
                             'loadrail':'gnd', 'opprail':'Vdd'},
                            var_functions)
        part.addPartChoice( amp_part,
                            {'Vin':'Vin', 'Iout':'Iout',
                             'loadrail':'Vdd', 'opprail':'gnd'},
                            var_functions )

        #build a summaryStr
        part.addToSummaryStr('loadrail is vdd','chosen_part_index')
        part.addToSummaryStr('input is pmos (rather than nmos)', 'input_is_pmos')
        part.addToSummaryStr('folded', 'chosen_part_index == input_is_pmos')
        part.addToSummaryStr('Ibias','Ibias')
        part.addToSummaryStr('Ibias2','Ibias2')
        part.addToSummaryStr('degen_choice (0=wire,1=resistor)', 'degen_choice')
        part.addToSummaryStr('load type (0=resistor,1=biasedMos,'
                             '2=ssIiLoad_Cascoded)', 'load_part_index')
        
        self._parts[name] = part
        return part

    def ssViAmp1_VddGndPorts_Fixed(self):
        """
        Description: Just like ssViAmp1, except it has 'Vdd' and 'gnd'
          as external ports, which are less flexible than ssViAmp1's
          ports of 'loadrail' and 'opprail'.  But it makes it directly
          interfaceable to the outside world (unlike ssViAmp1 on its own).
          
        Ports: Vin, Iout, Vdd, gnd
        
        Variables:
          -like dsViAmp, except replace its 'loadrail_is_vdd'
           with 'chosen_part_index'
        
        Variable breakdown:
          For overall part: chosen_part_index (==loadrail_is_vdd)
            0: set 'loadrail' of ssViAmp1 to 'gnd', and 'opprail' to 'Vdd'
            1: set 'loadrail' of ssViAmp1 to 'Vdd', and 'opprail' to 'gnd'
          For ssViamp1:
            All, a 1:1 mapping, except: loadrail_is_vdd=chosen_part_index
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        amp_part = self.ssViAmp1()

        #build the point_meta (pm)
        # -remember: don't need to add 'chosen_part_index' to pm
        #  because FlexParts do that automatically
        pm = PointMeta({})
        varmeta_map = amp_part.unityVarMap()
        pm = self.updatePointMeta(pm, amp_part, varmeta_map)
        del pm['loadrail_is_vdd']
        del pm['Vout']
        
        
        #build functions
        var_functions = varmeta_map
        var_functions['loadrail_is_vdd'] = 'chosen_part_index'
        var_functions['Vout']='0.9'

        # These switches allow to restrict some degrees of freedom
        # note that these are not fully correct,
        # especially wrt folding and fixing the input transistor
        fix_only_pmos_input=0
        fix_only_nmos_input=0
        disable_folding=0
        disable_degeneration=0
        disable_cascoding=0
        fix_input=0
        fix_ibias=0
        
        # use one of these
        fix_simple_load=0
        fix_active_load=0
        fix_cascoded_load=0
        
        fix_other=0

        assert not (fix_only_pmos_input and fix_only_nmos_input)

        # fix the input transistor type
        if fix_only_pmos_input:
            del pm['input_is_pmos']
            var_functions['input_is_pmos'] = '1'

        elif fix_only_nmos_input:
            del pm['input_is_pmos']
            var_functions['input_is_pmos'] = '0'

        elif disable_folding:
            del pm['input_is_pmos']
            var_functions['input_is_pmos'] = '1-chosen_part_index'

        # disable folding
        if disable_folding:
            del pm['folder_Vgs']
            var_functions['folder_Vgs'] = '0'
            del pm['folder_L']
            var_functions['folder_L'] = '0.18e-6'
            del pm['Ibias2']
            var_functions['Ibias2'] = '0'

        # fix the load to a resistor
        if fix_simple_load:
            del pm['load_part_index']
            var_functions['load_part_index']='0'
            del pm['load_L']
            var_functions['load_L']='.18e-6'
            del pm['load_Vgs']
            var_functions['load_Vgs']='0'
            
            del pm['loadcascode_Vgs']
            var_functions['loadcascode_Vgs']='0'
            del pm['loadcascode_L']
            var_functions['loadcascode_L']='0'
            del pm['loadcascode_recurse']
            var_functions['loadcascode_recurse']='0'
            del pm['load_fracLoad']
            var_functions['load_fracLoad']='0'

        # fix the load to a mos transistor
        elif fix_active_load:
            del pm['load_part_index']
            var_functions['load_part_index']='1'
            del pm['load_L']
            var_functions['load_L']='.18e-6'
            del pm['load_Vgs']
            var_functions['load_Vgs']='0.6'
            
            del pm['loadcascode_Vgs']
            var_functions['loadcascode_Vgs']='0'
            del pm['loadcascode_L']
            var_functions['loadcascode_L']='0'
            del pm['loadcascode_recurse']
            var_functions['loadcascode_recurse']='0'
            del pm['load_fracLoad']
            var_functions['load_fracLoad']='0'

        # fix the load to a cascoded mos transistor
        elif fix_cascoded_load:
            del pm['load_part_index']
            var_functions['load_part_index']='2'
            del pm['load_L']
            var_functions['load_L']='.18e-6'
            del pm['load_Vgs']
            var_functions['load_Vgs']='0.6'
            
            del pm['loadcascode_Vgs']
            var_functions['loadcascode_Vgs']='0.6'
            del pm['loadcascode_L']
            var_functions['loadcascode_L']='0.18e-6'
            del pm['loadcascode_recurse']
            var_functions['loadcascode_recurse']='0'
            del pm['load_fracLoad']
            var_functions['load_fracLoad']='0.5'   

        # fix the degeneration to no degeneration
        if disable_degeneration:
            del pm['degen_choice']
            var_functions['degen_choice']='0'
            del pm['degen_fracDeg']
            var_functions['degen_fracDeg']='0'

        # fix the input cascoding to no cascoding
        if disable_cascoding:
            del pm['inputcascode_is_wire']
            var_functions['inputcascode_is_wire']='1'
            del pm['inputcascode_recurse']
            var_functions['inputcascode_recurse']='0'
            del pm['inputcascode_L']
            var_functions['inputcascode_L']='0'
            del pm['inputcascode_Vgs']
            var_functions['inputcascode_Vgs']='0'

        # fix the input amplification stage
        if fix_input:
            del pm['ampmos_Vgs']
            var_functions['ampmos_Vgs']='0.7'
            del pm['ampmos_L']
            var_functions['ampmos_L']='.18e-6'
            del pm['ampmos_fracAmp']
            var_functions['ampmos_fracAmp']='0.5'

        if fix_ibias:
            del pm['Ibias']
            var_functions['Ibias']='0.1e-3'
        
        # fix all other design variables
        if fix_other:
            del pm['load_fracLoad']
            var_functions['load_fracLoad']='0.5'
        

        #build the main part
        part = FlexPart(['Vin', 'Iout', 'Vdd','gnd'], pm, name)

        # The with-pmos-unfolded case
        part.addPartChoice( amp_part,
                            {'Vin':'Vin', 'Iout':'Iout',
                             'loadrail':'gnd', 'opprail':'Vdd'},
                            var_functions)
            
        # The with-nmos-unfolded case
        part.addPartChoice( amp_part,
                            {'Vin':'Vin', 'Iout':'Iout',
                             'loadrail':'Vdd', 'opprail':'gnd'},
                            var_functions )

        #build a summaryStr
        part.addToSummaryStr('loadrail is vdd','chosen_part_index')
#         part.addToSummaryStr('input is pmos (rather than nmos)', 'input_is_pmos')
#         part.addToSummaryStr('folded', 'chosen_part_index == input_is_pmos')
        part.addToSummaryStr('Ibias','Ibias')
#         part.addToSummaryStr('Ibias2','Ibias2')
        part.addToSummaryStr('degen_choice (0=wire,1=resistor)', 'degen_choice')
#         part.addToSummaryStr('load type (0=resistor,1=biasedMos,'
#                              '2=ssIiLoad_Cascoded)', 'load_part_index')
        
        self._parts[name] = part
        return part

    def ssViAmp1mod_VddGndPorts_Fixed(self):
        """
        Description: Just like ssViAmp1, except it has 'Vdd' and 'gnd'
          as external ports, which are less flexible than ssViAmp1's
          ports of 'loadrail' and 'opprail'.  But it makes it directly
          interfaceable to the outside world (unlike ssViAmp1 on its own).
          
        Ports: Vin, Iout, Vdd, gnd
        
        Variables:
          -like dsViAmp, except replace its 'loadrail_is_vdd'
           with 'chosen_part_index'
        
        Variable breakdown:
          For overall part: chosen_part_index (==loadrail_is_vdd)
            0: set 'loadrail' of ssViAmp1 to 'gnd', and 'opprail' to 'Vdd'
            1: set 'loadrail' of ssViAmp1 to 'Vdd', and 'opprail' to 'gnd'
          For ssViamp1:
            All, a 1:1 mapping, except: loadrail_is_vdd=chosen_part_index
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        amp_part = self.ssViAmp1()

        #build the point_meta (pm)
        # -remember: don't need to add 'chosen_part_index' to pm
        #  because FlexParts do that automatically
        pm = PointMeta({})
        varmeta_map = amp_part.unityVarMap()
        pm = self.updatePointMeta(pm, amp_part, varmeta_map)
        del pm['loadrail_is_vdd']
        del pm['Vout']
        
        
        #build functions
        var_functions = varmeta_map
        var_functions['loadrail_is_vdd'] = 'chosen_part_index'
        var_functions['Vout']='0.9'

        # These switches allow to restrict some degrees of freedom
        # note that these are not fully correct,
        # especially wrt folding and fixing the input transistor
        fix_only_pmos_input=0
        fix_only_nmos_input=0
        disable_folding=0
        disable_degeneration=0
        disable_cascoding=0
        fix_input=0
        fix_ibias=0
        
        # use one of these
        fix_simple_load=1
        fix_active_load=0
        fix_cascoded_load=0
        
        fix_other=0

        assert not (fix_only_pmos_input and fix_only_nmos_input)

        # fix the input transistor type
        if fix_only_pmos_input:
            del pm['input_is_pmos']
            var_functions['input_is_pmos'] = '1'

        elif fix_only_nmos_input:
            del pm['input_is_pmos']
            var_functions['input_is_pmos'] = '0'

        elif disable_folding:
            del pm['input_is_pmos']
            var_functions['input_is_pmos'] = '1-chosen_part_index'

        # disable folding
        if disable_folding:
            del pm['folder_Vgs']
            var_functions['folder_Vgs'] = '0'
            del pm['folder_L']
            var_functions['folder_L'] = '0.18e-6'
            del pm['Ibias2']
            var_functions['Ibias2'] = '0'

        # fix the load to a resistor
        if fix_simple_load:
            del pm['load_part_index']
            var_functions['load_part_index']= '0'
            del pm['load_L']
            var_functions['load_L']='.18e-6'
            del pm['load_Vgs']
            var_functions['load_Vgs']='0'
            
            del pm['loadcascode_Vgs']
            var_functions['loadcascode_Vgs']='0'
            del pm['loadcascode_L']
            var_functions['loadcascode_L']='0'
            del pm['loadcascode_recurse']
            var_functions['loadcascode_recurse']='0'
            del pm['load_fracLoad']
            var_functions['load_fracLoad']='0'

        # fix the load to a mos transistor
        elif fix_active_load:
            del pm['load_part_index']
            var_functions['load_part_index']='1'
            del pm['load_L']
            var_functions['load_L']='.18e-6'
            del pm['load_Vgs']
            var_functions['load_Vgs']='0.6'
            
            del pm['loadcascode_Vgs']
            var_functions['loadcascode_Vgs']='0'
            del pm['loadcascode_L']
            var_functions['loadcascode_L']='0'
            del pm['loadcascode_recurse']
            var_functions['loadcascode_recurse']='0'
            del pm['load_fracLoad']
            var_functions['load_fracLoad']='0'

        # fix the load to a cascoded mos transistor
        elif fix_cascoded_load:
            del pm['load_part_index']
            var_functions['load_part_index']='2'
            del pm['load_L']
            var_functions['load_L']='.18e-6'
            del pm['load_Vgs']
            var_functions['load_Vgs']='0.6'
            
            del pm['loadcascode_Vgs']
            var_functions['loadcascode_Vgs']='0.6'
            del pm['loadcascode_L']
            var_functions['loadcascode_L']='0.18e-6'
            del pm['loadcascode_recurse']
            var_functions['loadcascode_recurse']='0'
            del pm['load_fracLoad']
            var_functions['load_fracLoad']='0.5'   

        # fix the degeneration to no degeneration
        if disable_degeneration:
            del pm['degen_choice']
            var_functions['degen_choice']='0'
            del pm['degen_fracDeg']
            var_functions['degen_fracDeg']='0'

        # fix the input cascoding to no cascoding
        if disable_cascoding:
            del pm['inputcascode_is_wire']
            var_functions['inputcascode_is_wire']='1'
            del pm['inputcascode_recurse']
            var_functions['inputcascode_recurse']='0'
            del pm['inputcascode_L']
            var_functions['inputcascode_L']='0'
            del pm['inputcascode_Vgs']
            var_functions['inputcascode_Vgs']='0'

        # fix the input amplification stage
        if fix_input:
            del pm['ampmos_Vgs']
            var_functions['ampmos_Vgs']='0.7'
            del pm['ampmos_L']
            var_functions['ampmos_L']='.18e-6'
            del pm['ampmos_fracAmp']
            var_functions['ampmos_fracAmp']='0.5'

        if fix_ibias:
            del pm['Ibias']
            var_functions['Ibias']='0.1e-3'
        
        # fix all other design variables
        if fix_other:
            del pm['load_fracLoad']
            var_functions['load_fracLoad']='0.5'
        

        #build the main part
        part = FlexPart(['Vin', 'Iout', 'Vdd','gnd'], pm, name)

        # The with-pmos-unfolded case
        part.addPartChoice( amp_part,
                            {'Vin':'Vin', 'Iout':'Iout',
                             'loadrail':'gnd', 'opprail':'Vdd'},
                            var_functions)
            
        # The with-nmos-unfolded case
        part.addPartChoice( amp_part,
                            {'Vin':'Vin', 'Iout':'Iout',
                             'loadrail':'Vdd', 'opprail':'gnd'},
                            var_functions )

        #build a summaryStr
        part.addToSummaryStr('loadrail is vdd','chosen_part_index')
#         part.addToSummaryStr('input is pmos (rather than nmos)', 'input_is_pmos')
#         part.addToSummaryStr('folded', 'chosen_part_index == input_is_pmos')
        part.addToSummaryStr('Ibias','Ibias')
#         part.addToSummaryStr('Ibias2','Ibias2')
        part.addToSummaryStr('degen_choice (0=wire,1=resistor)', 'degen_choice')
#         part.addToSummaryStr('load type (0=resistor,1=biasedMos,'
#                              '2=ssIiLoad_Cascoded)', 'load_part_index')
        
        self._parts[name] = part
        return part
                
    def ssViAmp1b_VddGndPorts(self):
        """
        
        ## HACK: this part doesn't have a fixed Vout=0.9
        
        Description: Just like ssViAmp1, except it has 'Vdd' and 'gnd'
          as external ports, which are less flexible than ssViAmp1's
          ports of 'loadrail' and 'opprail'.  But it makes it directly
          interfaceable to the outside world (unlike ssViAmp1 on its own).
          
        Ports: Vin, Iout, Vdd, gnd
        
        Variables:
          -like dsViAmp, except replace its 'loadrail_is_vdd'
           with 'chosen_part_index'
        
        Variable breakdown:
          For overall part: chosen_part_index (==loadrail_is_vdd)
            0: set 'loadrail' of ssViAmp1 to 'gnd', and 'opprail' to 'Vdd'
            1: set 'loadrail' of ssViAmp1 to 'Vdd', and 'opprail' to 'gnd'
          For ssViamp1:
            All, a 1:1 mapping, except: loadrail_is_vdd=chosen_part_index
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        amp_part = self.ssViAmp1()

        #build the point_meta (pm)
        # -remember: don't need to add 'chosen_part_index' to pm
        #  because FlexParts do that automatically
        pm = PointMeta({})
        varmeta_map = amp_part.unityVarMap()
        pm = self.updatePointMeta(pm, amp_part, varmeta_map)
        del pm['loadrail_is_vdd']
        
        #build functions
        var_functions = varmeta_map
        var_functions['loadrail_is_vdd'] = 'chosen_part_index'
        
        #build the main part
        part = FlexPart(['Vin', 'Iout', 'Vdd','gnd'], pm, name)

        part.addPartChoice( amp_part,
                            {'Vin':'Vin', 'Iout':'Iout',
                             'loadrail':'gnd', 'opprail':'Vdd'},
                            var_functions)
        part.addPartChoice( amp_part,
                            {'Vin':'Vin', 'Iout':'Iout',
                             'loadrail':'Vdd', 'opprail':'gnd'},
                            var_functions )
        
        self._parts[name] = part
        return part
        
        
    def currentMirror_Simple(self):
        """
        Description: simple 2-transistor current mirror
          
        Ports: Irefnode, Ioutnode, loadrail
        
        Variables:
            use_pmos, Iin, Iout, Vds_in, Vds_out, L, Vs
          
        Variable breakdown:
          For reference-input MOS (a mos3):
            use_pmos=use_pmos, Vgs=Vds=Vds_in, Ids=Iin, L=L
          For output MOS (a mos3):
            use_pmos=use_pmos, Vgs=Vds_in, Vds=Vds_out, Ids=Iout, L=L
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        ref_part = self.saturatedMos3()
        out_part = self.saturatedMos3()

        #build the point_meta (pm)
        pm = self.buildPointMeta({'use_pmos':'bool_var',
                                  'Iin':'Ids','Iout':'Ids',
                                  'Vds_in':'Vds','Vds_out':'Vds',
                                  'Vs':'Vs',
                                  'L':'L'})
        
        #build functions
        ref_functions = {'use_pmos':'use_pmos',
                         'Vgs':'Vds_in','Vds':'Vds_in',
                         'Ids':'Iin',
                         'L':'L'}
        out_functions = {'use_pmos':'use_pmos',
                         'Vgs':'Vds_in','Vds':'Vds_out',
                         'Ids':'Iout',
                         'L':'L'}
        
        #build the main part
        part = CompoundPart(['Irefnode','Ioutnode', 'oprail'], pm, name)

        part.addPart( ref_part, {'D':'Irefnode','G':'Irefnode','S':'oprail'},
                      ref_functions)
        part.addPart( out_part, {'D':'Ioutnode','G':'Irefnode','S':'oprail'},
                      out_functions)
        
        self._parts[name] = part
        return part

    def currentMirror_Cascode(self):
        """
        Description: standard cascode current mirror
          
        Ports: Irefnode, Ioutnode, loadrail
        
        Variables:
            use_pmos, Iin, Iout, Vds_in, 
            Vds_out, L, Vs, fracIn, fracOut,
            cascode_L
          
        Variable breakdown:

 
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cascoderef_part = self.saturatedMos3()
        mainref_part = self.saturatedMos3()
        cascodeout_part = self.saturatedMos3()
        mainout_part = self.saturatedMos3()

        #build the point_meta (pm)
        pm = self.buildPointMeta({'use_pmos':'bool_var',
                                  'Iin':'Ids','Iout':'Ids',
                                  'Vds_in':'Vds','Vds_out':'Vds',
                                  'fracIn':'frac','fracOut':'frac',
                                  'Vs':'Vs',
                                  'L':'L','cascode_L':'L'})
        
        #build the main part
        part = CompoundPart(['Irefnode','Ioutnode', 'oprail'], pm, name)
        
        n_ref = part.addInternalNode()
        n_out = part.addInternalNode()
        
        Vi_ref='(Vs + Vds_in * fracIn * ( 1 - 2 * ( use_pmos==1 )))'
        Vi_out='(Vs + Vds_out * fracOut * ( 1 - 2 * ( use_pmos==1 )))'

        Vds_casc_ref='(Vds_in * ( 1 - fracIn ))'
        Vds_casc_out='(Vds_out * ( 1 - fracOut ))'
        
        Vds_main_ref='(Vds_in * ( fracIn ))'
        Vds_main_out='(Vds_out * ( fracOut ))'
        
        part.addPart( cascoderef_part,
                      {'D':'Irefnode','G':'Irefnode','S':n_ref},
                      {'use_pmos':'use_pmos',
                       'Vgs':Vds_casc_ref,'Vds':Vds_casc_ref,
                       'Ids':'Iin',
                       'L':'cascode_L'} )

        part.addPart( mainref_part,
                      {'D':n_ref,'G':n_ref,'S':'oprail'},
                      {'use_pmos':'use_pmos',
                       'Vgs':Vds_main_ref,'Vds':Vds_main_ref,
                       'Ids':'Iin',
                       'L':'L'})

        part.addPart( cascodeout_part,
                      {'D':'Ioutnode','G':'Irefnode','S':n_out},
                      {'use_pmos':'use_pmos',
                       'Vgs':Vds_casc_ref,'Vds':Vds_casc_out,
                       'Ids':'Iout',
                       'L':'cascode_L'} )

        part.addPart( mainout_part,
                      {'D':n_out,'G':n_ref,'S':'oprail'},
                      {'use_pmos':'use_pmos',
                       'Vgs':Vds_main_ref,'Vds':Vds_main_out,
                       'Ids':'Iout',
                       'L':'L'} )
        
        self._parts[name] = part
        return part


    def currentMirror_LowVoltageA(self):
        """
        Description: low-voltage-A current mirror
          
        Ports: Irefnode, Ioutnode, oprail
        
        Variables:
            use_pmos, Iin, Iout, Vds_in,
            Vds_out, L, Vs, fracIn, fracOut,
            cascode_L, cascode_Vgs
          
        Variable breakdown:

          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cascoderef_part = self.biasedMos()
        mainref_part = self.saturatedMos3()
        cascodeout_part = self.biasedMos()
        mainout_part = self.saturatedMos3()

        #build the point_meta (pm)
        pm = self.buildPointMeta({'use_pmos':'bool_var',
                                  'Iin':'Ids','Iout':'Ids',
                                  'Vds_in':'Vds','Vds_out':'Vds',
                                  'fracIn':'frac','fracOut':'frac',
                                  'cascode_Vgs':'Vgs',
                                  'Vs':'Vs',
                                  'L':'L','cascode_L':'L'})
        
        #build the main part
        part = CompoundPart(['Irefnode','Ioutnode', 'oprail'], pm, name)
        
        n_ref = part.addInternalNode()
        n_out = part.addInternalNode()
        
        Vi_ref='(Vs + Vds_in * fracIn * ( 1 - 2 * ( use_pmos==1 )))'
        Vi_out='(Vs + Vds_out * fracOut * ( 1 - 2 * ( use_pmos==1 )))'

        Vds_casc_ref='(Vds_in * ( 1 - fracIn ))'
        Vds_casc_out='(Vds_out * ( 1 - fracOut ))'
        
        Vds_main_ref='(Vds_in * ( fracIn ))'
        Vds_main_out='(Vds_out * ( fracOut ))'
        
        part.addPart( cascoderef_part,
                      {'D':'Irefnode','S':n_ref},
                      {'use_pmos':'use_pmos',
                       'Vgs':'cascode_Vgs','Vds':Vds_casc_ref,
                       'Ids':'Iin','Vs':Vi_ref,
                       'L':'cascode_L'} )

        part.addPart( mainref_part,
                      {'D':n_ref,'G':'Irefnode','S':'oprail'},
                      {'use_pmos':'use_pmos',
                       'Vgs':'Vds_in','Vds':Vds_main_ref,
                       'Ids':'Iin',
                       'L':'L'})

        part.addPart( cascodeout_part,
                      {'D':'Ioutnode','S':n_out},
                      {'use_pmos':'use_pmos',
                       'Vgs':'cascode_Vgs','Vds':Vds_casc_out,
                       'Ids':'Iout','Vs':Vi_out,
                       'L':'cascode_L'} )

        part.addPart( mainout_part,
                      {'D':n_out,'G':'Irefnode','S':'oprail'},
                      {'use_pmos':'use_pmos',
                       'Vgs':'Vds_in','Vds':Vds_main_out,
                       'Ids':'Iout',
                       'L':'L'} )
        
        self._parts[name] = part
        return part

    def currentMirror(self):
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

        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cm_Simple = self.currentMirror_Simple()
        cm_Cascode = self.currentMirror_Cascode()
        cm_LowVoltageA = self.currentMirror_LowVoltageA()

        #build the point_meta (pm)
        pm = self.buildPointMeta({'use_pmos':'bool_var',
                                  'Iin':'Ids','Iout':'Ids',
                                  'Vds_in':'Vds','Vds_out':'Vds',
                                  'fracIn':'frac','fracOut':'frac',
                                  'cascode_Vgs':'Vgs',
                                  'Vs':'Vs',
                                  'L':'L','cascode_L':'L'})     
        
        
        #build the main part
        part = FlexPart(['Irefnode','Ioutnode', 'oprail'], pm, name)
        portmap = cm_Simple.unityPortMap()
        part.addPartChoice(cm_Simple, portmap,cm_Simple.unityVarMap())
        part.addPartChoice(cm_Cascode, portmap,cm_Cascode.unityVarMap())
        part.addPartChoice(cm_LowVoltageA, portmap,cm_LowVoltageA.unityVarMap())
        
        self._parts[name] = part
        return part

    def dsIiLoad(self):
        """
        Description: Differential-in current, single-ended out current load.
          This turns out to merely a wrapper for a current mirror with
          different ports, and a name name more appropriate for embedding
          in higher-level blocks.
          
        Ports: Iin1, Iin2, Iout, loadrail
          (Iin1 and Iin2 are differential.)
        
        Variables:
          Same as currentMirror
          except for the fact that use_pmos becomes loadrail_is_vdd 
        
        Variable breakdown:
          For currentMirror:
            All, a 1:1 mapping.
            use_pmos = loadrail_is_vdd 
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        cm_part = self.currentMirror()
        wire_part = self.wire()

        #build the point_meta (pm)
        pm = PointMeta({})
        cm_varmeta_map = cm_part.unityVarMap()
        wire_varmeta_map = wire_part.unityVarMap()
        pm = self.updatePointMeta(pm, cm_part, cm_varmeta_map)
        pm = self.updatePointMeta(pm, wire_part, wire_varmeta_map)

        del pm['use_pmos']
        pm.addVarMeta( self.buildVarMeta('bool_var', 'loadrail_is_vdd') )#add var
        
        #build functions
        cm_functions = cm_varmeta_map
        wire_functions = wire_varmeta_map

        cm_functions['use_pmos'] = 'loadrail_is_vdd'
        
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

    def ddViInput_stacked(self):
        """
        Description: ddViInput
          
        Ports: Vin1, Vin2, Iout1, Iout2, loadrail, opprail
        
        Variables:
          
        Variable breakdown:
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        ssvi_part = self.ssViInput()
        bias_part = self.biasedMos()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['fracVgnd'] = self.buildVarMeta('fracVgnd','fracVgnd')
        pm['folder_L'] = self.buildVarMeta('L','folder_L')
        pm['folder_Vgs'] = self.buildVarMeta('Vgs','folder_Vgs')
        pm['Vds1'] = self.buildVarMeta('Vds','Vds1')
        pm['Vds2'] = self.buildVarMeta('Vds','Vds2')
        pm['Vin_cmm'] = self.buildVarMeta('V','Vin_cmm')
       
        ssvi_varmap = ssvi_part.unityVarMap()
                
        bias_varmap = {'use_pmos':'input_is_pmos',
                       'Vds':'Vds1', 'Vgs':'inputbias_Vgs', 'Ids':'IGNORE', 'L':'inputbias_L', 'Vs':'Vs'
                       }
        pm = self.updatePointMeta(pm, ssvi_part, ssvi_varmap)
        pm = self.updatePointMeta(pm, bias_part, bias_varmap, True)
        del pm['Vds'];
        del pm['ampmos_Vgs'];
        
        #build the main part
        part = CompoundPart(['Vin1', 'Vin2', 'Iout1', 'Iout2',
                             'loadrail','opprail'], pm, name)
        virtual_ground = part.addInternalNode()
        
        #build the functions
        ssvi_functions = ssvi_varmap
        bias_functions = bias_varmap

        # the virtual ground node (#MAGIC: the min() is sort-of arbitrary)
        Vvgnd = '( Vs + min( Vds1, Vds2  ) * fracVgnd * (1 - 2 * (input_is_pmos==1)) )'

        # the amplifier gate node = Vin_cmm
        # this gives us the amplifier gate-source voltage
        Vgs_amp = '(abs( Vin_cmm - (' + Vvgnd  + ')))'
        
        ssvi_functions['ampmos_Vgs'] = Vgs_amp
        ssvi_functions['Vs'] = Vvgnd
        ssvi_functions['inputbias_L'] = 'folder_L' #unneeded
        ssvi_functions['inputbias_Vgs'] = 'folder_Vgs' #unneeded
        
        bias_Vds = "( min( Vds1, Vds2  ) * fracVgnd )"
        
        bias_functions['Vds'] = bias_Vds
        bias_functions['Ids'] = '2 * Ibias'
        
        ssvi_functions['Vds'] = 'Vds1 - (' + bias_Vds + ')'
        part.addPart( ssvi_part,
                      {'Vin':'Vin1', 'Iout':'Iout1',
                      'loadrail':'loadrail', 'opprail':virtual_ground},
                      ssvi_functions )
                      
        ssvi_functions['Vds'] = 'Vds2 - (' + bias_Vds + ')'
        part.addPart( ssvi_part,
                      {'Vin':'Vin2', 'Iout':'Iout2',
                      'loadrail':'loadrail', 'opprail':virtual_ground},
                      ssvi_functions)
      
        part.addPart( bias_part,
                      {'D':virtual_ground, 'S':'opprail'},
                      bias_functions )
                                             
        self._parts[name] = part
        return part
    
    def ddViInput_folded(self):
        """
        Description: ddViInput
          
        Ports: Vin1, Vin2, Iout1, Iout2, loadrail, opprail
        
        Variables:
          
        Variable breakdown:
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        ssvi_part = self.ssViInput()
        bias_part = self.biasedMos()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['fracVgnd'] = self.buildVarMeta('fracVgnd','fracVgnd')
        pm['folder_L'] = self.buildVarMeta('L','folder_L')
        pm['folder_Vgs'] = self.buildVarMeta('Vgs','folder_Vgs')
        pm['Vds1'] = self.buildVarMeta('Vds','Vds1')
        pm['Vds2'] = self.buildVarMeta('Vds','Vds2')
        pm['Vin_cmm'] = self.buildVarMeta('V','Vin_cmm')
       
        ssvi_varmap = ssvi_part.unityVarMap()
                
        bias_varmap = {'use_pmos':'input_is_pmos',
                       'Vds':'Vds1', 'Vgs':'inputbias_Vgs', 'Ids':'IGNORE', 'L':'inputbias_L', 'Vs':'Vs'
                       }
        pm = self.updatePointMeta(pm, ssvi_part, ssvi_varmap)
        pm = self.updatePointMeta(pm, bias_part, bias_varmap, True)
        del pm['Vds'];
        del pm['ampmos_Vgs'];
        
        #build the main part
        part = CompoundPart(['Vin1', 'Vin2', 'Iout1', 'Iout2',
                             'loadrail','opprail'], pm, name)
        virtual_ground = part.addInternalNode()
        
        #build the functions
        ssvi_functions = ssvi_varmap
        bias_functions = bias_varmap

        # the virtual ground node (#MAGIC: the min() is sort-of arbitrary)
        Vvgnd = '( Vs + min( Vds1, Vds2  ) * fracVgnd * (1 - 2 * (input_is_pmos==1)) )'

        # the amplifier gate node = Vin_cmm
        # this gives us the amplifier gate-source voltage
        Vgs_amp = '(abs( Vin_cmm - (' + Vvgnd  + ')))'
        
        ssvi_functions['ampmos_Vgs'] = Vgs_amp
        ssvi_functions['Vs'] = Vvgnd
        ssvi_functions['inputbias_L'] = 'folder_L'
        ssvi_functions['inputbias_Vgs'] = 'folder_Vgs'
        
        bias_Vds = "( min( Vds1, Vds2  ) * fracVgnd )"
        
        bias_functions['Vds'] = bias_Vds
        bias_functions['Ids'] = '2 * Ibias'
        
        ssvi_functions['Vds'] = 'Vds1 - (' + bias_Vds + ')'
        part.addPart( ssvi_part,
                      {'Vin':'Vin1', 'Iout':'Iout1',
                      'loadrail':virtual_ground, 'opprail':'opprail'},
                      ssvi_functions )
                      
        ssvi_functions['Vds'] = 'Vds2 - (' + bias_Vds + ')'
        part.addPart( ssvi_part,
                      {'Vin':'Vin2', 'Iout':'Iout2',
                      'loadrail':virtual_ground, 'opprail':'opprail'},
                      ssvi_functions)
      
        part.addPart( bias_part,
                      {'D':virtual_ground, 'S':'loadrail'},
                      bias_functions )

        self._parts[name] = part
        return part

    def ddViInput_mirrored(self):
        """
        Description: ddViInput with current mirror folding
          
        Ports: Vin1, Vin2, Iout1, Iout2, loadrail, opprail
        
        Variables:
          
        Variable breakdown:
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        ssvi_part = self.ssViInput()
        bias_part = self.biasedMos()
        mirror_part = self.currentMirror()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm['fracVgnd'] = self.buildVarMeta('fracVgnd','fracVgnd')
        pm['folder_L'] = self.buildVarMeta('L','folder_L')
        pm['folder_Vgs'] = self.buildVarMeta('Vgs','folder_Vgs')
        pm['Vds1'] = self.buildVarMeta('Vds','Vds1')
        pm['Vds2'] = self.buildVarMeta('Vds','Vds2')
        pm['Vin_cmm'] = self.buildVarMeta('V','Vin_cmm')

        # note: we have to hack into loadrail==vdd to make sure the part does not fold
        #
        ssvi_varmap = ssvi_part.unityVarMap()

        bias_varmap = {'use_pmos':'input_is_pmos',
                       'Vds':'Vds1', 'Vgs':'inputbias_Vgs', 'Ids':'IGNORE', 'L':'inputbias_L', 'Vs':'Vs'
                       }

        mirror_varmap = {
           'fracIn':'fracFolder','fracOut':'fracFolder',
           'cascode_Vgs':'foldercascode_Vgs',
           'L':'folder_L','cascode_L':'foldercascode_L',
           'chosen_part_index':'folder_use_cascode', #NOTE: collides with folderpart
           'Iin':'Ibias','Iout':'Ibias2',
           'use_pmos':'use_pmos',            ## function!
           'Vds_in':'Vds','Vds_out':'Vds',   ## function!
           'Vs':'Vs', ## vdd or vss          ## function!
        }

        pm = self.updatePointMeta(pm, ssvi_part, ssvi_varmap)
        pm = self.updatePointMeta(pm, bias_part, bias_varmap, True)
        pm = self.updatePointMeta(pm, mirror_part, mirror_varmap, True)

        del pm['Vds']
        del pm['ampmos_Vgs']
        del pm['use_pmos']

        #build the main part
        part = CompoundPart(['Vin1', 'Vin2', 'Iout1', 'Iout2',
                             'loadrail','opprail'], pm, name)

        #build the functions
        ssvi_functions = ssvi_varmap
        bias_functions = bias_varmap
        mirror_functions = mirror_varmap

        # the virtual ground node (#MAGIC: the min() is sort-of arbitrary)
        Vvgnd = '( Vs + min( Vds1, Vds2  ) * fracVgnd * (1 - 2 * (input_is_pmos==1)) )'

        # the amplifier gate node = Vin_cmm
        # this gives us the amplifier gate-source voltage
        Vgs_amp = '(abs( Vin_cmm - (' + Vvgnd  + ')))'
        
        ssvi_functions['ampmos_Vgs'] = Vgs_amp
        ssvi_functions['Vs'] = Vvgnd

        # make sure the input stage is not folded, since we
        # manually fold using the current mirror
        ssvi_functions['loadrail_is_vdd'] = '1-input_is_pmos'

        # these are not really needed:
        ssvi_functions['inputbias_L'] = 'folder_L' #unneeded
        ssvi_functions['inputbias_Vgs'] = 'folder_Vgs' #unneeded

        # the voltage of the opposite rail
        Voprail = '(' + str(self.ss.vdd) + ' * ( input_is_pmos == 0 ) + ' + str(self.ss.vss) + ' * ( input_is_pmos == 1 ) )'

        # mirror
        mirror_functions['use_pmos'] = '1-input_is_pmos' # not what the input stage uses
        mirror_functions['Vs'] = Voprail # voltage at opprail

        # bias part
        bias_Vds = "( min( Vds1, Vds2  ) * fracVgnd )"
        
        bias_functions['Vds'] = bias_Vds
        bias_functions['Ids'] = '2 * Ibias'
        
        # internal nodes
        virtual_ground = part.addInternalNode()
        nout1_int = part.addInternalNode()
        nout2_int = part.addInternalNode()

        # note: compared to a normal folded cascode, we have to switch the input terminals
        #       since the mirrored version is non-inverting
        # add one side
        ssvi_functions['Vds'] = 'Vds1 - (' + bias_Vds + ')'
        mirror_functions['Vds_in'] = 'abs('+str(self.ss.vdd)+'-'+str(self.ss.vss)+' - Vds1)'
        mirror_functions['Vds_out'] = 'abs('+str(self.ss.vdd)+'-'+str(self.ss.vss)+' - Vds1)'
        part.addPart( ssvi_part,
                      {'Vin':'Vin2', 'Iout':nout1_int,
                      'loadrail':'loadrail', 'opprail':virtual_ground},
                      ssvi_functions )

        part.addPart( mirror_part,
                      {'Irefnode':nout1_int, 'Ioutnode': 'Iout1', 'oprail': 'opprail'},
                      mirror_functions )

        # add other side
        ssvi_functions['Vds'] = 'Vds2 - (' + bias_Vds + ')'
        mirror_functions['Vds_in'] = 'abs('+str(self.ss.vdd)+'-'+str(self.ss.vss)+' - Vds2)'
        mirror_functions['Vds_out'] = 'abs('+str(self.ss.vdd)+'-'+str(self.ss.vss)+' - Vds2)'
        part.addPart( ssvi_part,
                      {'Vin':'Vin1', 'Iout':nout2_int,
                      'loadrail':'loadrail', 'opprail':virtual_ground},
                      ssvi_functions )

        part.addPart( mirror_part,
                      {'Irefnode':nout2_int, 'Ioutnode': 'Iout2', 'oprail': 'opprail'},
                      mirror_functions )

        # current source
        part.addPart( bias_part,
                      {'D':virtual_ground, 'S':'loadrail'},
                      bias_functions )

        self._parts[name] = part
        return part

    def ddViInput_FoldedOrMirrored(self):
        """
        Description: Chooses between folded and mirrored 
          
        Ports: Vin1, Vin2, Iout1, Iout2, loadrail,opprail
        
        Variables:
          
        Variable breakdown:
          For overall part: chosen_part_index (==use_folded?)
            0 : ddViInput_Stacked
            1 : ddViInput_Folded
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        folded_part = self.ddViInput_folded()
        mirrored_part = self.ddViInput_mirrored()

        #build the point_meta (pm)
        pm = PointMeta({})
        folded_varmap = folded_part.unityVarMap()
        mirrored_varmap = mirrored_part.unityVarMap()

        pm = self.updatePointMeta(pm, folded_part, folded_varmap)
        pm = self.updatePointMeta(pm, mirrored_part, mirrored_varmap, True)

        #build the main part
        part = FlexPart(['Vin1', 'Vin2', 'Iout1', 'Iout2',
                         'loadrail','opprail'], pm, name)

        part.addPartChoice( folded_part, folded_part.unityPortMap(), folded_varmap)
        part.addPartChoice( mirrored_part, mirrored_part.unityPortMap(), mirrored_varmap)

        self._parts[name] = part
        return part

    def ddViInput_Flex(self):
        """
        Description: Chooses between folded and stacked 
          
        Ports: Vin1, Vin2, Iout1, Iout2, loadrail,opprail
        
        Variables:
          
        Variable breakdown:
          For overall part: chosen_part_index (==use_folded?)
            0 : ddViInput_Stacked
            1 : ddViInput_FoldedOrMirrored (ddViInput_Folded or ddViInput_Mirrored)
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        stacked_part = self.ddViInput_stacked()
        fold_or_mirr_part = self.ddViInput_FoldedOrMirrored()

        #build the point_meta (pm)
        pm = PointMeta({})
        stacked_varmap = stacked_part.unityVarMap()
        fold_or_mirr_varmap = fold_or_mirr_part.unityVarMap()
        fold_or_mirr_varmap['chosen_part_index'] = 'use_mirror_to_fold'
        
        pm = self.updatePointMeta(pm, stacked_part, stacked_varmap)
        pm = self.updatePointMeta(pm, fold_or_mirr_part, fold_or_mirr_varmap, True)
        
        #build the main part
        part = FlexPart(['Vin1', 'Vin2', 'Iout1', 'Iout2',
                         'loadrail','opprail'], pm, name)

        part.addPartChoice( stacked_part, stacked_part.unityPortMap(), stacked_varmap)
        part.addPartChoice( fold_or_mirr_part, fold_or_mirr_part.unityPortMap(), fold_or_mirr_varmap)

        self._parts[name] = part
        return part

    def dsViAmp1(self):
        """
        Description: Differential-in voltage, single-ended out current amplifier.
          Combines together a ddViInput and dsIiLoad.
          
        Ports: Vin1, Vin2, Iout, loadrail, opprail
          (Vin1 and Vin2 are differential.)
        
        Variables:
            loadrail_is_vdd, input_is_pmos

        
        Variable breakdown:
          For ddViInput:
            1:1 mapping of all ddViInput vars
          For dsIiLoad:

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

            'chosen_folder_type':'chosen_folder_type',
            'use_mirror_to_fold':'use_mirror_to_fold',
            'folder_use_cascode':'folder_use_cascode',
            'foldercascode_L':'foldercascode_L',
            'foldercascode_Vgs':'foldercascode_Vgs',
            'fracFolder':'fracFolder',
            }

        load_varmeta_map = {
            'chosen_part_index':'load_chosen_part_index',
            'loadrail_is_vdd':'loadrail_is_vdd',
            'Iin':'Ibias','Iout':'Ibias', ## NOTE: this will require a hack because if it is folded,
                                          #        we need another bias current
            'Vds_in':'Vds_internal', # has to be VDD - Vds_internal for normal
            'Vds_out':'Vds_internal', # has to be VDD - Vds_out
            'fracIn':'load_fracIn',
            'fracOut':'load_fracOut',
            'Vs':'Vout',
            'L':'load_L',
            'cascode_L':'load_cascode_L',
            'cascode_Vgs':'load_cascode_Vgs'
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
        load_functions['Vds_in']="switchAndEval((loadrail_is_vdd==input_is_pmos), {" + \
                                 "1:'Vds_internal', " + \
                                 "0:'("+str(self.ss.vdd-self.ss.vss)+"- Vds_internal)'  })"
                                 
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
        
    def dsViAmp1_VddGndPorts(self):
        """
        Description: Just like dsViAmp1, except it has 'Vdd' and 'gnd'
          as external ports, which are less flexible than dsViAmp1's
          ports of 'loadrail' and 'opprail'.  But it makes it directly
          interfaceable to the outside world (unlike dsViAmp1 on its own).
          
        Ports: Vin1, Vin2, Iout, Vdd, gnd
          (Vin1 and Vin2 are differential.)
        
        Variables:
          -like dsViAmp1, except replace its 'loadrail_is_vdd'
           with 'chosen_part_index'
        
        Variable breakdown:
          For overall part: chosen_part_index (==loadrail_is_vdd)
            0: set 'loadrail' of dsViAmp1 to 'gnd', and 'opprail' to 'Vdd'
            1: set 'loadrail' of dsViAmp1 to 'Vdd', and 'opprail' to 'gnd'
          For dsViamp1:
            All, a 1:1 mapping, except: loadrail_is_vdd=chosen_part_index
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        amp_part = self.dsViAmp1()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm = self.updatePointMeta(pm, amp_part, amp_part.unityVarMap())
        
        # -remember: don't need to add 'chosen_part_index' to pm
        #  because FlexParts do that automatically

        varmap = amp_part.unityVarMap()
        
        del pm['loadrail_is_vdd']
        varmap['loadrail_is_vdd'] = 'chosen_part_index'
        
        del pm['Vout'] # HACK: this should be in Problems.py
        varmap['Vout']='0.9' # HACK: this should be in Problems.py
        
        del pm['Vin_cmm'] # HACK: this should be in Problems.py
        varmap['Vin_cmm']='0.9' # HACK: this should be in Problems.py
        
        #build the main part
        part = FlexPart(['Vin1', 'Vin2', 'Iout', 'Vdd','gnd'], pm, name)

        part.addPartChoice( amp_part,
                            {'Vin1':'Vin1', 'Vin2':'Vin2', 'Iout':'Iout',
                            'loadrail':'gnd', 'opprail':'Vdd'},
                            varmap)
        part.addPartChoice( amp_part,
                            {'Vin1':'Vin1', 'Vin2':'Vin2', 'Iout':'Iout',
                            'loadrail':'Vdd', 'opprail':'gnd'},
                            varmap )
        
        self._parts[name] = part
        
        #build a summaryStr
        part.addToSummaryStr('loadrail is vdd','chosen_part_index')
        
        part.addToSummaryStr('Ibias','Ibias')
        
        part.addToSummaryStr('input is pmos (rather than nmos)', 'input_is_pmos')
        part.addToSummaryStr('folded', 'chosen_part_index == input_is_pmos')
                
        part.addToSummaryStr('Ibias2','Ibias2')
        
        part.addToSummaryStr('degen_choice (0=wire,1=resistor)', 'degen_choice')
        part.addToSummaryStr('load type (0=simpleCM, 1=CascodedCM'
                                 '2=LowV CM)', 'load_chosen_part_index')                            
        part.addToSummaryStr('Virtual ground headroom fraction', 'fracVgnd')                            
        part.addToSummaryStr('Amplifier headroom fraction', 'fracAmp')
                            
        self._parts[name] = part
        return part
        
        
    def dsViAmp1b_VddGndPorts(self):
        """
        ##NOTE: this one is not necessary!!!
        
        ##HACK: the normal dsViAmp1_VddGndPorts has its output voltage fixed on 0.9
                this one is for the 2 stage design and therefore hasn't
        
        Description: Just like dsViAmp1, except it has 'Vdd' and 'gnd'
          as external ports, which are less flexible than dsViAmp1's
          ports of 'loadrail' and 'opprail'.  But it makes it directly
          interfaceable to the outside world (unlike dsViAmp1 on its own).
          
        Ports: Vin1, Vin2, Iout, Vdd, gnd
          (Vin1 and Vin2 are differential.)
        
        Variables:
          -like dsViAmp1, except replace its 'loadrail_is_vdd'
           with 'chosen_part_index'
        
        Variable breakdown:
          For overall part: chosen_part_index (==loadrail_is_vdd)
            0: set 'loadrail' of dsViAmp1 to 'gnd', and 'opprail' to 'Vdd'
            1: set 'loadrail' of dsViAmp1 to 'Vdd', and 'opprail' to 'gnd'
          For dsViamp1:
            All, a 1:1 mapping, except: loadrail_is_vdd=chosen_part_index
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        amp_part = self.dsViAmp1()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm = self.updatePointMeta(pm, amp_part, amp_part.unityVarMap())
        
        # -remember: don't need to add 'chosen_part_index' to pm
        #  because FlexParts do that automatically

        varmap = amp_part.unityVarMap()
        
        del pm['loadrail_is_vdd']
        varmap['loadrail_is_vdd'] = 'chosen_part_index'
        
        del pm['Vin_cmm'] # HACK: this should be in Problems.py
        varmap['Vin_cmm']='0.9' # HACK: this should be in Problems.py

        #build the main part
        part = FlexPart(['Vin1', 'Vin2', 'Iout', 'Vdd','gnd'], pm, name)

        part.addPartChoice( amp_part,
                            {'Vin1':'Vin1', 'Vin2':'Vin2', 'Iout':'Iout',
                            'loadrail':'gnd', 'opprail':'Vdd'},
                            varmap)
        part.addPartChoice( amp_part,
                            {'Vin1':'Vin1', 'Vin2':'Vin2', 'Iout':'Iout',
                            'loadrail':'Vdd', 'opprail':'gnd'},
                            varmap )
        
        self._parts[name] = part
        return part

    def ddIiLoad(self):
        """
        Description: diff-current-in, diff-current-out load.
          Merely twins up two ssIiLoads.
          
        Ports: Iout1, Iout2, loadrail, opprail
        
        Variables:
          chosen_part_index,
          loadrail_is_vdd,
          R,
          W, L, Vbias,
          loadcascode_recurse, loadcascode_W, loadcascode_L, loadcascode_Vbias
        
        Variable breakdown:
          For each ssIiLoad: 1:1 mapping

        Note: GRAIL pdf is wrong because it thinks ddIiLoad has Iin1,Iin2 when
          it actually doesn't
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        ss_part = self.ssIiLoad()

        #build the point_meta (pm)
        pm = PointMeta({})
        ss_varmap = ss_part.unityVarMap()
        pm = self.updatePointMeta(pm, ss_part, ss_varmap)
        
        #build the main part
        part = CompoundPart(['Iout1','Iout2','loadrail','opprail'],
                            pm, name)

        part.addPart( ss_part,
                      {'Iout':'Iout1','loadrail':'loadrail','opprail':'opprail'},
                      ss_varmap )
        part.addPart( ss_part,
                      {'Iout':'Iout2','loadrail':'loadrail','opprail':'opprail'},
                      ss_varmap )
        
        self._parts[name] = part
        return part

    def ddViAmp1(self):
        """
        Description: diff-voltage-in, diff-current-out 1-stage amp.
          Is a sequence of a ddViInput followed by a ddIiLoad.
          
        Ports: Vin1, Vin2, Iout1, Iout2, loadrail, opprail
        
        Variables:
          loadrail_is_vdd, input_is_pmos,
          cascode_W, cascode_L, cascode_Vbias, cascode_recurse, cascode_is_wire,
          ampmos_W, ampmos_L,
          degen_R, degen_choice,
          inputbias_W, inputbias_L, inputbias_Vbias
        
        Variable breakdown:
          For ddViInput:
            1:1 mapping of ddViInput vars
          For ddIiLoad:
            chosen_part_index=load_chosen_part_index,
            loadrail_is_vdd=loadrail_is_vdd,
            R=load_R,
            W=load_W, L=load_L, Vbias=load_Vbias,
            loadcascode_recurse, loadcascode_W, loadcascode_L, loadcascode_Vbias
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        input_part = self.ddViInput()
        load_part = self.ddIiLoad()

        #build the point_meta (pm)
        pm = PointMeta({})
        input_varmap = input_part.unityVarMap()
        load_varmap = {
            'chosen_part_index':'load_chosen_part_index',
            'loadrail_is_vdd':'loadrail_is_vdd',
            'R':'load_R', 'W':'load_W', 'L':'load_L', 'Vbias':'load_Vbias',
            'loadcascode_recurse':'loadcascode_recurse',
            'loadcascode_W':'loadcascode_W',
            'loadcascode_L':'loadcascode_L',
            'loadcascode_Vbias':'loadcascode_Vbias'}
            
        
        pm = self.updatePointMeta(pm, input_part, input_varmap)
        pm = self.updatePointMeta(pm, load_part, load_varmap, True)
        
        #build the main part
        part = CompoundPart(['Vin1','Vin2','Iout1','Iout2','loadrail','opprail'],
                            pm, name)

        part.addPart( input_part, input_part.unityPortMap(), input_varmap)
        part.addPart( load_part, load_part.unityPortMap(), load_varmap)
        
        self._parts[name] = part
        return part



    def ddViAmp1_VddGndPorts(self):
        """
        Description: Just like ddViAmp1, except it has 'Vdd' and 'gnd'
          as external ports
          
        Ports: Vin1, Vin2, Iout1, Iout2, Vdd, gnd
        
        Variables:
          -like ddViAmp1, except replace its 'loadrail_is_vdd'
           with 'chosen_part_index'
        
        Variable breakdown:
          For overall part: chosen_part_index (==loadrail_is_vdd)
            0: set 'loadrail' of ddViAmp1 to 'gnd', and 'opprail' to 'Vdd'
            1: set 'loadrail' of ddViAmp1 to 'Vdd', and 'opprail' to 'gnd'
          For ddViamp1:
            All, a 1:1 mapping, except: loadrail_is_vdd=chosen_part_index
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        amp_part = self.ddViAmp1()

        #build the point_meta (pm)
        pm = PointMeta({})
        pm = self.updatePointMeta(pm, amp_part, amp_part.unityVarMap())
        del pm['loadrail_is_vdd']
        # -remember: don't need to add 'chosen_part_index' to pm
        #  because FlexParts do that automatically

        #
        amp_functions = amp_part.unityVarMap()
        amp_functions['loadrail_is_vdd'] = 'chosen_part_index'
        
        #build the main part
        part = FlexPart(['Vin1', 'Vin2', 'Iout1', 'Iout2','Vdd','gnd'], pm, name)

        part.addPartChoice( amp_part,
                            {'Vin1':'Vin1', 'Vin2':'Vin2',
                             'Iout1':'Iout1', 'Iout2':'Iout2',
                             'loadrail':'gnd', 'opprail':'Vdd'},
                            amp_functions)
        part.addPartChoice( amp_part,
                            {'Vin1':'Vin1', 'Vin2':'Vin2',
                             'Iout1':'Iout1', 'Iout2':'Iout2',
                             'loadrail':'Vdd', 'opprail':'gnd'},
                            amp_functions )
        
        self._parts[name] = part
        return part

    def dsViAmp2_DifferentialMiddle_VddGndPorts(self):
        """
        Description: differential-voltage-in, single-ended-current out
          two-stage amplifier.  Communication between stages is
          DIFFERENTIAL (ie output of first stage is diff, and input to
          second stage is diff).  A level shifter can exist between
          the stages, as can feedback.
          
        Ports: Vin1, Vin2, Iout, Vdd, gnd
        
        Variables:
            stage1_loadrail_is_vdd, stage1_input_is_pmos,
            
            stage1_cascode_W, stage1_cascode_L, stage1_cascode_Vbias,
            stage1_cascode_recurse,cascode_is_wire,
            stage1_ampmos_W, stage1_ampmos_L,
            stage1_degen_R, stage1_degen_choice,
            stage1_inputbias_W, stage1_inputbias_L, stage1_inputbias_Vbias,

            stage2_loadrail_is_vdd, stage2_input_is_pmos,
            
            stage2_cascode_W, stage2_cascode_L, stage2_cascode_Vbias,
            stage2_cascode_recurse, stage2_cascode_is_wire,
            stage2_ampmos_W, stage2_ampmos_L,
            stage2_degen_R,
            stage2_degen_choice,
            stage2_inputbias_W, stage2_inputbias_L, stage2_inputbias_Vbias,
            stage2_load_chosen_part_index, stage2_load_base_W, stage2_load_ref_K,
            stage2_load_out_K,
            stage2_load_L, stage2_load_topref_usemos, stage2_load_topref_R,
            stage2_load_topref_K,
            stage2_load_middleref_K, stage2_load_bottomref_K,
            stage2_load_topout_K, stage2_load_bottomout_K,
            stage2_load_Vbias

            shifter_Drail_is_vdd,
            
            shifter_chosen_part_index,
            shifter_amp_W, shifter_amp_L,
            shifter_cascode_do_stack,
            shifter_cascode_D_W, shifter_cascode_D_L, shifter_cascode_D_Vbias,
            shifter_cascode_S_W, shifter_cascode_S_L, shifter_cascode_S_Vbias

            feedback_use_pmos,
            feedback_chosen_part_index, feedback_R, feedback_C,
            feedback_amp_W, feedback_amp_L,
            feedback_cascode_do_stack,
            feedback_cascode_D_W, feedback_cascode_D_L, feedback_cascode_D_Vbias,
            feedback_cascode_S_W, feedback_cascode_S_L, feedback_cascode_S_Vbias
        
        Variable breakdown:
          For ddViAmp1_VddGndPorts (stage 1): 1:1 mapping of ddViAmp1 variables,
            EXCEPT that each variable here has the prefix 'stage1_'
            
          For dsViAmp1_VddGndPorts (stage 2): 1:1 mapping of dsViAmp1 variables,
            EXCEPT with the prefix 'stage2_'
            
          For each levelShifterOrWire_VddGndPorts twin:
            use_pmos=stage1_loadrail_is_vdd,
            and all others are 1:1 with levelShifterOrWire
            EXCEPT with the prefix 'shifter_'
            
          For viFeedback_VddGndPorts: 1:1 mapping of viFeedback variables,
            EXCEPT with the prefix 'feedback_'
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        stage1_part = self.ddViAmp1_VddGndPorts()
        stage2_part = self.dsViAmp1_VddGndPorts()
        shifter_part = self.levelShifterOrWire_VddGndPorts()
        feedback_part = self.viFeedback_VddGndPorts()

        #build the point_meta (pm)
        pm = PointMeta({})
        stage1_varmap = {}
        for old_name in  stage1_part.point_meta.keys():
            stage1_varmap[old_name] = 'stage1_' + old_name
        stage2_varmap = {}
        for old_name in  stage2_part.point_meta.keys():
            stage2_varmap[old_name] = 'stage2_' + old_name
        shifter_varmap = {}
        for old_name in  shifter_part.point_meta.keys():
            shifter_varmap[old_name] = 'shifter_' + old_name
        feedback_varmap = {}
        for old_name in  feedback_part.point_meta.keys():
            feedback_varmap[old_name] = 'feedback_' + old_name
            
        pm = self.updatePointMeta(pm, stage1_part, stage1_varmap)
        pm = self.updatePointMeta(pm, stage2_part, stage2_varmap)
        pm = self.updatePointMeta(pm, shifter_part, shifter_varmap)
        pm = self.updatePointMeta(pm, feedback_part, feedback_varmap)

        #build functions
        stage1_functions = stage1_varmap
        stage2_functions = stage2_varmap
        shifter_functions = shifter_varmap
        feedback_functions = feedback_varmap
        
        #build the main part
        part = CompoundPart(['Vin1','Vin2', 'Iout','Vdd','gnd'], pm, name)

        stage1_out1 = part.addInternalNode()
        stage1_out2 = part.addInternalNode()
        stage2_in1 = part.addInternalNode()
        stage2_in2 = part.addInternalNode()

        part.addPart( stage1_part,
                      {'Vin1':'Vin1', 'Vin2':'Vin2','Iout1':stage1_out1,
                       'Vdd':'Vdd', 'gnd':'gnd'},
                      stage1_functions)
        part.addPart( stage2_part,
                      {'Vin1':stage2_in1, 'Vin2':stage2_in2, 'Iout':'Iout',
                       'Vdd':'Vdd', 'gnd':'gnd'},
                      stage2_functions)
        part.addPart( shifter_part,
                      {'Vin':stage1_out1, 'Iout':stage2_in1,
                       'Vdd':'Vdd', 'gnd':'gnd'},
                      shifter_functions) #twin #1
        part.addPart( shifter_part,
                      {'Vin':stage1_out2, 'Iout':stage2_in2,
                       'Vdd':'Vdd', 'gnd':'gnd'},
                      shifter_functions) #twin #2

        #FIXME: is feedback hooked up ok???
        # -especially loadrail, opprail?
        part.addPart( feedback_part,
                      {'Ifpos':stage1_out1, 'Ifneg':'gnd',
                       'VsensePos':'Iout', 'VsenseNeg':'gnd'},
                      feedback_functions)
        
        self._parts[name] = part
        return part

    def dsViAmp2_SingleEndedMiddle_VddGndPorts(self):
        """
        Description: differential-voltage-in, single-ended-current out
          two-stage amplifier.  Communication between stages is
          SINGLE-ended (ie output of first stage is single-ended, and input to
          second stage is single-ended).  A level shifter can exist between
          the stages, as can feedback.
          
        Ports: Vin1, Vin2, Iout, Vdd, gnd
        
        Variables:
            stage1_loadrail_is_vdd, stage1_input_is_pmos
            
            stage1_cascode_W, stage1_cascode_L, stage1_cascode_Vbias,
            stage1_cascode_recurse, stage1_cascode_is_wire,
            stage1_ampmos_W, stage1_ampmos_L,
            stage1_degen_R,
            stage1_degen_choice,
            stage1_inputbias_W, stage1_inputbias_L, stage1_inputbias_Vbias,
            stage1_load_chosen_part_index, stage1_load_base_W, stage1_load_ref_K,
            stage1_load_out_K,
            stage1_load_L, stage1_load_topref_usemos, stage1_load_topref_R,
            stage1_load_topref_K,
            stage1_load_middleref_K, stage1_load_bottomref_K,
            stage1_load_topout_K, stage1_load_bottomout_K,
            stage1_load_Vbias

            stage2_loadrail_is_vdd,  stage2_input_is_pmos,
            
            stage2_inputcascode_W, stage2_inputcascode_L,
            stage2_inputcascode_Vbias, stage2_inputcascode_recurse,
            stage2_ampmos_W, stage2_ampmos_L,
            stage2_degen_R, stage2_degen_choice,
            stage2_inputbias_W, stage2_inputbias_L, stage2_inputbias_Vbias
            stage2_load_part_index,
            stage2_load_R,
            stage2_load_W, stage2_load_L, stage2_load_Vbias,
            stage2_loadcascode_recurse, stage2_loadcascode_W,
            stage2_loadcascode_L, stage2_loadcascode_Vbias

            shifter_Drail_is_vdd,
            
            shifter_chosen_part_index,
            shifter_amp_W, shifter_amp_L,
            shifter_cascode_do_stack,
            shifter_cascode_D_W, shifter_cascode_D_L, shifter_cascode_D_Vbias,
            shifter_cascode_S_W, shifter_cascode_S_L, shifter_cascode_S_Vbias

            feedback_use_pmos,
            feedback_chosen_part_index, feedback_R, feedback_C,
            feedback_amp_W, feedback_amp_L,
            feedback_cascode_do_stack,
            feedback_cascode_D_W, feedback_cascode_D_L, feedback_cascode_D_Vbias,
            feedback_cascode_S_W, feedback_cascode_S_L, feedback_cascode_S_Vbias
        
        Variable breakdown:
          For stage 1 dsViAmp1_VddGndPorts (stage 1):
            1:1 mapping of ddViAmp1_VddGndPorts variables,
            EXCEPT that each variable here has the prefix 'stage1_'
            
          For stage 2 ssViAmp1_VddGndPorts (stage 2):
            1:1 mapping of ssViAmp1_VddGndPorts variables,
            EXCEPT with the prefix 'stage2_'
            
          For levelShifterOrWire_VddGndPorts:
            levelShifterOrWire_VddGndPorts
            EXCEPT with the prefix 'shifter_'
            
          For viFeedback:
            Currently a HACK to just use a capacitor!!
            1:1 mapping of viFeedback variables,
            EXCEPT with the prefix 'feedback_'
          
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        stage1_part = self.dsViAmp1b_VddGndPorts()
        stage2_part = self.ssViAmp1b_VddGndPorts()
#        shifter_part = self.levelShifterOrWire_VddGndPorts()
        feedback_part = self.capacitor() #note the HACK for just capacitor

        #build the point_meta (pm)
        pm = PointMeta({})
        stage1_varmap = {}
        for old_name in  stage1_part.point_meta.keys():
            stage1_varmap[old_name] = 'stage1_' + old_name
        stage2_varmap = {}
        for old_name in  stage2_part.point_meta.keys():
            stage2_varmap[old_name] = 'stage2_' + old_name
#        shifter_varmap = {}
#        for old_name in  shifter_part.point_meta.keys():
#            shifter_varmap[old_name] = 'shifter_' + old_name
        feedback_varmap = {}
        for old_name in  feedback_part.point_meta.keys():
            feedback_varmap[old_name] = 'feedback_' + old_name
            
        pm = self.updatePointMeta(pm, stage1_part, stage1_varmap)
        del pm['stage1_chosen_part_index']
        
        pm['stage1_loadrail_is_vdd']= self.buildVarMeta('bool_var',
                                                        'stage1_loadrail_is_vdd')
        
        pm = self.updatePointMeta(pm, stage2_part, stage2_varmap)
        del pm['stage2_chosen_part_index']
        pm['stage2_loadrail_is_vdd']= self.buildVarMeta('bool_var',
                                                        'stage2_loadrail_is_vdd')
        
#       pm = self.updatePointMeta(pm, shifter_part, shifter_varmap)
#       del pm['shifter_chosen_part_index']
#        pm['shifter_Drail_is_vdd'] = self.buildVarMeta('bool_var',
#                                                       'shifter_Drail_is_vdd')
        
        del pm['stage2_Vout']
        
        pm = self.updatePointMeta(pm, feedback_part, feedback_varmap)
        
        #build functions
        stage1_functions = stage1_varmap
        stage1_functions['chosen_part_index'] = 'stage1_loadrail_is_vdd' 
        
        stage2_functions = stage2_varmap
        stage2_functions['chosen_part_index'] = 'stage2_loadrail_is_vdd'
        
#         stage2_functions['Vin'] = "switchAndEval(shifter_use_wire, {1:'stage1_Vout', 0:'shifter_Vout'})"
        stage2_functions['Vout'] = '0.9'
        
#        shifter_functions = shifter_varmap
#        shifter_functions['chosen_part_index'] = 'shifter_Drail_is_vdd'
        
         
        feedback_functions = feedback_varmap
        
        #build the main part
        part = CompoundPart(['Vin1','Vin2', 'Iout', 'Vdd', 'gnd'], pm, name)

        stage1_out = part.addInternalNode()
#        stage2_in = part.addInternalNode()

#        part.addPart( stage1_part,
#                      {'Vin1':'Vin1', 'Vin2':'Vin2',
#                       'Iout':stage1_out, 'Vdd':'Vdd', 'gnd':'gnd'},
#                     stage1_functions)
# switch the inputs to make the noninverting amplifier inverting
        part.addPart( stage1_part,
                      {'Vin1':'Vin2', 'Vin2':'Vin1',
                       'Iout':stage1_out, 'Vdd':'Vdd', 'gnd':'gnd'},
                      stage1_functions)

        part.addPart( stage2_part,
#                       {'Vin':stage2_in, 'Iout':'Iout',
                       {'Vin':stage1_out, 'Iout':'Iout',
                       'Vdd':'Vdd', 'gnd':'gnd'},
                      stage2_functions)
#         part.addPart( shifter_part,
#                       {'Vin':stage1_out, 'Vout':stage2_in,
#                        'Vdd':'Vdd', 'gnd':'gnd'},
#                       shifter_functions)

        #Note: this is a HACK for just capacitor!
        part.addPart( feedback_part, {'1':'Iout','2':stage1_out},
                      feedback_functions)
        
        
        self._parts[name] = part
        return part
        
    def dsViAmp2_SingleEndedMiddle_VddGndPorts_Fixed(self):
        """
        Description: dsViAmp2_SingleEndedMiddle_VddGndPorts
        but with the possibility of fixing the design variables
        """ 
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        dss_part = self.dsViAmp2_SingleEndedMiddle_VddGndPorts()

        #build the point_meta (pm)
        pm = PointMeta({})
        varmap = dss_part.unityVarMap()
        
        pm = self.updatePointMeta(pm, dss_part, varmap)

        # These switches allow to restrict some degrees of freedom
        # note that these are not fully correct,
        # especially wrt folding and fixing the input transistor
        
        stage1_fix_only_pmos_input=0
        stage1_fix_only_nmos_input=0
        stage1_disable_folding=1
        stage1_disable_degeneration=1
        stage1_disable_cascoding=1
        stage1_fix_ibias=0
        stage1_fix_input_diffpair=1
        stage1_fix_simple_load=1
        stage1_fix_other=1
        
        stage2_fix_only_pmos_input=0
        stage2_fix_only_nmos_input=0
        stage2_disable_folding=1
        stage2_disable_degeneration=1
        stage2_disable_cascoding=1
        stage2_fix_ibias=0
        stage2_fix_input_stage=1
        stage2_fix_simple_load=1
        stage2_fix_other=1
        
        #stage 1
        assert not (stage1_fix_only_pmos_input and stage1_fix_only_nmos_input)
        
        # fix the input transistor type
        if stage1_fix_only_pmos_input:
            del pm['stage1_input_is_pmos']
            varmap['stage1_input_is_pmos'] = '1'
            
        elif stage1_fix_only_nmos_input:
            del pm['stage1_input_is_pmos']
            varmap['stage1_input_is_pmos'] = '0'
        
        elif stage1_disable_folding:
            del pm['stage1_input_is_pmos']
            varmap['stage1_input_is_pmos'] = '1-stage1_loadrail_is_vdd'
        
        # disable folding
        if stage1_disable_folding:
            del pm['stage1_folder_Vgs']
            varmap['stage1_folder_Vgs'] = '0'
            del pm['stage1_folder_L']
            varmap['stage1_folder_L'] = '0.18e-6'
            del pm['stage1_Ibias2']
            varmap['stage1_Ibias2'] = '0'
        
        # fix the load to a simple current mirror
        if stage1_fix_simple_load:
            del pm['stage1_load_chosen_part_index']
            varmap['stage1_load_chosen_part_index']='0'
            del pm['stage1_load_cascode_Vgs']
            varmap['stage1_load_cascode_Vgs']='0'
            del pm['stage1_load_cascode_L']
            varmap['stage1_load_cascode_L']='0'
            del pm['stage1_load_L']
            varmap['stage1_load_L']='.18e-6'
       
        # fix the degeneration to no degeneration
        if stage1_disable_degeneration:
            del pm['stage1_degen_choice']
            varmap['stage1_degen_choice']='0'
            del pm['stage1_degen_fracDeg']
            varmap['stage1_degen_fracDeg']='0'
        
        # fix the input cascoding to no cascoding
        if stage1_disable_cascoding:
            del pm['stage1_inputcascode_is_wire']
            varmap['stage1_inputcascode_is_wire']='1'
            del pm['stage1_inputcascode_recurse']
            varmap['stage1_inputcascode_recurse']='0'
            del pm['stage1_inputcascode_L']
            varmap['stage1_inputcascode_L']='0'
            del pm['stage1_inputcascode_Vgs']
            varmap['stage1_inputcascode_Vgs']='0'
        
        # fix the input amplification stage
        if stage1_fix_input_diffpair:
            del pm['stage1_ampmos_Vgs']
            varmap['stage1_ampmos_Vgs']='0.7'
            del pm['stage1_ampmos_L']
            varmap['stage1_ampmos_L']='.18e-6'
            del pm['stage1_fracAmp']
            varmap['stage1_fracAmp']='0.5'
            
            del pm['stage1_fracVgnd']
            varmap['stage1_fracVgnd']=str(0.2/1.8)
            
            del pm['stage1_inputbias_Vgs']
            varmap['stage1_inputbias_Vgs']='0.6'
            del pm['stage1_inputbias_L']
            varmap['stage1_inputbias_L']='.18e-6'
        
        if stage1_fix_ibias:
            del pm['stage1_Ibias']
            varmap['stage1_Ibias']='0.1e-3'
        
        # fix all other design variables
        if stage1_fix_other:
            del pm['stage1_load_fracOut']
            varmap['stage1_load_fracOut']='0.5'
            del pm['stage1_load_fracIn']
            varmap['stage1_load_fracIn']='0.5'
            
            del pm['stage1_Vds_internal']
            varmap['stage1_Vds_internal']='0.9'        
  
       ## Stage 2
        assert not (stage2_fix_only_pmos_input and stage2_fix_only_nmos_input)
        
        # fix the input transistor type
        if stage2_fix_only_pmos_input:
            del pm['stage2_input_is_pmos']
            varmap['stage2_input_is_pmos'] = '1'
            
        elif stage2_fix_only_nmos_input:
            del pm['stage2_input_is_pmos']
            varmap['stage2_input_is_pmos'] = '0'
        
        elif stage2_disable_folding:
            del pm['stage2_input_is_pmos']
            varmap['stage2_input_is_pmos'] = '1-stage2_loadrail_is_vdd'
        
        # disable folding
        if stage2_disable_folding:
            del pm['stage2_inputbias_Vgs']
            varmap['stage2_inputbias_Vgs'] = '0'
            del pm['stage2_inputbias_L']
            varmap['stage2_inputbias_L'] = '0.18e-6'
            del pm['stage2_Ibias2']
            varmap['stage2_Ibias2'] = '0'
        
        # fix the load to a simple current mirror
        if stage2_fix_simple_load:
            del pm['stage2_loadcascode_recurse']
            varmap['stage2_loadcascode_recurse']='0'
            del pm['stage2_loadcascode_Vgs']
            varmap['stage2_loadcascode_Vgs']='0'
            del pm['stage2_loadcascode_L']
            varmap['stage2_loadcascode_L']='0'
            del pm['stage2_load_part_index']
            varmap['stage2_load_part_index']='1'
            del pm['stage2_load_Vgs']
            varmap['stage2_load_Vgs']='0.65'            
            del pm['stage2_load_L']
            varmap['stage2_load_L']='.18e-6'
       
        # fix the degeneration to no degeneration
        if stage2_disable_degeneration:
            del pm['stage2_degen_choice']
            varmap['stage2_degen_choice']='0'
            del pm['stage2_degen_fracDeg']
            varmap['stage2_degen_fracDeg']='0'
        
        # fix the input cascoding to no cascoding
        if stage2_disable_cascoding:
            del pm['stage2_inputcascode_is_wire']
            varmap['stage2_inputcascode_is_wire']='1'
            del pm['stage2_inputcascode_recurse']
            varmap['stage2_inputcascode_recurse']='0'
            del pm['stage2_inputcascode_L']
            varmap['stage2_inputcascode_L']='0'
            del pm['stage2_inputcascode_Vgs']
            varmap['stage2_inputcascode_Vgs']='0'
        
        # fix the input amplification stage
        if stage2_fix_input_stage:
            del pm['stage2_ampmos_Vgs']
            varmap['stage2_ampmos_Vgs']='0.9'
            del pm['stage2_ampmos_L']
            varmap['stage2_ampmos_L']='.18e-6'
            del pm['stage2_ampmos_fracAmp']
            varmap['stage2_ampmos_fracAmp']='0.5'
            
        if stage2_fix_ibias:
            del pm['stage2_Ibias']
            varmap['stage2_Ibias']='0.1e-3'
        
        # fix all other design variables
        if stage2_fix_other:
            del pm['stage2_load_fracLoad']
            varmap['stage2_load_fracLoad']='0.5'
            
#             del pm['stage2_loadrail_is_vdd']
#             varmap['stage2_loadrail_is_vdd']=''
  
          
        ## Level shifter

        del pm['shifter_cascode_D_Vgs']
        varmap['shifter_cascode_D_Vgs']='0'
        del pm['shifter_Vout']
        varmap['shifter_Vout']='0'
        del pm['shifter_use_wire']
        varmap['shifter_use_wire']='0'
        del pm['shifter_Drail_is_vdd']
        varmap['shifter_Drail_is_vdd']='0'
        del pm['shifter_cascode_do_stack']
        varmap['shifter_cascode_do_stack']='0'
        del pm['shifter_cascode_S_Vgs']
        varmap['shifter_cascode_S_Vgs']='0'
        del pm['shifter_cascode_fracVi']
        varmap['shifter_cascode_fracVi']='0'
        del pm['shifter_Vin']
        varmap['shifter_Vin']='0'
        del pm['shifter_amp_L']
        varmap['shifter_amp_L']='0'
        del pm['shifter_Ibias']
        varmap['shifter_Ibias']='0'
        del pm['shifter_cascode_S_L']
        varmap['shifter_cascode_S_L']='0'
        del pm['shifter_cascode_D_L']
        varmap['shifter_cascode_D_L']='0'

        ## Feedback
#         del pm['feedback_C']
#         varmap['feedback_C']='0'

                            
        #build the main part
        part = FlexPart(dss_part.externalPortnames(), pm, name)
        
        part.addPartChoice( dss_part, dss_part.unityPortMap(), varmap)
        
        self._parts[name] = part
        return part

    def dsViAmp2_VddGndPorts(self):
        """
        Description: differential-voltage-in, single-ended-current out
          two-stage amplifier.  Can be instantiated with having
          differential communication between stages, or single-ended
          (it's a FlexPart).
          
        Ports: Vin1, Vin2, Iout, stage1_loadrail, stage2_loadrail,
          stage1_opprail, stage2_opprail
        
        Variables: A merge of the variables of dsViAmp2_DifferentialMiddle
         and of dsViAmp2_SingleEndedMiddle
         -Plus 'chosen_part_index'
        
        Variable breakdown:
          For overall part: chosen_part_index
            0 : choose dsViAmp2_DifferentialMiddle_VddGndPorts
            1 : choose dsViAmp2_SingleEndedMiddle_VddGndPorts
          For  dsViAmp2_DifferentialMiddle_VddGndPorts: 1:1 mapping of its vars
          For  dsViAmp2_SingleEndedMiddle_VddGndPorts: 1:1 mapping of its vars
        """ 
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        dds_part = self.dsViAmp2_DifferentialMiddle_VddGndPorts()
        dss_part = self.dsViAmp2_SingleEndedMiddle_VddGndPorts()

        #build the point_meta (pm)
        pm = PointMeta({})
        dds_varmap = dds_part.unityVarMap()
        dss_varmap = dss_part.unityVarMap()
        
        pm = self.updatePointMeta(pm, dds_part, dds_varmap)
        pm = self.updatePointMeta(pm, dss_part, dss_varmap, True)

        #build the main part
        part = FlexPart(dss_part.externalPortnames(), pm, name)
        
        part.addPartChoice( dds_part, dds_part.unityPortMap(), dds_varmap)
        part.addPartChoice( dss_part, dss_part.unityPortMap(), dss_varmap)
        
        self._parts[name] = part
        return part
 
    def dsViAmp_VddGndPorts(self):
        """
        Description: chooses between dsViAmp1_VddGndPorts and
          dsViAmp2_VddGndPorts
          
        Ports: Vin1, Vin2, Iout, Vdd, gnd
        
        Variables: same as dsViAmp2
        
        Variable breakdown:
          For overall part: chosen_part_index (=do_two_stage)
            0 : choose dsViAmp1_VddGndPorts
            1 : choose dsViAmp2_VddGndPorts
          For  dsViAmp1_VddGndPorts: use all the stage1_xxx variables to set
            its corresponding xxx variables          
          For  dsViAmp2_VddGndPorts: 1:1 mapping of its vars
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        onestage_part = self.dsViAmp1_VddGndPorts()
#        twostage_part = self.dsViAmp2_VddGndPorts()
        twostage_part = self.dsViAmp2_SingleEndedMiddle_VddGndPorts() # FIXME: DDS amp is ruled out here.

        #build the point_meta (pm)
        pm = PointMeta({})
        twostage_varmap = twostage_part.unityVarMap()
        onestage_varmap = {}
        for new_name in onestage_part.point_meta.keys():
            old_name = 'stage1_' + new_name
            onestage_varmap[new_name] = old_name
        
        # if the stage1_part of the 2 stage has loadrail_is_vdd,
        # we should keep it that way when switching over to one stage.
        # remove chosen_part_index from the onestage from the varmap as \
        # it does not have to be in the pm for this part. it is equal to
        # the stage1_loadrail_is_vdd of the 2 stage parameter.
        del onestage_varmap['chosen_part_index']
        
        #pm = self.updatePointMeta(pm, onestage_part, onestage_varmap)
        pm = self.updatePointMeta(pm, twostage_part, twostage_varmap, True)
        
        # make sure the chosen_part_index from the onestage is connected
        # right
        onestage_functions=onestage_varmap;
        onestage_functions['chosen_part_index']='stage1_loadrail_is_vdd'

        #build the main part
        part = FlexPart(['Vin1', 'Vin2', 'Iout', 'Vdd', 'gnd'], pm, name)

        part.addPartChoice( onestage_part, onestage_part.unityPortMap(),
                            onestage_varmap)
        part.addPartChoice( twostage_part, twostage_part.unityPortMap(),
                            twostage_varmap)
        #build a summaryStr
        part.addToSummaryStr('Number of stages Stages: ','chosen_part_index+1')
        part.addToSummaryStr('Stage 1: loadrail=vdd?   ','stage1_loadrail_is_vdd')
        part.addToSummaryStr('Stage 1: input=pmos?     ','stage1_input_is_pmos')
        part.addToSummaryStr('Stage 1: folded?         ','stage1_input_is_pmos==stage1_loadrail_is_vdd')
        #part.addToSummaryStr('Stage 1: input Vgs       ','stage1_ampmos_Vgs')
        part.addToSummaryStr('Stage 1: Ibias           ','stage1_Ibias')
        part.addToSummaryStr('Stage 1: Ibias2          ','stage1_Ibias2')
        part.addToSummaryStr('Stage 1: load type       ','stage1_load_chosen_part_index')
        part.addToSummaryStr('Stage 1: cascoded?       ','1-stage1_inputcascode_is_wire')
        part.addToSummaryStr('Stage 1: frac Virt. gnd  ','stage1_fracVgnd')
        part.addToSummaryStr('Stage 1: frac Amplifier  ','stage1_fracAmp')
	
        part.addToSummaryStr('Stage 2: loadrail=vdd?   ','stage2_loadrail_is_vdd')
        part.addToSummaryStr('Stage 2: input=pmos?     ','stage2_input_is_pmos')
        part.addToSummaryStr('Stage 2: folded?         ','stage2_input_is_pmos==stage2_loadrail_is_vdd')
        part.addToSummaryStr('Stage 2: input Vgs       ','stage2_ampmos_Vgs')
        part.addToSummaryStr('Stage 2: Ibias           ','stage2_Ibias')
        part.addToSummaryStr('Stage 2: Ibias2          ','stage2_Ibias2')
        part.addToSummaryStr('Stage 2: load type       ','stage2_load_part_index')
        part.addToSummaryStr('Stage 2: cascoded?       ','1-stage2_inputcascode_is_wire')
	       
        self._parts[name] = part
        return part

    def dsViAmp_VddGndPorts_TST(self):
        """
        Description: chooses between dsViAmp1_VddGndPorts and
          dsViAmp2_VddGndPorts
          
        Ports: Vin1, Vin2, Iout, Vdd, gnd
        
        Variables: same as dsViAmp2
        
        Variable breakdown:
          For overall part: chosen_part_index (=do_two_stage)
            0 : choose dsViAmp1_VddGndPorts
            1 : choose dsViAmp2_VddGndPorts
          For  dsViAmp1_VddGndPorts: use all the stage1_xxx variables to set
            its corresponding xxx variables          
          For  dsViAmp2_VddGndPorts: 1:1 mapping of its vars
        """
        name = whoami()
        if self._parts.has_key(name): return self._parts[name]

        #parts to embed
        onestage_part = self.dsViAmp1_VddGndPorts()
#        twostage_part = self.dsViAmp2_VddGndPorts()
        #twostage_part = self.dsViAmp2_SingleEndedMiddle_VddGndPorts() # FIXME: DDS amp is ruled out here.

        #build the point_meta (pm)
        pm = PointMeta({})
        #twostage_varmap = twostage_part.unityVarMap()
        onestage_varmap = {}
        for new_name in onestage_part.point_meta.keys():
            old_name = 'stage1_' + new_name
            onestage_varmap[new_name] = old_name
    
        pm = self.updatePointMeta(pm, onestage_part, onestage_varmap)
        #pm = self.updatePointMeta(pm, twostage_part, twostage_varmap, True)

        # if the stage1_part of the 2 stage has loadrail_is_vdd,
        # we should keep it that way when switching over to one stage.
        # remove chosen_part_index from the onestage from the varmap as \
        # it does not have to be in the pm for this part. it is equal to
        # the stage1_loadrail_is_vdd of the 2 stage parameter.
        del pm['stage1_chosen_part_index']
        del pm['stage1_use_mirror_to_fold']
        
        # always fold the first stage
        #del pm['stage1_input_is_pmos']

        # make sure the chosen_part_index from the onestage is connected
        # right
        onestage_functions = onestage_varmap;
        # fix it to a folded topology
        onestage_functions['chosen_part_index']='stage1_input_is_pmos'
        onestage_functions['use_mirror_to_fold']='1'
        #onestage_functions['input_is_pmos'] = 'stage1_loadrail_is_vdd'

        #build the main part
        part = FlexPart(['Vin1', 'Vin2', 'Iout', 'Vdd', 'gnd'], pm, name)

        part.addPartChoice( onestage_part, onestage_part.unityPortMap(),
                            onestage_varmap)
        #part.addPartChoice( twostage_part, twostage_part.unityPortMap(),
                            #twostage_varmap)

        self._parts[name] = part
        return part

