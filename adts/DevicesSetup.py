"""DevicesSetup.py

-DevicesSetup describes how to calc device area, do rnd generation, and netlist MOS models.
 It's stored in an overall ProblemSetup.

"""
import copy
import logging
import math
import types

from adts.Point import RndPoint
from util import mathutil
from util.ascii import asciiTo2dArray
import util.constants as constants

log = logging.getLogger('adts')

class DevicesSetup:
    """Describes, for transistors, resistors, and capacitors:
    -how area is calculated 
    -how random generation is done; including the rnd points themselves 
    -model netlisting for transistors

    @attributes

      always_nominal -- bool
      process -- e.g. 'UMC180', 'UMC90'.  Must be supported by constants.py
      nominal_rnd_point -- RndPoint -- cache of the nominal rnd point, so that it always has same ID
      all_rnd_points -- list of RndPoint -- list of non-nominal rnd points, loaded from a file

    @notes

      Nominal by default.  Call configure(rnd_setting) to change.

    """
        
    def __init__(self, process):
        assert process in constants.NMOS_TEMPLATE_MODEL.keys()
        
        self.always_nominal = True
        self.process = process

        self.nominal_rnd_point = RndPoint([])

        #all_rnd_points has 1 nominal rnd point, plus 30 or so non-nominal rnd points
        #
        #***NOTE: we load in the rnd point data from a file so that rnd points are always the same.
        #   This makes it easy to share them across slaves, and to have same rnd points when continuing a run.
        #
        #   The file has one row per var, and one column per rnd point.
        #   -There need to be enough rnd vars to cover (num_rndvars_per_transistor * max_num_transistors +
        #    num_res + num_cap) which is about (1 * 20 + 3 + 3) = 26.
        #   -There need to be enough points to cover the max expected num rnd points, e.g. 30.
        #   -It's ok if there are excess columns or rows in the file.
        #  
        #   The file can be generated via:
        #   1. in octave: X = randn(26, 30);
        #   2. in octave: save -text rnd_points.txt X;
        #   3. in a text editor, remove the first few lines which start with '#'
        #   4. make sure that 'rnd_points.txt' is in the proper directory such that the call below can see it.
        self.all_rnd_points = [self.nominal_rnd_point]
        X = asciiTo2dArray('adts/rnd_points.txt')
        for point_i in range(X.shape[1]):
            rnd_point = RndPoint(list(X[:,point_i]))
            self.all_rnd_points.append(rnd_point)

        #cache of approx mos models
        self.cached_approx_mos_models = None

        #postconditions
        all_rnd_IDs = [r.ID for r in self.all_rnd_points]
        assert len(all_rnd_IDs) == len(set(all_rnd_IDs))
        assert self.all_rnd_points[0].ID == self.nominal_rnd_point.ID

    def __str__(self):
        return 'DevicesSetup={always_nominal=%s, process=%s}' % \
               (self.always_nominal, self.process)
        
    def doRobust(self):
        """Returns if the problem requires that we do robust."""
        return (not self.always_nominal)
    
    def featureSize(self):
        """Returns feature size (in m).  E.g. 180e-9."""
        return constants.FEATURE_SIZE[self.process]

    def vdd(self):
        """Returns supply voltage (in volts).  E.g. 1.8."""
        return constants.VDD[self.process]

    def approxMosModels(self):
        """Loads and returns approx mos models, for use by OpLibrary/OpLibrary2.  Caches
        so that only one disk access is ever needed."""
        #compute cache if needed
        if self.cached_approx_mos_models is None:
            #late import to avoid circular reference
            from problems.OpLibrary import ApproxMosModels
            log.info('Load approx mos models: begin')
            self.cached_approx_mos_models = ApproxMosModels(
                constants.NMOS_APPROX_MOS_MODEL_FILEBASE[self.process],
                constants.PMOS_APPROX_MOS_MODEL_FILEBASE[self.process])
            log.info('Load approx mos models: done')

        #return cached data
        return self.cached_approx_mos_models

    #===========================================================================================
    #Configure self's data: change nominal vs. not, change process
    def makeNominal(self):
        self.always_nominal = True

    def makeRobust(self):
        self.always_nominal = False
    
    def setProcess(self, process):
        assert process in constants.NMOS_TEMPLATE_MODEL.keys()
        
        self.process = process
    
    #===========================================================================================
    #Get rnd points & IDs
    def rndIDs(self):
        """Returns the rnd IDs appropriate to self's problem: if always nominal, then just the nominal rnd ID;
        else return the ID of each rnd point in all_rnd_points."""
        if self.doRobust():
            return [r.ID for r in self.all_rnd_points]
        else:
            return [self.nominal_rnd_point.ID]

    def nominalRndPoint(self):
        """Returns the nominal rnd point"""
        return self.nominal_rnd_point

    def getRndPoint(self, target_rnd_ID):
        """Returns the rnd point having the target rnd_ID"""
        for r in self.all_rnd_points:
            if r.ID == target_rnd_ID:
                return r
        raise AssertionError, "do not have rnd point with ID=%s" % target_rnd_ID

    #===========================================================================================
    #Resistor
    def _techResistorArea(self):
        """Returns the minimum resistor area specified by the technology, so
        that matching is not horrible.  In m^2."""
        sigma_matching = constants.RES_SIGMA_MATCHING[self.process] * 1e-6 #units: % * m
        matching_spec = constants.RES_TARGET_MATCHING_SPEC #units: %
        tech_area = (sigma_matching / matching_spec)**2  #units: m^2
        return tech_area

    def _rBasedResistorArea(self, R):
        """Returns the minimum area that the input resistance R (in ohms) takes,
        if ignoring tech area.  In m^2.
        """
        area_per_sq = constants.RES_AREA_PER_SQUARE[self.process] #units: m^2/sq
        R_per_sq = constants.RES_SHEET_RESISTANCE[self.process] * 1e-3 #units: ohms / sq
        r_based_area = R * area_per_sq / R_per_sq #units: m^2
        return r_based_area
    
    def resistorArea(self, R):
        """Given the resistance, returns the area.  Adaptive such that a minimum matching spec
        is met."""
        tech_area = self._techResistorArea() #units: m^2
        r_based_area = self._rBasedResistorArea(R) #units: m^2
        area = max(tech_area, r_based_area) #units: m^2
        return area
    
    def varyResistance(self, R, rnd_val):
        """Returns a varied version of resistance R (in ohms), using the rnd_val to choose how much.
        rnd_val is drawn from N(0,1).  In ohms.
        """
        #preconditions
        assert mathutil.isNumber(R)
        assert mathutil.isNumber(rnd_val)
        assert R >= 0
        
        #corner case
        if R == 0:
            return 0.0
        
        #main case...
        Rnom = R #units: ohms
        sigma_matching = constants.RES_SIGMA_MATCHING[self.process] * 1e-6 #units: % * m
        area = self.resistorArea(Rnom) #units: m^2
        percent_variation = sigma_matching / math.sqrt(area) #units: %
        R = Rnom + Rnom * percent_variation * rnd_val #units: ohms
        R = max(R, 0.0) #disallow <0
        return R

    #===========================================================================================
    #Capacitor
    def _techCapacitorArea(self):
        """Returns the minimum capacitor area specified by the technology, so
        that matching is not horrible.  In m^2."""
        sigma_matching = constants.CAP_SIGMA_MATCHING[self.process] * 1e-6 #units: % * m
        matching_spec = constants.CAP_TARGET_MATCHING_SPEC #units: %
        tech_area = (sigma_matching / matching_spec)**2  #units: m^2
        return tech_area

    def _cBasedCapacitorArea(self, C):
        """Returns the minimum area that the input capacitance C (in farads) takes,
        if ignoring tech area.  In m^2.
        """
        density = constants.CAP_DENSITY[self.process] * 1e-15 / (1e-6)**2 #units: F/m^2
        c_based_area = C / density #units: m^2
        return c_based_area
    
    def capacitorArea(self, C):
        """Given the capacitance, returns the area.  Adaptive such that a minimum matching spec
        is met."""
        tech_area = self._techCapacitorArea() #units: m^2
        c_based_area = self._cBasedCapacitorArea(C) #units: m^2
        area = max(tech_area, c_based_area) #units: m^2
        return area
    
    def varyCapacitance(self, C, rnd_val):
        """Returns a varied version of capacitance C (in farads), using the rnd_val to choose how much.
        rnd_val is drawn from N(0,1).  In farads.
        """
        #preconditions
        assert mathutil.isNumber(C)
        assert mathutil.isNumber(rnd_val)
        assert C >= 0
        
        #corner case
        if C == 0:
            return 0.0
        
        #main case...
        Cnom = C #units: farads
        sigma_matching = constants.CAP_SIGMA_MATCHING[self.process] * 1e-6 #units: % * m
        area = self.capacitorArea(Cnom) #units: m^2
        percent_variation = sigma_matching / math.sqrt(area) #units: %
        C = Cnom + Cnom * percent_variation * rnd_val #units: farads
        C = max(C, 0.0) #disallow <0
        return C    

    #===========================================================================================
    #MOS models
    def mosArea(self, w, l, m):
        """Returns area (in m^2), given w (in m) and l (in m), and m (unitless)."""
        return (w * l * m)
    
    def nominalMosNetlistStr(self):
        """Returns a string that has one PMOS mos model, and one NMOS mos model, both with
        nominal settings.
        Model names are Pnom, Nnom (must be in line with EmbPart.spiceNetlistStr().
        """
        return self.mosNetlistStr(True, "nom", 1.0, 1.0, 0.0) + \
               self.mosNetlistStr(False, "nom", 1.0, 1.0, 0.0)

    def mosNetlistStr(self, is_pmos, part_name, w, l, rnd_value):
        """
        @description
        
          Generates a netlist declaring a SPICE .model, where the model parameters are set according to the inputs:

        @arguments
          is_pmos -- bool -- pmos or nmos?
          part_name -- str -- the name of the part in context of the circuit.  For non-nominal models,
            this will affect the model name
          w -- float -- transistor width
          l -- float -- transistor length
          rnd_value -- float -- in N(0,1); governs vth0 variation

        @return

          netlist -- string
            
        @notes
        
          The parameters here are based on UMC_MM180_REG18_V123, with randomness model
          taken by varying VTH0 based on 

        """
        #preconditions
        if constants.AGGR_TEST:
            assert isinstance(is_pmos, types.BooleanType)
            assert isinstance(part_name, types.StringType)
            assert mathutil.isNumber(w)
            assert mathutil.isNumber(l)
            assert mathutil.isNumber(rnd_value)
            if rnd_value != 0.0:
                assert self.doRobust()
        
        #main work
        if is_pmos:
            netlist = copy.copy(constants.PMOS_TEMPLATE_MODEL[self.process])
            base_VTH0 = constants.PMOS_BASE_VTH0[self.process]
            model_name = 'P%s' % part_name
        else:
            netlist = copy.copy(constants.NMOS_TEMPLATE_MODEL[self.process])
            base_VTH0 = constants.NMOS_BASE_VTH0[self.process]
            model_name = 'N%s' % part_name

        #magic number alert!
        # -by changing minL_compared_to_180, we can estimate variations at 90, 65, 45 nm:
        #  set it to 1.0 for 180nm, 2.0 for 90nm, 2.77 for 65nm, 4.0 for 45nm
        minL_compared_to_180 = constants.MINL_COMPARED_TO_180[self.process]
        AVT = constants.AVT_AT_180
        
        VTH_stddev = (AVT / math.sqrt(w * l)) * minL_compared_to_180
        VTH0 = base_VTH0 + VTH_stddev * rnd_value

        #Alternative formulation:
        # Vth variation for 2007, 2008, ..., 2022: (ITRS 2007)(for minimally-sized device; to
        #  have variation for larger devices it's proportional to area of course):
        #[31%, 35%, 40, 40, 40, 58, 58, 81, 81, 81, 81, 112, 112, 112, 112, 112]
        
        netlist = netlist.replace('REPLACEME_VTH0', '%9e' % VTH0)
        netlist = netlist.replace('REPLACEME_MODELNAME', model_name)
        
        #postconditions
        if constants.AGGR_TEST:
            assert isinstance(netlist, types.StringType)

        #done
        return netlist
