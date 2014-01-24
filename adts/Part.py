"""Part.py

A Part can either be
-an atomic 'primitive', e.g. a resistor or MOS, or
-a collection of EmbeddedParts (and therefore hierarchically composed

Thus, a (synthesized) circuit design is a PartEmbpart holding a CompoundPart.

A 'port' is what is at the Part end of a connection.
A 'node' connects to ports (i.e. like a hyperedge), e.g. internal_nodes.
"""
import copy
import logging
import random
import string
import types
import math

from adts.Metric import Metric
from adts.Point import *
from adts.DevicesSetup import DevicesSetup
from adts.Schema import Schema, Schemas, combineSchemasList
from util import mathutil
from util.constants import AGGR_TEST, BAD_METRIC_VALUE

log = logging.getLogger('part')

ATOMIC_PART_TYPE   = 0
COMPOUND_PART_TYPE = 1
FLEX_PART_TYPE     = 2
ALL_PART_TYPES = [0, 1, 2]

def switchAndEval(case, case2result):
    """
    @description

      This is an on-the-fly 'switch' statement, where 'case' is like
      the argument to a hardcoded case statement, and 'case2result'
      is what replaces the hardcoded statement itself.

      Example: 
      -case2result == {3:'4.2', 'yo':'7+2', 'p':'1/0', 'default':'400/9'}
      -then the equivalent hardcoded behavior would be:
      
       if case == 3:      return 4.2
       elif case == 'yo': return 9
       elif case == 'p':  return 1/0 (which raises a ZeroDivisionError!)
       else:              return 1000.0
       
      -so if the input was 3 then it would return 4.2
      -note the use of the special key 'default', for default behavior

    @arguments

      case -- a comparable_value, usually one of the key values in case2result
      case2result -- dict of comparable_value : result_value, ie case : result

    @return

      result -- one of the result_values in case2result

    @exceptions

    @notes

      It will not complain that there is a missing 'default' unless it
      does not find another key in case2result that matches 'case'.
      
    """
    if case2result.has_key(case):
        return eval(str(case2result[case]))
    else:
        return eval(str(case2result['default']))

class FunctionDOC:
    """
    @description

      FunctionDOC = Function-based Device Operating Constraint.
      
      E.g. to ensure that transistors stay in their proper operating region etc.
      'Function' DOCs can be measured with functions, ie without simulation.

    @attributes

      metric -- Metric object -- the metric associated with the DOC.  Should not
        have an objective on it.
      function_str -- string -- to compute the metric value from an
        input scaled_points dict.  Example: '(W - L)*2'
      
    @notes
    """
    def __init__(self, metric, function_str):
        """
        @description

          Constructor.
        
        @arguments
        
          metric -- Metric object -- see class description
          function_str -- string -- ''
        
        @return

          new_probe -- Probe object
    
        @exceptions
    
        @notes
        """
        #validate inputs
        if not isinstance(metric, Metric):
            raise ValueError(metric.__class__)
        if metric.improve_past_feasible:
            raise ValueError("no objectives allowed on DOC metric")
        if not isinstance(function_str, types.StringType):
            raise ValueError(function_str.__class__)

        #set attributes
        self.metric = metric
        self.function_str = function_str

    def resultsAreFeasible(self, scaled_point, part=None):
        """
        @description

          Is 'scaled_point' feasible?
        
        @arguments

          scaled_point -- Point object
          Part -- the part we're checking the DOC for
        
        @return

          feasible -- bool
    
        @exceptions
    
        @notes          
        """
        metric_value = evalFunction(scaled_point, self.function_str, part)

        retval = self.metric.isFeasible(metric_value)
        #log.info("DOC %s evaluates to %s (metric = %s)" % \
        #       ( str(self.metric.name), str(retval), str(metric_value) ) )
        #if not retval:
        #    log.info(" point: %s" % (str(scaled_point)))
            
        return retval

class SimulationDOC:
    """
    @description

      Simulation DOC = Device Operating Constraint found by simulation.
      E.g. to ensure that transistors
      stay in their proper operating region etc.

    @attributes

      metric -- Metric object -- the metric associated with the DOC.  Should not
        have an objective on it.
      function_str -- string -- to compute the metric value from an 
        input lis_results dict.  Example: '(vgs - vt)*2' 
      
    @notes
    """
    def __init__(self, metric, function_str, min_metric_value, max_metric_value):
        """
        @description

          Constructor.
        
        @arguments
        
          metric -- Metric object -- see class description
          function_str -- string -- ''
          min_metric_value -- float -- approximate minimum value.  Used for scaling a measured value
            into roughly [0,1] so that its constraint violation can be conveniently measured
          max_metric_value -- float -- approximate maximum value
        
        @return

          new_probe -- Probe object
    
        @exceptions
    
        @notes

          We only allow lowercase letters in function_str so that
          it matches up more readily with SPICE-simulation-extracted info.
        """
        #preconditions
        if not isinstance(metric, Metric):
            raise ValueError(metric.__class__)
        if metric.improve_past_feasible:
            raise ValueError("no objectives allowed on DOC metric")
        if not isinstance(function_str, types.StringType):
            raise ValueError(function_str.__class__)
        if function_str != string.lower(function_str):
            raise ValueError("function_str needs to be lowercase: %s" % function_str)
        assert mathutil.isNumber(min_metric_value)
        assert mathutil.isNumber(max_metric_value)

        #set attributes
        self.metric = metric
        self.function_str = function_str
        self.min_metric_value = min_metric_value
        self.max_metric_value = max_metric_value
        self.metric_range = (max_metric_value - min_metric_value)

    def constraintViolation01(self, lis_results, device_name):
        """
        @description

          Violation of this DOC, scaled to roughly a range [0,1].  Returns 0.0 if no violation.
        
        @arguments

          lis_results -- dict of 'lis__device_name__measure_name' : lis_value 
          device_name -- name of the device that we're interested in
        
        @return

          feasible -- bool
    
        @exceptions
    
        @notes          
        """
        metric_value = self.evaluateFunction(lis_results, device_name)
        feasible = self.metric.isFeasible(metric_value)
        if feasible: #early exit
            return 0.0
        elif self.metric_range == 0.0:
            return 1.0
        else:
            violation = self.metric.constraintViolation(metric_value)
            violation01 = violation / self.metric_range

            #give it a fixed component and non-fixed component, therefore we induce a preference
            # to minimize number of DOCs violated, before minimizing the degree of violation of
            # remaining infeasible DOCs.  Think "rank-based" plus finer granularity for improved search.
            violation01 = 0.5 + violation01 / 2.0 
            return violation01

    def resultsAreFeasible(self, lis_results, device_name):
        """
        @description

          Are 'lis_results' feasible for this DOC on the specified device?
        
        @arguments

          lis_results -- dict of 'lis__device_name__measure_name' : lis_value 
          device_name -- name of the device that we're interested in
        
        @return

          feasible -- bool
    
        @exceptions
    
        @notes          
        """
        metric_value = self.evaluateFunction(lis_results, device_name)
        return self.metric.isFeasible(metric_value)

    def evaluateFunction(self, lis_results, device_name):
        """
        @description

          Evaluate the value of 'lis_results' on the specified device
        
        @arguments

          lis_results -- dict of 'lis__device_name__measure_name' : lis_value --
            Note that this will contain many device names that are not
            equal to the next argument ('device_name')
          device_name -- name of the device that we're interested in evaluating
            this DOC on
        
        @return

          evaluated_value -- float (or int)
    
        @exceptions
    
        @notes

          WARNING: because 'lis_results' are extracted from a SPICE output
           and SPICE puts everything to lowercase, we lowercase the device_name
           within this routine.  
        """
        #Build up a lis_point that function_str can interpret
        #Example: 'lis__INSTANCENAME__vgs' => 'vgs', wherever INSTANCENAME
        # matches device_name
        device_name = string.lower(device_name)
        lis_point = {}
        for big_lis_name, lis_value in lis_results.items():
            prefix = 'lis__' + device_name + '__'
            if prefix == big_lis_name[:len(prefix)]:
                lis_name = big_lis_name[len(prefix):]
                lis_point[lis_name] = lis_value

        #Now evaluate it!
        value = evalFunction(lis_point, self.function_str)

        #Done
        return value


class Part(object):
    """
    @description

      A Part can be an atomic 'primitive', e.g. a resistor or MOS,
      or it can be built up by other types of parts, eg CompoundPart.
      (The Part class is abstract.)
  
      Note that because non-Atomic Parts have point_meta
      rather than just variable names, it means that they bound
      variables from its higher level, but also there will be other
      bounds as values get computed going towards the bottom AtomicParts.
      E.g.: a higher-level 'Vbias' may be more constraining than
      the lowest-level 'DC' voltage.
        
      But it can be the other way too, e.g. a higher-level multiplier 'K'
      may be less constraining than the 'K' on a specific type of current
      mirror.  Or a higher level 'W*L' variable is still subject
      to lower-level constraints on 'W' and 'L' individually.
      
    @attributes

      name -- string -- this part's name
      point_meta -- PointMeta object -- defines the variable names and ranges
        that this part has
      parttype -- one of ATOMIC_PART_TYPE, COMPOUND_PART_TYPE, FLEX_PART_TYPE

      function_DOCs -- list of DOC -- these are the DOCs that can
        be found merely by evaluating a funcion (no simulation necessary).
        Work for any level of Part, not just atomic parts.
        Can be very nice for quickly determining if a circuit will
        be feasible, without doing simulation.
        Example: have Vgs-0.4 > 0.2 on
        an operating-point driven formulation where Vgs is a var in the
        part's point_meta and 0.4 / 0.2 are approximations of Vgs / Vod.
        
      simulation_DOCs -- list of DOC -- these are DOCs that will be found
        on AtomicParts during simulation.

      probes -- list of Probe -- used for measuring and constraining
        device operating conditions
      
      _external_portnames -- list of string -- names for each port that is
        visible to other parts that use this.  Order is important.
        
      _internal_nodenames -- list of string -- names for each node that
        is defined internally by this Part.  Order is important

      _summary_str_tuples -- list of (label, func_str) -- these can
        be set such that additional info is put at the beginning of a netlist
        See summaryStr().
      
    @notes
    
      Each Part created get a unique ID.  This is implemented
      by the class-level attribute '_ID_counter' in combination with
      the 'ID' property() call.

      Methods that all Parts provide:
        externalPortnames() : list of string
        internalNodenames() : list of string
        embeddedParts(scaled_point) : part_list
        internalNodenames() : name_list
        portNames() : list of string
        unityVarMap() : dict of varname : varname
        unityPortMap() : dict of {ext_port1:ext_port1, ext_port2:ext_port, ...}
        __str__() : string
        str2(tabdepth, full_info) : string
      
    """ 
    _ID_counter = 0L
    all_choice_var_names = set([]) #class-level list of all choice_var names

    def __init__(self, external_portnames, point_meta, parttype, name=None):
        """
        @description

          Constructor.
        
        @arguments
        
          external_portnames -- list of strings (order is important!!) --
            names for each port that is visible to other parts that use this
          point_meta -- PointMeta object -- describes varnames & ranges that
            this part has
          name - string -- this part's name; auto-generated if 'None' is input
        
        @return

          new_part -- Part object
    
        @exceptions
    
        @notes          
        """
        self._ID = Part._ID_counter
        Part._ID_counter += 1

        assert mathutil.allEntriesAreUnique(external_portnames)
        self._external_portnames = external_portnames

        self._internal_nodenames = []

        assert isinstance(point_meta, PointMeta)
        self.point_meta = point_meta

        if name is None:
            self.name = 'part_ID' + str(self.ID)
        else:
            assert isinstance(name, types.StringType)
            self.name = name

        self.parttype = parttype

        self.function_DOCs = []
        self.simulation_DOCs = []

        self._summary_str_tuples = []

        self.attr_to_keep_as_ref = []
        
    ID = property(lambda s: s._ID)

    def __getstate__(self):
        """Remove some attributes that consume too much when pickled
        """
        refs = {}
        keys_to_remove = ['approx_mos_models']
        for key in keys_to_remove:
            if key in self.__dict__.keys():
                # save reference
                refs[key] = getattr(self, key)
                log.debug("removing %s from %s for pickling, was %s" % (str(key), self.name, str(refs[key])))
                                    
                # remove
                setattr(self, key, None)
        
        d = copy.copy(self.__dict__)
        
        # restore the references in self
        for key in keys_to_remove:
            if key in self.__dict__.keys():
                setattr(self, key, refs[key])
                log.debug("restored %s to %s" % (str(key),str(getattr(self, key)) ))
        return d
    
    def __setstate__(self, d):
        for key, value in d.items():
            setattr(self, key, value)

    def __deepcopy__(self, memo={} ):
        """Override the deepcopying of some contents and
           keep using references for them
        """
        # create a new instance of ourself
        result = self.__class__.__new__(self.__class__)
            
        memo[id(self)] = result
        for key, value in self.__dict__.iteritems():
            if key in self.attr_to_keep_as_ref:
                log.info("keeping reference to %s as %s" % (key, value))
                setattr(result, key, value)
                pass
            else:
                setattr(result, key, copy.deepcopy(value, memo))

        return result

    def reattachAttribute(self, key, attr, key_has_to_exist = True, parts_done = []):
        """re-attaches a reference to the Part and to its embedded parts.
        """
        top_level = (len(parts_done) == 0)
            
        # already done
        if self.ID in parts_done:
            log.debug("part %s already done" % self.ID)
            return
        
        if not key_has_to_exist:
            log.debug("restoring %s of part %s" % (str(key), self.name))
            
            setattr(self, key, attr)
        else:
            if key in self.__dict__.keys():
                log.info("restoring %s of part %s %s to %s" % (str(key), self.__class__, self.name, attr))
                setattr(self, key, attr)

        parts_done.append(self.ID)
        
        # recurse down
        if 'embedded_parts' in self.__dict__.keys():
            for p in self.embedded_parts:
                p.reattachAttribute( key, attr, key_has_to_exist, parts_done )
        if 'part_choices' in self.__dict__.keys():
            for p in self.part_choices:
                p.reattachAttribute( key, attr, key_has_to_exist, parts_done )

        if top_level:
            log.debug("parts done %s" % str(parts_done))
            # this is to account for a very nasty side-effect
            # in which parts_done persists between subsequent calls
            # when no value has been passed.
            del parts_done[:]

    def sortedChoiceVarnames(self):
        """Return point_meta's choice keys, sorted.  Exploit caching."""
        varnames= [varname for varname,varmeta in self.point_meta.iteritems()
                   if varmeta.isChoiceVar()]
        return sorted(varnames)

    def validate(self):
        """
        Raises an exception if:
        -any possible embedded part's connections-to-parent not in self's ports
        -...
        """
        raise NotImplementedError, 'implement in child'

    def validateEmbPartConnections(self):
        """
        Raises an exception if any possible embedded part's
        connections-to-parent not in self's ports
        """
        if not AGGR_TEST: return
        for embpart in self.possibleEmbeddedParts():
            parent_portnames_used_by_embpart = set(embpart.connections.values())
            parent_portnames = set(self.portNames())
            extra_ports = mathutil.listDiff(parent_portnames_used_by_embpart,
                                            parent_portnames)
            assert not extra_ports, 'extra_ports = %s\n' \
                   'parent_portnames = %s\n' \
                   'parent_portnames_used_by_embpart = %s\n' \
                   % (extra_ports, parent_portnames, parent_portnames_used_by_embpart)

    def mayContainPartWithID(self, compare_ID):
        """Returns True if this Part or a possible sub-part has compareID.
        Checks all the way down the hierarchy"""
        if self.ID == compare_ID:
            return True
        for poss_emb_part in self.possibleEmbeddedParts():
            if poss_emb_part.part.mayContainPartWithID(compare_ID):
                return True
        return False
        
    def addFunctionDOC(self, function_DOC):
        """
        @description

          Adds DOC_instance to self.function_DOCs
        
        @arguments

          function_DOC -- FunctionDOC object
        
        @return

          <<none>>
    
        @exceptions
    
        @notes
        """
        assert isinstance(function_DOC, FunctionDOC)
        self.function_DOCs.append(function_DOC)

    def addSimulationDOC(self, simulation_DOC):
        """
        @description

          Adds DOC_instance to self.simulation_DOCs
        
        @arguments

          simulation_DOC -- SimulationDOC object
        
        @return

          <<none>>
    
        @exceptions
    
        @notes
        """
        assert isinstance(simulation_DOC, SimulationDOC)
        self.simulation_DOCs.append(simulation_DOC)

    def externalPortnames(self):
        """
        @description

          Returns a list of self's external portnames
        
        @arguments

          <<none>>
        
        @return

          external_portnames -- list of string --
    
        @exceptions
    
        @notes

          Implemented here, therefore no need to implement in children.
        """
        return self._external_portnames
        
    def internalNodenames(self):
        """
        @description

          Returns a list of self's internal nodenames.
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes
        
          Abstract method -- child to implement.
        """ 
        raise NotImplementedError('implement in child')  

    def numSubpartPermutations(self):
        """
        @description

          Returns the number of possible permutations of different
          subparts, i.e. total size of the (structural part) of topology
          space for this part.
        
        @arguments

          <<none>>
        
        @return

          count -- int
    
        @exceptions
    
        @notes

        """
        return self.schemas().numPermutations()

    def schemas(self):
        """
        @description

          Returns a list of possible structures, in a compact fashion.
          Useful to count the total possible number of topologies,
          and also for a more fair random generation of individuals.
        
        @arguments

          <<none>>
        
        @return

          schemas -- Schemas object
    
        @exceptions
    
        @notes

        """
        raise NotImplementedError('implement in child')
    
    def schemasWithVarRemap(self, emb_part):
        """
        @description

          Returns the schemas that come from 'emb_part', but remapping
          the vars of that to line up with self's point_meta vars.
          Assumes that emb_part can be found in self.embedded_parts

        @arguments

          emb_part -- EmbeddedPart --
        
        @return

          schemas -- Schemas object
                
        @exceptions
    
        @notes

          This is currently a helper function used by CompoundPart and FlexPart
        """
        remap_schemas = Schemas()
        for emb_schema in emb_part.part.schemas():
            remap_schema = Schema()
            for (emb_schema_var, emb_schema_vals) in emb_schema.items():
                emb_schema_func = emb_part.functions[emb_schema_var]
                if emb_schema_func in self.point_meta.keys(): #1:1 mapping
                    remap_var = emb_schema_func
                    poss_vals = emb_schema_vals
                    remap_schema[remap_var] = poss_vals
                elif isSimpleEqualityFunc(emb_schema_func):
                    remap_var1, remap_var2 = varsOfSimpleEqualityFunc(emb_schema_func)
                    assert remap_var1 != None and remap_var2 != None
                    poss_vals1 = self.point_meta[remap_var1].possible_values
                    remap_schema[remap_var1] = poss_vals1
                    
                    poss_vals2 = self.point_meta[remap_var2].possible_values
                    remap_schema[remap_var2] = poss_vals2
                elif emb_part.varHasNumberFunc(emb_schema_var):
                    pass #nothing to do because we don't have remap var
                elif isInversionFunc(emb_schema_func):
                    remap_var = varOfInversionFunc(emb_schema_func)
                    assert remap_var is not None
                    if emb_schema_vals == [0,1]: poss_vals = [0,1]
                    elif emb_schema_vals == [1]: poss_vals = [0]
                    elif emb_schema_vals == [0]: poss_vals = [1]
                    else: raise "shouldn't get here"
                    remap_schema[remap_var] = poss_vals
                elif isSimpleFunc(emb_schema_func):
                    pass
                else:
                    import pdb; pdb.set_trace()
                    raise AssertionError("general case not handled yet")
            remap_schema.checkConsistency()
            remap_schemas.append(remap_schema)
            
        remap_schemas.checkConsistency()
        assert len(remap_schemas) > 0, "should just contain a Schemas with empty Schema inside"
        return remap_schemas
    
    def embeddedParts(self, scaled_point):
        """        
          Abstract method -- child to implement.
        """ 
        raise NotImplementedError('implement in child') 

    def possibleEmbeddedParts(self):
        """        
          Abstract method -- child to implement.
        """ 
        raise NotImplementedError('implement in child') 

    def portNames(self):  
        """
        @description

          Returns list of external_port_names + internal_node_names
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes       
        
          Abstract method -- child to implement.   
        """ 
        raise NotImplementedError('implement in child')

    def unityVarMap(self):
        """
        @description

          Returns a dict of {varname1:varname1, varname2:varname2, ...}
          Which can be useful for the 'functions' arg of EmbeddedParts
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes

          Implemented here, therefore no need to implement in children.  
        """
        varnames = self.point_meta.keys()
        return dict(zip(varnames, varnames))

    def unityPortMap(self):
        """
        @description
        
          Returns a dict of {ext_port1:ext_port1, ext_port2:ext_port, ...}
          Which can be useful for the 'connections' arg of Embedded Parts
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes

          Implemented here, therefore no need to implement in children.  
        """
        ports = self.externalPortnames()
        return dict(zip(ports, ports))

    def __str__(self):
        raise NotImplementedError('implement in child')

    def str2(self, tabdepth):
        raise NotImplementedError('implement in child')

    def summaryStr(self, scaled_point):
        """This is helpful for quickly identifying the structural
        characteristics of a netlist defined by the point of this Part."""
        if len(self._summary_str_tuples) == 0: return ''
        s = '\n* ==== Summary for: %s ====\n' % self.name
        for label, func_str in self._summary_str_tuples:
            if func_str == '':
                s += '* %s\n' % label
            else:
                s += '* %s = %s\n' % (label, evalFunction(scaled_point,func_str))
        s += '* ==== Done summary ====\n\n'
        return s

    def addToSummaryStr(self, label, func_str):
        self._summary_str_tuples.append((label, func_str))
        
class AtomicPart(Part):
    """
    @description

      An AtomicPart can be instantiated directly as a SPICE primitive
      
    @attributes
        
      spice_symbol -- string -- what's used in SPICE netlisting;
        e.g. 'R' for resistor and 'G' for vccs
      external_portnames -- see Part
      point_meta --  see Part
      name -- string -- see Part
      
    @notes
    """

    def __init__(self, spice_symbol, external_portnames, point_meta,  name = None):
        """
        @description
        
        @arguments
        
          spice_symbol -- string -- what's used in SPICE netlisting
          external_portnames -- see Part
          point_meta --  see Part
          name -- string -- see Part
        
        @return
    
        @exceptions
    
        @notes          
        """
        Part.__init__(self, external_portnames, point_meta, ATOMIC_PART_TYPE, name)
                
        assert isinstance(spice_symbol, types.StringType)
        assert len(spice_symbol) == 1
        self.spice_symbol = spice_symbol

    def validate(self):
        """
        Raises an exception if:
        -...
        """
        if not AGGR_TEST: return
        pass

    def internalNodenames(self):
        """
        @description

          Returns a list of self's internal nodes.

          Because this is an AtomicPart, the list is empty.
        
        @arguments

          <<none>>
        
        @return

          internal_nodenames -- list of string --
    
        @exceptions
    
        @notes          
        """ 
        return []
    
    def schemas(self):
        """
        @description

          Returns a list of possible structures, in a compact fashion.
          Useful to count the total possible number of topologies,
          and also for a more fair random generation of individuals.
        
        @arguments

          <<none>>
        
        @return

          schemas -- Schemas object
    
        @exceptions
    
        @notes

        """
        return Schemas([Schema()])

    def embeddedParts(self, scaled_point):
        """     
        """
        return []

    def possibleEmbeddedParts(self):
        """     
        """
        return []

    def portNames(self):
        """
        @description
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes          
        """ 
        return self.externalPortnames() + self.internalNodenames()
    
    def __str__(self):
        """
        @description
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes          
        """ 
        return self.str2()
        
    def str2(self, tabdepth=0, full_info=True):
        """
        @description
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes          
        """ 
        s = ''
        s += tabString(tabdepth)
        s += 'AtomicPart={'
        s += ' name=%s' % self.name
        s += '; ID=%s' % self.ID
        if full_info:
            s += '; spice_symbol=%s' % self.spice_symbol
            s += '; external_portnames=%s' % self.externalPortnames()
        s += '; point_meta=%s' % self.point_meta.str2(full_info)
        if full_info:
            s += '; function_DOCs=%s' % str(self.function_DOCs)
            s += '; simulation_DOCs=%s' % str(self.simulation_DOCs)
        s += ' /AtomicPart} '
        return s

def atomicPartToCompoundPart(part):
    """Creates a CompoundPart, given an AtomicPart."""
    assert part.parttype == ATOMIC_PART_TYPE
    new_part = CompoundPart(part.externalPortnames(),
                            copy.deepcopy(part.point_meta),
                            novelPartName(part.name))
    new_part.addPart(part,
                     part.unityPortMap(),
                     part.unityVarMap())
    if hasattr(part, 'approx_mos_models'):
        new_part.approx_mos_models = part.approx_mos_models
    return new_part

def copyCompoundPart(part):
    """Creates a copy of a CompoundPart: a shallow copy, except
    for ensuring a unique name and ID.  Therefore it's mutation-ready"""
    assert part.parttype == COMPOUND_PART_TYPE
    new_part = CompoundPart(part.externalPortnames(), copy.deepcopy(part.point_meta), novelPartName(part.name))
    new_part.embedded_parts = [copyEmbeddedPart(emb_part) for emb_part in part.embedded_parts]
    new_part._internal_nodenames = copy.copy(part._internal_nodenames)
    if hasattr(part, 'approx_mos_models'):
        new_part.approx_mos_models = part.approx_mos_models
    return new_part

def copyEmbeddedPart(emb_part):
    """Safe copy of an EmbeddedPart.  Meant for use as a helper
    function to copyCompoundPart, copyFlexPart; therefore it only
    does a safe copy of connections and functions, but not 'part'."""
    new_emb_part = EmbeddedPart(emb_part.part,
                                copy.deepcopy(emb_part.connections),
                                copy.deepcopy(emb_part.functions))
    return new_emb_part
    
        
class CompoundPart(Part):
    """
    @description

      A CompoundPart is a collection of other parts and
      their connections.  It can have internal ports.  It cannot be directly
      instantiated directly as a spice primitive.

      After __init___, a CompoundPart has no embedded parts/connections or
      internal nodes.  New parts/connections are embedded via addPart(),
      and new internal nodes are added via addInternalNode().
      
    @attributes

      embedded_parts -- list of EmbeddedPart -- the actual parts (order not
       important, except in making netlisting consistent)
      
    @notes
    """

    def __init__(self, external_portnames, point_meta, name = None):
        """
        @description
        
        @arguments
        
          external_portnames -- see Part.  Remember, it's only the ports that
            the outside world sees, not internal nodes.
          point_meta -- see Part.
          name -- see Part
        
        @return

          new_compound_part -- CompoundPart object
    
        @exceptions
    
        @notes          
        """
        Part.__init__(self, external_portnames, point_meta,
                      COMPOUND_PART_TYPE, name)
        
        # Each entry of embedded_parts is an EmbeddedPart.
        # Order is not important, _except_ to make netlisting consistent.
        self.embedded_parts = []

    def validate(self):
        """
        Raises an exception if:
        -any internal node is connected <2 times
        -any possible embedded part's connections-to-parent not in self's ports
        -any possible embedded part does not validate
        """
        #check: any internal node is connected <2 times
        for internal_node in self.internalNodenames():
            count = 0
            for emb_part in self.embedded_parts:
                count += sum([1
                              for self_portname in emb_part.connections.values()
                              if self_portname == internal_node])
            if count < 2:
                raise AssertionError(
                    "internal_node '%s' in part '%s' only had %d connections"
                    % (internal_node, self.name, count))

        #check: embedded part's connections-to-parent not in self's ports
        self.validateEmbPartConnections()
            
        #check: validate embedded parts
        #note: the following does NOT recurse and is not supposed to
        # (too expensive)
        if AGGR_TEST:
            for emb_part in self.possibleEmbeddedParts():
                emb_part.validate()
    
    def addInternalNode(self):
        """
        @description

          Adds a new internal port, and returns its name.
        
        @arguments

          <<none>>
        
        @return

          name -- string -- name of new internal port
    
        @exceptions
    
        @notes

          Modifies self.
        """
        name = NodeNameFactory().build()
        self._internal_nodenames.append(name)
        return name

    def addPart(self, part_to_add, connections, functions):
        """
        @description

          Adds part_to_add to this Part using connections / functions as
          the 'how to add'.
        
        @arguments
        
          part_to_add -- Part object -- a description of the sub-part to add
          connections --
            -- dict of part_to_add_ext_portname : self_intrnl_or_ext_portname --
            -- how to wire part_to_add's external ports from
            self's external_ports or self's internal_nodes. 
          functions --
            -- dict of subpart_to_add_varname : str_func_of_self_var_names
            -- stores how to compute sub-part_to_add's vars from self.
            
        @return

          <<nothing>>
    
        @exceptions
    
        @notes
        """ 
        #preconditions
        assert part_to_add.parttype in ALL_PART_TYPES
        for sub_port, self_port in connections.items():
            if self_port not in self.portNames():
                raise ValueError("self_port=%s not in self.portnames=%s" % (self_port, self.portNames()))
        for var_name, function in functions.items():
            if function == 'IGNORE':
                raise ValueError('Forgot to specify the function for var: %s'% var_name)
                                                  
        #main work       
        embpart = EmbeddedPart(part_to_add, connections, functions)
        self.embedded_parts.append(embpart)

        if AGGR_TEST:
            validateFunctions(functions, self.point_meta.minValuesScaledPoint())

    def internalNodenames(self):
        """
        @description

          Returns a list of self's internal nodes.

          Because this is an AtomicPart, the list is empty.
        
        @arguments

          <<none>>
        
        @return

          internal_nodenames -- list of string --
    
        @exceptions
    
        @notes          
        """ 
        return self._internal_nodenames
    
    def schemas(self):
        """
        @description

          For this CompoundPart, returns a list of possible structures,
          in a compact fashion (i.e. effectively a list of Schema objects).
          
          Useful to count the total possible number of topologies,
          and also for a more fair random generation of individuals.
        
        @arguments

          <<none>>
        
        @return

          schemas -- Schemas object
    
        @exceptions
    
        @notes

        """
        #gather the schemas per emb part, and keep a count per emb part
        remapped_emb_schemas = [] # list of Schemas objects
        for emb_part in self.embedded_parts:
            schemas = self.schemasWithVarRemap(emb_part)
            remapped_emb_schemas.append(schemas)

        #now combine the list of Schemas.  This does the 'combinatorial explosion'
        # of possibilities, but removes redundancies as needed.
        if remapped_emb_schemas:
            schemas = combineSchemasList(remapped_emb_schemas)
        else:
            schemas = Schemas([Schema()])
        
        return schemas
    

    def embeddedParts(self, scaled_point):
        """
        @description

          Returns the embedded parts that arise when 'scaled_point' is input. 
        
        @arguments

          scaled_point -- Point object -- 
        
        @return

          embedded_parts -- list of EmbeddedPart objects
    
        @exceptions
    
        @notes

          It takes in scaled_point in order to maintain a consistent
          interface with other Parts, which actually need such an input
          in order to determine the current embedded parts (e.g. FlexPart).
          But for this part (CompoundPart) it actually does not use scaled_point.
          
        """ 
        return self.embedded_parts

    def possibleEmbeddedParts(self):
        """
        @description

          Returns all possible embedded parts, independent of any input point. 
        
        @arguments
        
        @return

          embedded_parts -- list of EmbeddedPart objects
    
        @exceptions
    
        @notes
        """ 
        return self.embedded_parts
    
    def portNames(self):
        """
        @description

          Returns a concatenated list of external portnames and internal
          nodenames.
        
        @arguments

          <<none>>
        
        @return

          port_names -- list of string
    
        @exceptions
    
        @notes          
        """ 
        return self.externalPortnames() + self.internalNodenames()
    
    def __str__(self):
        """
        @description

          Override of str()
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes
        """ 
        return self.str2()
        
    def str2(self, tabdepth=0, full_info=True):
        """
        @description

          'Nice' tabbed string object.  
        
        @arguments

          tabdepth -- int -- how deep in the hierarchy are we, and therefore
            how many tabs do we want to space our output?
        
        @return

          string_rep -- string
    
        @exceptions
    
        @notes

          This is not directly implemented in str() because we wanted
          the extra argument of 'tabdepth' which makes it easier
          to figure out hierarchy of complex Parts.
        """ 
        s = ''
        s += tabString(tabdepth)
        s += 'CompoundPart={'
        s += ' ID=%s' % self.ID
        s += "; name='%s'" % self.name
        if full_info:
            s += '; externalPortnames()=%s' % self.externalPortnames()
        s += '; point_meta=%s' % self.point_meta.str2(full_info)
        if full_info:
            s += '; function_DOCs=%s' % str(self.function_DOCs)
            s += '; simulation_DOCs=%s' % str(self.simulation_DOCs)
        s += '; # embedded_parts=%d' % len(self.embedded_parts)
        s += '; actual embedded_parts of part:\n'

        if len(self.embedded_parts) == 0: s += '(None)'

        for embpart in self.embedded_parts:
            descr_s = "('%s' within '%s')(partID=%d within partID=%d) :" % \
                      (embpart.part.name, self.name, embpart.part.ID, self.ID)
            
            s += '\n%ssub-EMBPART: %s \n%s%s' % \
                 (tabString(tabdepth+1), descr_s, ' ' * (tabdepth+2),
                  embpart.str2(tabdepth=tabdepth+2, full_info=full_info))
            
            s += '\n%sPART of sub-embpart: %s \n%s' % \
                 (tabString(tabdepth+1), descr_s,
                  embpart.part.str2(tabdepth=tabdepth+2, full_info=full_info))

            s += '\n'

        s += tabString(tabdepth)
        s += ' /CompoundPart (ID=%s)} ' % self.ID
        return s


def copyFlexPart(part):
    """Creates a copy of a FlexPart: a shallow copy, except
    for ensuring a unique name and ID.  Therefore it's mutation-ready"""
    #preconditions
    assert part_to_add.parttype == FLEX_PART_TYPE
    part.validateChoices()

    #main work
    dummy_pm = PointMeta({})
    new_part = FlexPart(part.externalPortnames(), dummy_pm, novelPartName(part.name), copy.copy(part.choice_var_name))
    new_part.part_choices = [copyEmbeddedPart(emb_part) for emb_part in part.part_choices]
    new_part.point_meta = copy.deepcopy(part.point_meta)
    if hasattr(part, 'approx_mos_models'):
        new_part.approx_mos_models = part.approx_mos_models

    #postconditions
    new_part.validateChoices()

    #return
    return new_part

class FlexPart(Part):
    """
    @description

      A FlexPart can implement one of many sub-parts, depending on
      the value that its self.choice_var_name has.

      It is akin to an 'interface' that holds all the possible implementations.
      
    @attributes

      part_choices -- list of EmbeddedPart --
      choice_var_name -- string -- this is the varname on self.point_meta
        that is used to choose from self.part_choices
      <<plus the attributes inherited from Part>>
      
    @notes
    """
    
    def __init__(self, external_portnames, point_meta, name = None,
                 choice_var_name = None):
        """
        @description

          Constructor.
        
        @arguments
        
          external_portnames -- see Part
          point_meta -- see Part.  Note that an extra VarMeta will be added,
            named self.choice_var_name with range (0,1,...,num_choices-1)
          name -- see Part
        
        @return

          FlexPart object.
    
        @exceptions
    
        @notes          
        """
        #preconditions
        #(none)

        #main work
        self_point_meta = copy.deepcopy(point_meta)
        if choice_var_name is None:
            self.choice_var_name = 'chosen_part_index' #default
        else:
            self.choice_var_name = choice_var_name
        Part.all_choice_var_names |= set([self.choice_var_name])
        
        self_point_meta.addVarMeta(ChoiceVarMeta([], None, self.choice_var_name))
        
        Part.__init__(self, external_portnames, self_point_meta, FLEX_PART_TYPE, name)
        
        self.part_choices = []

        #postconditions
        self.validate(need_some_choices = False)

    def validate(self, need_some_choices=True):
        """
        Raises an exception if:
        -validateChoices(need_some_choices) fails
        -any possible embedded part's connections-to-parent not in self's ports
        -any possible embedded part does not validate
        """
        self.validateChoices(need_some_choices)

        #check: embedded part's connections-to-parent not in self's ports
        self.validateEmbPartConnections()

        #note: the following does NOT recurse and is not supposed to
        # (too expensive)
        if AGGR_TEST:
            for emb_part in self.possibleEmbeddedParts():
                emb_part.validate()

    def validateChoices(self, need_some_choices=True):
        """Raises exception if
        -choice varmeta fails validate()
        -num choices in choice varmeta != len(part_choices)
        -have 0 choices in choice varmeta
        -have 0 trustworthy choices in choice varmeta
        """
        varmeta = self.choiceVarMeta()
        assert varmeta.name == self.choice_var_name
        varmeta.validate()
        assert len(varmeta.possible_values) == len(self.part_choices)
        if need_some_choices:
            assert len(varmeta.possible_values) > 0, "FlexPart should have >0 choices"
            assert len(varmeta.nonNovelValues()) > 0, "FlexPart should have >0 trustworthy choices"

    def addPartChoice(self, part_choice_to_add, connections, functions, is_novel = False):
        """
        @description

          Add a candidate part, returns the embedded_part created

          The external_portnames of part_choice_to_add must be
          a subset of self.externalPortnames().
        
        @arguments
        
          part_choice_to_add -- Part object -- another 'part choice' 
          connections -- dict of part_to_add_ext_portname : self_ext_portname --
            -- how to wire part_to_add's external ports from self's external_ports
          functions -- dict of subpart_to_add_varname : str_func_of_self_var_names
            -- stores how to compute sub-part_to_add's vars from self.
          is_novel -- bool -- if True, then the choiceVarMeta will track that part choice as novel.
        
        @return
    
        @exceptions
    
        @notes

          The 'connections' and 'functions' arguments are identical to those of CompoundPart.addPart(),
          except for 'connections' we actually don't have to use all of a FlexPart's external ports and
          there are no possible internal ports of 'self' to connect to.
          
          This implementation is nearly identical to CompoundPart.addPart()
        """
        #preconditions
        assert part_choice_to_add.parttype in ALL_PART_TYPES
        for sub_port, self_port in connections.items():
            if self_port not in self.portNames():
                raise ValueError("self_port=%s not in self.portnames=%s" % (self_port, self.portNames()))
        for var_name, function in functions.items():
            if function == 'IGNORE':
                raise ValueError('Forgot to specify the function for var: %s'% var_name)

        #main work
        for self_port in connections.values():
            assert self_port in self.portNames(), (self_port, self.portNames())
                                                         
        embpart = EmbeddedPart(part_choice_to_add, connections, functions)
        self.part_choices.append(embpart)

        # calculate the random generation weight of the part choice
        # the weight is defined as the number of permutations the schemas
        # of a choice can take.
        # The idea is that this number of permutations
        # is proportional to the difficulty of generating a good random instance
        # of this embedded part. When uniform weighting is used for the selection
        # between part choices, this results in a bias towards the simpler
        # structures. The weighting tries to undo this bias.
        w = part_choice_to_add.schemas().numPermutations()
        log.debug("Adding part %s to part %s, weight %s" % (part_choice_to_add.name, self.name, str(w)))
        
        new_v = len(self.part_choices) - 1
        self.point_meta[self.choice_var_name].addNewPossibleValue(new_v, w, is_novel)

        #postconditions
        if AGGR_TEST:
            validateFunctions(functions, self.point_meta.minValuesScaledPoint())

        #return
        return embpart

    def choiceVarMeta(self):
        """Returns the VarMeta in self.point_meta that has the choice var"""
        return self.point_meta[self.choice_var_name]

    def randomChoice(self, allow_novel):
        """Return an embedded part, selected randomly.  Can only include
        novel parts in the list of possibilities if allow_novel is True"""
        self.validateChoices()
        varmeta = self.choiceVarMeta()
        unscaled_choice = varmeta.createRandomUnscaledVar(allow_novel)
        choice = varmeta.scale(unscaled_choice)
        return self.part_choices[choice]

    def chosenPart(self, scaled_point):
        """
        @description

          Returns the part that's chosen according to the input point.

          Specifically: uses index = scaled_point[self.choice_var_name] to
          return self.part_choices[index]
        
        @arguments

          scaled_point -- Point object --
        
        @return

          chosen_part -- EmbeddedPart object --
    
        @exceptions
    
        @notes          
        """
        assert scaled_point.is_scaled
        assert len(self.part_choices) > 0, "need to have added choices to self"
        
        index = scaled_point[self.choice_var_name]
        
        if (index < 0) or (index >= len(self.part_choices)):
            raise IndexError('List index out of range: index=%d; len(self.part_choices)=%d'
                             '\nscaled_point=%s  \npart=%s' % 
                             (index, len(self.part_choices), str(scaled_point), str(self)))

        #main work
        return self.part_choices[index]

    def internalNodenames(self):
        """
        @description

          Returns a list of all the internal nodenames.  Because this is a FlexPart, that means
          it returns an empty list.
        
        @arguments

          <<none>>
        
        @return

          internal_nodenames -- list of string -- in this case []
    
        @exceptions
    
        @notes          
        """ 
        return []

    def portNames(self):
        """
        @description
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes          
        """ 
        return self.externalPortnames() + self.internalNodenames()

    def schemas(self):
        """
        @description

          For this FlexPart, returns a list of possible structures, in a compact fashion. Useful to count
          the total possible number of topologies, and also for a more fair random generation of individuals.
        
        @arguments

          <<none>>
        
        @return

          schemas -- Schemas object
    
        @exceptions
    
        @notes

        """
        schemas = Schemas()
        for emb_part_i, emb_part in enumerate(self.part_choices):
            next_schemas = self.schemasWithVarRemap(emb_part)
            for next_schema in next_schemas:
                next_schema[self.choice_var_name] = [emb_part_i]
            schemas.extend(next_schemas)

        schemas.merge()

        if not schemas:
            schemas = Schemas([Schema()])

        return schemas
    
    def embeddedParts(self, scaled_point):
        """
        @description

          Returns the embedded parts that arise when 'scaled_point' is input.

          Because this is a FlexPart, it generates the lists ON THE FLY
          according to the value of self.choice_var_name in scaled_point
        
        @arguments

          scaled_point -- Point object --
        
        @return

          embedded_parts -- list of EmbeddedPart objects
            
        @exceptions
    
        @notes

          Being a FlexPart, the list of embedded_parts will always have exactly one entry.
        """
        return [self.chosenPart(scaled_point)]

    def possibleEmbeddedParts(self):
        """
        @description

          Returns all possible embedded parts, independent of any input point. 
        
        @arguments
        
        @return

          embedded_parts -- list of EmbeddedPart objects
    
        @exceptions
    
        @notes
        """ 
        return self.part_choices
    
    def __str__(self):
        """
        @description
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes          
        """ 
        return self.str2()
        
    def str2(self, tabdepth=0, full_info=True):
        """
        @description

          'Nice' tabbed string object.  
        
        @arguments

          tabdepth -- int -- how deep in the hierarchy are we, and therefore
            how many tabs do we want to space our output?
        
        @return

          string_rep -- string
    
        @exceptions
    
        @notes

          This is not directly implemented in str() because we wanted the extra argument of 'tabdepth'
          which makes it easier to figure out hierarchy of complex Parts.
        """ 
        s = ''
        s += tabString(tabdepth)
        s = 'FlexPart={'
        s += ' ID=%s' % self.ID
        s += "; name='%s'" % self.name
        s += "; choice_var_name='%s'" % self.choice_var_name
        if full_info:
            s += '; external_portnames=%s' % self.externalPortnames()
        s += '; point_meta=%s' % self.point_meta.str2(full_info)
        if full_info:
            s += '; function_DOCs=%s' % str(self.function_DOCs)
            s += '; simulation_DOCs=%s' % str(self.simulation_DOCs)
        s += '; # part_choices=%d' % len(self.part_choices)
        s += '; actual part_choices:\n'

        if len(self.part_choices) == 0: s += '(None)'
        
        for embpart in self.part_choices:
            descr_s = "('%s' option for '%s')(partID=%d option for partID=%d) :"\
                      % (embpart.part.name, self.name, embpart.part.ID, self.ID)
            
            s += '\n%spart_choice EMBPART: %s \n%s%s' % \
                 (tabString(tabdepth+1), descr_s, ' '*(tabdepth+2),
                  embpart.str2(tabdepth=tabdepth+2, full_info=full_info))
            
            s += '\n%sPART of part_choice: %s \n%s' % \
                 (tabString(tabdepth), descr_s,
                  embpart.part.str2(tabdepth+2, full_info))

            s += '\n'

        s += tabString(tabdepth)
        s += ' /FlexPart (ID=%s)} ' % self.ID
        return s


class EmbeddedPart:
    """
    @description

      An EmbeddedPart holds a Part _instantiated_ in the context of a parent Part.  It tells how
      that sub part's ports are connected to the parent part's ports, and how to compute the sub part's variables
      from the parent's variables.
      
    @attributes
    
      part - Part object -- for this sub part
      connections -- dict of self_portname : parent_portname -- how to wire self's external ports from parent ports
      functions -- dict of self_varname : str_func_of_parent_var_names -- how to compute self's variables from parent.  
      
    @notes
    """
    _partnum = 0 #used to get unique part numbers when netlisting
    
    def __init__(self, part, connections, functions):
        """
        @description

          Constructor.
        
        @arguments

          part -- see 'attributes' section above
          connections -- see 'attributes' section above
          functions -- see 'attributes' section above
                      
        @return

          new_embedded_part -- EmbeddedPart object
    
        @exceptions
    
        @notes

          If this EmbeddedPart is going to be the top-level part, then set the function dict's values
          to scaled_var_values (numbers),not str funcs. That's all that's needed in order to propagate
          number values to all the child blocks, thus instantiating the whole part.
          
        """
        #preconditions
        if AGGR_TEST:
            for conn_value in connections.values():
                assert isinstance(conn_value, types.StringType)
            validateVarLists(connections.keys(), part.externalPortnames(), 'connections', 'part.externalPortnames')

        #set main attributes
        self.part = part
        self.connections = connections
        self.functions = functions

        #caching for speed
        # -_cached_isNumberFunc is dict of function_string : bool.
        # -NOT dict of varname : bool because the function_string itself can
        #  change from a number to non-number.
        self._cached_isNumberFunc = {} 

        #postconditions
        self.validate()
        
    def reattachAttribute(self, key, attr, key_has_to_exist = True, parts_done = []):
        """re-attaches a reference to this and the embedded Parts
        """
        # pass on the request to the part embedded
        self.part.reattachAttribute( key, attr, key_has_to_exist, parts_done)
            
    def __eq__(self, other):
        """
        Returns True if
        -self.part.ID == other.part.ID (i.e. doesn't dive deeper into Part)
        -connections and functions are the same
        """
        return self.part.ID == other.part.ID and \
               self.connections == other.connections and \
               self.functions == other.functions

    def validate(self):
        """Raises exception if:
        -_validatePointMetaFunctionConsistency fails
        -connection keys do not line up with self.part.externalPortnames
        -connection values are not strings
        """
        if AGGR_TEST:
            for conn_value in self.connections.values():
                assert isinstance(conn_value, types.StringType)
            
            self._validatePointMetaFunctionConsistency()
            validateVarLists(self.connections.keys(),
                             self.part.externalPortnames(),
                             'embpart.connections',
                             'embpart.part.externalPortnames')

    def _validatePointMetaFunctionConsistency(self):
        part = self.part
        functions = self.functions
        fnames = set(functions.keys())
        pnames = set(part.point_meta.keys())
        
        if fnames != pnames:
            fnames = sorted(functions.keys())
            pnames = sorted(part.point_meta.keys())
            extra_fnames = sorted(mathutil.listDiff(fnames, pnames))
            extra_pnames = sorted(mathutil.listDiff(pnames, fnames))
            s = "\n\n"
            s += "=================================================\n"
            s += "ERROR:\n"
            
            s += "embpart.part.ID = %d" % part.ID
            s += "; embpart.part.name = %s\n" % part.name
            s += "\n"
            
            s += "Var names in embpart.functions and "
            s += "embpart.part.point_meta (=%s.point_meta) don't align\n" % \
                 part.name
            s += "\n"
            
            s += "extra names in embpart.functions"
            s += " (or missing names in %s.point_meta) = %s\n" % \
                 (part.name, extra_fnames)
            s += "\n"
            
            s += "extra names in %s.point_meta" % part.name
            s += " (or missing names in embpart.functions) = %s\n"  % \
                 extra_pnames
            s += "\n"
            
            s += "embpart.functions = %s" % niceFunctionsStr(functions)
            s += "\n"

            print s
            import pdb; pdb.set_trace()
            raise ValueError(s)

    def varHasNumberFunc(self, emb_schema_var):
        """Returns True if self.functions[emb_schema_var]
        is a number function according to 'isNumberFunc'.
        Leverages caching for speed.

        Handles extra whitespace, and () around the number.
        """
        func = self.functions[emb_schema_var]
        if not self._cached_isNumberFunc.has_key(func):
            self._cached_isNumberFunc[func] = isNumberFunc(func)

        return self._cached_isNumberFunc[func]

    def _validateFlexStability(self, vars_before):
        """Helper for wrapEachNonFlexPartWithFlexPart"""
        if AGGR_TEST:
            if vars_before is not None:
                validateVarLists(vars_before, self.part.point_meta.keys(),
                                 'vars_before', 'vars_after')
            for flex_part in self.flexParts():
                flex_part.validateChoices()

        if AGGR_TEST: #_really_ slow, should maybe just turn off
            #- call to subPartsInfo can break
            for emb_part in self._embeddedPartsOrderedBottomUp():
                log.info("Validate emb_part holding '%s'" % emb_part.part.name)
                for round_i in range(5):
                    scaled_point = emb_part.part.point_meta.createRandomScaledPoint(True)
                    info = emb_part.subPartsInfo(scaled_point)

    def embeddedAtomicParts(self):
        """Returns ALL embedded Atomic Parts in self's hierarchy"""
        return self._embeddedPartsOfTargetPartType(ATOMIC_PART_TYPE)
    
    def embeddedCompoundParts(self):
        """Returns ALL embedded Compound Parts in self's hierarchy"""
        return self._embeddedPartsOfTargetPartType(COMPOUND_PART_TYPE)

    def embeddedFlexParts(self):
        """Returns ALL embedded Flex Parts in self's hierarchy"""
        return self._embeddedPartsOfTargetPartType(FLEX_PART_TYPE)
        
    def _embeddedPartsOfTargetPartType(self, target_part_type):
        """Returns a flat list of all EmbeddedPart objects in the hierarchy
        whose 'part' is the target_part_type of AtomicPart, FlexPart, 
        CompoundPart, or 'any'.

        Note that this is NOT dependent on an input point, i.e. it
        gives the whole hierarchy rather than just a subset defined by 'point'.
        """
        #preconditions
        if AGGR_TEST:
            assert (target_part_type == 'any') or (target_part_type in ALL_PART_TYPES)
        
        #we'll be building this up
        emb_parts_found = []

        #recurse
        for embpart in self.part.possibleEmbeddedParts():
            emb_parts_found += embpart._embeddedPartsOfTargetPartType(
                target_part_type)

        #process current node
        if (target_part_type == 'any') or (self.part.parttype == target_part_type):
            emb_parts_found.append(self)

        #postconditions
        if AGGR_TEST:        
            for embpart in emb_parts_found:
                assert isinstance(embpart, EmbeddedPart), str(embpart)
                if target_part_type != 'any':
                    assert embpart.part.parttype == target_part_type

        #done
        return emb_parts_found

    def _filterPartsToTargetPartType(self, emb_parts, target_part_type):
        """Returns a subset of 'emb_parts' -- the ones of 'target_part_type'
        which can be AtomicPart, FlexPart, CompoundPart, or 'any'. """
        if target_part_type == 'any':
            return emb_parts
        else:
            return [emb_part for emb_part in emb_parts
                    if emb_part.part.parttype == target_part_type]
        

    def parts(self):
        """Returns all Atomic, Compound, and Flex Parts in self's hierarchy"""
        return self._partsOfTargetPartType('any')
    
    def atomicParts(self):
        """Returns all Atomic Parts in self's hierarchy"""
        return self._partsOfTargetPartType(ATOMIC_PART_TYPE)
    
    def compoundParts(self):
        """Returns all Compound Parts in self's hierarchy"""
        return self._partsOfTargetPartType(COMPOUND_PART_TYPE)
    
    def flexParts(self):
        """Returns all Flex Parts in self's hierarchy"""
        return self._partsOfTargetPartType(FLEX_PART_TYPE)

    def _partsOfTargetPartType(self, target_part_type):
        """Returns a flat list of all Part objects in the hierarchy, where the 'part' is the
        target_part_type of AtomicPart, FlexPart, CompoundPart, or 'any'.

        Note that this is NOT dependent on an input point, i.e. it
        gives the whole hierarchy rather than just a subset defined by 'point'.
        """
        parts, IDs = [], []
        for emb_part in self._embeddedPartsOfTargetPartType(target_part_type):
            class_matches = (target_part_type == 'any') or (emb_part.part.parttype == target_part_type)
            if class_matches and (emb_part.part.ID not in IDs):            
                IDs.append(emb_part.part.ID)
                parts.append(emb_part.part)

        return parts

    def updateOptPointMetaFlexPartChoices(self, broadening_means_novel):
        """
        @description

          Traverses self's hierarchy and pulls up all FlexPart choices upwards if they are not
          already considered somehow.

          If the choice is considered, but the range is not broad enough for all the choices, update that.
        
        @arguments
        
          broadening_means_novel -- bool -- if True and if we have to broaden a var meta's ranges,
            then all new values are considered 'novel'
        
        @return
    
        @exceptions
    
        @notes
        """
        #for each emb_part = EmbeddedPart in bottom-up fashion:
        #  for each child
        #    for each choice var of child:
        #      if (var is not accounted for in child's functions)
        #         add that choice index to functions
        #
        #      if (var is not accounted for in emb_part.point_meta)
        #         add that choice index to emb_part.point_meta
        #
        #      if (child_pm's var's range is larger than emb_part.point_meta's
        #       range for that var)
        #          broaden the range of emb_part.point_meta's range

        log.debug('updateOptPointMetaFlexPartChoices(): begin')
        num_before = len(self.part.point_meta)

        #for each EmbeddedPart in bottom-up fashion:
        for emb_part in self._embeddedPartsOrderedBottomUp():
            emb_pm = emb_part.part.point_meta

            #for each choice var of each child...
            for child_emb_part in emb_part.part.possibleEmbeddedParts():
                child_pm = child_emb_part.part.point_meta
                for child_varname in child_pm.choiceVars():
                    
                    #get base info
                    child_varmeta = child_pm[child_varname]
                    if not child_emb_part.functions.has_key(child_varname):
                        child_emb_part.functions[child_varname] = None

                    #corner case
                    if child_emb_part.varHasNumberFunc(child_varname):
                        continue
                    
                    #set function in child
                    func = child_emb_part.functions[child_varname]
                    if func is None:
                        child_emb_part.functions[child_varname] = child_varname
                        log.debug("Added child_emb_part.functions['%s']='%s' "
                                  "(child_emb_part is a '%s')" %
                                  (child_varname, par_varname,
                                   child_emb_part.part.name))

                    #if (emb_part.part.point_meta doesn't have par_varname), add
                    par_varnames = child_emb_part.varsGoverningChildVar(
                        emb_pm.keys(), child_varname, False)
                    if len(par_varnames) > 0: #bug if >1?
                        par_varname = par_varnames[0] 
                    else:
                        par_varname = child_varname

                    if not emb_pm.has_key(par_varname):
                        if isinstance(child_varmeta, DiscreteVarMeta):
                            new_varmeta = DiscreteVarMeta(child_varmeta.possible_values, par_varname)
                            new_varmeta.novel_values = child_varmeta.novel_values
                        elif isinstance(child_varmeta, ContinuousVarMeta):
                            new_varmeta = ContinuousVarMeta(
                                child_varmeta.logscale, child_varmeta.min_unscaled_value,
                                child_varmeta.max_unscaled_value, name=par_varname,
                                use_eq_in_netlist=child_varmeta.use_eq_in_netlist)
                        else:
                            raise AssertionError
                        emb_pm.addVarMeta(new_varmeta)
                        log.debug("Added '%s' to %s emb_part's point_meta" %
                                  (par_varname,emb_part.part.name))

                        if broadening_means_novel:
                            print "shouldn't be here because we don't want to add vars with novelty operators"
                            import pdb; pdb.set_trace()

                    #if (child_pm's var's range >  emb_part.point_meta's),
                    # then broaden.  Also update novelty if needed.
                    child_discrete = isinstance(child_varmeta, DiscreteVarMeta)
                    parent_discrete = isinstance(emb_pm[par_varname], DiscreteVarMeta)
                    if child_discrete and parent_discrete:
                        possvals_child = child_varmeta.possible_values
                        possvals_par = emb_pm[par_varname].possible_values
                        missing_vals = mathutil.listDiff(possvals_child, possvals_par)
                        for missing_val in missing_vals:
                            emb_pm[par_varname].addNewPossibleValue(missing_val, broadening_means_novel)

                        if missing_vals:
                            log.debug("Broadened range of choices for var '%s' of %s emb_part" %
                                      (par_varname, emb_part.part.name))
                    elif (not child_discrete) and (not parent_discrete):
                        emb_pm[par_varname].min_unscaled_value = min(
                            child_varmeta.min_unscaled_value, emb_pm[par_varname].min_unscaled_value)
                        emb_pm[par_varname].max_unscaled_value = max(
                            child_varmeta.max_unscaled_value, emb_pm[par_varname].max_unscaled_value)
                        emb_pm[par_varname].updateScaledMinMax()

                    else:
                        raise AssertionError("cannot handle a mix of child and parent being discrete/"
                                             "continuous.  child_discrete=%s, parent_discrete=%s" %
                                             (child_discrete, parent_discrete))                        
                            

        #also need to bring variables from self.part.point_meta into
        # self.functions
        for pm_var in self.part.point_meta.iterkeys():
            if not self.functions.has_key(pm_var):
                self.functions[pm_var] = None

        log.debug('Num vars before update=%d' % num_before)
        log.debug('Num vars after update=%d' % len(self.part.point_meta))
        log.debug('updateOptPointMetaFlexPartChoices(): done')

        #postconditions
        if False: #AGGR_TEST: #_really_ slow
            for emb_part in self._embeddedPartsOrderedBottomUp():
                emb_part._validatePointMetaFunctionConsistency()
                
    def _embeddedPartsOrderedBottomUp(self):
        """Returns 'emb_parts' = list of EmbeddedPart, in an order such that:
        if j > i, emb_parts[i] is never a parent of emb_parts[j].  
        """
        #we'll be building this up
        return_emb_parts = []

        #recurse
        for embpart in self.part.possibleEmbeddedParts():
            return_emb_parts += embpart._embeddedPartsOrderedBottomUp()

        #append only now ("post-order" node processing in tree traversal)
        return_emb_parts.append(self)

        #done
        return return_emb_parts    
        

    def numAtomicParts(self, scaled_point):
        """
        @description

          Returns the number of atomic parts in the instantiation of this Part defined by 'point',
          which overrides self.functions.
          
          Includes all child parts.
        
        @arguments

          scaled_point -- dict of varname : var_value
        
        @return

          num_parts -- int -- total number of atomic parts
    
        @exceptions
    
        @notes

          Be careful: this routine modifies self.functions to be 'point'!!
        """
        assert scaled_point.is_scaled

        if self.part.parttype == ATOMIC_PART_TYPE:
            return 1
        
        else: # CompoundPart or FlexPart
            emb_parts = self.part.embeddedParts(scaled_point)
            
            num_atomic_parts = 0                    
            for embpart in emb_parts:
                embpart_scaled_point = self._embPoint(embpart, scaled_point)
                num_atomic_parts += embpart.numAtomicParts(embpart_scaled_point)
                
            return num_atomic_parts

    def varsUsed(self, scaled_point):
        """Returns the vars in scaled_point which are used
        (Some vars are ineffective because of certain part choices)"""
        #Rule: For a var to be "used", it has to be used by at least one of its emb parts.
        #Exceptions:
        # 1. AtomicPart -- no emb parts, but all vars at self's level are used (subPartsInfo handles)
        # 2. FlexPart.choice_var_name -- affects choices below but nothing below sees it.
        

        #main work
        if self.part.parttype == ATOMIC_PART_TYPE:
            return scaled_point.keys()
        
        else: # CompoundPart or FlexPart
            vars_used = set([])
            all_vars = scaled_point.keys()
            
            for emb_part in self.part.embeddedParts(scaled_point):
                emb_scaled_point = self._embPoint(emb_part, scaled_point)

                #recurse
                emb_vars_used = emb_part.varsUsed(emb_scaled_point)

                #translate 'emb_vars_used' into the current level
                for cand_var in all_vars:
                    for (embpart_varname, f) in emb_part.functions.iteritems():
                        if (embpart_varname in emb_vars_used) and functionUsesVar(cand_var, f, all_vars):
                            vars_used.add(cand_var)
                            break

            if self.part.parttype == FLEX_PART_TYPE:
                vars_used.add(self.part.choice_var_name)
            
        return list(vars_used)

    def subPartsInfo(self, scaled_point, only_novel_subparts=False, at_top=True):
        """
        @description

          Returns info about each sub part, sub-sub-part, etc.
        
        @arguments

          scaled_point -- dict of varname : var_value
          only_novel_subparts -- bool -- if True, only return the _novel_ subparts (otherwise returns all)
        
        @return

          info_list -- list of (sub_EmbeddedPart, sub_scaled_point, toplevel_vars_used_by_sub_EmbeddedPart)
    
        @exceptions
    
        @notes

          Be careful: this routine modifies self.functions to be 'point'!!
        """
        #preconditions
        assert scaled_point.is_scaled
        
        #main work
        if self.part.parttype == ATOMIC_PART_TYPE:
            if at_top:
                info_list = [(None, None, set(scaled_point.keys()))]
            else:
                info_list = []
        
        else: # CompoundPart or FlexPart
            info_list = []
            flat_emb_parts = self.part.embeddedParts(scaled_point)

            for emb_part in flat_emb_parts:
                #prepare for recurse
                emb_scaled_point = self._embPoint(emb_part, scaled_point)
                
                #recurse
                emb_info_list = emb_part.subPartsInfo(emb_scaled_point, at_top=False)

                par_vars = self._varsUsedByEmbPart(emb_part, scaled_point)
                
                #modify recurse info such that toplevel vars are
                # for _this_ level rather than a sub-level.  Then
                # add to info_list.
                for (sub_emb_part, sub_scaled_point, sub_par_vars) in emb_info_list:
                    emb_vars_used_by_sub_emb_part = sub_par_vars
                    par_vars_used = self._varsUsedByEmbVarsOfEmbPart( \
                        emb_part, emb_vars_used_by_sub_emb_part, scaled_point)
                    if (not only_novel_subparts) or sub_emb_part._isNovel(sub_scaled_point):
                        tup = (sub_emb_part, sub_scaled_point, par_vars_used)
                        info_list.append(tup)
                
                #(maybe) build up at this level
                if (not only_novel_subparts) or emb_part._isNovel(emb_scaled_point):
                    tup = (emb_part, emb_scaled_point, par_vars)
                    info_list.append(tup)

        #postconditions
        all_toplevel_vars = self.part.point_meta.keys()
        for (dummy, dummy, some_toplevel_vars) in info_list:
            validateIsSubset(some_toplevel_vars, all_toplevel_vars,
                             'some_toplevel_vars', 'all_toplevel_vars')
            
        #return    
        return info_list

    def _isNovel(self, self_scaled_point):
        """Helper to subPartsInfo.  Returns True if 'self' is novel given this sub_scaled_point
        (needs to be _at_ the location of novelty, i.e. returns False if it's actually a child
        that has the novelty that this is passing on. emb_scaled_point is the point appropriate
        to self.part.point_meta)"""
        assert set(self_scaled_point.keys()) == set(self.part.point_meta.keys())
        
        if self.part.parttype != FLEX_PART_TYPE:
            return False

        varmeta = self.part.choiceVarMeta()
        val = self_scaled_point[varmeta.name]
        is_novel = varmeta.scaledValueIsNovel(val)
        return is_novel
            
        
    def mutatableFlexTuples(self, scaled_point):
        """
        @description

          Returns the embedded flex parts that that come from 'emb_part' when it is instantiated with
          scaled_point.  Also returns the toplevel varname that controls each of those flex parts.

          Ignore the non-mutatable flex parts, i.e. ones that are governed by non-unity var mappings to the top.

          Ignores other non-mutatable parts:
          -parts that are already novel (too mean)
          -wires, open circuits, etc

        @arguments

          scaled_point -- 
        
        @return

          tuples -- list of (sub-EmbeddedPart, toplevel var)
                
        @exceptions
    
        @notes

        """
        #preconditions
        assert scaled_point.is_scaled

        #main work
        tuples = []

        #Atomic Part - nothing to do
        if self.part.parttype == ATOMIC_PART_TYPE:
            pass
        
        else: # CompoundPart or FlexPart
            flat_emb_parts = self.part.embeddedParts(scaled_point)

            for emb_part in flat_emb_parts:
                #prepare for recurse
                emb_scaled_point = self._embPoint(emb_part, scaled_point)

                #ignore novel parts
                if emb_part._isNovel(emb_scaled_point):
                    continue

                #recurse
                emb_tuples = emb_part.mutatableFlexTuples(emb_scaled_point)

                #modify recurse info such that toplevel vars are
                # for _this_ level rather than a sub-level.  Then
                # add to tuples.
                for (sub_emb_part, sub_par_var) in emb_tuples:
                    #identify a par_var that has a 1:1 mapping
                    # (avoid the 1-choice_index mappings, and others)
                    par_var = None
                    for emb_part_var,emb_part_func in emb_part.functions.items():
                        if emb_part_func == sub_par_var: #is it 1:1 ?
                            par_var = emb_part_var

                    #keep the tuple if it has retained the 1:1 mapping
                    if par_var is not None:
                        tuples.append((sub_emb_part, par_var))
                
            #build up at this level
            if self.part.parttype == FLEX_PART_TYPE:
                self_emb_part = self
                tup = (self_emb_part, self.part.choice_var_name)
                tuples.append(tup)

        #filter away useless tuples, which include
        # all one-port parts (e.g. dcvs)
        # 'wire' in name
        # 'open' in name
        # 'short' in name
        #-also disallow (DsViAmp2_SingleEndedMiddle_VddGndPorts because that
        # was causing problems in returning chosenPart(it was highest level part;
        # will need to add checks for other highest level parts if they are used)
        tuples = [(emb_part, dummy) for (emb_part, dummy) in tuples
                  if \
                  ('WIRE' not in string.upper(emb_part.part.name)) and \
                  ('OPEN' not in string.upper(emb_part.part.name)) and \
                  ('SHORT' not in string.upper(emb_part.part.name)) and \
                  (string.upper('DsViAmp2_SingleEndedMiddle_VddGndPorts') \
                   not in string.upper(emb_part.part.name)) and \
                  (len(emb_part.part.portNames()) > 1)
                  ]
        
        #postconditions
        all_toplevel_vars = set(self.part.point_meta.keys())
        for (dummy, tup_toplevel_var) in tuples:
            assert tup_toplevel_var in all_toplevel_vars
            
        #return    
        return tuples

    def varsGoverningChildVar(self, cand_parent_vars, child_part_var,
                              complain_if_not_found):
        """Returns list of the vars found in self's functions
        that controls the value of 'child_part_var', a var in
        self.part.point_meta
        """        
        #preconditions
        # -ensure child_part_var in self's point_meta
        assert child_part_var in self.part.point_meta.keys()

        #main work: set 'par_var'
        self_func = self.functions[child_part_var]
        self_func = removeWhitespaceAndBrackets(self_func)
        
        #1:1 mapping -- easy to handle
        if self_func in cand_parent_vars:
            return [self_func]

        #no variables govern the value of 'child_part_var' because
        # it is merely a number
        elif self.varHasNumberFunc(child_part_var):
            return []
            
        #'1-child_part_var' mapping -- easy to handle    
        elif isInversionFunc(self_func):
            remap_var = varOfInversionFunc(self_func)
            assert remap_var in cand_parent_vars
            return [remap_var]

        elif isSimpleFunc(self_func):
            #handles: <,<=,>,>=,==,+,-,*,/
            remap_vars = varsOfSimpleFunc(self_func)
            for var in remap_vars:
                assert var in cand_parent_vars, (var, remap_vars, self_func)
            return remap_vars
            
        
        #can't do much with unknown functions / unexpected var names
        else:
            if complain_if_not_found:
                raise AssertionError('cannot extract vars from func: %s' % self_func)
            else:
                return []


    def novelSubPartsInfo(self, scaled_point, at_top=True):
        """
        @description

          Returns info about each _novel_ sub part, sub-sub-part, etc.
        
        @arguments

          scaled_point -- dict of varname : var_value
        
        @return

          info_list -- list of (sub_EmbeddedPart, sub_scaled_point, toplevel_vars_used_by_sub_EmbeddedPart),
            but unlike subPartsInfo, it only returns the _novel_ subparts.
    
        @exceptions
    
        @notes

          Be careful: this routine modifies self.functions to be 'point'!!
        """
        #preconditions
        assert scaled_point.is_scaled

        #main work
        if self.part.parttype == ATOMIC_PART_TYPE:
            info_list = []
        
        else: # CompoundPart or FlexPart
            info_list = []
            flat_emb_parts = self.part.embeddedParts(scaled_point)

            for emb_part in flat_emb_parts:
                #prepare for recurse
                emb_scaled_point = self._embPoint(emb_part, scaled_point)

                #recurse
                emb_info_list = emb_part.subPartsInfo(emb_scaled_point, at_top=False)

                par_vars = self._varsUsedByEmbPart(emb_part, scaled_point)
                
                #modify recurse info such that toplevel vars are
                # for _this_ level rather than a sub-level.  Then
                # add to info_list.
                for (sub_emb_part, sub_scaled_point, sub_par_vars) in emb_info_list:
                    emb_vars_used_by_sub_emb_part = sub_par_vars
                    par_vars_used = self._varsUsedByEmbVarsOfEmbPart( \
                        emb_part, emb_vars_used_by_sub_emb_part, scaled_point)
                    tup = (sub_emb_part, sub_scaled_point, par_vars_used)
                    info_list.append(tup)
                
                #build up at this level, ONLY if novel
                if self.part.parttype == FLEX_PART_TYPE:
                    scaled_val = emb_scaled_point[self.choice_var]
                    if self.choiceVarMeta().scaledValueIsNovel(scaled_val):
                        tup = (emb_part, scaled_point, par_vars)
                        info_list.append(tup)

        #postconditions
        all_toplevel_vars = self.part.point_meta.keys()
        for (dummy, dummy, some_toplevel_vars) in info_list:
            validateIsSubset(some_toplevel_vars, all_toplevel_vars,
                             'some_toplevel_vars', 'all_toplevel_vars')
            
        #return    
        return info_list

    def novelty(self, scaled_point):
        """Returns the novelty of this part.
        Note that we don't count every variable in the point, but instead _only_ the variables that
        actually have an effect (some will have no effect because of other vars, e.g. if we've selected
        a 1-stage amp then 2-stage vars have no effect.)"""
        assert scaled_point.is_scaled
        novelty = 0
        for var in self.varsUsed(scaled_point):
            var_meta = self.part.point_meta[var]
            if var_meta.isChoiceVar() and var_meta.scaledValueIsNovel(scaled_point[var]):
                novelty += 1
        return novelty

    def area(self, scaled_point, devices_setup=None):
        """Returns the area of this part and all sub-parts.  Includes M, C, R.  Units: m^2. """
        assert scaled_point.is_scaled

        #the toplevel embedded part has a reference to devices setup; see Problems.addArea()
        if devices_setup is None: 
            devices_setup = self.devices_setup
        
        if self.part.parttype == ATOMIC_PART_TYPE:
            if scaled_point.has_key('W') and scaled_point.has_key('L'):
                w, l = scaled_point['W'], scaled_point['L'] #units: m
                if scaled_point.has_key('M'): m = scaled_point['M'] #units: unitless
                else:                         m = 1
                return devices_setup.mosArea(w, l, m)
            elif scaled_point.has_key('C'):
                C = scaled_point['C'] #units: farads
                return devices_setup.capacitorArea(C)
            elif scaled_point.has_key('R'):
                R = scaled_point['R'] #units: ohms
                return devices_setup.resistorArea(R)
            else:
                return 0.0
        
        else: # CompoundPart or FlexPart
            emb_parts = self.part.embeddedParts(scaled_point)
            
            tot_area = 0.0           
            for embpart in emb_parts:
                embpart_scaled_point = self._embPoint(embpart, scaled_point)
                tot_area += embpart.area(embpart_scaled_point, devices_setup)
                
            return tot_area

    def areaLog10(self, scaled_point, devices_setup=None):
        return math.log10(self.area(scaled_point, devices_setup))

    def numFailingFunctionDOCs(self, scaled_point):
        """
        @description

          Returns the number of function DOCs that are failing at this scaled_point.
          Recurses.
        
        @arguments

          scaled_point -- Point object
        
        @return

          num_failing -- int.
    
        @exceptions
    
        @notes
        """
        #preconditions
        assert scaled_point.is_scaled

        #main work...
        num_failing = 0
        
        did_railbin = False #use this to determine if/when to do just-in-time railbinning
            
        #evaluate the function DOCs of this part
        for function_DOC in self.part.function_DOCs:
            if not did_railbin:
                self.part.point_meta.inPlaceRailbinScaled(scaled_point); did_railbin = True
            if not function_DOC.resultsAreFeasible(scaled_point, self.part):
                num_failing += 1
        
        #case: AtomicPart, so no recursion and nothing left to do
        if self.part.parttype == ATOMIC_PART_TYPE:
            pass

        # case: CompoundPart or FlexPart, so recurse
        else: 
            if not did_railbin:
                self.part.point_meta.inPlaceRailbinScaled(scaled_point); did_railbin = True
            emb_parts = self.part.embeddedParts(scaled_point)
                    
            for embpart in emb_parts:
                #compute scaled_point for sub-part.  Don't need to railbin because it
                # will recursively call this function, and this function does railbinning.
                embpart_scaled_point = self._embPoint(embpart, scaled_point, do_railbin=False)

                num_failing += embpart.numFailingFunctionDOCs(embpart_scaled_point)
                
        return num_failing

    def functionDOCsAreFeasible(self, scaled_point):
        """
        @description

          Returns True only if all of this part's AND its sub-parts' function DOCs have been met.
        
        @arguments

          scaled_point -- Point object
        
        @return

          feasible -- bool
    
        @exceptions
    
        @notes
        """
        #preconditions
        assert scaled_point.is_scaled

        #main work...
        did_railbin = False #use this to determine if/when to do just-in-time railbinning
            
        #evaluate the function DOCs of this part
        for function_DOC in self.part.function_DOCs:
            if not did_railbin:
                self.part.point_meta.inPlaceRailbinScaled(scaled_point); did_railbin = True
            if not function_DOC.resultsAreFeasible(scaled_point, self.part):
                return False
        
        #case: AtomicPart, so no recursion and nothing left to do
        if self.part.parttype == ATOMIC_PART_TYPE:
            return True

        # case: CompoundPart or FlexPart, so recurse
        else: 
            if not did_railbin:
                self.part.point_meta.inPlaceRailbinScaled(scaled_point); did_railbin = True
            emb_parts = self.part.embeddedParts(scaled_point)
                    
            for embpart in emb_parts:
                #compute scaled_point for sub-part.  Don't need to railbin because it
                # will recursively call this function, and this function does railbinning.
                embpart_scaled_point = self._embPoint(embpart, scaled_point, do_railbin=False)

                f = embpart.functionDOCsAreFeasible(embpart_scaled_point)
                if not f:
                    return False
                
            return True
                        
    def simulationDOCsCost(self, lis_results):
        """
        @description

          Returns the cost of simulation DOCs.  When cost == 0, all DOCs are met.
        
        @arguments

          lis_results -- dict of 'lis__device_name__measure_name' : lis_value -- used to compute DOCs higher up
        
        @return

          cost -- float in [0, Inf] but typically in [0,1]
    
        @exceptions
    
        @notes
        """
        #setup
        self.__class__._partnum = 0
        scaled_point = Point(True, self.functions)
        sim_DOCs = []

        #main call
        (success, sum_cost, num_seen) = self._simulationDOCsCost_helper(scaled_point, sim_DOCs, lis_results)
        
        #wrapup
        if not success:
            cost = BAD_METRIC_VALUE #arbitarily big number
        elif num_seen == 0:
            cost = 0.0
        else:
            cost = sum_cost / num_seen
            
        self.__class__._partnum = 0
        return cost
        
    def _simulationDOCsCost_helper(self, scaled_point0, sim_DOCs, lis_results):
        """
        @description

          Helper function for simulationDOCsCost().

        @arguments

          scaled_point0 -- Point object -- 
          sim_DOCs -- list of DOC objects -- for each building block that led to this part, include its DOCs.
            All its DOCs have to ultimately be calculatable with device-level measurements
          lis_results -- dict of 'lis__device_name__measure_name' : lis_value -- these values are compared
            with the DOC needs to see if all DOCs have been met
        
        @return

          success -- bool -- successfully found and computed?
          sum_cost -- float -- total cost so far (not normalized by num_seen)
          num_seen -- int -- number of DOCs that were encountered at this level and below.  
    
        @exceptions
    
        @notes
        """
        #case: bad lis_results
        if len(lis_results) == 0 and len(sim_DOCs) > 0:
            return (False, None, None)

        assert scaled_point0.is_scaled
        scaled_point = self.part.point_meta.railbin(scaled_point0)

        sum_cost, num_seen = 0.0, 0

        #case: AtomicPart, so we actually have to measure the sim_DOCs
        if self.part.parttype == ATOMIC_PART_TYPE:
            new_DOCs = sim_DOCs[:] + self.part.simulation_DOCs
            name_s = self._atomicPartInstanceName(scaled_point)
            
            #subcase: it's Atomic, but not a MOS (only measure MOSes right now)
            if not self.isMOS():
                #it's ok to pass when we have novelty, because we may
                # have been adding parts to higher-level MOSes
                pass
            
                #assert len(new_DOCs) == 0, \
                #      "shouldn't have sim_DOCs passed to part:\n '%s'" % self.part
                
            #subcase: it's a MOS, so test it
            else:
                for DOC_instance in new_DOCs:
                    sum_cost += DOC_instance.constraintViolation01(lis_results, name_s)
                    num_seen += 1

            self.__class__._partnum += 1
            return (True, sum_cost, num_seen)

        # case: CompoundPart or FlexPart, so recurse and add sim_DOCs
        else: 
            emb_parts = self.part.embeddedParts(scaled_point)
                    
            for embpart_i, embpart in enumerate(emb_parts):
                embpart_scaled_point = self._embPoint(embpart, scaled_point)

                new_DOCs = sim_DOCs[:] + self.part.simulation_DOCs
                (success, next_sum_cost, next_num_seen) = \
                          embpart._simulationDOCsCost_helper(embpart_scaled_point, new_DOCs, lis_results)
                if not success:
                    return (False, None, None)
                sum_cost += next_sum_cost
                num_seen += next_num_seen
                
            return (True, sum_cost, num_seen)

    def isMOS(self):
        """Returns True if self.part is an NMOS or PMOS atomic part"""
        return (self.part.parttype == ATOMIC_PART_TYPE) and ('MOS' in string.upper(self.part.name))
    
    def isPMOS(self):
        """Returns True if self.part is a PMOS atomic part"""
        return (self.part.parttype == ATOMIC_PART_TYPE) and ('PMOS' in string.upper(self.part.name))

    def spiceNetlistStr(self, annotate_bb_info=False, add_infostring=False, variation_data=None,
                        models_too=False):
        """
        @description

          Returns a SPICE-simulatable netlist of self, INCLUDING all
          the child blocks and their blocks etc.  This means that if
          we call this from the very top-level block, via recursion we generate
          a SPICE netlist for the whole circuit.
        
        @arguments

          annotate_bb_info -- bool -- annotate with building block information?
          add_infostring -- bool -- add a summaryStr of the part to the netlist (facilitates quick examination of topo.)
          variation_data -- (RndPoint, EnvPoint, DevicesSetup) or None -- 
            Contains how to do random variation.  Only use the None to keep some unit tests compact.                models_too -- bool -- return a second value, of models string
          
        @return

          spice_netlist_string -- string -- a netlist of 'self' and its sub-blocks
          (maybe) models_string -- string -- defines MOS models.  If nominal, has one for PMOS
            and one for NMOS.  If non-nominal, has one per transistor.
    
        @exceptions
    
        @notes

          We can get away with no needed arguments, rather than an input point,
          because 'self' is already an _instantiated_ part.  Either at this
          level or an ancestor's level, the parameters have been set as numbers.
        """
        #condition inputs
        if variation_data is None:
            (rnd_point, env_point, devices_setup) = (RndPoint([]), EnvPoint(is_scaled=True), DevicesSetup('UMC180'))
        else:
            (rnd_point, env_point, devices_setup) = variation_data
        num_rnd_values_before = len(rnd_point.values_list)
        
        #preconditions
        if AGGR_TEST:
            assert isinstance(rnd_point, RndPoint)
            assert isinstance(env_point, EnvPoint)
            #assert isinstance(devices_setup, DevicesSetup) #don't check, to avoid import dependencies...
            assert devices_setup is not None                   #...but at least have a low-dependency check
            for value in self.functions.itervalues():
                assert mathutil.isNumber(value)
        
        #initialize data
        scaled_point = Point(True, self.functions)
        
        if annotate_bb_info:
            bb_list = []
        else:
            bb_list = None

        #reset part and node names (need this for checking unique netlists)
        self.__class__._partnum = 0 
        NodeNameFactory._port_counter = 0L 

        #do actual work
        rnd_point_to_deplete = copy.deepcopy(rnd_point) #this will get diminished on-the-fly in the helper
        (netlist_list, models_list) = self.spiceNetlistStr_helper(
            scaled_point, self.connections, bb_list, rnd_point_to_deplete, env_point, devices_setup)
        netlist = "".join(netlist_list)

        #if
        models_string = "".join(models_list)
        doing_nominal = rnd_point_to_deplete.isNominal()
        if doing_nominal:
            assert not models_string
            models_string = devices_setup.nominalMosNetlistStr()

        #

        #reset (for safety)
        self.__class__._partnum = 0 
        NodeNameFactory._port_counter = 0L 
        
        if add_infostring:
            netlist = self.part.summaryStr(scaled_point) + netlist

        #postconditions
        num_rnd_values_after = len(rnd_point.values_list)
        assert num_rnd_values_before == num_rnd_values_after, "rnd point depletion accidentally happened during" \
               "call to spiceNetlistStr_helper; went from %d values to %d values" % \
               (num_rnd_values_before, num_rnd_values_after)

        #done
        if models_too:
            return (netlist, models_string)
        else:
            return netlist
        

    def spiceNetlistStr_helper(self, scaled_point0, subst_connections, bb_list,
                               rnd_point_to_deplete, env_point, devices_setup):
        """
        @description

          This is the worker function for spiceNetlistStr().
        
        @arguments
        
          scaled_point0 -- scaled Point == dict of self_varname : scaled_value_computed_from_above
            -- how to assign self's vars based on values from parent
          subst_connections -- dict of self_ext_portname : portname_from_above
            -- tells how to substitute connections going from above to this level
          bb_list -- list of (Part,point) -- list of building blocks that led to this part.
            Gets added to as it recursively dives.  If None, ignore; else make part of netlist.
          rnd_point_to_deplete -- RndPoint -- gets depleted as different sub-blocks use its values_list
          env_point -- scaled EnvPoint --
          devices_setup -- DevicesSetup --
        
        @return
        
          netlist_list -- list of string -- if self's Part is atomic, it returns the [spice string]
            else returns a list of spice strings for all sub-parts (recursively calculated)
          models_list -- list of string -- if nominal, it's empty.  If non-nominal, it
            has one MOS model per transistor
    
        @exceptions
    
        @notes          
        """
        #preconditions
        if AGGR_TEST:
            assert isinstance(scaled_point0, Point)
            assert scaled_point0.is_scaled
            assert isinstance(subst_connections, types.DictType)
            assert (bb_list is None) or isinstance(bb_list, types.ListType)
            assert isinstance(rnd_point_to_deplete, RndPoint)
            assert isinstance(env_point, EnvPoint)
            assert env_point.is_scaled
            #assert isinstance(devices_setup, DevicesSetup) #don't check, to avoid import dependencies...
            assert devices_setup is not None                   #...but at least have a low-dependency check
            if devices_setup.always_nominal:
                assert rnd_point_to_deplete.isNominal()
            if not rnd_point_to_deplete.isNominal():
                assert not devices_setup.always_nominal
                assert devices_setup.doRobust()

        #main work...
        scaled_point = self.part.point_meta.railbin(scaled_point0)

        # -this can be true even if devices_setup.doRobust() is True, because it may be nom rnd point
        doing_nominal = rnd_point_to_deplete.isNominal()
        
        if self.part.parttype == ATOMIC_PART_TYPE:
            #build spice_netlist_str = one line declaring the part composed of name, ports, modelname, params

            # -name_s
            name_s = self._atomicPartInstanceName(scaled_point)

            # -ports_s
            portnames = [subst_connections[port] for port in self.part.externalPortnames()]
            if self.part.name == 'dcvs': # -handle special case for dcvs which has only one port.
                portnames.append('0')
            ports_s = string.join(portnames)

            # -model_s
            model_s = ''
            if self.isMOS():
                #these must be in line with DevicesSetup's naming of models
                if doing_nominal:
                    if self.isPMOS(): model_s = 'Pnom'
                    else:             model_s = 'Nnom'
                else:
                    if self.isPMOS(): model_s = 'P%s' % name_s
                    else:             model_s = 'N%s' % name_s 

            # -var_s
            #   -if R or C, and doing non-nominal, then pop a rnd value and to vary the resistance or capacitance
            cp_scaled_point = copy.copy(scaled_point)
            if scaled_point.has_key('R') and (not doing_nominal):
                rnd_value = rnd_point_to_deplete.values_list.pop()
                cp_scaled_point['R'] = devices_setup.varyResistance(scaled_point['R'], rnd_value)
            elif scaled_point.has_key('C') and (not doing_nominal):
                rnd_value = rnd_point_to_deplete.values_list.pop()
                cp_scaled_point['C'] = devices_setup.varyCapacitance(scaled_point['C'], rnd_value)

            if self.part.name == 'wire':
                vars_s = 'R=0'
            else:
                vars_s  = self.part.point_meta.spiceNetlistStr(cp_scaled_point)

            # -put it together
            spice_netlist_str = name_s + ' ' + ports_s + ' ' + model_s + ' ' + vars_s + '\n'
            if bb_list is not None: 
                bb_list.append((self.part, scaled_point))
                spice_netlist_str = self._annotatedAtomicPartStr(bb_list) + spice_netlist_str

            #build models_string
            # -only MOS+robust needs device-specific model definitions
            if self.isMOS() and (not doing_nominal):
                rnd_value = rnd_point_to_deplete.values_list.pop()
                models_string = devices_setup.mosNetlistStr(
                    self.isPMOS(), name_s, cp_scaled_point['W'], cp_scaled_point['L'], rnd_value)
            else:
                models_string = ''

            #wrapup
            self.__class__._partnum += 1
            return ([spice_netlist_str], [models_string])
        
        else: # CompoundPart or FlexPart
            emb_parts = self.part.embeddedParts(scaled_point)
            internal_nodenames = self.part.internalNodenames()
            netlist_list, models_list = [], []
            global_internal_nodenames = {} #local_nodename : global_nodename
            for nodename in internal_nodenames:
                global_internal_nodenames[nodename] = NodeNameFactory().build()
                    
            for embpart_i, embpart in enumerate(emb_parts):     
                #substitute values into funcs to make sub-point              
                embpart_scaled_point = self._embPoint(embpart, scaled_point)

                #substitute ports from parent ports and this' internal ports
                embpart_subst_connections = {}
                for embpart_portname, parent_portname in embpart.connections.items():
                    if subst_connections.has_key(parent_portname):
                        subst_portname = subst_connections[parent_portname]
                    else: 
                        subst_portname = global_internal_nodenames[parent_portname]
                    embpart_subst_connections[embpart_portname] = subst_portname

                if bb_list is None:
                    new_bb_list = None
                else:
                    new_bb_list = bb_list[:] + [(self.part, scaled_point)]
                    
                (embpart_netlist_list, embpart_models_list) = embpart.spiceNetlistStr_helper(
                    scaled_point0=embpart_scaled_point, subst_connections=embpart_subst_connections, bb_list=new_bb_list,
                    rnd_point_to_deplete=rnd_point_to_deplete, env_point=env_point, devices_setup=devices_setup)

                netlist_list += embpart_netlist_list
                models_list += embpart_models_list
                
            return (netlist_list, models_list)

    def _embPoint(self, embpart, scaled_point, do_railbin=True):
        """
        @description

          Substitute scaled_point's values into embpart.functions to create sub-point for this embpart.

        @arguments

          embpart -- embeddedPart -- from self's computed embedded_parts
          scaled_point -- Point -- holds the values which embpart will use to do the computing
          do_railbin -- bool -- rail & bin before returning?  (Should always be True unless we know for sure
            that the caller will be railing this)

        @return

          embpart_scaled_point -- Point object --

        @exceptions

        @notes
        
          Helper for spice netlisting, and elsewhere.    
        """
        scaled_d = {}
        for embpart_varname, f in embpart.functions.items():
            try:
                v = evalFunction(scaled_point, f, self.part)
            except:
                s = "The call to evalFunction() broke.  Details:\n\n"
                s += "-We were trying to compute the value for " \
                     "embedded_part_varname of '%s'\n\n" % embpart_varname
                s += "-The function f is: '%s'\n\n\n" % f
                s += "-The part name is: %s\n" % self.part.name
                s += "-The (scaled) input point which broke it was: %s\n\n" % \
                     scaled_point.nicestr()
                print s
                import pdb; pdb.set_trace()
                v = evalFunction(scaled_point, f, self.part)
                raise ValueError(s)

            #turn off for speed
            #assert mathutil.isNumber(v), (v, scaled_point, f)
                
            scaled_d[embpart_varname] = v
        scaled_p = Point(True, scaled_d)
        if do_railbin:
            embpart.part.point_meta.inPlaceRailbinScaled(scaled_p)
        return scaled_p

    def _varsUsedByEmbPart(self, embpart, scaled_point):
        """
        @description

          Returns the list of var names in 'scaled_point' that embpart depends on for its computations.
            embpart is a sub-part of 'self'.

          NOT recursive; i.e. only operates at this level.

        @arguments

          embpart -- embeddedPart -- from self's computed embedded_parts
          scaled_point -- Point -- holds the values which embpart will use to do the computing

        @return

          vars_used -- set of string -- subset of variable names found in 'scaled_point'

        @exceptions

        @notes
           
        """
        #preconditions
        if AGGR_TEST:
            validateVarLists(
                embpart.part.point_meta.keys(), embpart.functions.keys(),
                'vars that pm expects filled','vars that functions fill')
            
        #exploit caching if possible
        if not hasattr(embpart, '_vars_used'):
            # (choice_var, choice_value)'s key : set_of_vars
            embpart._vars_used = {} 

        #can be uniquely identified by (sorted) choice var values
        choices_key = str([scaled_point[var_name]
                           for var_name in self.part.sortedChoiceVarnames()])
        
        if embpart._vars_used.has_key(choices_key):
            return embpart._vars_used[choices_key]
            
        #main work
        vars_used = set([])
        all_varnames = scaled_point.keys()
        for cand_var in scaled_point.iterkeys():
            for embpart_varname, f in embpart.functions.iteritems():
                if functionUsesVar(cand_var, f, all_varnames):
                    vars_used.add(cand_var)
                    break

        #set cache
        embpart._vars_used[choices_key] = vars_used

        #postconditions
        validateIsSubset(vars_used, scaled_point.keys(), 'vars used', 'scaled_point.keys()')
        
        return vars_used

    def _varsUsedByEmbVarsOfEmbPart(self, embpart, embvars, scaled_point):
        """Like _varsUsedByEmbPart, except only considers embvars = a subset
        of the vars of embpart (as opposed to all the vars)"""
        
        #preconditions
        validateIsSubset(embvars, embpart.functions.keys(), 'embvars', 'embpart.functions.keys()')
            
        #exploit caching if possible
        if not hasattr(embpart, '_vars_used_ee'):
            # [embvars_key][(choice_var, choice_value)'s key] : set_of_vars
            embpart._vars_used_ee = {}
            
        embvars_key = str(sorted(embvars))
        if not embpart._vars_used_ee.has_key(embvars_key):
            embpart._vars_used_ee[embvars_key] = {}
        cache = embpart._vars_used_ee[embvars_key]

        #can be uniquely identified by (sorted) choice var values
        choices_key = str([scaled_point[var_name]
                           for var_name in self.part.sortedChoiceVarnames()])

        if cache.has_key(choices_key):
            return cache[choices_key]
            
        #main work
        vars_used = set([])
        all_varnames = scaled_point.keys()
        for cand_var in scaled_point.iterkeys():
            for embpart_varname, f in embpart.functions.iteritems():
                if embpart_varname not in embvars: continue
                if functionUsesVar(cand_var, f, all_varnames):
                    vars_used.add(cand_var)
                    break

        #set cache
        cache[choices_key] = vars_used

        #postconditions
        validateIsSubset(vars_used, scaled_point.keys(),'vars used', 'scaled_point.keys()')
                
        return vars_used

    def _atomicPartInstanceName(self, scaled_point):
        """Build the name string of an Atomic Part, for use
        in SPICE netlisting and maybe elsewhere"""
        assert self.part.parttype == ATOMIC_PART_TYPE
        
        name_s = self.part.spice_symbol
        
        #  -special case: make it easy to identify wires
        if self.part.spice_symbol=='R' and (len(scaled_point)==0 or scaled_point['R']==0.0):
            name_s += 'wire'
            
        name_s  += str(self.__class__._partnum)
        
        return name_s

            
    def _annotatedAtomicPartStr(self, bb_list):
        """
        @description

          Helper function for netlisting with bb annotations.

          Returns a string describing the info to annotate prior
          to the actual instantiating line of an atomic part.

          Note: this doesn't just give the leaf-level part in annotated form,
          but instead it gives _all_ the parts that lead to the leaf-level part
        """
        bb_s = '\n*--------------------------------------------\n'
        for level, (list_part, list_point) in enumerate(bb_list):
            bb_s += '* ' + ' '*level*2 + list_part.name
            bb_s += ' (ID=' + str(list_part.ID) + ')'
            bb_s += ': '
            bb_s += self._bbPointStr(list_point)
            bb_s +='\n'
        return bb_s
            
    def _bbPointStr(self, list_point):
        """
        @description

          Helper function for _annotatedAtomicPartStr()

          Returns a string describing a bb list_point that can fit onto a line.
          Tries to focus on the vars with topology info, and if there
          is still space, if adds other vars.
    
        """
        #set magic numbers
        max_num_vars = 700000 #magic number alert
        subvars_to_avoid = ['W','L','R','Vbias','K','C','Ids','GM','DC_V',
                            'Ibias', 'Ibias2']

        #intialize
        vars_covered = []
        s = ''

        #first priority: include a choice var if it's there
        for choice_var in Part.all_choice_var_names:
            if choice_var in list_point.keys():
                var, val = choice_var, list_point[choice_var]
                s += '%s=%g, ' % (var, val)
                vars_covered.append(var)

        #next priority: get all vars except ones to 'avoid', up to max_num_vars
        for var, val in list_point.items():
            if len(vars_covered) >= max_num_vars: break
            if var in vars_covered: continue
            if self._doAvoidVar(var, subvars_to_avoid):
                continue
            s += '%s=%g, ' % (var, val)
            vars_covered.append(var)

        #if still space, fill up some vars that were avoided
        for var, val in list_point.items():
            if len(vars_covered) >= max_num_vars: break
            if var in vars_covered: continue
            s += '%s=%g, ' % (var, val)
            vars_covered.append(var)
            
        return s

    def _doAvoidVar(self, var, subvars_to_avoid):
        """
        @description

          Returns True if the tail characters of 'var' line up with any of the subvars to avoid.

        """
        for subvar in subvars_to_avoid:
            if var[-len(subvar):] == subvar:
                return True
        return False
                    
    def __str__(self):
        """
        @description

          Override str()
    
        @notes          
        """
        return self.str2(tabdepth=0, full_info=True)

    def str2(self, tabdepth, full_info):
        s = ''
        s += 'EmbeddedPart={'
        s += " partname='%s'" % self.part.name
        s += '; partID=%s' % self.part.ID
        if full_info:
            s += '; connections=%s' % self.connections

        if full_info:
            show_functions = self.functions
        else:
            show_functions = {}
            for varname in self.part.point_meta.choiceVars():
                show_functions[varname] = self.functions[varname]
                
        s += '; functions=%s' % str(show_functions) #niceFunctionsStr()
        s += ' /EmbeddedPart}'

        return s    


def validateFunctions(functions, scaled_point):
    """
    @description

      Validate functions by substituting scaled_point into
      the function, and ensuring that it can be then evaluated into a
      numeric value.

      Useful for making sure that a Library is defined well.

    @arguments

      functions --
        -- dict of varname : str_func_of_func_varnames
        -- stores how to compute sub-part_to_add's vars from self.
      scaled_point -- Point object -- holds a number for each func_varname

    @return

    @exceptions

    @notes          
    """
    if not scaled_point.is_scaled:
        raise ValueError
    
    if not AGGR_TEST: return
    
    for embpart_varname, f in functions.iteritems():
        #anything that calls 'part.' gets a free pass in this validation
        if isinstance(f, types.StringType) and f[:5] == 'part.':
            continue

        #but we test the rest!
        try:
            embpart_varval = evalFunction(scaled_point, f)
        except:
            s = "func to compute '%s' (='%s') is bad\n" % (embpart_varname,f)
            s += "  scaled_point=%s\n  f=%s\n" % (scaled_point, f)
            s += "  can only use vars: %s" % scaled_point.keys()
            raise ValueError(s)


#this list includes: numbers, lowercase alphabet, uppercase alphabet, '_', '.', ':'
_alphanumeric = '1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM_.:'

def evalFunction(point, func_str, part=None):
    """
    @description

      Substitutes all values of point into func_str, then return eval(func_str).
      E.g: if point is {'a':1.0, 'b':2.0} and func_str is 'b*2.0', returns 4.0.
      E.g: if func_str is merely a number, returns that number (nothing to subst)

    @arguments

      point -- Point object --
      func_str -- string --
      part -- Part object -- makes 'part' visible to eval,
        such that we can call part.REQUESTS,
        e.g. for part.approx_mos_models.estimateNmosWidth(Ids, Vs, Vd, Vbs, L)

    @return

    @exceptions

      If func_str is '', then merely return the string '' rather than a number.
      
    @notes          
    """
    if func_str == '': return ''
    if mathutil.isNumber(func_str):
        return func_str
    
    #early exit?
    for point_var in point.iterkeys():
        if point_var == func_str:
            return point[point_var]
        
    f = copy.copy(func_str)
    
    try:
        #Algorithm:
        #while not out of letters:
        #  identify start of cand word
        #  identify end of cand word
        #  calc word
        #  if word is in point.keys(), replace it (otherwise assume it's a func)
        #  repeat sequence, starting from end of cand word + 1

        #while not out of letters:
        len_f = len(f)
        st = 0
        while True:
            
            #  identify start of cand word
            letters_left = True
            while True:
                if f[st] in _alphanumeric: break
                st += 1
                if st >= len_f:
                    letters_left = False
                    break
            if not letters_left: break
            
            #  identify end of cand word
            fin = st+1
            while True:
                if fin >= len_f:
                    break
                if f[fin] not in _alphanumeric: break
                fin += 1
                
            #  calc word
            word = f[st:fin]

            # if word is in point.keys(), replace it
            # (otherwise assume it's a func)
            new_st = fin + 1
            for cand_var, cand_val in point.items():
                if cand_var == word:
                    f = f[:st] + str(cand_val) + f[fin:]
                    old_len = (fin-st)
                    new_len = len(str(cand_val))
                    new_st = fin + 1 - old_len + new_len
                    len_f = len(f)
                    break
                
            #  repeat sequence, starting from end of cand word + 1
            st = new_st
            if st >= len_f:
                break
                
        # The remaining string should be evaluatable using
        # the recursiveEval function

        try:
            return recursiveEval(point, f, part)
        except:
            raise ValueError
    
    except:
        s = "Encountered an error in evalFunction()\n"
        s + "orig func_str = %s\n" % func_str
        s += "point = %s\n" % point
        s += "func_str with values subst. = %s" % f
        raise ValueError(s)

def recursiveEval(point, func_str, part=None):
    """
    @description

      Evaluates a function string taking into account the fact that  there can be user defined function
      names in there. These cause the need to define an evaluation order since not everything can
      be evaluated as one big string.
      
      The sub-evaluations can be defined by placing them between <$ $> brackets.

    @arguments

      point -- Point object --
      func_str -- string --
      part -- Part object -- makes 'part' visible to eval, such that we can call part.REQUESTS,

    @return
        
      the evaluated value
      
    @exceptions
      
    @notes          
    """
    # if it evaluates just fine, why spend the effort?
    try:
        retval = eval(str(func_str))
        return retval
    except:
        # bad luck... it didn't        
        # so now find all occurrences of matching <$ ... $> pairs
        # and call recursiveEval on the contained string
      
        scan_str = copy.copy(func_str)
        new_str = ""
        while len(scan_str):
            first_bracket_pos = scan_str.find('<$')
            
            if first_bracket_pos != -1:
                # opening bracket found
                assert scan_str[first_bracket_pos] == '<', "No bracket at first bracket position"
                assert scan_str[first_bracket_pos+1] == '$', "No bracket at first bracket position"
                
                # find the matching bracket
                idx = first_bracket_pos + 2
                open_brackets = 1
                while idx < len(scan_str) - 1 and open_brackets > 0:
                    curr = scan_str[idx]
                    next = scan_str[idx+1]
                    
                    if curr == '<' and next == "$":
                        open_brackets += 1
                    if curr == '$' and next == ">":
                        open_brackets -= 1
                    
                    idx += 1
                
                matching_bracket_pos = idx-1
                
                assert matching_bracket_pos < len(scan_str),  "No matching bracket found"
                assert scan_str[matching_bracket_pos] == '$', "No bracket at matching bracket position"
                assert scan_str[matching_bracket_pos+1] == '>', "No bracket at matching bracket position"
                
                # get what's inbetween the brackets
                eval_string = scan_str[first_bracket_pos+2:matching_bracket_pos]
                
                # don't forget the section before the evalstring
                new_str += scan_str[:first_bracket_pos]
                
                # recursively evaluate what's inbetween the brackets
                if eval_string:
                    new_str += str(recursiveEval(point, eval_string, part))
                
                # the remainder is still to be scanned for matched brackets
                scan_str = scan_str[matching_bracket_pos+2:]
            else:
                # there are no more matched brackets to eval, so add
                # the remainder of the original string to the one to eval
                new_str += scan_str
                scan_str = ""
                
        # There should not be anything left in the string that
        # is not evaluatable.
        return eval(new_str)
               
def functionUsesVar(compare_var, func_str, all_vars):
    """
    @description

      Returns True if 'func_str' uses 'compare_var', using similar functionality to that of evalFunction().
      
    @arguments

      compare_var -- string --
      func_str -- string -- the function
      all_vars -- list of string -- all varnames available that the function might have

    @return

    @exceptions
      
    @notes          
    """
    if AGGR_TEST:
        assert compare_var
        assert func_str
        assert all_vars
        assert compare_var in all_vars
    
    if func_str == compare_var: return True #early exit?
    if compare_var not in func_str: return False #early exit?
    if mathutil.isNumber(func_str): return False

    f = copy.copy(func_str)

    try:
        #Algorithm:
        #while not out of letters:
        #  identify start of cand word
        #  identify end of cand word
        #  calc word
        #  if word is in point.keys(), replace it (otherwise assume it's a func)
        #  repeat sequence, starting from end of cand word + 1

        #while not out of letters:
        len_f = len(f)
        st = 0
        while True:
            
            #  identify start of cand word
            letters_left = True
            while True:
                if f[st] in _alphanumeric: break
                st += 1
                if st >= len_f:
                    letters_left = False
                    break
            if not letters_left: break
            
            #  identify end of cand word
            fin = st+1
            while True:
                if fin >= len_f:
                    break
                if f[fin] not in _alphanumeric: break
                fin += 1
                
            #  calc word
            word = f[st:fin]

            # found it?
            if word == compare_var:
                return True

            # if word is in all_vars, replace it (otherwise assume it's a func)
            new_st = fin + 1
            for cand_var in all_vars:
                if cand_var == word:
                    f = f[:st] + 'XXXXX'+ f[fin:]
                    old_len = (fin-st)
                    new_len = 5 #5 X's
                    new_st = fin + 1 - old_len + new_len
                    len_f = len(f)
                    break
                
            #  repeat sequence, starting from end of cand word + 1
            st = new_st
            if st >= len_f:
                break

        return False
    
    except:
        s = "Encountered an error in functionUsesVar()\n"
        s + "orig func_str = %s\n" % func_str
        s += "func_str with values subst. = %s" % f
        raise ValueError(s)
   
def flattenedTupleList(tup_list):
    """
    @description

      Flatten the input list of tuples by returning a non-tuple list that has all the entries of the
      left-item-of-tuple, followed by all the entries of the right-item-of-tuple.

    @arguments

      tup_list -- shaped like [(a1,b1), (a2,b2), ...]

    @return

      flattened_list -- shaped like [a1,a2,...,b1,b2,...]

    @exceptions

    @notes          
    """ 
    return [a for (a,b) in tup_list] + [b for (a,b) in tup_list]

class NodeNameFactory:
    """
    @description

      Builds unique node names.  This is helpful for internally generated
      node names.
      
    @attributes
      
    @notes
    """

    _port_counter = 0L
    def __init__(self):
        """
        @description
        
        """ 
        pass

    def build(self):
        """
        @description

          Return a unique node name.

        @arguments

          <<none>>

        @return

          new_node_name -- string -- 

        @exceptions

        @notes          
        """ 
        self.__class__._port_counter += 1
        return 'n_auto_' + str(self.__class__._port_counter)
    

def replaceAutoNodesWithXXX(netlist_in):
    """
    @description

      Replaces all nodes having name 'n_auto_NUM' with 'XXX'.
      Makes unit-testing easier.

    @arguments

      netlist_in -- string -- a SPICE netlist

    @return

      modified_netlist_in -- string -- 

    @exceptions

    @notes

      Nodes with a name like 'n_auto_NUM' come from NodeNameFactory
    """ 
    netlist = copy.copy(netlist_in)
    while True:
        Istart = string.find(netlist, 'n_auto')
        if Istart == -1: break
        Iend = min(string.find(netlist, ' ', Istart),
                   string.find(netlist, '\n', Istart))
        netlist = netlist[:Istart] + 'XXX' + netlist[Iend:]
    return netlist
    
def tabString(tabdepth):
    """
    @description

      Returns a string with 2 x tabdepth '.' symbols.  Used by str2().

    @arguments

      tabdepth -- int --

    @return

      tab_string -- string

    @exceptions

    @notes          
    """ 
    return '.' * tabdepth * 2

ALPH_SET_EXCEPT_E = set('abcdfghijklmnopqrstuvwxyzABCDFGHIJKLMNOPQRSTUVWXYZ')
    
def isNumberFunc(func):
    """Returns True if this function is of the form:
    '3.2' or '5.8 - 6.2', etc.
    That is, returns True if the function can
    be evaluated to a number.
    """
    #corner case
    if func is None:
        return False

    #corner case
    if AGGR_TEST:
        assert isinstance(func, types.StringType), "should never have a raw number or other here: '%s'" % func

    #main case...
    # -using eval() etc is slow, so the strategy here is to
    # remove all characters that a number might use
    # -have to catch variables like e, ee, e12, ee12, etc.
    s = func

    #remove whitespace and surrounding brackets
    s = removeWhitespaceAndBrackets(s)
                
    # -remove first sign
    if s and (s[0] == '-' or s[0] == '+'):
        s = s[1:]

    #early return: first entry isn't a number
    if s and s[0] == 'e':
        return False

    #early return: it's a number
    s = s.rstrip(' 0123456789.+-e')
    if not s:
        return True

    #early return: has non-number characters (ignore test for e)
    if set(s).intersection(ALPH_SET_EXCEPT_E):
        return False

    #main work: use eval()
    try:
        return mathutil.isNumber(eval(func))
    except NameError:
        return False
    except:
        print "want to catch this error explicitly; if we can't then just return False"
        import pdb; pdb.set_trace()

def isInversionFunc(func):
    """Returns True if this function is of the form:
    '(1-VAR)' or '1-VAR'.  Whitespace is ignored.
    """
    if func is None:
        return False
    elif isNumberFunc(func):
        return False
    v = varOfInversionFunc(func)
    return (v is not None)

def varOfInversionFunc(func):
    """If 'func' is of the form:
    '(1-VAR)' or '1-VAR', then it returns the name of VAR
    If it cannot find that form, then it returns None.
    Whitespace is ignored.
    """
    func = removeWhitespaceAndBrackets(func)

    #ensure whitespace is around the first '-'
    Istart = string.find(func, '-')
    if Istart == -1: return None
    func = func[:Istart] + ' - ' + func[Istart+1:]

    #split into 3 parts and analyze
    s = string.split(func)
    if len(s) != 3: return None
    if s[0] != '1': return None
    if s[1] != '-': return None
            
    return s[2]

def removeWhitespaceAndBrackets(func):
    """
    -Removes all whitespace from 'func': leading edge, internal, trailing edge
    -Removes >0 outer groups of brackets, e.g. (x), ((x)), etc => x
    """
    #remove whitespace
    import re
    func = re.sub("\s+", "", func)

    #remove brackets
    while func and func[0]=='(':
        assert func[-1] == ')'
        func = func[1:-1]
    return func
        

def isSimpleEqualityFunc(func):
    """
    Returns True if this function is of the form 'VAR1 == VAR2' where VAR1 and VAR2 are any string.
    Whitespace is ignored.
    """
    if mathutil.isNumber(func):
        return False
    v1, v2 = varsOfSimpleEqualityFunc(func)
    return (v1 is not None)

def varsOfSimpleEqualityFunc(func):
    """Assuming that 'func' is a simpleEquality func of the form '(VAR1 == VAR2)' or 'VAR1 == VAR2',
    then this returns the var names for VAR1 and VAR2. If it cannot find that form, then it returns (None, None)
    """
    #remove whitespace
    func = removeWhitespaceAndBrackets(func)

    #ensure whitespace is around the first '=='
    Istart = string.find(func, '==')
    if Istart == -1: return (None,None)
    func = func[:Istart] + ' == ' + func[Istart+2:]

    #split into 3 parts and analyze
    s = string.split(func)
            
    if len(s) != 3: return (None, None)
    
    return (s[0], s[2])

def isSimpleFunc(func):
    """Returns True if this is a simple func (see varsOfSimpleFunc for details)""" 
    return varsOfSimpleFunc(func) is not None

def varsOfSimpleFunc(func):
    """In a function of the form 'var1 OP var2' or '(var1 OP var2)',
    where OP is <,<=,>,>=,==,+,-,*,/, it returns a list of [var1, var2] or [var1] if both vars are the same.
    If the function is not that form, it returns None.
    """    
    op = None
    for cand_op in ["<=","<",">=",">","==","+","-","*","/"]:
        if cand_op in func:
            op = cand_op
            break
    if op is None:
        return None

    #remove whitespace
    func = removeWhitespaceAndBrackets(func)

    #ensure whitespace is around the first op
    Istart = string.find(func, op)
    if Istart == -1:
        return None
    func = func[:Istart] + ' ' + op + ' ' + func[Istart+len(op):]

    #split into 3 parts and analyze
    s = string.split(func)
            
    if len(s) != 3:
        return None

    ret_vars = [s[0], s[2]]
    ret_vars = list(set(ret_vars)) #uniquify
    return ret_vars

def niceFunctionsStr(functions):
    max_num_show = 17 #magic number alert
                
    ll = []
    ll.append('functions={\n')
    for varname, f in functions.items()[:max_num_show]:
        ll.append("  '%s' = '%s'\n" % (varname, f))
    if len(functions) > max_num_show:
        ll.append('  ...\n')
    ll.append('  /functions}')
    return ''.join(ll)
            
def novelPartName(base_name):
    return PartNameFactory().create(base_name)

class PartNameFactory:
    """
    @description

      Auto-generates a Part name, with the guarantee that each auto-generated name is unique compared
      to every other auto-generated name (but does NOT compare to manually generated names)
      
    @attributes
      
    @notes
    """
    
    _name_counter = 0L
    def __init__(self):
        """
        @description
          
        """ 
        pass

    def create(self, base_name):
        """
        @description

          Returns an auto-created var name.
        
        @arguments

          <<none>>
        
        @return

          new_var_name -- string
    
        @exceptions
    
        @notes
          
        """ 
        self.__class__._name_counter += 1
        return 'mutated' + str(self.__class__._name_counter) + '_' + base_name
    
