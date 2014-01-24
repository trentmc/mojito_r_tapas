"""Library.py

"""
import types
import copy
import math
import sys
import string

import numpy

from util import mathutil

from adts import *
from util.constants import *

import logging
log = logging.getLogger('library')

def whoami():
    """
    @description

      Returns the name of the function calling 'whoami'.

    @arguments

      <<none>>

    @return

      name_of_function_calling_whoami -- string -- 

    @exceptions

    @notes
    
      This is useful for naming each part according to the function that
      defines it, instead of having to specify its name as a string
      within the code itself (which is error-prone).  
    """ 
    return sys._getframe(1).f_code.co_name

class Library:
    """
    @description

      A 'Library' is an abstract class with worker routines to help
      its children such as SizesLibrary and OpLibrary.
      
    @attributes

      _ref_varmetas -- dict of generic_var_name : varmeta.  
      _parts -- dict of part_name : Part object
      
    @notes
    
      Generic var names in ref_varmetas are: W, L, K, R, C, GM, DC_I, Ibias,
        DC_V, discrete_Vbias, cont_Vbias
    """

    def __init__(self):
        """
        @description

          Constructor.
        
        @arguments
        
        @return

          new_library -- Library object
    
        @exceptions
    
        @notes

          This constructor method doesn't bother building each possible Part
          right now; rather, it defers that to a 'just in time' basis
          for the first request for a given part.  Once it builds the
          requested part, it _does_ store it in self._parts, such that
          subsequent calls for the same part do not need to rebuild the part.
        """
        self._ref_varmetas = {}
        self._parts = {}

    #====================================================================
    #The following routines simplify the construction of VarMeta and
    # PointMeta objects in Parts
    def buildVarMeta(self, ref_varname, new_varname=None):
        """
        @description

          Returns a COPY of self._ref_varmetas[ref_varname], so that
          CompoundPart does not refer to the same varmeta and thus
          the newly created VarMeta can be changed without disturbing the old.
          
          If there is more than one of these in a CompoundPart, then
          so that each varMeta has unique names, specify 'new_varname'
          (otherwise new_varname will == ref_varname)

        @arguments

          ref_varname -- string -- indicates what varmeta we wish to copy
          new_varname -- string -- see description.

        @return

          new_varmeta -- VarMeta object -- copy of a VarMeta in self, and
            possibly with a different varname.

        @exceptions

        @notes          
        """
        new_varmeta = copy.deepcopy(self._ref_varmetas[ref_varname])
        if new_varname is not None:
            new_varmeta.name = new_varname
        return new_varmeta

    def buildPointMeta(self, varname_list_or_dict):
        """
        @description

          Simple way to construct PointMeta objects, by leveraging
          self's reference varmetas.
          
            If the input is a list:
              Returns a PointMeta with a VarMeta named for each entry
              Example: buildPointMeta(['W','L']) 
             
            Else the input is a dict:
              Returns a PointMeta where each VarMeta's new_varname is the key
                and ref_varname is the corresponding value
              Example: buildPointMeta({'DC':'DC_I', 'W':'W'})
            
        @arguments

          varname_list_or_dict -- either:
            (a) dict of new_varname : ref_varname, or
            (b) list of new_varname strings

        @return

          new_point_meta -- PointMeta object

        @exceptions

        @notes

          Uses self.buildVarMeta() as a helper function.
        """
        
        #list
        if isinstance(varname_list_or_dict, types.ListType):
            varname_list = varname_list_or_dict
            assert mathutil.allEntriesAreUnique(varname_list)
            return PointMeta([self.buildVarMeta(varname)
                              for varname in varname_list])

        #dict
        else:
            varname_dict = varname_list_or_dict
            assert mathutil.allEntriesAreUnique(varname_dict.keys())
            return PointMeta([self.buildVarMeta(ref_name, new_name)
                              for new_name, ref_name in varname_dict.items()])

    def updatePointMeta(self, base_point_meta, extra_part, varmeta_map,
                        only_add_nonoverlapping_vars=False):
        """
        @description

          Adds extra_part.point_meta's varmetas to base_point_meta, renamed
          via varmeta_map.  Makes it simple to construct PointMetas for some parts.

          Specifically, does:
          1. pm = copy of base_point_meta
          2. add each VarMeta in extra_part.point_meta, with new name
             from varmeta_map as:
             VarMeta.newname = varmeta_map[VarMeta.oldname]
             (ie varmeta_map is old_varname : new_varname)
          3. returns pm

          This is another way to simplify construction of PointMetas.

        @arguments

          base_point_meta -- PointMeta object --
          extra_part -- Part object -- use this part's point_meta attribute
          varmeta_map -- dict of old_name_string : new_name_string -- how
            to change the variable names in extra_part.point_meta
          only_add_nonoverlapping_vars -- bool --
            If False: if we are about to overlap a var, it will raise an error
            If True: before adding a var, checks for overlap, and only adds
              if no overlap
            (This refers to base_point_meta's vars, i.e. old_name_string)

        @return

          updated_point_meta -- PointMeta object -- like base_point_meta but
            has an extra VarMeta for each var in extra_part.point_meta

        @exceptions

        @notes

          Can map to a newname of 'IGNORE' which is there to state
          that the old_varname is accounted for, but that the new var's
          varmeta isn't wanted.  This guarantees that the user is acknowledging
          the existence of the oldvar (helps with debugging).
        """
        extra_point_meta = extra_part.point_meta
        if sorted(extra_point_meta.keys()) != sorted(varmeta_map.keys()):
            pnames, vnames = extra_point_meta.keys(), varmeta_map.keys()
            extra_vnames = sorted(mathutil.listDiff(vnames, pnames))
            extra_pnames = sorted(mathutil.listDiff(pnames, vnames))
            name = extra_part.name
            s = "Var names in 'varmeta_map.keys()' and '%s.point_meta' do " \
                "not align; \n\nextra names in varmeta_map " \
                " (or missing names in %s.point_meta) = %s" \
                "\n\nextra names in %s.point_meta " \
                " (or missing names in 'functions') = %s\n" \
                "\nextra_part.name = %s\n" \
                "\nIf you want to ignore the extra point_meta var, " \
                " have POINT_META_VAR : 'IGNORE' " \
                % (name,
                   name, extra_vnames, 
                   name, extra_pnames,
                   extra_part.name)
            raise ValueError(s)

        pm = copy.deepcopy(base_point_meta)
        for old_name, extra_vm in extra_point_meta.items():
            extra_vm = copy.deepcopy(extra_vm)
            new_name = varmeta_map[extra_vm.name]
            extra_vm.name = new_name

            #corner case: don't care about this var
            if new_name == 'IGNORE':
                continue

            #
            if self._ref_varmetas.has_key(new_name) and \
               extra_vm != self._ref_varmetas[new_name]:
                s = "Adding a varmeta with (new) name of %s" % new_name
                s += " by copying it from input part's varmeta[old_name=%s]" % \
                     old_name
                s += ", BUT that new_name is already a reference varmeta"
                s += ", and the reference varmeta != the input part's varmeta"
                log.warning(s)

            #corner case: overlap
            if pm.has_key(extra_vm.name): 
                if only_add_nonoverlapping_vars:
                    #already have the var; nothing to do
                    pass 
                else:
                    #error
                    raise AssertionError('already have var: %s' % extra_vm.name)

            #main case: no overlap
            else:
                pm.addVarMeta( extra_vm )
                
        return pm
 
    def wrapEachNonFlexPartWithFlexPart(self):
        """
        -For each Compound/AtomicPart in the library
          -convert it to a new FlexPart with that Compound/AtomicPart as a sub-part
          -the new FlexPart has all the parameters of the previous part,
           plus of course a choice varname
         """
        log.info("wrapEachNonFlexPartWithFlexPart: begin")
        log.info("All part names: %s" % sorted(self._parts.keys()))

        #main work
        parts_to_wrap = [part for part in self._parts.itervalues()
                         if not isinstance(part, FlexPart)]
        log.info("Parts to wrap: %s" % sorted([part.name for part in parts_to_wrap]))

        for part_to_wrap in parts_to_wrap:
            self.wrapWithFlexPart(part_to_wrap)

        #postconditions
        for part in self._parts.itervalues():
            part.validate()
            for embpart in part.possibleEmbeddedParts():
                embpart.validate()
            
        log.info("wrapEachNonFlexPartWithFlexPart: done")

    def wrapWithFlexPart(self, part_to_wrap):
        #build wrapper_part = a FlexPart which has 'part_to_wrap' as its single current part choice
        wrapper_part_name = "wrapper" + "_" + part_to_wrap.name
        log.info("Wrap '%s' with flex part '%s': begin" % (part_to_wrap.name, wrapper_part_name))
        choice_varname = wrapper_part_name + "_" + "choice"
        wrapper_part = FlexPart(part_to_wrap.externalPortnames(), copy.deepcopy(part_to_wrap.point_meta),
                                wrapper_part_name, choice_varname)
        wrapper_part.addPartChoice(part_to_wrap, part_to_wrap.unityPortMap(),
                                   part_to_wrap.unityVarMap())

        #for each "other_part" in the libary
        #  for each of other_part's embedded_parts which point to "part_to_wrap"
        #    replace that pointer with a pointer to "wrapper_point"
        num_replaced = 0
        for parent_part in self._parts.itervalues():
            for emb_part in parent_part.possibleEmbeddedParts():
                if emb_part.part.ID == part_to_wrap.ID:
                    emb_part.part = wrapper_part
                    num_replaced += 1
        log.debug("Replaced %d occurrences of '%s' with '%s'" %
                  (num_replaced, part_to_wrap.name, wrapper_part_name))

        #actually add the self's parts list
        self._parts[wrapper_part_name] = wrapper_part
            
        #update opt point metas for the whole library
        #  -therefore every part that contains the wrapper_part will have some variable
        #   that determines the value of wrapper_part name
        #  -if it has >1 instance of wrapper_part in its hierarchy, it needs to have multiple
        #   versions of that variable
        #  -therefore a safe way is to make sure that when we propagate the name upwards,
        #   each level of propagation takes a new name based on its context below
        new_variables = set([choice_varname])
        made_changes = True
        while made_changes:
            (made_changes, new_variables) = self._oneRoundPropagateVariablesUpwards(new_variables)

        log.debug("Wrap '%s' with flex part '%s': done" % (part_to_wrap.name, wrapper_part_name))

        #postconditions
        for part in self._parts.itervalues():
            part.validate()

    def _oneRoundPropagateVariablesUpwards(self, new_variables):
        """Does one round of propogating 'new_variables' from children to parents."""
        
        made_changes = False
        newest_variables = copy.copy(new_variables)
        for new_variable in new_variables:
            (next_made_changes, next_newest_variables) = \
                                self._oneRoundPropagateVariableUpwards(new_variable)
            made_changes = made_changes or next_made_changes
            newest_variables |= next_newest_variables
        return (made_changes, newest_variables)
        
    def _oneRoundPropagateVariableUpwards(self, new_variable):
        """If a part has a new variable 'new_var', then this
        routine ensures that all parent parts of that part have a variable 'PARENTNAME_new_var'
        (1) in their pointmeta, and (2) a mapping from 'PARENTNAME_new_var' to 'new_var' in
        their functions.
        """
        made_changes = False
        newest_variables = set([new_variable])
        for parent_part in self._parts.itervalues():
            for (emb_part_i, emb_part) in enumerate(parent_part.possibleEmbeddedParts()):
                child_part = emb_part.part
                if child_part.point_meta.has_key(new_variable):
                    child_variable_name = new_variable
                    #compact version
                    parent_variable_name = "%d.%d:%s" % (parent_part.ID, emb_part_i, child_variable_name)

                    #non-compact version
                    #parent_variable_name = parent_part.name + "." + "emb" + str(emb_part_i) + \
                    #                       "." + child_variable_name
                    newest_variables.add(parent_variable_name)

                    #(1) update point meta if needed
                    if not parent_part.point_meta.has_key(parent_variable_name):
                        child_varmeta = child_part.point_meta[child_variable_name]
                        parent_varmeta = copy.deepcopy(child_varmeta)
                        parent_varmeta.name = copy.copy(parent_variable_name)
                        parent_part.point_meta[parent_variable_name] = parent_varmeta
                        #log.debug("For part '%s': added variable '%s' to point_meta." %
                        #          (parent_part.name, parent_variable_name))
                        made_changes = True

                    #(2) update var mapping if needed
                    if not emb_part.functions.has_key(child_variable_name):
                        emb_part.functions[child_variable_name] = copy.copy(parent_variable_name)
                        #log.debug("For (parent part '%s', child '%s'), added mapping '%s' = '%s'" %
                        #          (parent_part.name, child_part.name,
                        #           child_variable_name, parent_variable_name))
                        made_changes = True
                                                    
        return (made_changes, newest_variables)


def replaceAfterMRWithBlank(s):
    """
    @description

      With input string 's', everything after M=, or R= on each line is replaced
      with nothing.

      Helper for _compareStrings3() of library unit tests.

    @arguments

      s -- string

    @return

      modified_s -- string

    @exceptions

    @notes
    """
    s = _replaceAfterStrWithBlank(s, "M=")
    s = _replaceAfterStrWithBlank(s, "R=")
    return s
    
def _replaceAfterStrWithBlank(s, cutoff_str):
    """
    @description

      With input string 's', everything after cutoff_str on each line is replaced
      with nothing.

      Helper for replaceAfterMRWithBlank.

    @arguments

      s -- string

    @return

      modified_s -- string

    @exceptions

    @notes

    """

    #find ranges of 's' to delete
    del_ranges = [] #(st,fin)
    newline_loc = 0
    while True:    
        m_loc = s.find(cutoff_str, newline_loc)
        if m_loc == -1:
            break
        
        newline_loc = s.find("\n", m_loc)
        if newline_loc == -1:
            newline_loc = len(s)-1

        del_ranges.append((m_loc, newline_loc))

    #make a list of locs, ordered as [st0, fin0, st1, fin1, ...]
    locs = [0]
    for m_loc, newline_loc in del_ranges:
        locs.extend([m_loc-1, newline_loc])
    locs.append(len(s))

    #from list of locs, construct new_s_list which has all the wanted segments
    num_pairs = len(locs)/2
    new_s_list = []
    for pair_i in range(num_pairs):
        st = locs[pair_i*2]
        fin = locs[pair_i*2+1]
        new_s_list.append(s[st:fin])

    #merge the list into a string, and return it
    new_s = string.join(new_s_list)
    return new_s
   
def replaceSummaryStrWithBlank(s):
    """
    @description

      With input string 's', everything line between and including the
      '* ==== Summary' line and the '* ==== Done summary ====' line is deleted.

      Helper for _compareStrings4() of library unit tests.

    @arguments

      s -- string

    @return

      modified_s -- string

    @exceptions

    @notes

    """
    st_loc = s.find('* ==== Summary')
    
    if st_loc == -1:
        return s

    end_str = '* ==== Done summary ===='
    end_loc = s.find(end_str) + len(end_str)

    s2 = s[:st_loc] + s[end_loc+1:]
    
    return s2
    
def saturationSimulationDOCs():
    """Return a list of the recommended SimulationDOCs for saturation MOS's should use.
    (e.g. inside definition of saturatedMos3()).
    """
    # the SimulationDOC's are DOC's derived from simulation 'lis' output.

    #This is the original approach.  It does not say degree of feas/infeas, but is most accurate.
    metric = Metric('operating_region', REGION_SATURATION, REGION_SATURATION, False, 0, 1)
    function = 'region'
    doc1 = SimulationDOC(metric, function, REGION_SATURATION, REGION_SATURATION)

    # Goal is "get violation of saturation constraint to be 0.0",
    # where 'sat_violation' is a special measure in Analysis.py, using vgs, vth, vdsat, is_pmos
    #(added this DOC to help guide finding DOCs)
    metric = Metric('sat_violation', -INF, 0.0, False, 0, 10)
    function = 'sat_violation'
    doc2 = SimulationDOC(metric, function, 0.0, 10.0)

    return [doc1, doc2]

    
