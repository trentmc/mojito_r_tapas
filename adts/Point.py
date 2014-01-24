"""Point.py
Defines Point and PointMeta types
"""
import types

from Var import *
from util.constants import AGGR_TEST

class PointMeta(dict):
    """
    @description

      Defines the bounds for a space, that points can occupy.
      
    @attributes

      inherited__dict__ maps var_name : VarMeta
      
    @notes
    """ 
    
    def __init__(self, list_of_varmetas):
        """
        @description

            Constructor.
        
        @arguments
        
            list_of_varmetas -- list of VarMeta objects -- collectively
              the varmetas will fully describe the point's space
        
        @return
    
        @exceptions
    
        @notes
        
          Order does not matter in 'list_of_varmetas' as a dict
          is stored internally.
          
        """ 
        dict.__init__(self,{})

        for varmeta in list_of_varmetas:
            self[varmeta.name] = varmeta

        #dict of num_vars : list_of_varnames
        #(it's a dict of lists, rather than a single list, to support
        # when more vars are added)
        self._cached_choice_vars = {}
        self._cached_non_choice_vars = {}
    
    def addVarMeta(self, varmeta):
        """
        @description

          Add another variable (dimension) to this space.
        
        @arguments

          varmeta -- VarMeta object -- defines the new dimension
        
        @return

          <<none>>, but alters self's __dict__
    
        @exceptions
    
        @notes
        """
        if varmeta.name in self.keys():
            raise ValueError("Cannot add a varmeta with name '%s' because "
                             "we already have a var with that name (vars=%s)" %
                             (varmeta.name, self.keys()))
        assert isinstance(varmeta, ContinuousVarMeta) or \
               isinstance(varmeta, DiscreteVarMeta)
        self[varmeta.name] = varmeta

    def varsWithMultipleOptions(self):
        return [var_name
                for (var_name, var_meta) in self.iteritems()
                if var_meta.hasMultipleOptions()]

    def choiceVars(self):
        """Returns a list of the var names which are have isChoiceVar()== True"""
        numvars = len(self)
        if not self._cached_choice_vars.has_key(numvars):
            self._cached_choice_vars[numvars] = [
                var_name
                for (var_name, var_meta) in self.iteritems()
                if var_meta.isChoiceVar()]
        return self._cached_choice_vars[numvars]
    
    def nonChoiceVars(self):
        """Returns a list of the var names which are have isChoiceVar()==False"""
        numvars = len(self)
        if not self._cached_non_choice_vars.has_key(numvars):
            self._cached_non_choice_vars[numvars] = [
                var_name
                for (var_name, var_meta) in self.iteritems()
                if not var_meta.isChoiceVar()]
        return self._cached_non_choice_vars[numvars]

    def choiceVarsWithMultipleOptions(self):
        """Returns choice vars which hvae >1 possible values"""
        return [var_name
                for (var_name, var_meta) in self.iteritems()
                if var_meta.isChoiceVar() and \
                len(var_meta.possible_values) > 1]
    
    def __str__(self):
        """
        @description

          Override str().
        
        @arguments
        
        @return
    
        @exceptions
    
        @notes
        """
        return self.str2()

    def str2(self, show_all_vars=True):
        s = ''
        s += 'PointMeta={'
        s += '%d vars (%d are choice vars)\n' % (len(self), len(self.choiceVars()))
        if show_all_vars:
            varnames = sorted(self.keys())
        else:
            varnames = self.choiceVars()

        for (i, varname) in enumerate(varnames):
            if len(varnames) > 1: s += '\n   '
            s += str(self[varname])
            if i < len(varnames)-1: s += ', '
            
        if len(varnames) > 1: s += '\n'
        s += '/PointMeta}'
        return s

    def unityVarMap(self):
        """
        @description

          Returns a dict of {varname1:varname1, varname2:varname2, ...}
          for all vars of self.
        
        @arguments

          <<none>>
        
        @return

          unity_var_map -- dict of string : string --
    
        @exceptions
    
        @notes
        """
        names = self.keys()
        return dict(zip(names, names))

    def createRandomScaledPoint(self, with_novelty):
        unscaled_p = self.createRandomUnscaledPoint(with_novelty)
        scaled_p = self.scale(unscaled_p)
        return scaled_p

    def createRandomUnscaledPoint(self, with_novelty, use_weights = False):
        """
        @description

          Draw an unscaled Point, with uniform bias, from the space
          described by this PointMeta.
        
        @arguments

          with_novelty -- bool -- for vars that may or may not have
            novelty, do we allow the possibility of randomly choosing
            novelty?
        
        @return

          unscaled_point -- Point object
    
        @exceptions
    
        @notes

          This is completely different than RndPoint.
        """
        unscaled_d = {}
        for varname, varmeta in self.items():
            unscaled_d[varname] = varmeta.createRandomUnscaledVar(with_novelty, use_weights)

        return Point(False, unscaled_d)
        

    def spiceNetlistStr(self, scaled_point):
        """
        @description

          Returns 'scaled_point' as a SPICE-netlist ready string.
        
        @arguments

          scaled_point -- Point object -- 
        
        @return

          netlist_string -- string --
    
        @exceptions
    
        @notes
        """ 
        s = ''
        for i, (varname, scaled_varvalue) in enumerate(scaled_point.items()):
            s += self[varname].spiceNetlistStr(scaled_varvalue)
            if i < len(scaled_point)-1:
                s += ' '
        return s

    def minValuesScaledPoint(self):
        """
        @description

          Returns a scaled point, where the value of each dimension is
          its minimum value.
        
        @arguments

          <<none>>
        
        @return

          scaled_point -- Point object --
    
        @exceptions
    
        @notes
        """ 
        unscaled_d = {}
        for varname, varmeta in self.items():
            unscaled_d[varname] = varmeta.min_unscaled_value
        unscaled_p = Point(False, unscaled_d) 
        scaled_p = self.scale( unscaled_p )
        return scaled_p
        
    def inPlaceRailbinScaled(self, scaled_point):
        """
        @description

          Does an in-place railbin on scaled_point.  Assumes that it's already scaled.
        
        @arguments

          scaled_point -- Point object -- point needing railbinning
          
        @return

          <<none>> but scaled_point may be internally modified
    
        @exceptions
    
        @notes

          -Be careful when calling this, in-place can be dangerous!
          -It does NOT check to see if scaled_point is actually scaled (for speed reasons)
          -The payoff of all this, of course, is speed.
        """ 
        for (varname, varmeta) in self.iteritems():
            old_val = scaled_point[varname]
            new_val = varmeta.railbinScaled(old_val)
            if new_val != old_val:
                scaled_point[varname] = new_val
                
    def railbin(self, unscaled_or_scaled_point):
        """
        @description

          Rails then bins the input point, which may be scaled or unscaled.
          If it was unscaled coming in, it's unscaled coming out.
          If it was scaled coming in, it's scaled coming out.
          I.e. output maintains the same scaling as input.
        
        @arguments

          unscaled_or_scaled_point -- Point --
        
        @return

          railbinned_point -- Point -- the same info as the incoming point,
            but it has been railed (brought to [min,max]) and binned
    
        @exceptions
    
        @notes

          Always does the actual railing/binning in UNSCALED var space
          (very significant for logspace continuous vars and discrete vars!).
        """
        if unscaled_or_scaled_point.is_scaled:
            return self._railbinScaled(unscaled_or_scaled_point)
        else:
            return self._railbinUnscaled(unscaled_or_scaled_point)
        
        
    def _railbinScaled(self, scaled_point):
        """
        @description

          Helper function for railbin(); works only on scaled points.
        
        @arguments

          scaled_point -- Point object -- point needing railbinning
          
        @return

          railbinned_scaled_point -- Point object that's been railbinned
    
        @exceptions
    
        @notes
        """ 
        try:
            scaled_d = {}
            for (varname, varmeta) in self.iteritems():                    
                scaled_d[varname] = varmeta.railbinScaled(scaled_point[varname])
                
        except:
            validateVarLists(self.keys(), scaled_point.keys(), 'point_meta_keys', 'point_keys')
            import pdb; pdb.set_trace()

        return Point(True, scaled_d)
        
    def _railbinUnscaled(self, unscaled_point):
        """
        @description

          Helper function for railbin(); works only on unscaled points.
        
        @arguments

          unscaled_point -- Point object -- point needing railbinning
          
        @return

          railbinned_unscaled_point -- Point object that's been railbinned
    
        @exceptions
    
        @notes
        """
        unscaled_d = {}
        for (varname, varmeta) in self.iteritems():                    
            unscaled_d[varname] = varmeta.railbinUnscaled(unscaled_point[varname])
        
        return Point(False, unscaled_d)

    def unscale(self, unscaled_or_scaled_point):
        """
        @description

          Returns an unscaled version of the incoming point, which
          may or not be scaled.
        
        @arguments

          unscaled_or_scaled_point -- Point object
        
        @return

          scaled_point -- Point object
    
        @exceptions
    
        @notes

          Doesn't have to do any work if the incoming point is already scaled.
          
        """ 
        p = unscaled_or_scaled_point
        if not p.is_scaled:
            return p
        else:
            unscaled_d = {}
            for (varname, varmeta) in self.iteritems():
                unscaled_d[varname] = varmeta.unscale(p[varname])
            return Point(False, unscaled_d)

    def scale(self, unscaled_or_scaled_point):
        """
        @description

          Returns a scaled version of the incoming point, which
          may or not be scaled.
        
        @arguments

          unscaled_or_scaled_point -- Point object
        
        @return

          scaled_point -- Point object
    
        @exceptions
    
        @notes

          Doesn't have to do any work if the incoming point is already scaled.
          
        """ 
        p = unscaled_or_scaled_point
        if p.is_scaled:
            return p
        else:
            scaled_d = {}
            for (varname, varmeta) in self.iteritems():
                scaled_d[varname] = varmeta.scale(p[varname])
            return Point(True, scaled_d)
            
class Point(dict):
    """
    @description

      A point in a space.  That space is often defined by PointMeta.
      Can be unscaled or scaled (keeps track of what it is).
      
    @attributes

      ID -- int -- unique ID for this point
      __dict__ -- maps var_name : var_value.  These values can all
        be scaled or unscaled.
      is_scaled -- bool -- are all the var_values scaled, or unscaled?
      
    @notes

      Each var_value may be float or int type.
      
    """
    
    # Each point created get a unique ID
    _ID_counter = 0L
    
    def __init__(self, is_scaled, *args):
        """
        @description

          Constructor.
        
        @arguments

          is_scaled -- bool -- is this point scaled or unscaled?
          *args -- whatever is wanted for dict constructor.  Typically
            this is a dict of varname : var_value.
        
        @return
    
        @exceptions
    
        @notes
        """
        #manage 'ID'
        self._ID = self.__class__._ID_counter
        self.__class__._ID_counter += 1
        
        #validate inputs
        assert isinstance(is_scaled, types.BooleanType)

        #initialize parent class
        dict.__init__(self,*args)

        #set values
        self.is_scaled = is_scaled

    ID = property(lambda s: s._ID)

    def nicestr(self):
        """Override str()"""
        ll = []
        ll.append('\n\nPoint={ID=%d, is_scaled=%d, values:\n\n' % (self.ID, self.is_scaled))
        for var_name, var_value in self.iteritems():
            ll.append( '%s = %g\n\n' % (var_name, var_value))
        ll.append('\n/Point}\n\n')
        return ''.join(ll)
        

class EnvPoint(Point):
    """
    @description

      A Point in environmental variable space.
      
    @attributes
      
    @notes

      EnvPoints keep their own ID counters.  Which means
      that an EnvPoint may have the same ID as another non-EnvPoint;
      but it does not make them the same point!
    """ 
    _ID_counter = 0L
    
class RndPoint(object):
    
    # Each point created get a unique ID
    _ID_counter = 0L
    
    def __init__(self, values_list):
        """
        @description

          Constructor.
        
        @arguments

          values_list -- list of float
        
        @return
    
        @exceptions
    
        @notes
        """
        #preconditions
        assert isinstance(values_list, types.ListType)
        for value in values_list:
            assert mathutil.isNumber(value)
        
        #manage 'ID'
        self._ID = self.__class__._ID_counter
        self.__class__._ID_counter += 1
        
        #set values
        self.values_list = values_list

    ID = property(lambda s: s._ID)

    def isNominal(self):
        """Reports if self is nominal.  Can be nominal if self.values_list == [] or has all zeros
        -self.values_list only contains zeros
        """
        if self.values_list == []:
            return True
        else:
            for value in self.values_list:
                if value != 0:
                    return False
            return True

    def nicestr(self):
        """Override str()"""
        ll = []
        ll.append('\n\nRndPoint={ID=%d, is_nominal=%d, values: %s\n' %
                  (self.ID, self.is_nominal, self.values_list))

        return "".join(ll)
        

def validateVarLists(varlist1, varlist2, descr1, descr2):
    """Compares sorted(varlist1) with sorted(varlist2);
    If they are not equal then it prints descriptive output and
    raises a ValueError"""
    if not AGGR_TEST: return
    if set(varlist1) != set(varlist2):
        varlist2 = sorted(varlist2)
        varlist1 = sorted(varlist1)
        s = "Error: Vars in two lists do not line up\n"
        s += "Vars in %s but not in %s: %s\n" % \
             (descr1, descr2, mathutil.listDiff(varlist1, varlist2))
        s += "Vars in %s but not in %s: %s\n" % \
             (descr2, descr1, mathutil.listDiff(varlist2, varlist1))
        print s
        import pdb; pdb.set_trace()
        raise ValueError(s)

def validateIsSubset(sub_list, full_list, sub_list_descr, full_list_descr):
    """Complains if vars in sub_list are not in full_list.
    Does _not_ complain the other direction.
    """
    if not AGGR_TEST: return
    if not set(sub_list).issubset(set(full_list)):
        extra_vars = mathutil.listDiff(sub_list, full_list)
        s = "Error: Sublist has some vars that full list doesn't\n"
        s += "Vars in sublist '%s' but not in full list '%s': %s\n" % \
             (sub_list_descr, full_list, extra_vars)
        s += "All vars in sublist = %s" % str(sub_list)
        s += "All vars in full list = %s" % str(full_list)
        print s
        import pdb; pdb.set_trace()
        raise ValueError(s)
        
    
    
