"""ADTs for Dyt and Cyt (Discrete and Continuous versions of Yuret optimizer, respectively.

Has classes:
-YTVarMeta (base), YTDiscreteVarMeta (child), YTContinuousVarMeta (child)
-YTPointMeta
-YTPoint
"""
import logging
import math

import numpy

from adts import *

log = logging.getLogger('yt')

class YTVarMeta(object):
    """
    @description

      Abstract class describing the space for a single variable, which can be
      continuous or discrete.
      
    @attributes

      name -- string -- the variable's name
      
    @notes
    """
    
    def __init__(self, name):
        """Constructor.
                
          name -- string -- name for the var. If None, then a name is
            auto-generated which will have the word 'auto' in it.
        """ 
        self.name = name

    def __str__(self):
        raise NotImplementedError('Implement in child')
        
    def railbinUnscaled(self, unscaled_var_value):
        """
        @description

          Abstract.
          If continuous, rails the var if necessary.
          If discrete var, bins the var if necessary.
          Does NOT scale!  Does NOT check to see if it's truly unscaled before.
        
        @arguments

          unscaled_var_value -- float or int --
        
        @return

          railbinned_unscaled_value -- float or int -- 
    
        @exceptions
    
        @notes
          
        """ 
        raise NotImplementedError('Implement in child')

    def scale(self, unscaled_var_value):
        """
        @description

          Abstract.  Scales the input value.
        
        @arguments

          unscaled_var_value -- float or int
        
        @return

          scaled_var_value -- float or int
    
        @exceptions
    
        @notes
          
        """ 
        raise NotImplementedError('Implement in child')
        
    #def railbinThenScale(self, unscaled_var_value):
    #    raise NotImplementedError('Implement in child')

    def unscale(self, scaled_var_value):
        """
        @description

          Abstract.
          Unscales the input value.
        
        @arguments

          scaled_var_value -- float or int
        
        @return

          unscaled_var_value -- float or int
    
        @exceptions
    
        @notes
          
        """ 
        raise NotImplementedError('Implement in child')
        
        

class YTDiscreteVarMeta(YTVarMeta):
    """
    @description

      Describes the set of possible discrete values that a variable can take.
      
    @attributes
    
      name -- string -- the variable's name
        
      possible_values  -- list of numbers
      min_unscaled_value -- int -- always 0
      _is_choice_var -- cached value to speed isChoiceVar() calcs
      
    @notes

      An 'unscaled_value' for a discrete var is always one of the
      integers indexing into its list of possible_values.
      
      It does not use (or need) the notion of logscaling because that can
      be handled directly by values stored in the possible_values.
    """
    
    def __init__(self, possible_values, name):
        """
        @description

          Constructor
        
        @arguments

          possible_values -- list of number (float or int) -- the values
            that this var can take.  Must be sorted in ascending order.
          name -- string -- see doc for YTVarMeta __init__
        
        @return

          YTDiscreteVarMeta object
    
        @exceptions
    
        @notes
          
        """ 
        YTVarMeta.__init__(self, name)

        if sorted(possible_values) != possible_values:
            raise ValueError("expect possible_values to be sorted")
        
        #DO NOT have the following check, because we want to be able
        # to add possible values after creation of the YTDiscreteVarMeta
        #if len(possible_values) == 0:
        #    raise ValueError("need >0 possible values")
        
        self.possible_values = possible_values
        
        self._is_choice_var = None

    def __getMinScaledValue(self):
        return self.possible_values[0]
    def __setMinScaledValue(self, v):
        raise AssertionError("not allowed to set this")
    min_scaled_value = property(__getMinScaledValue, __setMinScaledValue)
    
    def __getMaxScaledValue(self):
        return self.possible_values[-1]
    def __setMaxScaledValue(self, v):
        raise AssertionError("not allowed to set this")
    max_scaled_value = property(__getMaxScaledValue, __setMaxScaledValue)

    def __getMinUnscaledValue(self):
        return 0
    def __setMinUnscaledValue(self, v):
        raise AssertionError("not allowed to set this")
    min_unscaled_value = property(__getMinUnscaledValue, __setMinUnscaledValue)
    
    def __getMaxUnscaledValue(self):
        return len(self.possible_values) - 1
    def __setMaxUnscaledValue(self, v):
        raise AssertionError("not allowed to set this")
    max_unscaled_value = property(__getMaxUnscaledValue, __setMaxUnscaledValue)
        
    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = ''
        s += 'YTDiscreteVarMeta={'
        s += ' name=%s' % self.name
        
        n = len(self.possible_values)
        ndisp = 6
        s += '; # possible values=%d' % n
        s += '; possible_values=['
        for i,possible_value in enumerate(self.possible_values[:ndisp]):
            s += '%.3g' % possible_value
            if i < ndisp-1: s += ', '
        if n > ndisp+1: s += ', ...'
        if n > ndisp:   s += ', %.2g' % self.possible_values[-1]
        s += ']'
        s += '; min/max_unscaled_value=%g/%g' % \
             (self.min_unscaled_value, self.max_unscaled_value)
        s += ' /YTDiscreteVarMeta}'
        return s

    def addNewPossibleValue(self, scaled_var_value):
        """
        @description

          Add another possible value.  It will ensure that the
          possible values stay sorted.  Note that if there are
          existing unscaled_points, then they may end up referring to a
          different value now!  That can be avoided by scaling them first
        
        @arguments

          scaled_var_value -- float or int -- new value to add
        
        @return

          <<none>> but updates self.possible_values
    
        @exceptions
    
        @notes
          
        """ 
        assert scaled_var_value not in self.possible_values
        self.possible_values = sorted(self.possible_values + [scaled_var_value])
        
        self._is_choice_var = None
        #self.max_unscaled_value does not need updating because it's a
        # function of self.possible_values
        
    def railbinUnscaled(self, unscaled_var_value):
        """
        @description

          Bins the var to closest allowable integer (index) value
        
        @arguments

          unscaled_var_value -- int --
        
        @return

          railbinned_unscaled_var_value -- int --
    
        @exceptions
    
        @notes
        """ 
        max_index = len(self.possible_values) - 1
        index = int(round(unscaled_var_value))
        index = max(0, min(max_index, index))
        return index

    def scale(self, unscaled_var_value):
        """
        @description

          Returns self.possible_values[unscaled_var_value]
        
        @arguments

          unscaled_var_value -- int --          
        
        @return

          scaled_var_value -- float or int --    
    
        @exceptions
    
        @notes
          
        """
        return self._railbinThenScale(unscaled_var_value) #safer 
 
    def _railbinThenScale(self, unscaled_var_value):
        """
        @description

          Helper function which railbins, then scales, the input value.
        
        @arguments

          unscaled_var_value -- int
        
        @return

          railbinned_scaled_var_value -- float or int
    
        @exceptions
    
        @notes
        """ 
        unscaled_val = self.railbinUnscaled(unscaled_var_value)
        safe_index = unscaled_val
        scaled_val = self.possible_values[safe_index]        
        return scaled_val

    def unscale(self, scaled_var_value):
        """
        @description

          Unscales the scaled input value.

          Returns the index corresponding to the item that
          self.possible_values that scaled_var_value is closest
          to in non-log space
          Example:
           If self.possible_values = [10,100,1000] and scaled_var_value = 400
           then unscaled_var_value = 1 (corresponding to varval = 100, NOT 1000)
        
        @arguments

          scaled_var_value -- float or int --    
        
        @return

          unscaled_var_value -- int
    
        @exceptions
    
        @notes
          
        """
        return numpy.argmin([abs(scaled_var_value - v)
                               for v in self.possible_values])



class YTContinuousVarMeta(YTVarMeta):
    """
    @description

      A continuous var meta.
      -Minimum unscaled value is 0, maximum unscaled value is 1.0
      
    @attributes
    
      name -- string -- the variable's name
      min_scaled_value -- float -- minimum scaled value
      max_scaled_value -- float -- maximum scaled value
      
    @notes

      An 'unscaled_value' for a discrete var is always one of the
      integers indexing into its list of possible_values.
      
      It does not use (or need) the notion of logscaling because that can
      be handled directly by values stored in the possible_values.
    """
    
    def __init__(self, min_scaled_value, max_scaled_value, name):
        """
        @description

          Constructor
        
        @arguments

          possible_values -- list of number (float or int) -- the values
            that this var can take.  Must be sorted in ascending order.
          name -- string -- see doc for YTVarMeta __init__
        
        @return

          YTContinuousVarMeta object
    
        @exceptions
    
        @notes
          
        """
        YTVarMeta.__init__(self, name)

        #preconditions
        assert max_scaled_value > min_scaled_value, (min_scaled_value, max_scaled_value)

        self.min_scaled_value = min_scaled_value
        self.max_scaled_value = max_scaled_value

    def __getMinUnscaledValue(self):
        return 0.0
    def __setMinUnscaledValue(self, v):
        raise AssertionError("not allowed to set this")
    
    def __getMaxUnscaledValue(self):
        return 1.0
    def __setMaxUnscaledValue(self, v):
        raise AssertionError("not allowed to set this")
    
    min_unscaled_value = property(__getMinUnscaledValue, __setMinUnscaledValue)
    max_unscaled_value = property(__getMaxUnscaledValue, __setMaxUnscaledValue)
        
    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = ''
        s += 'YTContinuousVarMeta={'
        s += ' name=%s' % self.name
        s += '; min/max_scaled_value=%.3e/%.3e' % (self.min_scaled_value, self.max_scaled_value)
        s += '; min/max_unscaled_value=%g/%g' % (self.min_unscaled_value, self.max_unscaled_value)
        s += ' /YTContinuousVarMeta}'
        return s
        
    def railbinUnscaled(self, unscaled_var_value):
        """
        @description

          Rails the variable to within (min=0.0, max=1.0).  Does no binning b/c continuous.
        
        @arguments

          unscaled_var_value -- int --
        
        @return

          railbinned_unscaled_var_value -- int --
    
        @exceptions
    
        @notes
        """
        return max(0.0, min(1.0, unscaled_var_value))

    def scale(self, unscaled_var_value):
        """
        @description

          Returns a scaled version of the unscaled input value
        
        @arguments

          unscaled_var_value -- int --          
        
        @return

          scaled_var_value -- float or int --    
    
        @exceptions
    
        @notes
          
        """
        return self._railbinThenScale(unscaled_var_value) #safer 
 
    def _railbinThenScale(self, unscaled_value):
        """
        @description

          Helper function which railbins, then scales, the input value.
        
        @arguments

          unscaled_var_value -- int
        
        @return

          railbinned_scaled_var_value -- float or int
    
        @exceptions
    
        @notes
        """ 
        unscaled_value = self.railbinUnscaled(unscaled_value)
        scaled_value = self.min_scaled_value + \
                       unscaled_value * (self.max_scaled_value - self.min_scaled_value)
        return scaled_value

    def unscale(self, scaled_value):
        """
        @description

          Unscales the scaled input value.

          Returns the index corresponding to the item that
          self.possible_values that scaled_value is closest
          to in non-log space
          Example:
           If self.possible_values = [10,100,1000] and scaled_value = 400
           then unscaled_value = 1 (corresponding to varval = 100, NOT 1000)
        
        @arguments

          scaled_value -- float or int --    
        
        @return

          unscaled_value -- int
    
        @exceptions
    
        @notes
          
        """
        unscaled_value = (scaled_value - self.min_scaled_value) / \
                         (self.max_scaled_value - self.min_scaled_value)
        unscaled_value = self.railbinUnscaled(unscaled_value)
        return unscaled_value

def _nonzeroValuesStr(point):
    s = '{'
    for var_name, var_value in point.items():
        if float(var_value) != 0.0:
            s += '%s=%g,' % (var_name, var_value)
    s += '}'
    return s
    

class YTPointMeta(dict):
    """
    @description

      Defines the bounds for a space, that points can occupy.
      
    @attributes

      inherited__dict__ maps var_name : YTVarMeta
      
    @notes
    """  
    def __init__(self, list_of_varmetas):
        """
        @description

            Constructor.
        
        @arguments
        
            list_of_varmetas -- list of YTVarMeta objects -- collectively
              the varmetas will fully describe the point's space
        
        @return
    
        @exceptions
    
        @notes
        
          Order does not matter in 'list_of_varmetas' as a dict
          is stored internally.
          
        """ 
        dict.__init__(self,{})

        for varmeta in list_of_varmetas:
            assert varmeta.name not in self.keys(), (varmeta.name, self.keys())
            self[varmeta.name] = varmeta

    def continuousVarNames(self):
        """Returns a list of the continuous varmeta names"""
        return [varmeta.name
                for varmeta in self.values()
                if isinstance(varmeta, YTContinuousVarMeta)]

    def continuousVaryingVarNames(self):
        """Returns a list of the continuous, varying varmeta names"""
        return [varmeta.name
                for varmeta in self.values()
                if isinstance(varmeta, YTContinuousVarMeta)
                and varmeta.max_scaled_value > varmeta.min_scaled_value]

    def discreteVarNames(self):
        """Returns a list of the discrete varmeta names"""
        return [varmeta.name
                for varmeta in self.values()
                if isinstance(varmeta, YTDiscreteVarMeta)]

    def discreteVaryingVarNames(self):
        """Returns a list of discrete, varying vars"""
        return [varmeta.name
                for varmeta in self.values()
                if isinstance(varmeta, YTDiscreteVarMeta) and \
                len(varmeta.possible_values)>1]
        
    
    def addVarMeta(self, varmeta):
        """Update self by adding another variable (dimension) to this space."""
        assert varmeta.name not in self.keys(), (varmeta.name, self.keys())
        assert isinstance(varmeta, YTContinuousVarMeta) or \
               isinstance(varmeta, YTDiscreteVarMeta)
        self[varmeta.name] = varmeta

    def __str__(self):
        s = ''
        s += 'YTPointMeta={'
        for i, varmeta in enumerate(self.values()):
            s += str(varmeta)
            if i < len(self)-1: s += ', '
        s += '/YTPointMeta}'
        return s


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
        p = unscaled_or_scaled_point
        if p.is_scaled:
            unscaled_d = dict([(varname, varmeta.unscale( p[varname]) )
                               for varname, varmeta in self.items()])
            unscaled_p = YTPoint(False, unscaled_d)
            railbinned_unscaled_p = self._railbinUnscaled( unscaled_p )
            railbinned_scaled_p = self.scale(railbinned_unscaled_p)
            return railbinned_scaled_p
        else:
            railbinned_unscaled_p = self._railbinUnscaled(p)
            return railbinned_unscaled_p
        
    def _railbinUnscaled(self, unscaled_point):
        """Returns value is like unscaled_point, except railed and binned"""
        ru_d = dict([(varname, varmeta.railbinUnscaled(unscaled_point[varname]))
                     for varname, varmeta in self.items()])
        p = YTPoint(False, ru_d)
        return p

    def scale(self, unscaled_or_scaled_point):
        """Returns a scaled version of the input point,
        which may be scaled OR unscaled."""
        p = unscaled_or_scaled_point
        if p.is_scaled:
            return p
        else:
            scaled_d = dict([(varname, varmeta.scale(p[varname]))
                             for varname,varmeta in self.items()])
            p = YTPoint(True, scaled_d)
            return p
  
    def unscale(self, scaled_point):
        """Returns an unscaled version of the input 'scaled_point'"""
        assert scaled_point.is_scaled
        d = {}
        for varname, varmeta in self.items():
            d[varname] = varmeta.unscale(scaled_point[varname])
        point = YTPoint(False, d)
        return point

class YTPoint(Point):
    """YTPoint is either
    (a) a scaled opt_point that is in line with ps.opt_point_meta, OR
    (b) an unscaled opt_point that the algorithm manipulates during its search.
    They are distinguished by self.is_scaled
    """
    def __init__(self, is_scaled, d):
        Point.__init__(self, is_scaled, d)

    def discreteStepsize(self):
        """Returns stepsize.  Only appropriate for points with discrete varmetas."""
        assert not self.is_scaled
        assert len(self) > 0
        return int(sum(abs(v) for v in self.itervalues()))
    
    def continuousStepsize(self):
        """Returns stepsize.  Only appropriate for points with continuous varmetas."""
        assert not self.is_scaled
        sumsq = 0.0
        return math.sqrt(sum((v**2) for v in self.itervalues()))

    def __add__(self, other):
        """Overload the '+' operator for elementwise add, scalar add"""
        assert self.is_scaled == other.is_scaled
        assert len(self) == len(other)
        
        #make a copy of self with new ID
        s = YTPoint(self.is_scaled, self)
        for var_name in s.keys():
            s[var_name] = s[var_name] + other[var_name]
        return s

    def __mul__(self, scalarVal):
        """Overload the '*' operator for just scalar multiply"""
        m = {}
        for var_name in self.keys():
            m[var_name] = self[var_name] * scalarVal
        new_point = self.__class__(self.is_scaled, m)
        return new_point
    


def buildDytPointMeta(ps_opm):
    """Return a YT_point_meta version of 'ps_opm'.  Unlike 'ps_opm', the
    output pm knows how to scale and unscale.
    """
    assert isinstance(ps_opm, PointMeta)
    
    varmetas = []
    for ps_varname, ps_varmeta in ps_opm.items():
        yt_varmeta = YTDiscreteVarMeta(ps_varmeta.possible_values, ps_varname)
        varmetas.append( yt_varmeta )
    yt_opm = YTPointMeta(varmetas)
    return yt_opm



def buildCytPointMeta(ps_opm):
    """Return a YT_point_meta version of 'ps_opm'.  Unlike 'ps_opm', the
    output pm knows how to scale and unscale.
    """
    assert isinstance(ps_opm, PointMeta)
    
    varmetas = []
    for ps_varname, ps_varmeta in ps_opm.items():
        yt_varmeta = YTContinuousVarMeta(ps_varmeta.min_unscaled_value,
                                         ps_varmeta.max_unscaled_value,
                                         ps_varname)
        varmetas.append( yt_varmeta )
    yt_opm = YTPointMeta(varmetas)
    return yt_opm
