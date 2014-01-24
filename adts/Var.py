"""Var.py

Holds:
-VarMeta
-Var

"""
import math
import types
import random

import numpy

from util import mathutil

class VarMeta:
    """
    @description

      Abstract class describing the space for a single variable, which can be
      continuous or discrete.
      
    @attributes

      name -- string -- the variable's name
      use_eq_in_netlist -- bool -- when doing SPICE netlisting, use an '=' ?
      
    @notes
    """
    
    def __init__(self, name=None, use_eq_in_netlist=True):
        """
        @description
        
        @arguments
        
          name -- string -- name for the var. If None, then a name is
            auto-generated which will have the word 'auto' in it.
            For SPICE vars, this must be the SPICE name, e.g. 'R' for resistors.
            
          use_eq_in_netlist -- bool -- use '=' (vs ' ') when SPICE netlisting
            (only matters for PointMetas on AtomicParts)
        
        @return

          VarMeta object
    
        @exceptions
    
        @notes
          
        """ 
        if name is None:
            self.name = VarMetaNameFactory().create()
        else:
            if not isinstance(name, types.StringType):
                raise ValueError("'name' %s is not a string" % name)
            self.name = name

        assert isinstance(use_eq_in_netlist, types.BooleanType)
        self.use_eq_in_netlist = use_eq_in_netlist

    def __eq__(self, other):
        """
        @description

          Abstract.
          Override '==' operator
          
        """
        raise NotImplementedError("implement in child")
    
    def __ne__(self, other):
        return not self.__eq__(other)
        

    def __str__(self):
        """
        @description

          Abstract.
          Override str()
          
        """ 
        raise NotImplementedError('Implement in child')

    def hasMultipleOptions(self):
        """Returns True if it can take on >1 value"""
        raise NotImplementedError('Implement in child')

    def spiceNetlistStr(self, scaled_value):
        """
        @description

          Gives string version of this var that can be used in SPICE.
        
        @arguments

          scaled_value -- float or int -- _scaled_ value of the variable
        
        @return

          SPICE_string_rep -- string
    
        @exceptions
    
        @notes
        
          Assumes scaling (and railing / binning) has already been done!

        """ 
        if self.use_eq_in_netlist: eq_s = '='
        else: eq_s = ' '
        s = '%s%s%g' % (self.name, eq_s, scaled_value)
        return s
        
    def railbinUnscaled(self, unscaled_value):
        """
        @description

          Abstract.
          If continuous, rails the var if necessary.
          If discrete var, bins the var if necessary.
          Does NOT scale!  Does NOT check to see if it's truly unscaled before.
        
        @arguments

          unscaled_value -- float or int --
        
        @return

          railbinned_unscaled_value -- float or int -- 
    
        @exceptions
    
        @notes
          
        """ 
        raise NotImplementedError('Implement in child')

    def scale(self, unscaled_value):
        """
        @description

          Abstract.
          Scales the input value.
        
        @arguments

          unscaled_value -- float or int
        
        @return

          scaled_value -- float or int
    
        @exceptions
    
        @notes
          
        """ 
        raise NotImplementedError('Implement in child')
        
    #def railbinThenScale(self, unscaled_value):
    #    raise NotImplementedError('Implement in child')

    def unscale(self, scaled_value):
        """
        @description

          Abstract.
          Unscales the input value.
        
        @arguments

          scaled_value -- float or int
        
        @return

          unscaled_value -- float or int
    
        @exceptions
    
        @notes
          
        """ 
        raise NotImplementedError('Implement in child')

    def createRandomUnscaledVar(self, allow_novelty, use_weights = False):
        """
        @description

          Draw an unscaled var, with uniform bias, from the 1-d space
          described by this VarMeta.
        
        @arguments

          allow_novelty -- bool
        
        @return

          unscaled_var -- float or int (depending if Continuous or Discrete)
    
        @exceptions
    
        @notes

          Abstract.
        """
        raise NotImplementedError('Implement in child')

    def mutate(self, unscaled_value, stddev, allow_novelty):
        """
        @description

          Abstract.
          Mutates the var value.
        
        @arguments

          unscaled_value -- float (depending on if Cont or Discrete)
          stddev -- float  in [0,1] -- amount to vary the float or int;
            0.0 means no vary, 0.05 or 0.01 is reasonable, 1.0 is crazy vary.
          allow_novelty -- bool
        
        @return

          unscaled_value -- float (depending on if Cont or Discrete)
    
        @exceptions
    
        @notes

          The returned value will be binned and scaled appropriately.
        """ 
        raise NotImplementedError('Implement in child')
        
        
class ContinuousVarMeta(VarMeta):
    """
    @description

      A VarMeta that is continuous, i.e. has an infinite number of
      possible values (within a bounded range).
      
    @attributes
    
      name -- string -- the variable's name
      use_eq_in_netlist -- bool -- when doing SPICE netlisting, use an '=' ?
      
      logscale -- bool -- is scaled_value = 10**unscaled_value ?  Tells
        how to scale the var.
      min_unscaled_value -- float or int -- lower bound
      max_unscaled_value -- float or int -- upper bound.  With min_unscaled_value
        tells how to rail the var.
      
    @notes

      min_unscaled_value and max_unscaled_value define how to rail the var.
      
    """
    
    def __init__(self, logscale, min_unscaled_value, max_unscaled_value,
                 name=None, use_eq_in_netlist=True):
        """
        @description

          Constructor.
        
        @arguments

          logscale -- bool -- is scaled_value = 10**unscaled_value ?
          min_unscaled_value -- float or int -- lower bound
          max_unscaled_value -- float or int -- upper bound
          name -- string -- see doc for VarMeta __init__
          use_eq_in_netlist -- bool -- see doc for VarMeta __init__
        
        @return

          ContinuousVarMeta object
    
        @exceptions
    
        @notes

        """
        VarMeta.__init__(self, name, use_eq_in_netlist)

        #validate inputs
        if not isinstance(logscale, types.BooleanType):
            raise ValueError("'logscale' must be boolean: %s" % logscale)
        if not mathutil.isNumber(min_unscaled_value):
            raise ValueError("'min_unscaled_value' must be a number: %s" %
                             min_unscaled_value)
        if not mathutil.isNumber(max_unscaled_value):
            raise ValueError("'max_unscaled_value' must be a number: %s" %
                             max_unscaled_value)
        if min_unscaled_value > max_unscaled_value:
            raise ValueError("min_unscaled_value(=%s) was > max (=%s)" %
                             (min_unscaled_value, max_unscaled_value))

        #set values
        self.logscale = logscale
        self.min_unscaled_value = min_unscaled_value
        self.max_unscaled_value = max_unscaled_value

        # -we don't _need_ min/max_scaled_value but they save a lot of time when railing
        self.updateScaledMinMax()

    def updateScaledMinMax(self):
        """Sets/updates self.min_scaled_value, self.max_scaled_value"""
        mn, mx = self.scale(self.min_unscaled_value), self.scale(self.max_unscaled_value)
        if mn > mx:
            mn, mx = mx, mn
        self.min_scaled_value = mn
        self.max_scaled_value = mx
        

    def __eq__(self, other):
        """
        @description

          Abstract.
          Override '==' operator
          
        """
        return self.__class__ == other.__class__ and \
               self.name == other.name and \
               self.use_eq_in_netlist == other.use_eq_in_netlist and \
               self.logscale == other.logscale and \
               self.min_unscaled_value == other.min_unscaled_value and \
               self.max_unscaled_value == other.max_unscaled_value
    
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = ''
        s += 'ContinuousVarMeta={'
        s += ' name=%s' % self.name
        s += '; logscale? %d' % self.logscale
        s += '; unscaled range = [%10g, %10g]' % \
             (self.min_unscaled_value, self.max_unscaled_value)
        s += " ; use_'='? %s" % self.use_eq_in_netlist
        s += ' /ContinuousVarMeta}'
        return s

    def isContinuous(self):
        return True

    def isDiscrete(self):
        return False

    def hasMultipleOptions(self):
        """Returns True if it can take on >1 value"""
        return (self.min_unscaled_value < self.max_unscaled_value)
        
    def railbinUnscaled(self, unscaled_value):
        """
        @description

          Rails the unscaled input value to within
          [self.min_unscaled_value, self.max_unscaled_value].
          
          Does not need to 'bin' because that has no meaning for
          continuous variables (but 'bin' is in the method name
          in order to have a common interface with discreteVarMeta).
        
        @arguments

          unscaled_value -- float or int
        
        @return

          railbinned_unscaled_value -- float or int
    
        @exceptions
    
        @notes
          
        """ 
        return max(self.min_unscaled_value,
                   min(self.max_unscaled_value, unscaled_value))
        
    def railbinScaled(self, scaled_value):
        """
        @description

          Rails the scaled input value to within
          [self.min_scaled_value, self.max_scaled_value].
          
          Does not need to 'bin' because that has no meaning for
          continuous variables (but 'bin' is in the method name
          in order to have a common interface with discreteVarMeta).
        
        @arguments

          scaled_value -- float or int
        
        @return

          railbinned_scaled_value -- float or int
    
        @exceptions
    
        @notes
          
        """ 
        return max(self.min_scaled_value,
                   min(self.max_scaled_value, scaled_value))

    def scale(self, unscaled_value):
        """
        @description

          Scales the unscaled var.
          -If self.logscale is False, returns unscaled_value
          -If self.logscale is True, returns 10^unscaled_value
        
        @arguments

          unscaled_value -- float or int
        
        @return

          scaled_value -- float or int
    
        @exceptions
    
        @notes

          Does NOT rail!  (And 'bin' has no meaning of course.)
        """ 
        assert mathutil.isNumber(unscaled_value), unscaled_value
        if self.logscale:
            return 10 ** unscaled_value
        else:
            return unscaled_value

    #def railbinThenScale(self, unscaled_value):
    #    return self.scale( self.railbinUnscaled( unscaled_value ) )

    def unscale(self, scaled_value):
        """
        @description

          Unscales the unscaled var.
          -If self.logscale is False, returns scaled_value
          -If self.logscale is True, returns log10(unscaled_value)
        
        @arguments

          scaled_value -- float or int
        
        @return

          unscaled_value -- float or int
    
        @exceptions
    
        @notes

          Provides no guarantee that the var is railed properly.
          (And 'bin' has no meaning of course.)
        """
        if self.logscale:
            try:
                return math.log10(float(scaled_value))
            except OverflowError:
                s = 'math range error: scaled_value=%g, varmeta=self=%s' %\
                    (scaled_value, self)
                raise OverflowError(s)
        else:
            if not mathutil.isNumber(scaled_value):
                raise ValueError(scaled_value)
            return scaled_value

    def isChoiceVar(self):
        """
        Is this ContinuousVarMeta a choice var?
        (always returns False because only DiscreteVarMetas can be choice vars)
        """
        return False

    def createRandomUnscaledVar(self, allow_novelty, use_weights = False):
        """
        @description

          Draw an unscaled var, with uniform bias, from the 1-d space
          described by this VarMeta.
        
        @arguments

          allow_novelty -- bool -- allow novelty for vars that might be novel?
        
        @return

          unscaled_var -- float in range [self.min_unscaled_value, self.max..]
    
        @exceptions
    
        @notes
        """
        return random.uniform(self.min_unscaled_value, self.max_unscaled_value)

    def mutate(self, unscaled_value, stddev, allow_novelty, use_weights = False):
        """
        @description

          Mutates the var value.
        
        @arguments

          unscaled_value -- float
          stddev -- float  in [0,1] -- amount to vary the float;
            0.0 means no vary, 0.05 or 0.01 is reasonable, 1.0 is crazy vary.
          allow_novelty -- bool
        
        @return

          unscaled_value -- float or int (depending on if Cont or Discrete)
    
        @exceptions
    
        @notes

          The returned value will be binned and scaled appropriately.
          The input value can be out of the range.
        """
        if not (0.0 <= stddev <= 1.0):
            raise ValueError("stddev=%g is not in [0,1]" % stddev)
        
        #a fraction of the time, choose the value uniformly
        if random.random() < stddev:
            return self.createRandomUnscaledVar(allow_novelty, use_weights = use_weights)
        else:
            rng = self.max_unscaled_value - self.min_unscaled_value
            new_value = random.gauss(unscaled_value, stddev*rng)
            new_value = self.railbinUnscaled(new_value)
            return new_value
    
class DiscreteVarMeta(VarMeta):
    """
    @description

      Describes the set of possible discrete values that a variable can take.
      
    @attributes
    
      name -- string -- the variable's name
      use_eq_in_netlist -- bool -- when doing SPICE netlisting, use an '=' ?
        
      possible_values  -- list of numbers
      novel_values -- subset of possible_values
      min_unscaled_value -- int -- always 0
      _is_choice_var -- cached value to speed isChoiceVar() calcs
      
    @notes

      An 'unscaled_value' for a discrete var is always one of the
      integers indexing into its list of possible_values.
      
      It does not use (or need) the notion of logscaling because that can
      be handled directly by values stored in the possible_values.
    """
    
    def __init__(self, possible_values, name=None, use_eq_in_netlist=True):
        """
        @description

          Constructor
        
        @arguments

          possible_values -- list of number (float or int) -- the values
            that this var can take.  Must be sorted in ascending order.
          name -- string -- see doc for VarMeta __init__
          use_eq_in_netlist -- bool -- see doc for VarMeta __init__
        
        @return

          DiscreteVarMeta object
    
        @exceptions
    
        @notes
          
        """ 
        VarMeta.__init__(self, name, use_eq_in_netlist)

        if not mathutil.allEntriesAreNumbers(possible_values):
            raise ValueError("Each value of possible_values must be a number")
        if sorted(possible_values) != possible_values:
            raise ValueError("expect possible_values to be sorted")
        
        #DO NOT have the following check, because we want to be able
        # to add possible values after creation of the DiscreteVarMeta
        #if len(possible_values) == 0:
        #    raise ValueError("need >0 possible values")
        
        self.possible_values = possible_values
        
        self.min_unscaled_value = 0
        #self.max_unscaled_value is defined via a property (see below)
        
        # -we don't _need_ min/max_scaled_value but they save a lot of time when railing
        #self.min_scaled_value is defined via a property (see below)
        #self.max_scaled_value is defined via a property (see below)
        
        self._is_choice_var = None

        #this is a subset of possible_values which, if taken, are considered
        # 'novel'
        self.novel_values = []

    def __eq__(self, other):
        """
        @description

          Abstract.
          Override '==' operator
          
        """
        return self.__class__ == other.__class__ and \
               self.name == other.name and \
               self.use_eq_in_netlist == other.use_eq_in_netlist and \
               self.possible_values == other.possible_values and \
               self.novel_values == other.novel_values
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def validate(self):
        """Raises an exception  if:
        -duplicate possible values
        -duplicate novel values
        -values in novel values that are not in possible vals
        """
        poss_vals = self.possible_values
        novel_vals = self.novel_values
        assert len(poss_vals) == len(set(poss_vals))
        assert len(novel_vals) == len(set(novel_vals))
        assert set(novel_vals).issubset(set(poss_vals))

    def _maxUnscaledValue(self):
        """
        @description
        
          Helper func to implement self.max_unscaled_value such
          that it's always correct.  Works in conjucntion with a
          property() call (see below) to achieve this functionality.
        
        @arguments

          <<none>>
        
        @return

          max_unscaled_value -- int
    
        @exceptions
    
        @notes
        """ 
        return len(self.possible_values) - 1
    max_unscaled_value = property(_maxUnscaledValue)

    def _minScaledValue(self):
        assert self.possible_values
        return self.possible_values[0]
    min_scaled_value = property(_minScaledValue)
    
    def _maxScaledValue(self):
        assert self.possible_values
        return self.possible_values[-1]
    max_scaled_value = property(_maxScaledValue)
        
    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = ''
        s += 'DiscreteVarMeta={'
        s += ' name=%s' % self.name

        s += valsListStr(self.possible_values, 'possible_values')
        s += valsListStr(self.novel_values, 'novel_values')

        #don't need the following because 'possible_values' covers it
        #s += '; min/max_unscaled_value=%g/%g' % \
        #     (self.min_unscaled_value, self.max_unscaled_value)
        
        s += ' /DiscreteVarMeta}'
        return s

    def isContinuous(self):
        return False

    def isDiscrete(self):
        return True
    
    def hasMultipleOptions(self):
        """Returns True if it can take on >1 value"""
        return (len(self.possible_values) > 1)

    def addNewPossibleValue(self, scaled_value, is_novel=False):
        """
        @description

          Add another possible value.  It will ensure that the
          possible values stay sorted.  Note that if there are
          existing unscaled_points, then they may end up referring to a
          different value now!  That can be avoided by scaling them first
        
        @arguments

          scaled_value -- float or int -- new value to add
          is_novel -- bool -- if True, then the this var value will
            also be added to the list of novel values
        
        @return

          <<none>> but updates self.possible_values and self.novel_values
    
        @exceptions
    
        @notes
          
        """
        #preconditions
        self.validate()
        assert scaled_value not in self.possible_values
        assert mathutil.isNumber(scaled_value)
        assert scaled_value not in self.possible_values
        assert scaled_value not in self.novel_values

        #main work
        self.possible_values = sorted(self.possible_values + [scaled_value])
        if is_novel:
            self.novel_values = sorted(self.novel_values  + [scaled_value])
        
        self._is_choice_var = None
        #self.max_unscaled_value does not need updating because it's a
        # function of self.possible_values

        #postconditions
        self.validate()
        
    def railbinUnscaled(self, unscaled_value):
        """
        @description

          Bins the var to closest allowable integer (index) value
        
        @arguments

          unscaled_value -- int --
        
        @return

          railbinned_unscaled_value -- int --
    
        @exceptions
    
        @notes
        """ 
        max_index = len(self.possible_values) - 1
        index = int(round(unscaled_value))
        index = max(0, min(max_index, index))
        return index
    
    def railbinScaled(self, scaled_value):
        """
        @description

          Rails the scaled input value to within
          [self.min_scaled_value, self.max_scaled_value].

          The value is also binned.
          
        @arguments

          scaled_value -- float or int
        
        @return

          railbinned_scaled_value -- float or int
    
        @exceptions
    
        @notes
          
        """
        #corner case
        if scaled_value <= self.min_scaled_value:
            return self.min_scaled_value

        #corner case
        elif scaled_value >= self.max_scaled_value:
            return self.max_scaled_value

        #main case
        else:
            eps = 1e-20
            closest_value = self.min_scaled_value
            smallest_dist = abs(scaled_value - closest_value)
            for possible_value in self.possible_values:
                dist = abs(scaled_value - possible_value)
                #getting futher away, can do early exit
                if dist > smallest_dist: 
                    break

                #improving
                elif dist < smallest_dist:
                    closest_value, smallest_dist = possible_value, dist

                #can stop early if perfect match
                if dist < eps:
                    break
                
            return closest_value

    def scale(self, unscaled_value):
        """
        @description

          Returns self.possible_values[unscaled_value]
        
        @arguments

          unscaled_value -- int --          
        
        @return

          scaled_value -- float or int --    
    
        @exceptions
    
        @notes
          
        """
        return self._railbinThenScale(unscaled_value) #safer 
 
    def _railbinThenScale(self, unscaled_value):
        """
        @description

          Helper function which railbins, then scales, the input value.
        
        @arguments

          unscaled_value -- int
        
        @return

          railbinned_scaled_value -- float or int
    
        @exceptions
    
        @notes
        """ 
        unscaled_val = self.railbinUnscaled(unscaled_value)
        safe_index = unscaled_val
        try:
            scaled_val = self.possible_values[safe_index]
        except:
            import pdb; pdb.set_trace()
            scaled_val = self.possible_values[safe_index]            
        return scaled_val

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
        return numpy.argmin([abs(scaled_value - v)
                               for v in self.possible_values])
    def isChoiceVar(self):
        """
        @description

          Is this Discrete VarMeta a 'choice' var?>
          ie are its possible values integers that are identical to it indices?
          i.e. are its possible values [0,1,2,...,n-1] ?

          Auto-detects based on its possible values.
        
        @arguments

         <<none>>
        
        @return

          is_choice_var -- bool
    
        @exceptions
    
        @notes

          Caches self._is_choice_var, or uses the cache if it's there
        """
        if self._is_choice_var is not None:
            return self._is_choice_var
        
        self._is_choice_var = True
        for index, poss_value in enumerate(self.possible_values):
            if not isinstance(poss_value, types.IntType) or \
               poss_value != index:
                self._is_choice_var = False
                break

        return self._is_choice_var

    def scaledValueIsNovel(self, scaled_value):
        """Returns True if the scaled_value is one of the novel_values"""
        return (scaled_value in self.novel_values)

    def createRandomUnscaledVar(self, allow_novelty, use_weights = False):
        """
        @description

          Draw an unscaled var, with uniform bias, from the 1-d space
          described by this VarMeta.
        
        @arguments

          allow_novelty -- bool -- for vars that may or may not have
            novelty, do we allow the possibility of randomly choosing
            novelty?
        
        @return

          unscaled_var -- float or int (depending if Continuous or Discrete)
    
        @exceptions
    
        @notes

        """
        if allow_novelty:
            num_choices = len(self.possible_values)
            assert num_choices > 0
            unscaled_value = random.randint(0, num_choices-1)
            return unscaled_value
        else:
            poss_choices = self.nonNovelValues()
            scaled_value = random.choice(poss_choices)
            unscaled_value = self.unscale(scaled_value)
            
        return unscaled_value

    def nonNovelValues(self):
        """Returns self.possible_values, except for the novel ones;
        i.e. returns the 'trustworthy' values"""
        return mathutil.listDiff(self.possible_values, self.novel_values)

    def mutate(self, unscaled_value, stddev, allow_novelty, use_weights = False):
        """
        @description

          Mutates the var value.
        
        @arguments

          unscaled_value -- int
          stddev -- float  in [0,1] -- amount to vary the int;
            0.0 means no vary, 0.05 or 0.01 is reasonable, 1.0 is crazy vary.
          allow_novelty -- bool
        
        @return

          unscaled_value -- int (depending on if Cont or Discrete)
    
        @exceptions
    
        @notes

          Does not care if the input value is not binned and scaled.
          The returned value will be binned and scaled appropriately.
        """
        
        #corner case:
        #'choice' vars have no concept of small change from one value to next;
        #  so just choose something different than the incoming value with
        #  uniform bias
        if self.isChoiceVar():
            poss_values = [poss_value
                           for poss_value in self.possible_values
                           if poss_value != unscaled_value]
            if len(poss_values) == 0:
                return unscaled_value
            else:
                return random.choice(poss_values)

        #main case: ...
        
        #a fraction of the time, choose the value uniformly
        if random.random() < stddev:
            return self.createRandomUnscaledVar(allow_novelty, use_weights = use_weights)

        #the rest of the time, change value by +1 or -1 with equal
        # probability.  (There are special cases to handle too, of course)
        else:
            unscaled_value = self.railbinUnscaled(unscaled_value)
            num_choices = len(self.possible_values)
            if num_choices == 1:
                return unscaled_value
            elif unscaled_value <= 0: #at min value
                return 1
            elif unscaled_value >= (num_choices-1): #at max value
                return unscaled_value - 1
            else:
                if random.random() < 0.5:
                    return unscaled_value + 1
                else:
                    return unscaled_value - 1

class ChoiceVarMeta(DiscreteVarMeta):
    """
    @description

      VarMeta to be used for choice variables
      
    @attributes
    
      name -- string -- the variable's name
      use_eq_in_netlist -- bool -- when doing SPICE netlisting, use an '=' ?
        
      possible_values  -- list of numbers
      novel_values -- subset of possible_values
      min_unscaled_value -- int -- always 0
      _is_choice_var -- cached value to speed isChoiceVar() calcs
      
    @notes

      is a DiscreteVarMeta with some extra attributes
    """
    
    def __init__(self, possible_values, weights=None, name=None, use_eq_in_netlist=True):
        """
        @description

          Constructor
        
        @arguments

          possible_values -- list of number (float or int) -- the values
            that this var can take.  Must be sorted in ascending order.
          name -- string -- see doc for VarMeta __init__
          use_eq_in_netlist -- bool -- see doc for VarMeta __init__
        
        @return

          ChoiceVarMeta object
    
        @exceptions
    
        @notes
          
        """ 
        DiscreteVarMeta.__init__(self, possible_values, name, use_eq_in_netlist)

        self.use_weights = (weights != None)
        self._weights = []
        self._raw_weights = []
        self.updateWeights(weights)

    def updateWeights(self, weights):
        if weights != None:
            assert len(weights) == len(self.possible_values)
            self.use_weights = True
            self._raw_weights = weights

            if len(weights):
                # make sure the weights don't get too biased
                # idea: make sure the weight of one branch is not more than a certain
                # percentage of the total weight
        
                sum = 0.0
                for w in self._raw_weights:
                    sum += float(w)
        
                normalized = numpy.array(self._raw_weights) / sum
                nb_weights = float(len(normalized))
                # give everyone at least a certain percentage
                # XX% of the interval should be distributed over all
                # weights. The remaining 100-XX% is according to the 
                # raw weight
                min_pct = 0.2 # magic number alert (20%)
                corrected = min_pct/nb_weights + (1.0 - min_pct) * normalized

                self._weights = list(corrected)
                #print "sum: %d, raw: %s => corr: %s, norm: %s, %s" % (sum, self._raw_weights, self._weights, normalized, corrected)

        else: 
            self.use_weights = False
            self._raw_weights = []
            for p in self.possible_values:
                self._raw_weights.append(1)
            assert len(self._raw_weights) == len(self.possible_values)

    def getWeights(self):
        return self._weights
    
    def addNewPossibleValue(self, scaled_value, weight=None, is_novel=False):
        """
        @description

          Add another possible value.  It will ensure that the
          possible values stay sorted.  Note that if there are
          existing unscaled_points, then they may end up referring to a
          different value now!  That can be avoided by scaling them first
        
        @arguments

          scaled_value -- float or int -- new value to add
          is_novel -- bool -- if True, then the this var value will
            also be added to the list of novel values
        
        @return

          <<none>> but updates self.possible_values and self.novel_values
    
        @exceptions
    
        @notes
          
        """
        DiscreteVarMeta.addNewPossibleValue(self, scaled_value, is_novel)
        
        if weight == None:
            # we only allow adding unweighted items when no weighting is used
            assert self.use_weights == False
            
            self.updateWeights( None )
        else:
            # this is the first possible value,
            # hence it determines whether weighting is used
            if len(self.possible_values) == 1:
                self.use_weights = True
                raw_weights = []
            else:
                raw_weights = self._raw_weights
            # only allow adding weighted possible values when weighting is used
            assert self.use_weights == True
            raw_weights.append( weight )
            self.updateWeights( raw_weights )

    def createRandomUnscaledVar(self, allow_novelty, use_weights = False):
        """
        @description

          Draw an unscaled var that takes into account the size of the
          underlying schema.
        
        @arguments

          allow_novelty -- bool -- for vars that may or may not have
            novelty, do we allow the possibility of randomly choosing
            novelty?
        
        @return

          unscaled_var -- int
    
        @exceptions
    
        @notes

        """
        if allow_novelty:
            return mathutil.randIndex(self.getWeights())
        else:
            # construct a new weight vector that has zero weight
            # for every novel choice index
            w = self.getWeights()
            poss_choices = self.nonNovelValues()
            tmp_weights = [0 for i in range(0, len(w))]
            for p in poss_choices:
                if use_weights:
                    tmp_weights[p] = w[p]
                else:
                    tmp_weights[p] = 1

            # use the new weight vector to generate the value
            return mathutil.randIndex(tmp_weights)
    
class VarMetaNameFactory:
    """
    @description

      Auto-generates a varmeta name, with the guarantee that each
      auto-generated name is unique compared to every other auto-generated
      name (but does NOT compare to manually generated names)
      
    @attributes
      
    @notes
    """
    
    _name_counter = 0L
    def __init__(self):
        """
        @description
          
        """ 
        pass

    def create(self):
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
        return 'av' + str(self.__class__._name_counter)
    
def valsListStr(vals, descr_str):
    n = len(vals)
    ndisp = 6  #magic number
    ndisp = min(len(vals), ndisp)
    s = ""
    s += '; # %s=%d' % (descr_str, n)
    if n > 0:
        s += '; %s=[' % descr_str
        for i, val in enumerate(vals[:ndisp]):
            s += '%.3g' % val
            if i < (ndisp-1): s += ', '
        if n > ndisp+1: s += ', ...'
        if n > ndisp:   s += ', %.2g' % vals[-1]
        s += ']'
    return s
    
