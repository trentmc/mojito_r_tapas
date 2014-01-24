"""mathutil.py
"""

import copy
import operator
import math
import random
import types

import numpy

from constants import BAD_METRIC_VALUE, INF

def stddev(x):
    """
    @description

      Returns standard deviation of vector x.
      
    @arguments

      x -- list of ints and/or floats, or 1d array -- 

    @return

      stddev_of_x -- float

    @exceptions

      'x' must have >1 entries.

    @notes
    """
    if len(x) <= 1:
        raise ValueError("input x is too short (len=%d)" % len(x))
    
    x = numpy.asarray(x)
    if len(x.shape) != 1:
        raise ValueError("only 1d arrays allowed")
    
    mean_x = numpy.average(x)
    SSE = sum( (x - mean_x)**2 ) #"sum-squared error"
    N = len(x)
    if N == 1:   denom = float(1)
    else:        denom = float(N - 1)
    s = math.sqrt( SSE / denom )
    return s

def average(x):
    return numpy.average(x)

def median(x):
    """Computes the median in O(n*log(n)) time.
    Note: while there are algorithms for median in O(n), this version
    uses the C-optimized python sort(), and benchmarks show it's
    faster to do this up to n = approx 20,000.
    """
    #bias n toward 0.0 to make each rank a < or = test
    n = len(x)
    n = n - (1.0 / float(n))
    K = int(0.5 * (n))
    
    return sorted(x)[K]

def findZeroCrossings(v):
    lte0 = v > 0
    dlte0 = abs(lte0[0:-1] - lte0[1:])
    return numpy.compress(numpy.greater(dlte0, 0), range(len(dlte0)))

def hasBAD(x):
    """Returns true if any entry in vector or 1d array x has a BAD_METRIC_VALUE in it"""
    for xi in x:
        if xi == BAD_METRIC_VALUE:
            return True
    return False

def hasNan(x):
    """Returns true if any entry in vector or 1d array x has a 'nan' in it"""
    for xi in x:
        if isNan(xi):
            return True
    return False

def isNan(xi):
    return ((xi == xi) != True)

def hasInf(x):
    """Returns true if any entry in vector or array x has an 'inf' in it"""
    for xi in x:
        if xi == INF:
            return True
    return False

def hasNanOrInf(x):
    """Returns true if any entry in vector or array x has a 'nan' or
    'inf' in it"""
    for xi in x:
        if (xi == INF) or (xi == -INF) or isNan(xi):
            return True
    return False
    
def allEntriesAreUnique(x):
    """Returns true if every entry in x is unique"""
    return len(x) == len(set(x))

def listDiff(list_a, items_to_remove):
    """Returns list_a, minus 'items_to_remove'
    """
    return [entry_a for entry_a in list_a if entry_a not in items_to_remove]

def listIntersect(list_a, list_b):
    """Returns the items that list_a and list_b share"""
    set_a, set_b = set(list_a), set(list_b)
    return list( set_a.intersection(set_b) )

def listsOverlap(list_a, list_b):
    """Returns True if list_a and list_b share at least one common item '
    """
    set_a, set_b = set(list_a), set(list_b)
    return len( set_a.intersection(set_b) ) > 0

def isNumber( x ):
    """Returns True only if x is a Float, Int, or Long, and NOT complex"""
    #turned off the check for LongType, ComplexType for speed, because we never use them
    # and isNumber() is a bottleneck.
    return \
           (
        isinstance(x, types.FloatType) or \
        isinstance(x, types.IntType) #or \
        #isinstance(x, types.LongType) or
        #isinstance(x, types.ComplexType)
        ) \
        and (not isNan(x))

def allEntriesAreNumbers( xs ):
    """Returns true if every entry in this list, 1-d array, or set is
    a NumberType"""
    for x in xs:
        if not isNumber( x ):
            return False
    return True

def randIndexFromDict(biases_dict):
    """
    @description

      Randomly chooses a key from biases_dict, where
      with a bias towards higher values of biases which are in dict's values.
      
    @arguments

      biases_dict -- dict of cand_return_key : corresponding_bias;
        each bias is a float or int

    @return

      chosen_return_key -- one of the keys of biases_dict

    @exceptions

    @notes
    """
    cand_keys = biases_dict.keys()
    biases = [biases_dict[key] for key in cand_keys]
    chosen_i = randIndex(biases)
    return cand_keys[chosen_i]

def randIndex(biases):
    """
    @description

      Randomly chooses an int in range {0,1,....,len(biases)-1), where
      with a bias towards higher values of biases
      
    @arguments

      biases -- list of [bias_for_index0, bias_for_index1, ...] where
        each bias is a float or int

    @return

      chosen_index -- int -- 

    @exceptions

    @notes
    """
    #validate inputs
    if len(biases) == 0:
        raise ValueError("Need >0 biases")
    if min(biases) < 0.0:
        raise ValueError("All biases must be >=0")

    #corner case: every bias is zero
    if float(min(biases)) == float(max(biases)) == 0.0:
        return random.randint(0, len(biases)-1)

    #main case
    accbiases = numpy.add.accumulate(biases)
    maxval = accbiases[-1]
    thr = maxval * random.random()
    
    for i, accbias in enumerate(accbiases):
        if accbias > thr or accbias == maxval:
            return i

    #should never get here, but just in case...
    return i

def niceValuesStr( d ):
    """
    @description

      Given a dict of key : number_value, output as a string
      where the values are printed with '%g'.
      
    @arguments

      d -- dict

    @return

      s -- string

    @exceptions

      If number_value is a BAD_METRIC_VALUE, it will print that instead
      of applying %g.

    @notes
    """
    s = '{'
    for index, (key, value) in enumerate(d.items()):
        s += '%s:' % key
        if value == BAD_METRIC_VALUE:
            s += str(BAD_METRIC_VALUE)
        else:
            s += '%g' % value
        if (index+1) < len(d):
            s += ','
    s += '}'
    return s
    


def uniqueStringIndices(strings_list):
    """
    @description

      Returns a list of indices of strings_list such
      that there is only one id.  Always returns the index
      of the first unique element that occurs.

    @arguments

      strings_list -- list of string objects

    @return

      I -- list of indices into strings_list

    @exceptions

    @notes
    """
    if not isinstance(strings_list, types.ListType):
        raise ValueError("argument needs to be a list, not a %s" %
                         (strings_list.__class__))
    
    #The trick: python dictionaries are very efficient at knowing
    # what keys they already have.  So exploit that for O(NlogN) efficiency.
    #(Otherwise the algorithm is O(N^2)
    
    I = []
    strings_dict = {}
    for i,s in enumerate(strings_list):
        if not isinstance(s, types.StringType):
            raise ValueError("an entry was %s rather than string" %
                             (s.__class__))
        len_before = len(strings_dict)
        strings_dict[s] = 1
        len_after = len(strings_dict)
        if len_after > len_before:
            I.append(i)
    return I


def minPerRow(X):
    """
    @description

      Returns the minimum value found in each row

    @arguments

      X -- 2d array of numbers

    @return

      min_per_row -- 1d array of numbers

    @exceptions

    @notes
    """
    assert len(X.shape) == 2, "should be 2d array"
    return numpy.array([min(X[row_i,:]) for row_i in range(X.shape[0])],
                         dtype=float)

def maxPerRow(X):
    """
    @description

      Returns the maximum value found in each row

    @arguments

      X -- 2d array of numbers

    @return

      max_per_row -- 1d array of numbers

    @exceptions

    @notes
    """
    assert len(X.shape) == 2, "should be 2d array"
    return numpy.array([max(X[row_i,:]) for row_i in range(X.shape[0])],
                         dtype=float)

def scaleTo01(X, min_x, max_x):
    """
    @description

      Assuming that min_x and max_x define the min and max values for
      each row of X, then this will return values of X that are each
      scaled to within [0,1].

    @arguments

      X -- 2d array of numbers [# input vars][# samples] -- data to be scaled
      min_x -- 1d array of numbers [# input vars] -- minimum bounds of X
      max_x -- 1d array of numbers [# input vars] -- maximum bounds of X

    @return

      scaled_X -- 2d array of numbers [# input vars][# samples] -- like X,
        but each value is in [0,1] according to min_x and max_x
      
    @exceptions

      If min_x and max_x do not correspond precisely to the
      min and max values of X, that's ok, it just means that the
      output won't be precisely in [0,1]

    @notes
    """
    assert len(X.shape) == 2, "X should be a 2d array"
    assert len(min_x) == len(max_x) == X.shape[0]
    for var_i in range(len(min_x)):
        assert min_x[var_i] < max_x[var_i]

    scaled_X = numpy.zeros(X.shape, dtype=float)
    for var_i in range(X.shape[0]):
        scaled_X[var_i,:] = (X[var_i,:] - min_x[var_i]) / \
                            (max_x[var_i] - min_x[var_i])

    return scaled_X
    
    

def distance(x1, x2):
    """
    @description

      Returns the euclidian distane between x1 and x2

    @arguments

      x1, x2 -- 1d array [input variable #] -- input points

    @return

      d -- float -- the distance

    @exceptions

    @notes

      Does _not_ scale to a [0,1] range
    """
    assert len(x1) == len(x2) > 0
    return math.sqrt( sum([(x1[i] - x2[i])**2 for i in range(len(x1))]))



def epanechnikovQuadraticKernel(distance01, lambd):
    """
    @description

      A popular kernel function for local regression (and other apps).
      The smaller that 'distance' is, the closer to 1.0 the output comes.
      And if 'distance' is too large, then output is 0.0.

    @arguments

      distance01 -- float -- expect this to be scaled in [0,1]
      lambd -- float -- 'bandwidth'

    @return

      k -- float -- kernel output

    @exceptions

    @notes

      Reference: Hastie, Tibhsirani and Friedman, 2001, page 167
    """
    t = distance01 / lambd
    if abs(t) <= 1.0:
        return 3.0/4.0 + (1.0 - t**2)
    else:
        return 0.0

def permutations(var_bases):
    """
    @description

      Returns all permutations as specified by var_bases.  

    @arguments

      var_bases -- list of int -- var_bases[i] > 0 specifies the number of
        values that this base can take on.  len(var_bases) = num vars = n

    @return

      perms -- list of perm, where perm is a list of n values, one
        for each var.  0 <= perm[var] < var_bases[var]

    @exceptions

    @notes
    
      var_bases[0] is most significant digit, and var_bases[-1] is least
        significant.
    """
    perms = []
    cur_number = [0 for base in var_bases]
    overflowed = False
    while not overflowed:
        perms.append(cur_number)
        cur_number, overflowed = baseIncrement(cur_number, var_bases)

    return perms

def baseIncrement(cur_number, var_bases):
    """
    @description

      Increments cur_number according to the base-system of 'var_bases'

      If cur_number is at its maximum possible value, and this is called,
      then (None, overflowed=True) is returned.

    @arguments

      cur_number -- list of int -- len(cur_number) = num_vars = n
      var_bases -- list of int -- var_bases[i] > 0 specifies the number of
        values that this base can take on.  len(var_bases) = n

    @return

      next_number -- list of int --
      overflowed -- bool -- only True if cur_number is at max possible value

    @exceptions

    @notes

      Assumes that cur_number is within the proper range as specified
       by var_bases
    """
    if len(var_bases)==0:
        return None, True

    active_var = len(var_bases)
    next_number = copy.copy(cur_number)
    next_number[active_var-1] += 1
    while True:
        if next_number[active_var-1] >= var_bases[active_var-1]:
            next_number[active_var-1] = 0
            active_var -= 1
            if active_var-1 < 0:
                return None, True
            next_number[active_var-1] += 1
        else:
            break

    return next_number, False
    

def uniquifyVector( v):
    """
    @description

      Returns a vector which has no duplicate entries.

    @arguments

      v -- 1d array -- 

    @return

      unique_v -- 1d array

    @exceptions

    @notes
    
      Side effect: vector gets sorted.
    """
    if v.shape[0] <= 1:
        # if there are zero or one element(s)
        # they are always unique
        return v

    unique_v = numpy.array(sorted(set(v)))
    return unique_v

def nmse(yhat, y, min_y, max_y):
    """Return normalized mean-square error.  Good for assessing
    'goodness' of a regressor'
    """
    #preconditions
    assert len(yhat) > 0
    assert len(yhat) == len(y)

    #corner case
    if (max(yhat) == min(yhat) == max(y) == min(y)):
        return 0.0

    #corner case
    assert max_y > min_y

    #main case
    yhat, y = numpy.asarray(yhat), numpy.asarray(y)
    y_range = float(max_y) - float(min_y)
    
    nmse = math.sqrt(numpy.average(((yhat - y) / y_range) ** 2))

    return nmse

def removeConstantRows(X, I):
    """
    @description

        From among possible rows I, return the indices of rows of X 
        that do not have the same constant value in all columns.

    @arguments

        X: data set
        I: list of rows (indices) in X to be evaluated.
        
    @return

        new_I: list of indices of rows that do not contain a constant value.

    @exceptions

    @notes

    """ 
    # If the max value in the row == min value in the row
    # then the values in the row are constant.
    new_I = [i for i in I if max(X[i, :]) > min(X[i, :])]
    new_I.sort()
    return new_I


                          

def mostImportantVars(infl_per_var, target_perc_impact):
    """
    @description
    
      In the general case, returns a list holding indices or names of variables 
      whose total influence is as close as possible to target_perc_impact. When
      type(infl_per_var) == list, it returns variable indices and when 
      type(infl_per_var) == dict, it returns variable names. 
    
    @arguments
    
      infl_per_var -- list or dict -- see mostImportantVars_list
          and mostImportantVars_dict
      target_perc_impact -- float -- scalar in range [0, 1]
        
    @return
    
      list -- see mostImportantVars_list and mostImportantVars_dict
        
    @exceptions
      
    @notes
      
    """    
    
    if len(infl_per_var) == 0:
        return []
    elif type(infl_per_var) == dict:
        return mostImportantVars_dict(infl_per_var, target_perc_impact)
    else:
        return mostImportantVars_list(infl_per_var, target_perc_impact)
    
def mostImportantVars_list(list_infl_per_var, target_perc_impact):
    """
    @description
    
      In the general case, returns a list holding indices of variables whose 
      total influence is as close as possible to target_perc_impact. The 
      special case where all elements of list_infl_per_var equal 0 returns the 
      list of all variables.
    
    @arguments
    
      list_infl_per_var -- list of float -- entry i holds influence of var i
      target_perc_impact -- float -- scalar in range [0, 1]
        
    @return
    
      impt_vars -- list of strings -- the most important vars
        
    @exceptions
      
    @notes

      If sum(infls) == 0, it returns ALL variables.  It could have returned
      no variables, but for the current usage, returning all variables
      is more useful.
    """   
    
    infls = list_infl_per_var
    
    if min(infls) < 0.0:
        raise ValueError('min(infls): invalid value %s' % min(infls))
    if not (0.0 <= target_perc_impact <= 1.0):
        raise ValueError('target_perc_impact: invalid value %s' % target_perc_impact)
    if sum(infls) == 0: 
        return range(len(infls)) 
    
    cur_impact = 0.0
    impt_vars = []
    
    # calculate the enumeration once to save time and ensure it is constant
    enum = [(i, infl) for (i, infl) in enumerate(infls)]

    #initialize remaining vars and impacts
    vars_left = [i for (i, infl) in enum if infl > 0.0]
    impacts_left = [infl for (i, infl) in enum if infl > 0.0]

    # loop while the target impact is not met and there are 
    # still variables to look at
    while cur_impact / sum(infls) < target_perc_impact and vars_left:
        # find most important var and its corresponding impact
        I = numpy.argmax(impacts_left)
        (next_var, next_impact) = (vars_left[I], impacts_left[I])
        
        # add this next most important variable to the current list
        impt_vars.append(next_var)
        
        # update the impact to reflect the newly added variable
        cur_impact += next_impact
        
        # remove the newly added variable and impact from consideration
        del vars_left[I]
        del impacts_left[I]
        
    return impt_vars
    
def mostImportantVars_dict(dict_infl_per_var, target_perc_impact):
    """
    @description
    
      In the general case, returns a list holding names of variables whose 
      total influence is as close as possible to target_perc_impact. 
    
    @arguments
    
      dict_infl_per_var -- dict of string:float -- each entry tells the
         relative influence of a variable
      target_perc_impact -- float -- scalar in range [0, 1]
        
    @return
    
      impt_vars -- list of strings -- the most important vars
        
    @exceptions
      
    @notes
    """
    items = dict_infl_per_var.items()
    list_infl_per_var = [item[1] for item in items]
    impt_vars = mostImportantVars_list(list_infl_per_var, target_perc_impact)
    return [items[i][0] for i in impt_vars]


def integerValue(binary_digits):
    """Converts a binary number (represented here as a list) into
    an integer.  MSB is leftmost index."""
    num_digits = len(binary_digits)
    integer_value = 0
    for (i,digit) in enumerate(binary_digits):
        bit_index = num_digits-i-1
        integer_value += (digit * (2**bit_index))
    return integer_value

def binaryValue(integer_value, num_digits):
    """Converts an integer value to a binary (represented here as a list)"""
    assert 0 <= integer_value <= (2**num_digits - 1)
    remaining_value = integer_value
    digits = [0] * num_digits
    for i in range(num_digits):
        bit_index = num_digits-i-1
        (remaining_value, digits[bit_index]) = divmod(remaining_value, 2)
    return digits



def correlation(xs, ys):
    """
    @description
    
      Returns the Pearson Correlation Coefficient of the two input
      vectors/lists xs and ys.
    
    @arguments
    
      xs -- vector/list -- input vector 1
      ys -- vector/list -- input vector 2
        
    @return
    
      r -- float -- the correlation coefficient
        
    @exceptions
      
    @notes
    
      Pearson's r coefficient, i.e. 'correlation coefficient' is described at
      http://mathworld.wolfram.com/CorrelationCoefficient.html
    
    """   
    
    if len(xs) != len(ys):
        raise ValueError('len(xs) = %d and len(ys) = %d must be equal' 
                         % (len(xs), len(ys)))

    N = len(xs)
    if N == 0:
        raise ValueError('len(xs) / len(ys) = %d must be >0' % N)
    
    # AUDIT: should we be using ieee NaN/Inf/-Inf?
    
    # this shortcuts all the calculations
    if N == 1:
        # return float('NaN')
        return 0.0
    
    sum_xy = 0.0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0

    for (x, y) in ((xs[i], ys[i]) for i in range(N)):
        sum_xy += x * y
        sum_x += x
        sum_y += y
        sum_x2 += x ** 2
        sum_y2 += y ** 2
    
    numerator = sum_xy - (sum_x * sum_y) / N

    #tlm: the 'max' is to avoid numerical error which might
    # otherwise make the argument to sqrt() negative
    denominator = math.sqrt(max(0.0, (sum_x2 - (sum_x ** 2) / N) 
                                * (sum_y2 - (sum_y ** 2) / N)))
                          
    if numerator == 0.0 and denominator == 0.0:
        # return float('Nan')
        return 0.0
    elif numerator > 0. and denominator == 0.0:
        # return INF
        return 0.0
    elif numerator < 0. and denominator == 0.0:
        # return INF
        return 0.0
    else:
        return numerator / denominator


def correlation_2(y, t):
    """Returns the correlation between vectors y and t.
    Correlation can be thought of as a way of
    measuring the agreement between two waveforms, that
    is independent of the waveforms' scale and bias

    WARNING: does not handle corner cases, e.g. is susceptible
    to divide-by-zero errors
    """
    n = len(y)
    m1 = numpy.average(y)
    m2 = numpy.average(t)
    s1 = stddev(y)
    s2 = stddev(t)
    y = numpy.asarray(y) - m1
    t = numpy.asarray(t) - m2
    v = numpy.dot(t,y)
    r = v / ((n-1) * s1 *s2)
    return r



def correlation_3(y, t):
    """This version of correlation may be buggy?
    """
    n = len(y)
    m = numpy.average(y)
    y = y - m
    v = 1.0 / n * numpy.dot(y,y)
    w = 1.0 / math.sqrt(v)
    r = 1.0 / n * w * numpy.dot(y, t)
    return r


def averagePerRow(X):
    """
    @description

       Computes and returns the average for each row in 2d array X.
       This routine exists to help clarify some code.

    @arguments

        X -- 2d array, size n x N -- 
        
    @return

       avg_per_row -- 1d array, length n --

    @exceptions

    @notes        
    """
    if len(X.shape) != 2:
        raise ValueError("Need a 2d array input; shape was %s" % X.shape)
    if X.shape[1] == 0:
        raise ValueError("Need >0 entries to compute average")
    
    return numpy.average(X, 1)

def stddevPerRow(X):
    """
    @description

       Computes and returns the stddev for each row in 2d array X.
       This routine exists to help clarify some code.

    @arguments

        X -- 2d array, size n x N -- 
        
    @return

       stddev_per_row -- 1d array, length n --

    @exceptions

    @notes        
    """
    if len(X.shape) != 2:
        raise ValueError("Need a 2d array input; shape was %s" % X.shape)
    if X.shape[1] == 0:
        raise ValueError("Need >0 entries to compute stddev")
    
    return numpy.array([stddev(X[i,:]) for i in range(X.shape[0])])



def normality(x):
    """
    @description

        Returns a value in the range [0, 1] answering the question:
        'How close to a Gaussian distribution is the vector x?'
        The returned value is interpreted as 0.0 = not normal, 1.0= very normal

        Implemented by measuring the corre

    @arguments

        x: a list of values
        
    @return

        A floating point value in the range [0, 1] -- see Description

    @exceptions

    @notes

    This return a value > 0.9 for uniform distributions and > 0.95
    for normal distributions. are these values acceptable?
        
    """
    N = len(x)
    if N <= 1:
        return 0.0

    gauss_vals = [inverseNormalCumul((i + 0.5) / N) for i in range(N)]
    
    #plotting data_vals vs. gauss_vals would be a QQ plot
    r = correlation(sorted(x), gauss_vals)
    return abs(r ** 2)





def inverseNormalCumul(p, u=0.0, std=1.0):
    """
    @description
        
        Given a probability 'p' in a standard normal distribution with
        mean 'u' and standard deviation 'std', returns the value x such that 
        probability(x <= X) = p.
    
    @arguments
      
        p -- number -- the probability in a normal distribution
        u -- number -- the mean of a normal distribution (default = 0.0)
        std -- number -- the stand. dev. of a normal distribution (default = 1.0)
        
    @return
    
        float -- see Description
        
    @exceptions
        
    @notes
    """
    
    if std < 0.0:
        raise ValueError('std: invalid value %s' % std)
    
    return u + _inverseNormalCumul(p) * std 
    
def _inverseNormalCumul(p):
    """
    @description
        
        Given a probabilty 'p' in a standard normal distribution with
        mean=0.0 and stddev=1.0, returns the value x such that 
        probability(x <= X) = p.
    
    @arguments
      
        p -- number -- the probability in a normal distribution
        
    @return
    
        float -- see Description
        
    @exceptions
      
    @notes
    
    http://home.online.no/~pjacklam/notes/invnorm/#Python
    http://home.online.no/~pjacklam/notes/invnorm/impl/field/ltqnorm.txt
    
    :COPYRIGHT: cc - the file .../data/legal/acklam.txt credits the author
    as required for distribution of this code.
    """

    if not (0.0 < p < 1.0):
        raise ValueError('p: invalid value %s' % p)

    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
          2.445134137142996e+00, 3.754408661907416e+00)

    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow

    # Rational approximation for lower region:
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
                    (
                        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4])
                        * q 
                        + c[5]
                    ) 
                    /
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    # Rational approximation for upper region:
    if phigh < p:
       q = math.sqrt(-2 * math.log(1 - p))
       return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    # Rational approximation for central region:
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def gaussianPdf(x, u, std_dev):
    """
    @description

        For a gaussian (normal) distribution characterized by u and std_dev,
        returns the probability that it will sample x
        
    @arguments


        u -- float -- mean of the distribution
        std_dev -- float -- standard deviation of the distribution
        x -- float -- threshold value
        
    @return

        float: the probability of x -- see Description

    @exceptions

    @notes
    """
    if std_dev == 0:
        raise ValueError('gaussianPdf: std_dev = 0')
    
    z = (x - u) / std_dev
    numer = math.exp(-(z * z) / 2.0)
    denom = math.sqrt(2.0 * math.pi)
    return numer / denom

def gaussianCdf(x, u, std_dev):
    """
    @description

        For a gaussian (normal) distribution characterized by u and std_dev,
        returns the probability that it will sample <= x
        (i.e. area under the gaussian pdf in range [-inf, X]
        
    @arguments

        u -- float -- mean of the distribution
        std_dev -- float -- standard deviation of the distribution
        x -- float -- threshold value
        
    @return

        prob_lte_x -- float -- probability that a sample will be <=x

    @exceptions

        std_dev cannot be 0.0
        
    @notes

      Now that we use scipy, consider using scipy.stat.norm.cdf(z)
      instead of _gaussianCdfPositiveZ(z)
    """
    if std_dev == 0:
        raise ValueError('gaussianCdf: std_dev was 0.0')
    
    z = (x - u) / std_dev
    if z >= 0.0:
        return _gaussianCdfPositiveZ(z)
    else:
        return 1.0 - _gaussianCdfPositiveZ(-z)


#from Lapin, Probability and Statistics For Modern Engineering,
#'Table D: Cumulative Probability Distribution Function for
# the Standard Normal Distribution', pp.767-768
# 'z' = "number of std deviations"
#Currently it only has about 1/5 of the entries that the table has;
# for more accuracy use more data (or use scipy.stats.norm.cdf)
_z_to_prob = {
    0.00 : 0.5000, 0.05 : 0.5199,
    0.10 : 0.5398, 0.15 : 0.5596,
    0.20 : 0.5793, 0.25 : 0.5987, 
    0.30 : 0.6179, 0.35 : 0.6368,
    0.40 : 0.6554, 0.45 : 0.6736,
    
    0.50 : 0.6915, 0.55 : 0.7088,
    0.60 : 0.7257, 0.65 : 0.7422,
    0.70 : 0.7580, 0.75 : 0.7734,
    0.80 : 0.7881, 0.85 : 0.8023,
    0.90 : 0.8159, 0.95 : 0.8289,
    
    1.00 : 0.8431, 1.05 : 0.8531,
    1.10 : 0.8643, 1.15 : 0.8749,
    1.20 : 0.8849, 1.25 : 0.8944,
    1.30 : 0.9032, 1.35 : 0.9115,
    1.40 : 0.9192, 1.45 : 0.9265,

    1.50 : 0.9332, 1.55 : 0.9394,
    1.60 : 0.9452, 1.65 : 0.9505,
    1.70 : 0.9554, 1.75 : 0.9599,
    1.80 : 0.9641, 1.85 : 0.9678,
    1.90 : 0.9713, 1.95 : 0.9744,
    
    2.00 : 0.9772,
    2.10 : 0.9821,
    2.20 : 0.9861,
    2.30 : 0.9893,
    2.40 : 0.9918,
    
    2.50 : 0.9938,
    2.60 : 0.9953,
    2.70 : 0.9965,
    2.80 : 0.9974,
    2.90 : 0.9981,
    
    3.00 : 0.9986,
    3.10 : 0.9990,
    3.40 : 0.99966,
    3.70 : 0.99989,

    4.0 : 0.99997,
    4.5 : 1.00000,
}
left_z = 0.00
for right_z in sorted(_z_to_prob.keys())[1:]:
    assert _z_to_prob[left_z] < _z_to_prob[right_z]
    left_z = right_z

def _gaussianCdfPositiveZ(z):
    if z < 0:
        raise ValueError('_gaussianCdfPositiveZ needs z>0')
    z_keys = sorted(_z_to_prob.keys())

    #corner case 1: minimum value
    if z == 0.0:
        return _z_to_prob[z]

    #corner case 2: maximum value
    elif z >= z_keys[-1]:
        return _z_to_prob[z_keys[-1]]

    #main case: z will be between two keys, so do linear interpolation
    else:
        z_a = z_keys[0]
        for z_b in sorted(_z_to_prob.keys())[1:]:
            #case 1: exact match to a key
            if z == z_b:
                return _z_to_prob[z_b]

            #case 2: match -- between two keys
            elif z_a < z < z_b:
                p_a = _z_to_prob[z_a]
                p_b = _z_to_prob[z_b]
                p = p_a + (z - z_a) * (p_b - p_a) / (z_b - z_a)
                return p

            #case 3: no match, so prep for next loop, and re-loop
            z_a = z_b



def drawFromPyramidDistribution(d):
    """
    @description

      Draw from a distribution in range [0, d], where
      the probability is highest at x=d/2, then linearly decreases
      to both sides down to a probability of 0.0 at x = 0.0 and x = d.
    
    @arguments

      d -- float -- maximum value 
            
    @return

      x -- float -- randomly drawn number in [0, d]
        
    @exceptions

      d must be >= 0.
      
    @notes
    """
    if d < 0:
        raise ValueError('"d" must be >= 0')
    elif d == 0:
        return 0.0

    #use a single-drawn random variable for two purposes
    r = random.random()

    #first use: go left y/n?
    go_left = (r < 0.5)

    #for the second use, we need to ensure that r is in [0,1]
    if r < 0.5:
        r *= 2.0
    else:
        r = (r - 0.5)*2.0

    #replace the following line with the two below it (for speed)
    #x_transformed = drawFromLinearIncreasingDistribution(d / 2.0)
    cum_p_drawn = r * ((d/2.0)**2)
    x_transformed = math.sqrt(cum_p_drawn)
    
    if go_left:
        x = x_transformed
    else:
        x = d - x_transformed
    return x


def dictsAlmostEqual(dict1, dict2, tol = 1.0e-4):
    """Returns True if all values in dict1 are almost equal to all values in dict2.
    Dicts must have same keys.
    """
    #preconditions
    assert set(dict1.keys()) == set(dict2.keys())

    #main work
    for k in dict1.iterkeys():
        if not scalarsAlmostEqual(dict1[k], dict2[k], tol):
            return False
    return True

def scalarsAlmostEqual(v1, v2, tol = 1.0e-4):
    """Returns True if the scalar v1 and v2 are almost equal, within a tolerance"""
    
    #corner case: denominator is zero
    if (v2 == 0.0):
        return abs(v1) <= tol
    else:
        return ((v1 / v2) - 1.0) <= tol
        


class MultiBaseCounterIterator:
    """
    @description

        Iterator class used with MultiBaseCounter below.

    @arguments

        The MultiBaseCounter instance to iterate over.

    @return

        -- each call to next() returns the next value in the counter.

    @exceptions

    @notes

    """
    def __init__(self, counter):
        self.counter = counter
        self.finished = False

    def __iter__(self):
        return self

    def next(self):
        if self.finished:
            raise StopIteration
        v = self.counter.currentValue()
        try:
            self.counter.increment()
        except ValueError:
            self.finished = True
        return v


class MultiBaseCounter:
    """
    @description

        Class that provides iteration over a sequence of ranges,
        each with an independent numerical base.

    @arguments

        The list of bases for each position in the counter.
        Each must be a positive integer.

    @return

        -- each call to next() returns the next value in the counter.

    @exceptions

    @notes

        Iteration over this object instantiated with
        bases == (3, 4, 2) would generate values:

           (0, 0, 0)
           (0, 0, 1)
           (0, 1, 0)
           ...
           (0, 3, 1)
           (1, 0, 0)
           ...
           (2, 3, 1)

        Virtually all interaction with this class should be simply
        creating an instance and iterating over it.

        Note that it is not safe to have two different simulateous
        iterators accessing the same MultiBaseCounter object.

    """
    def __init__(self, bases):
        assert type(bases) in (list, tuple), 'bases must be sequence'
        assert bases, 'bases cannot be empty'
        self.bases = []
        self.values = []
        for base in bases:
            assert base > 0, 'base must be >0'
            assert isinstance(base, int), 'base must be int'
            self.bases.append(base)
            self.values.append(0)

        self.num_permutations = reduce(operator.mul, self.bases)

    def reset(self):
        self.values = [0 for val in self.bases]

    def currentValue(self):
        # Don't return original list, as caller could mutate it.
        return tuple(self.values)

    def increment(self, position=None):
        if position is None:
            position = len(self.bases) - 1
        if position == -1:
            raise ValueError('last combination already generated')
        self.values[position] += 1
        if self.values[position] == self.bases[position]:
            self.values[position] = 0
            self.increment(position - 1)

    def __iter__(self):
        return MultiBaseCounterIterator(self)

def rail(x, minX, maxX):
    """
    @description

        Returns a vector x2 where every element is an element of vector x that
        is constrained between minX and maxX.

        ie. x2[i]=max(minX[i], min(maxX[i], x[i])) for all i in [0..len(x))

    @arguments

        x -- vector -- the input vector
        minX -- vector -- the input min vector
        maxX -- vector -- the input max vector

    @return

        vector -- see Description

    @exceptions

    @notes

    """
    if len(x) != len(minX):
        raise ValueError('len(x) = %d and len(minX) = %d must be equal'
                         % (len(x), len(minX)))
    if len(x) != len(maxX):
        raise ValueError('len(x) = %d and len(maxX) = %d must be equal'
                         % (len(x), len(maxX)))

    # check that every element of minX is <= every element of maxX, otherwise
    # the equation below is nonsense.
    #
    # ie. minX = 1.0, maxX = -1.0, x = 0.0
    #     rail = max(minX, min(maxX, x)) = max(1.0, min(-1.0, 0.0)) = 1.0 (bad!)
    less_equal = numpy.less_equal(minX, maxX)
    if numpy.sum(less_equal) != len(less_equal):
        bad_indices = [i
                       for i in range(len(less_equal))
                       if less_equal[i] == 0]
        raise ValueError('minX[i] > maxX[i] at for indices = %s' % bad_indices)

    x2 = numpy.zeros(x.shape[0]) * 0.0
    for (i, (xi, mn, mx)) in enumerate(zip(x, minX, maxX)):
        x2[i] = max(mn, min(mx, xi))
    return x2


def removeDuplicateRows(X, I, costs=None):
    """
    @description

        From among possible rows I, return the indices of rows of X
        such that there are no duplicated rows.  When choosing between
        duplicate rows, choose on the basis of cost[row i].

    @arguments

        X: data set
        I: list of rows (indices) in X to be evaluated.

    @return

        new_I: list of indices of rows that do not contain a constant value.

    @exceptions

    @notes

    """
    # short circuit because we don't want to look at anything
    if len(I) == 0:
        return I

    # initialize default costs. Create a vector of 0's the same
    # length as the number of rows in X
    if costs is None:
        costs = [0] * X.shape[0]

    # Sort X, using the Comparator class(method), returning a list
    # of indices in I that access X in the desired sort order.
    I.sort(Comparator(X))
    new_I = []

    # find the duplicates in O(n)
    groups = [] # a list of (lists of duplicate rows (curr_group(s))
    curr_group = None # list of current indices containing duplicates
    # :AUDIT: range(len(I) - **2** ???) because of i_next?
    for index in range(len(I) - 1):
        i = I[index]
        i_next = I[index + 1]

        # if the entries are equal, add to a group
        # X[i] == X[i_next] is a piece-wise comparison of the entries in X
        # that returns a '1' if equal and a '0' if not.
        # sum(X[i] == X[i_next]) is a count of the number of equal entries
        # if the sum matches the length, then all entries are equal
        if sum(X[i] == X[i_next]) == len(X[i]):
            if curr_group is None:
                curr_group = []# create current group of duplicates
                groups.append(curr_group) # and add it to the master list
                curr_group.append(i) # include row i as the base case
            # and the next row since they are equal
            # and the next row every time it is equal in the future
            curr_group.append(i_next)

        # otherwise add to the list of non-duplicated rows
        else:
            if curr_group is not None:
                curr_group = None # must restart looking for duplicate rows
            else:
                new_I.append(i) # this is a non-duplicate row

    # add the last element if not in a duplicate group
    if curr_group is None or curr_group[-1:] != I[-1:]:
        new_I.append(I[-1:][0])

    # pick the duplicate with the least cost in O(n)
    # and add it to new_I
    for group in groups:
        costs_of_group = [costs[i] for i in group]
        best_i = group[numpy.argmin(costs_of_group)]
        new_I.append(best_i)

    # sort and return the new list of indices in O(n log n)
    new_I.sort()
    return new_I


class Comparator:
    """
    @description

        Class based implementation of an array comparison function.

    @arguments


    @return

        int: negative if x < y; 0 if x == y, else positive

    @exceptions

    @notes

    """
    def __init__(self, X):
        """Creates a local reference to the data set X.
        """
        self.X = X

    def __call__(self, x, y):
        dotxy = numpy.dot(self.X[x], self.X[y])
        dotxx = numpy.dot(self.X[x], self.X[x])
        dotyy = numpy.dot(self.X[y], self.X[y])
        if dotxy == dotxx and dotxy == dotyy:
            return 0
        else:
            # :AUDIT: duplicates line before the if statement
            # dotyy = numpy.dot(self.X[y], self.X[y])
            return cmp(dotxx, dotyy)

def normalizeVector(v, n=2):
    """ normalizes the vector to length 1 according to its n-norm """
    vv = numpy.array(v, dtype=float)
    vx = copy.copy(vv)
    for i in range(n-1):
        vx = vx*vv
    norm = (numpy.sum(vx))**(1/float(n))
    return vv / norm
