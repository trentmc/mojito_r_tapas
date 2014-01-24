"""Hypervolume.py

Measures hypervolume of a pareto front =  S-measure = Lebesque measure.

Supplies these specific routines:
-hypervolumeMinimize() -- use when trying to minimize objective functions
-hypervolumeMaximize() -- use when trying to maximize objective functions

Note: Non-exact version, using Monte Carlo sampling is O(n*d*N) where N is number
 of monte carlo samples.  Error drops off with sqrt(N).

See bottom of page for a commented-out fully fast deterministic version
that treats the approach as a Klee's metric problem. (Warning: buggy!)
"""

import copy
from itertools import izip
import logging
import math
import random
import types

import numpy

from constants import AGGR_TEST
from engine.EngineUtils import coststr
import mathutil
from util.octavecall import plotScatterAndPause

log = logging.getLogger('hypervolume')

def hypervolumeMinimize(pareto_front, r, use_fast_2d=True):
    """
      Like hypervolumeMaximize, except assumes we try to _minimize_ all.
    """
    (pareto_front2, r2) = _flipProblemSigns(pareto_front, r)
    return hypervolumeMaximize(pareto_front2, r2, use_fast_2d)
    
def hypervolumeMaximize(pareto_front, r, use_fast_2d=True, max_num_mc=5000):
    """
    @description

      Returns volume of the input pareto front.  Assumes we try to maximize all
      dimensions; therefore 'r' by definition has to be <= the lowest value
      of the pareto front, for each dimension.

      Does it by mixture importance sampling:
      -most of the time, it samples from a mixture of gaussians
       where each gaussian's center value is defined by a pareto front point
      -the remainder of the time, it samples from a uniform distribution

    @arguments

      pareto_front -- list of points that we want to calculate the volume of
      r -- point -- reference point
      use_fast_2d -- bool -- if the problem is 2d, then use a
        quick deterministic heuristic to do it.
      max_num_mc -- int -- when using monte carlo integration, this
        is the number of monte carlo samples

    @return

     volume -- float

    @exceptions

    @notes

      Each point is a list with 'd' entries; each entry is integer- or float-
      valued.
    """
    #calculate 'num_dim', 'pf_minx', 'maxx' _and_ preconditions
    # -'r' is the minimum of values the whole region, and 'pf_minx' is the minimum
    # values found in the pareto front.  'maxx' is for whole region, and
    # also for pareto front because it is calculated from that.
    assert len(pareto_front) > 0
    num_dim = len(r)

    for pareto_point in pareto_front:
        assert len(pareto_point) == num_dim
        assert _isNondominated(pareto_point, pareto_front)

    whole_minx = numpy.asarray(r)
    pf_minx = _minx(pareto_front)
    maxx = _maxx(pareto_front)
    log.info("whole_minx = %s" % coststr(list(whole_minx)))
    log.info("pf_minx = %s" % coststr(list(pf_minx)))
    log.info("maxx = %s" % coststr(list(maxx)))
    
    for (whole_mn, pf_mn, mx) in izip(whole_minx, pf_minx, maxx):
        assert whole_mn <= pf_mn <= mx, (whole_mn, pf_mn, mx)

    #corner case: can use fast 2d
    if use_fast_2d and num_dim == 2:
        return _hypervolumeMaximize2d(pareto_front, r)

    #corner case: there is no differentiation between pareto front
    # values for some dimensions.  So cut down the dimensionality.
    have_equal_dims = False
    for (pf_mn, mx) in izip(pf_minx, maxx):
        if pf_mn == mx:
            have_equal_dims = True
            break
    if have_equal_dims:
        reduced_pareto_front = [_reducedPoint(point, pf_minx, maxx)
                                for point in pareto_front]
        reduced_r = _reducedPoint(r, pf_minx, maxx)
        reduced_num_dim = len(reduced_r)
        if reduced_num_dim == 0:
            reduced_vol = 1.0
        elif reduced_num_dim == 1:
            reduced_maxx = _maxx(reduced_pareto_front)
            reduced_pf_minx = _minx(reduced_pareto_front)
            reduced_vol = (reduced_maxx[0] - reduced_pf_minx[0])
        else:
            reduced_vol = hypervolumeMaximize(
                reduced_pareto_front, reduced_r, use_fast_2d)
        
        vol = reduced_vol
        for (whole_mn, pf_mn, mx) in izip(whole_minx, pf_minx, maxx):
            if pf_mn == mx:
                vol *= (mx - whole_mn)
        return vol
        
    #corner case: pfcube is empty because it's defined by just one point
    # (note: _need_ to have this _after_ dimension-reduction corner case above)
    vol_pf_cube = numpy.product(maxx - pf_minx)
    vol_whole_cube = numpy.product(maxx - whole_minx)
    if vol_pf_cube == 0.0: 
        vol = vol_whole_cube
        return vol

    #main case.
    for (pf_mn, mx) in izip(pf_minx, maxx):
        assert pf_mn < mx

    #prepare for iterative monte carlo loop
    num_mc = 0
    num_dominated = 0
    #if AGGR_TEST:
    #    X = numpy.zeros((num_dim, max_num_mc), dtype=float)
    #    I_dominated = []

    #do iterative monte carlo loop
    while True:
        x = [random.uniform(pf_mn, mx)
             for (pf_mn, mx) in izip(pf_minx, maxx)]
        
        if _isDominated(x, pareto_front):
            num_dominated += 1
            
        num_mc += 1

        #if AGGR_TEST:
        #    X[:,num_mc-1] = x
        #    if x_dominated:
        #        I_dominated.append(num_mc-1)

        #on-the-fly calculation and output
        if ((num_mc % 500) == 0) and (num_mc > 0):
            vol = _volume(vol_whole_cube, vol_pf_cube, num_dominated, num_mc)
            log.info("Num_mc=%4d, current volume=%.6e" % (num_mc, vol))

        #maybe stop
        if num_mc >= max_num_mc:
            log.info("Stop: num_mc == max")
            break

    #if AGGR_TEST and num_dim == 2:
    #    pass
    #    #plotScatterAndPause(X, numpy.transpose(numpy.array(pareto_front)))
    #    #X_dominated = numpy.take(X, I_dominated, 1)
    #    #plotScatterAndPause(X, X_dominated)

    vol = _volume(vol_whole_cube, vol_pf_cube, num_dominated, num_mc)
    return vol

def _hypervolumeMaximize2d(pareto_front, r):
    """Fast, simple, reliable approach"""
    points = pareto_front
    #preconditions
    for point in points:
        assert len(point) == len(r)
        assert mathutil.allEntriesAreNumbers(point)
            
    #re-order points, in order of descending value in 0th dimension
    I = numpy.argsort([point[0] for point in points])
    points = [points[i] for i in I]

    #now count volume
    vol = 0.0
    num_points = len(points)
    for i in range(num_points):
        if i == 0: left = r[0]
        else:      left = points[i-1][0]
        right = points[i][0]
        lower = r[1]
        upper = points[i][1]
        
        assert left <= right
        assert lower <= upper
        vol += ((right - left) * (upper - lower))

    #postconditions
    assert mathutil.isNumber(vol)
    
    return vol

def _flipProblemSigns(pareto_front, r):
    """To go back and forth from hypervolumeMinimize <=> maximize,
    returns a flipped version of 'pareto_front' and 'r'
    """
    pareto_front2 = []
    for point in pareto_front:
        point2 = [-pi for pi in point]
        pareto_front2.append(point2)
    r2 = [-ri for ri in r]

    return (pareto_front2, r2)

def _atCorner(point, minx, maxx):
    """Returns True if point[i] == minx[i] or == maxx[i],
    for each i.
    """
    for (val, mn, mx) in izip(point, minx, maxx):
        if (val == mn) or (val == mx):
            pass
        else:
            return False
    return True

def _volume(vol_whole_cube, vol_pf_cube, num_dominated, num_mc):
    """Returns estimated volume (when maximizing objectives)
    """
    proportion_dominated = float(num_dominated) / num_mc
        
    vol_pf = vol_pf_cube * proportion_dominated
    vol = vol_whole_cube - vol_pf_cube + vol_pf 
    
    return vol

def _minx(points):
    """Returns a 1d array where entry i is the minimum value
    encountered in the ith dimension from among the points."""
    assert points
    num_dim = len(points[0])
    minx = numpy.array([min([point[var_j] for point in points])
                          for var_j in range(num_dim)])
    return minx

def _maxx(points):
    """Returns a 1d array where entry i is the maximum value
    encountered in the ith dimension from among the points."""
    assert points
    num_dim = len(points[0])
    maxx = numpy.array([max([point[var_j] for point in points])
                          for var_j in range(num_dim)])
    return maxx

def _isDominated(point, pareto_front):
    """Returns True if any of the points in 'pareto_front'
    dominate 'point' or are equal to it.
    We are trying to _maximize_ values in the point.
    """    
    for pareto_point in pareto_front:
        if _dominates(pareto_point, point):
            return True
    return False

def _isNondominated(point, pareto_front):
    """Returns True if none of the points in 'pareto_front'
    dominate 'point' or are equal to it.
    We are trying to _maximize_ values in the point.
    """    
    for pareto_point in pareto_front:
        if _dominates(pareto_point, point):
            return False

    for pareto_point in pareto_front:
        if _vecEqual(pareto_point, point):
            return True
        
    return True

def _reducedPoint(point, minx, maxx):
    """Returns a version of point, with entries wherever minx[i] != maxx[i]"""
    reduced_point = [val for (val, mn, mx) in izip(point, minx, maxx)
                     if mn != mx]
    return reduced_point

def _dominates(point1, point2):
    """Returns True if point1 dominates point2.
    We are trying to _maximize_ the values in points.

    In order for point1 to dominate:
    -we need to have found at least 1 dimension of point1 that is better
    -and all remaining dimensions of point2 have to be equivalent or better

    If any dimension of point1 is worse than point2, then we know
    immediately that point1 does not dominate point2
    """
    found_better = False
    for val1, val2 in izip(point1, point2):
        if val1 < val2:
            return False
        
        if val1 > val2:
            found_better = True

    return found_better

def _vecEqual(point1, point2):
    """Returns True only if point1 and point2 have same values.
    Works regardless of if each point is a list or 1d array
    """
    for (v1, v2) in izip(point1, point2):
        if v1 != v2:
            return False
    return True

def _inBounds(x, minx, maxx):
    """Returns true if minx[i] <= x[i] <= maxx[i] for each i"""
    assert len(x) == len(minx) == len(maxx)
    for (xi, minxi, maxxi) in izip(x, minx, maxx):
        if not (minxi <= xi <= maxxi):
            return False
    return True

# LOWER = 0
# UPPER = 1

# PRINT = False

# class KLEE_HypervolumeCalculator:
#     """
#     tlm -- WARNING -- this code is buggy!!!!!!  It is not giving correct
#     answers.
    
#     Uses the technique:

#     N. Beume and G. Rudolph, 'Faster S-metric Calculation by Considering Dominated
#     Hypervolume as Klee's Measure Problem', IASTED 2006.   On google search.

#     Worst-case complexity is: O(n*log(n) + n^(d/2)) where
#     n = number of datapoints
#     d = number of dimensions (objectives).

#     As of July 2007, this is the fastest-known approach for computing an
#     exact version of hypervolume.
#     """
    
#     def __init__(self, points, r):
#         #preconditions
#         for point in points:
#             assert len(point) == len(r)
#             assert mathutil.allEntriesAreNumbers(point)
            
#         self.n = len(points) #number of samples
#         self.d = len(r)      #full number of input dimensions 
#         self.r = r           # reference point

#     def volume(self, region, points, splitvar, cover_val, depth):
#         """
#         @description

#           Main workhorse routine for calculating volume.  

#         @arguments

#           region -- 2d array of [LOWER, UPPER][point index] --
#             contains the upper and lower bounds of the region
#           points -- list -- store the points whose induced cuboids partially or
#             completely cover 'region'
#           splitvar -- int  -- dimension at which 'region' is to cut in order to
#             generate two child regions
#           cover_val -- float -- this is the value of the dth coordinate of the
#             first cuboid that covers the parent node's region
#           depth -- int -- depth of recursion
          
#         @return

#          volume -- float

#         @exceptions

#         @notes

#           Like 'Algorithm 1', except indices here start at 0, not 1.
#         """
#         d = self.d
#         assert 1 <= splitvar <= d
        
#         #corner case
#         if self.n == 0:
#             return 0

#         #main case...
#         if depth == 0:
#             #sort points first according to dimension 'd' (only do this once)
#             I = numpy.argsort([point[d-1] for point in points])
#             points = [points[i] for i in I]

#         if PRINT:
#             print "\ncall volume with: region=%s, points=%s, splitvar=%s" \
#                   ", cover_val=%s, depth=%s" % \
#                   (region, points, splitvar, cover_val, depth)
                   
#         volume = 0

#         # is region completely covered?
#         # -also determine cover_index, new_cover_val
#         cover_index = 1 #follow algorithm values, not python values
#         new_cover_val = cover_val
#         while (new_cover_val == cover_val) and (cover_index != len(points)):
#             if self.pointFullyCoversRegion(points[cover_index-1], region):
#                 new_cover_val = points[cover_index-1][d-1]
#                 trel_vol = self.trellisVolume(points[cover_index-1][:d-1],region)
#                 volume += (trel_vol * (cover_val - new_cover_val))
#             else:
#                 cover_index += 1

#         if cover_index == 1:
#             if PRINT: print "1 return with volume=%d\n" % volume
#             return volume

#         # do the cuboids form a trellis?
#         # -if yes, then we can stop partitioning this space
#         # -recall: in a trellis, the cuboids that intersect the region
#         #  cover it completely in each of the d-1 dimensions except one
#         all_cuboids_are_piles = True
#         for point in points[:cover_index-1]:
#             if not self.isPile(point, region):
#                 all_cuboids_are_piles = False
#                 break

#         if all_cuboids_are_piles:
#             #yes, cuboids form a trellis, so compute volume of trellis and return
            
#             # calculate volume by sweeping along dimension d
#             #   -a trellis can be fully characterized by its lower boundaries
#             #    because upper boundaries are always r

#             #   -initilize trellis's lower bound to max values here; later on we
#             #    will update the lower bound as we progress...
#             trellis_lower = self.r[:d-1] 
#             i = 1 #this 'i' follows same values as paper (not python values)
#             while True: #for each segment of the trellis, calculate its volume...
#                 current_val = points[i-1][d-1]
#                 while True: #identify a new lower bound for each
                
#                     #update the trellis' lower bound at pile_var
#                     #  (pile_var is the sole dimension in 'region' that
#                     #   is not completely covered by points[i])
#                     pile_var = self.checkPile(points[i-1], region)
#                     assert pile_var != -1, "expected all cuboids to be piles"
#                     if points[i-1][pile_var-1] < trellis_lower[pile_var-1]:
#                         trellis_lower[pile_var-1] = points[i-1][pile_var-1]
#                     i += 1
#                     if i <= cover_index:#fix original paper
#                         next_val = points[i-1][d-1]
#                     else:
#                         next_val = new_cover_val
                    
#                     if current_val != next_val: break

#                 trel_vol = self.trellisVolume(trellis_lower, region) 
#                 volume += (trel_vol * (next_val - current_val))
#                 if PRINT:
#                     print "  append to volume with with region [%s,%s] in var %d"%\
#                           (current_val, next_val, pile_var)
#                 if next_val == new_cover_val: break

#             if PRINT: print "2 return with volume=%d\n" % volume
#             return volume

#         else:
#             #no, cuboids do not form a trellis, so split & recurse...
            
#             # split region into two children regions
#             # -find a good split boundary 'splitval'; change splitvar if needed
#             # -via detection of intersections i.e. finding which points
#             #  that induce boundaries
#             splitval = -1
#             if PRINT:
#                 print "  cover_index=%d, num_points=%d" % \
#                       (cover_index, len(points))
#             while True:
#                 if PRINT: print "  try splitvar = %d" % splitvar
#                 #holds boundaries for pts that induce a boundary for i<splitvar
#                 intersect_vals = []
#                 non_intersect_vals = []
#                 for point in points[:cover_index-1]:
#                     res = self.testIntersect(point, region, splitvar)
#                     if res == 1:   intersect_vals.append(point[splitvar-1])
#                     elif res == 0: non_intersect_vals.append(point[splitvar-1])

#                 if PRINT:
#                     print "  # int=%d, # nonint=%d, sqrt(n)=%g" % \
#                           (len(intersect_vals),
#                            len(non_intersect_vals), math.sqrt(self.n))

#                 if intersect_vals:
#                     splitval = mathutil.median(intersect_vals)
#                 elif (len(non_intersect_vals) > math.sqrt(self.n)):
#                     splitval = mathutil.median(non_intersect_vals)
#                 else:
#                     splitvar += 1

#                     if PRINT: print "  need to increment splitvar"
#                     assert 1 <= splitvar <= d
                    
#                 if splitval != -1: break

#             if PRINT:
#                 print "  choose to split region[%d] at boundary=%d" % \
#                       (splitvar, splitval)

#             # recurse on the two children regions
#             #  -child1 region = upper half of region split at (splitvar,splitval)
#             #  -note: don't bother creating new child 'region' data structure;
#             #   just change one entry in 'region' and restore it after recurse
#             old_val = region[UPPER][splitvar-1] #store
#             region[UPPER][splitvar-1] = splitval
#             pointsC = [point
#                        for point in points[:cover_index-1]
#                        if self.pointPartiallyCoversRegion(point, region)]
#             if pointsC:
#                 if PRINT: print "  recurse to left"
#                 volume += self.volume(region, pointsC, splitvar,
#                                         new_cover_val, depth+1)
#                 if PRINT: print "  return from recurse to left"
#             region[UPPER][splitvar-1] = old_val #restore

#             #  -child2 region = lower half of region split at (splitvar,splitval)
#             old_val = region[LOWER][splitvar-1] #store
#             region[LOWER][splitvar-1] = splitval 
#             pointsC = [point
#                        for point in points[:cover_index-1]
#                        if self.pointPartiallyCoversRegion(point, region)]
#             if pointsC:
#                 if PRINT: print "  recurse to right"
#                 volume += self.volume(region, pointsC, splitvar-1,
#                                       new_cover_val, depth+1)
#                 if PRINT: print "  return from recurse to right"
#             region[LOWER][splitvar-1] = old_val #restore

#             if PRINT: print "3 return with volume=%d\n\n" % volume
#             return volume

#     def pointPartiallyCoversRegion(self, point, region):
#         """Returns True if 'point' partially covers 'region'

#         'point' is points[i]
#         """
#         for var_j in range(1, self.d - 1 + 1):
#             if point[var_j-1] >= region[UPPER][var_j-1]:
#                 return False
#         return True

#     def pointFullyCoversRegion(self, point, region):
#         """Returns True if 'point' fully covers 'region'

#         'point' is points[i]
#         """
#         for var_j in range(1, self.d - 1 + 1):
#             if point[var_j-1] > region[LOWER][var_j-1]:
#                 return False
#         return True

#     def testIntersect(self, point, region, splitvar):
#         """Detects whether a point's split boundary is a candidate for the
#         splitting line that partitions the region into two child regions.  
#         """
#         assert 1 <= splitvar <= self.d

#         #not a candidate
#         if region[LOWER][splitvar-1] >= point[splitvar-1]:
#             return -1  

#         #is a candidate; provide further info on how to split
#         if splitvar >= 2:
#             import pdb; pdb.set_trace()
#         for var_j in range(1, splitvar - 1 + 1):
#             if point[var_j-1] > region[LOWER][var_j-1]:
#                 return 1

#         return 0

#     def isPile(self, point, region):
#         """Returns True if the cuboid induced by 'point' is a pile.
#         -An i-pile is a cuboid that does not cover the ith dimension completely,
#          i.e. a cuboid that only partially covers the ith dimension.
#         """
#         pile_var = self.checkPile(point, region)
#         return (pile_var != -1)

#     def checkPile(self, point, region):
#         """
#         Helper functoin to getPile() and isPile().

#         Recall that a cuboid is a pile w.r.t 'region' if it covers the
#         region completely in all but one dimensions. 

#         Behavior of this routine:
#           If the cuboid induced by the point is a pile, 
#             return the sole dimension that is not completely covered.
#           Else
#             returns -1 as an indicator of failure

#         """
#         pile_var = -1
#         for var_j in range(1, self.d - 1 + 1):
#             if (point[var_j-1] > region[LOWER][var_j-1]) and (pile_var != -1):
#                 return -1
#             pile_var = var_j
#         return pile_var

#     def trellisVolume(self, trellis_lower, region):
#         """
#         Calculates the (d-1)-dimensional volume formed by a trellis.

#         Calculates via the inclusion-exclusion principle:
#           -each summand is 
#           -the ith factor is either the size of the region in the dimension i,
#            or the value of a 1-dimensional KMP of the contained i-piles
#           -the sign of a summand depends on the number of KMP factors
#         -the following implementation uses an index vector 'indicator'
#          to determine whether a factor is to be a 1-dimensional KMP of the
#          i-piles, or the size of the region in dimension i.
#           -this way, all possible varaitions can be processed by assigning the
#            indicator vector the binary represenation of the numbers from 1 to q

#          -'trellis_lower' completely characterizes the trellis, because
#           its upper bound is defined by self.r.  trellis_lower[i] is therefore
#           the minimal ith coordinate of the i-piles.
#         """
#         volume = 0
#         indicator = [1]*(self.d - 1)
#         num_summands = mathutil.integerValue(indicator)
        
#         for summand_i in range(1, num_summands + 1):
#             #each summand is additively composed of (d-1) factors
#             # corresponding to the d-1 dimensions.
#             summand = 0
            
#             #indicator[summand_i] determines what the factor is:
#             # if 1: factor is a 1-dim KMP of the i-piles
#             # if 0: factor is the size of the region in dimension i
#             indicator = mathutil.binaryValue(summand_i, self.d-1)

#             #iteratively add each factor to the summand
#             one_counter = 0
#             for var_j in range(1, self.d - 1 + 1):
#                 if indicator[var_j-1] == 1:
#                     # factor is a 1-dimensional KMP of the i-piles
#                     factor = region[UPPER][var_j-1] - trellis_lower[var_j-1]
#                     summand += factor
#                     one_counter += 1
                    
#                 else:
#                     # factor is the size of the region in dimension i
#                     factor = region[UPPER][var_j-1] - region[LOWER][var_j-1]
#                     summand += factor

#             if (one_counter % 2) == 0:
#                 volume -= summand
#             else:
#                 volume += summand
        
#         return volume

