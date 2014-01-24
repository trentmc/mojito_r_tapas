""" Caff.py: Builds and simulates CAFFEINE models

References:
 1. McConaghy, Eeckelaert, and Gielen, 'CAFFEINE', DATE 2005
 1. Hornby, 'ALPS', GECCO 2006

Attributes of this version:
 -tree-based; children are a _set_ not a list
 -on-tree caching
 -regularized linear learning
 -has unity operator
 -for non-walter data, chooses training data by Korns' vertical slicing
 -generate random bases 1 at a time, and discard them immediately if poor
  (gives far higher probability of getting trees with >>1 bases)
  (makes huge difference)
-rational functions trick (makes huge difference)
-have biased generation of var_exprs: create all possibilities of
 {var_i} x {op_j} x {numerator,denominator}, then prune down with reglin;
 then those have the highest chance of being used as expr's in
 random gen of trees. (have higher bias for better-perf. ones too)
 (makes huge difference)
-add more (smooth) operators:
  -new crossover operator: copy a tree from one set into another tree's set
   (encourages growing)
  -add a simple basis function
  -copy a tree from one basis function, then mutate it (similar to MARS;
   encourages specialization
-have option of mobj vs. sobj (not sure if mobj helps!!)
-make initial popsize a multiplier x the usual popsize


BACKLOG:
-to note: simplest regressor has highest train nmse, but usually
 a _really_ good test nmse (often best)
   -perhaps we can _always_ just return 1 regressor with lowest test_nmse...
    (that's a valid flow for the designer!!)

-make a fraction of inds in initial generation = just 'good bases'
 plus a mutation or two

-run rfor first to identify most important variables AND variable
interactions (and use both in next linear regression step)

-post-evolution parameter tuning

-framework to calculate & remember test_nmse, including for whole PF
-benchmark on walter
-benchmark on MOS
-benchmark on larger circuit
"""
import copy
import logging
import math
import random
import types

import numpy

from LinearModel import LinearModelFactory, LinearBuildStrategy
import RegressorUtils 
from util import mathutil
from util.constants import INF
from util.Hypervolume import hypervolumeMaximize

log = logging.getLogger('caff')

#variable           num   depth(s)
SUM_SYMBOL         = 1   # 0  3
PRODUCT_SYMBOL     = 2   # 1  4
NONLIN_EXPR_SYMBOL = 3   # 2  /
NUMBER_SYMBOL      = 4   # 2  5
VAR_EXPR_SYMBOL    = 5   # 2  5

NONTERM_SYMBOLS = [SUM_SYMBOL, PRODUCT_SYMBOL, NONLIN_EXPR_SYMBOL]
TERM_SYMBOLS = [NUMBER_SYMBOL, VAR_EXPR_SYMBOL]

SUM_CHILD_SYMBOLS = [PRODUCT_SYMBOL]
PRODUCT_CHILD_SYMBOLS = [NONLIN_EXPR_SYMBOL, NUMBER_SYMBOL, VAR_EXPR_SYMBOL]

OP_NOP   = 1
OP_ABS   = 2
OP_UNITY = 3
OP_MAX0  = 4
OP_MIN0  = 5
OP_LOG10 = 6
OPS = [1, 2, 3, 4, 5, 6]

WEIGHT_BASE_RANGE = 10.0

class Placeholder:
    pass

def randomRationalAllocation():
    return RationalAllocation(random.choice([True, False]),
                              random.choice([True, False]))

class RationalAllocation:

    def __init__(self, assign_to_numerator, assign_to_denominator):
        assert isinstance(assign_to_numerator, types.BooleanType)
        assert isinstance(assign_to_denominator, types.BooleanType)
        self.assign_to_numerator = assign_to_numerator
        self.assign_to_denominator = assign_to_denominator

    def assignToNumerator(self):
        return self.assign_to_numerator

    def assignToDenominator(self):
        return self.assign_to_denominator

    def mutate(self):
        """Flip either the numerator or denominator assignment"""
        if random.random() < 0.5:
            self.assign_to_numerator = (not self.assign_to_numerator)
        else:
            self.assign_to_denominator = (not self.assign_to_denominator)

class Tree:

    _ones = None # = numpy.ones(len(y), dtype=float)
    _infs = None # = inf * ones
    _y_per_var = {} # [var_index][var_exponent] : y
    
    def __init__(self, symbol, data):
        #preconditions
        assert (symbol in NONTERM_SYMBOLS) or (symbol in TERM_SYMBOLS)

        #set attributes
        self.symbol = symbol

        # -the exact form of the data depends on what the symbol is
        # (see 'simulate()' for info)
        self.data = data
        
        self.is_terminal = (symbol in TERM_SYMBOLS)

        #cached_simdata is either None or (cached_y, id_of_X)
        self.cached_simdata = None

        #cached_complexity is either None or a dict of depth_ind : compl_float
        self.cached_complexity = None

    def __setstate__(self, state):
        """Override what gets done in a shallow copy (or in pickling)"""
        self.symbol = state['symbol']
        self.is_terminal = state['is_terminal']

        #cached_simdata is NOT what the state has, but rather is re-initalized
        # because there is a good chance that after a copy, more things will
        # be done to this
        self.cached_simdata = None

        self.cached_complexity = None

        #we need to shallow-copy the data rather than just get the reference,
        # because we have sets and tuples, and those may be changed, and we
        # don't want to accidentally change the originals
        state_data = state['data']
        if self.symbol in [SUM_SYMBOL, PRODUCT_SYMBOL]:
            self.data = (copy.copy(state_data[0]), copy.copy(state_data[1]))
        elif self.symbol in [NONLIN_EXPR_SYMBOL, NUMBER_SYMBOL,
                             VAR_EXPR_SYMBOL]:
            self.data = copy.copy(state_data)
        else:
            raise AssertionError('Unknown symbol')

    def __str__(self):
        """Override str()"""
        return self.str2(0)

    def str2(self, depth):
        """Helper for str, but has an extra argument of depth"""
        if self.is_terminal:
            if self.symbol == NUMBER_SYMBOL:
                #'data' is merely a scalar
                return '%g' % scale_w(self.data)
            
            elif self.symbol == VAR_EXPR_SYMBOL:
                #'data' is a tuple of (variable_index, variable_exponent)
                (var_i, var_exp) = self.data
                if var_exp < 0:
                    return 'x%d^(%g)' % (var_i, var_exp)
                elif var_exp == 1:
                    return 'x%d' % var_i
                else: #var_exp >= 0, except 1
                    return 'x%d^%g' % (var_i, var_exp)
                                
            else:
                raise AssertionError('Unknown terminal symbol')
            
        else: #nonterminal symbol
            if self.symbol == SUM_SYMBOL:
                #'data' is (set of PRODUCT_SYMBOL trees, Placeholder)
                s = []

                child_svec = []
                for child in self.children():
                    child_s = child.str2(depth + 1)
                    if child_s not in ['', '0.0', '0']:
                        child_svec.append(child_s)
                child_svec.sort() #to keep output order consistent
                
                s = []
                for (child_i, child_s) in enumerate(child_svec):
                    if depth == 0:
                        s += ['\n + %s' % (child_s)]
                    else:
                        s += ['%s' % (child_s)]
                        if child_i < (len(child_svec)-1):
                            s += [' + ']
                        
                    
                return ''.join(s)
            
            elif self.symbol == PRODUCT_SYMBOL:
                #'data' is (set of trees, RationalAllocation)
                # >=0 NUMBER_SYMBOL trees, >=0 VAR_EXPR_SYMBOL trees,
                # >=0 NONLIN_EXPR_SYMBOL trees
                data2 = []
                children = self.children()
                data2.extend([c for c in children
                              if c.symbol == NUMBER_SYMBOL])
                data2.extend([c for c in children
                              if c.symbol == VAR_EXPR_SYMBOL])
                data2.extend([c for c in children
                              if (c.symbol == NONLIN_EXPR_SYMBOL)
                              and (c.data[0] != OP_NOP)])

                child_svec = []
                for child in data2:
                    child_s = child.str2(depth + 1)
                    if child_s not in ['', '1.0', '1']:
                        child_svec.append(child_s)
                child_svec.sort() #to keep output order consistent
                
                s = []
                for (child_i, child_s) in enumerate(child_svec):
                    next_s = [child_s]
                    if child_i < (len(child_svec)-1):
                        next_s += [' * ']
                    s += [''.join(next_s)]
                return ''.join(s)

            elif self.symbol == NONLIN_EXPR_SYMBOL:
                #'data' is a tuple of (nonlin_op, SUM_SYMBOL tree, nonlin_exp)
                (op, sum_tree, op_exp) = self.data

                if op_exp == 0:
                    return '1.0'

                if op_exp < 0:
                    op_exp_str = '^(%g)' % op_exp
                elif op_exp == 1:
                    op_exp_str = ''
                else: #op_exp > 0 (except 1)
                    op_exp_str = '^%g' % op_exp

                sum_str = sum_tree.str2(depth + 1)
                if op == OP_NOP:
                    raise AssertionError('Should not get this deep with nop')
                elif op == OP_ABS:
                    return 'abs(%s)%s' % (sum_str, op_exp_str)
                elif op == OP_UNITY:
                    return '(%s)%s' % (sum_str, op_exp_str)
                elif op == OP_MAX0:
                    return 'max(0, %s)%s' % (sum_str, op_exp_str)
                elif op == OP_MIN0:
                    return 'min(0, %s)%s' % (sum_str, op_exp_str)
                elif op == OP_LOG10:
                    return 'log10(%s)%s' % (sum_str, op_exp_str)
                else:
                    raise 'Unknown op %d' % op
            else:
                raise AssertionError('Unknown nonterminal symbol')

    def complexity(self, X, cur_depth):
        """Returns a number describing self's complexity (recursive)
        """
        #calculate and store into cache if it is not there
        if (self.cached_complexity is None):
            self.cached_complexity = {}
        if not self.cached_complexity.has_key(cur_depth):
            complexity = self._calcComplexity(X, cur_depth)
            assert complexity is not None
            self.cached_complexity[cur_depth] = complexity

        #return value from cache
        return self.cached_complexity[cur_depth]
    
    def _calcComplexity(self, X, cur_depth):
        """Returns a number describing self's complexity (recursive)
        """
        if self.symbol == NUMBER_SYMBOL:
            return 1.0
            
        elif self.symbol == VAR_EXPR_SYMBOL:
            (var_i, var_exp) = self.data
            if var_exp == -1:     return 1.0
            elif var_exp < 0:     return 2.0
            elif var_exp == 0:    return 0.0 #whole expr has no effect
            elif var_exp == 1:    return 1.0
            else:                 return 1.5
        
        elif self.symbol == SUM_SYMBOL:
            if cur_depth == 0:
                raise AssertionError, "ind should have handled this"
#                 complexity = 0
#                 have_some_numerators = False
#                 have_some_denominators = False
#                 for child in self.children():
#                     child_complexity = child.complexity(X, cur_depth + 1)
#                     if child.data[1].assignToNumerator():
#                         complexity += child_complexity
#                         have_some_numerators = True
#                     if child.data[1].assignToDenominator():
#                         complexity += child_complexity
#                         have_some_denominators = True
#                 if have_some_numerators:
#                     complexity += 1.5
#                 if have_some_denominators:
#                     complexity += 3.0
            else:
                complexity = 0.0
                yhat = self.simulate(X)
                if min(yhat) == max(yhat) == 0.0:
                    pass
                else:
                    #recurse & sum
                    for product_child in self.children():
                        complexity += product_child.complexity(X, cur_depth+1)

            return complexity

        elif self.symbol == PRODUCT_SYMBOL:
            yhat = self.simulate(X)
            if min(yhat) == max(yhat) == 1.0:
                return 0.0
            else:
                #recurse & sum
                complexity = 0.0
                for expr_child in self.children():
                    complexity += expr_child.complexity(X, cur_depth+1)
                return complexity
        
        elif self.symbol == NONLIN_EXPR_SYMBOL:
            #'data' is a tuple of (nonlin_op, SUM_SYMBOL tree, nonlin_exp)
            (op, sum_tree, op_exp) = self.data

            complexity = 0.0
            
            if   op == OP_NOP:     return 0.0 #whole expr has no effecct
            elif op == OP_ABS:     complexity += 1.5
            elif op == OP_UNITY:   complexity += 0.0
            elif op == OP_MAX0:    complexity += 2.0
            elif op == OP_MIN0:    complexity += 2.0
            elif op == OP_LOG10:   complexity += 2.0
            else: raise 'Unknown op %d' % op
            
            if   op_exp == -1:     complexity += 1.0
            elif op_exp < 0:       complexity += 2.0
            elif op_exp == 0:      return 0.0 #whole expr has no effect
            elif op_exp == 1:      complexity += 0.0
            else:                  complexity += 1.5

            #recurse & sum
            complexity += sum_tree.complexity(X, cur_depth+1) 

            return complexity
        else:
            raise AssertionError('Unknown symbol')

    def validate(self, depth):
        """Recursively go down 'self' and raise an error if a
        self-inconsistency is detected"""

        if self.is_terminal:
            if self.symbol == NUMBER_SYMBOL:
                #'data' is merely a scalar
                assert mathutil.isNumber(self.data)
                assert depth == 2 or depth == 5
            
            elif self.symbol == VAR_EXPR_SYMBOL:
                #'data' is a tuple of (variable_index, variable_exponent)
                (var_i, var_exp) = self.data
                assert isinstance(var_i, types.IntType)
                assert mathutil.isNumber(var_exp)
                assert depth == 2 or depth == 5
                                
            else:
                raise AssertionError('Unknown terminal symbol')
            
        else: #nonterminal symbol
            if self.symbol == SUM_SYMBOL:
                #'data' is (set of PRODUCT_SYMBOL trees, Placeholder)
                assert type(self.data[0]) == type(set([]))
                assert isinstance(self.data[1], Placeholder)
                assert depth == 0 or depth == 3
                assert len(self.children()) >= 1
                for child in self.children():
                    assert child.symbol in SUM_CHILD_SYMBOLS
                    child.validate(depth + 1) #recurse
            
            elif self.symbol == PRODUCT_SYMBOL:
                #'data' is (set of trees, RationalAllocation)
                # >=0 NUMBER_SYMBOL trees, >=0 VAR_EXPR_SYMBOL trees,
                # >=0 NONLIN_EXPR_SYMBOL trees
                assert type(self.data[0]) == type(set([]))
                assert isinstance(self.data[1], RationalAllocation)
                assert depth == 1 or depth == 4
                for child in self.children():
                    assert child.symbol in PRODUCT_CHILD_SYMBOLS, child.symbol
                    child.validate(depth + 1) #recurse
                    
            elif self.symbol == NONLIN_EXPR_SYMBOL:
                #'data' is a tuple of (nonlin_op, SUM_SYMBOL tree, nonlin_exp)
                (op, sum_tree, op_exp) = self.data
                assert op in OPS
                assert sum_tree.symbol == SUM_SYMBOL
                assert mathutil.isNumber(op_exp)
                assert depth == 2, depth
                sum_tree.validate(depth + 1) #recurse
                
            else:
                raise AssertionError('Unknown nonterminal symbol')


    def replaceChild(self, old_child, new_child):
        """Replace the subtree 'old_child' with subtree 'new_child';
        and set new_child's parent to 'self'"""
        assert not self.is_terminal
        assert isinstance(new_child, Tree)
        
        if (self.symbol == SUM_SYMBOL):
            children = self.children()
            children.remove(old_child)
            children.add(new_child)

        elif (self.symbol == PRODUCT_SYMBOL):
            children = self.children()
            assert old_child in children
            children.remove(old_child)
            children.add(new_child)
            
        elif self.symbol == NONLIN_EXPR_SYMBOL:
            assert old_child == self.data[1]
            self.data = (self.data[0], new_child, self.data[2])
            
        else:
            raise AssertionError("unknown symbol")
        
    def addChild(self, new_child):
        """Add new_child as one of self's children"""
        assert isinstance(new_child, Tree)
        
        if self.symbol == SUM_SYMBOL:
            assert new_child.symbol in SUM_CHILD_SYMBOLS
            self.children().add(new_child)
            
        elif self.symbol == PRODUCT_SYMBOL:
            assert new_child.symbol in PRODUCT_CHILD_SYMBOLS
            self.children().add(new_child)
            
        elif self.symbol in [NONLIN_EXPR_SYMBOL, NUMBER_SYMBOL,
                             VAR_EXPR_SYMBOL]:
            raise ValueError('only works for sum and product symbols')
            
        else:
            raise AssertionError("unknown symbol: %s" % self.symbol)
        

    def children(self):
        """Return a set of children"""
        if self.is_terminal:
            return set([])
        elif (self.symbol == SUM_SYMBOL):
            return self.data[0]
        elif (self.symbol == PRODUCT_SYMBOL):
            return self.data[0]
        elif self.symbol == NONLIN_EXPR_SYMBOL:
            (dummy, sum_tree, dummy) = self.data
            return set([sum_tree])
        else:
            raise AssertionError("unknown symbol")

    def childOfId(self, target_id):
        """Returns the child with id 'target_id'; raises error if not found.
        """
        for child in self.children():
            if id(child) == target_id:
                return child
        raise ValueError('child with target_id not found')

    def removeChild(self, child_to_remove):
        assert isinstance(child_to_remove, Tree)
        
        if self.is_terminal:
            raise ValueError
        elif (self.symbol == SUM_SYMBOL):
            self.children().remove(child_to_remove)
        elif (self.symbol == PRODUCT_SYMBOL):
            self.children().remove(child_to_remove)
        elif self.symbol == NONLIN_EXPR_SYMBOL:
            raise ValueError
        else:
            raise AssertionError("unknown symbol")

    def pathsToTarget(self, target_subtree, target_depth, cur_depth):
        """Returns a list of paths, where each path is a
        list of Tree nodes which are the path from self to target subtree.
        (There can be 0, 1, or more occurrences of target_subtree in self).
        In each path, the first entry will be root_tree, and
        last entry will be target_subtree"""
        if self == target_subtree:
            paths = [[self]]
        elif (cur_depth + 1 > target_depth):
            paths = []
        else:
            paths = []
            for child in self.children():
                sub_paths = child.pathsToTarget(target_subtree, target_depth,
                                                cur_depth + 1)
                for sub_path in sub_paths:
                    path = [self] + sub_path
                    paths.append(path)
                
        return paths

    def subtreeInfo(self, cur_depth):
        """Return a list of (subtree reference, depth).
        Recursively called.
        """
        tuples = []

        #info of 'self'
        tuples.append((self, cur_depth))

        #recurse with info for each child
        for child in self.children():
            tuples.extend(child.subtreeInfo(cur_depth + 1))

        return tuples

    def isTall(self):
        """Returns True if this tree is 'tall',
        i.e. have significant distance from self to bottom of tree.
        The basic test is: is there a NONLIN_EXPR_SYMBOL from self to bottom?
        """
        if self.symbol == SUM_SYMBOL:
            for child in self.children():
                if child.isTall():
                    return True
            return False
        elif self.symbol == PRODUCT_SYMBOL:
            for child in self.children():
                if child.isTall():
                    return True
            return False
        elif self.symbol == NONLIN_EXPR_SYMBOL:
            return True
        elif self.symbol in [NUMBER_SYMBOL, VAR_EXPR_SYMBOL]:
            return False
        else:
            raise AssertionError('unknown symbol')

    def clearCachedData(self):
        """Top-down clear self.cached_simdata"""
        self.cached_simdata = None
        self.cached_complexity = None
        for child in self.children(): #recurse
            child.clearCachedData()

        self.__class__._ones = None 
        self.__class__._infs = None 
        self.__class__._y_per_var = {} 

    def simulateBases(self, X):
        """
        @description

          Simulate each child of 'self' and return its output.
          This is typically only called for top-level trees.
          
        @arguments

          X -- 2d array of [# input vars][# samples] -- inputs

        @return

          numerator_ys -- dict of id(base) : 1d array of [# samples] --
            output per base that's in numerator
          denominator_ys -- dict of id(base) : 1d array of [# samples] --
            output per base that's in denominator

        @exceptions

          Only works if 'self' is a SUM_SYMBOL

        @notes

          There may be >= 0 entries for numerator_ys.
          There may be >= 0 entries for denominator_ys.
          numerator_ys and denominator_ys may have duplicate entries.
        """
        assert self.symbol == SUM_SYMBOL
        (numerator_ids, denominator_ids) = self.rationalAllocations()
        numerator_ys, denominator_ys = {}, {}
        for numerator_id in numerator_ids:
            child = self.childOfId(numerator_id)
            numerator_ys[id(child)] = child.simulate(X)
        for denominator_id in denominator_ids:
            child = self.childOfId(denominator_id)
            denominator_ys[id(child)] = child.simulate(X)

        return (numerator_ys, denominator_ys)

    def rationalAllocations(self):
        """Returns (sorted_list_of_numerator_tree_ids,
                    sorted_list_of_denominator_tree_ids)
        """
        assert self.symbol == SUM_SYMBOL
        numerator_ids = sorted([id(child)
                                for child in self.children()
                                if child.data[1].assignToNumerator()])
        denominator_ids = sorted([id(child)
                                  for child in self.children()
                                  if child.data[1].assignToDenominator()])
        return (numerator_ids, denominator_ids)
        
    def simulate(self, X):
        """Given a 2d array 'X' where rows are variables and columns
        are datapoints, simulate 'self' to return a 1d array 'y' which
        has one entry for each datapoint.
        """
        
        #use cached data if available (and same 'X')
        if (self.cached_simdata is not None):
            (cached_y, cached_id_X) = self.cached_simdata
            if (id(X) == cached_id_X):
                return cached_y
            else:
                self.clearCachedData()

        #compute y; cache it then return it
        if self.__class__._ones is None:
            self.__class__._ones = numpy.ones(X.shape[1], dtype=float)
        if self.__class__._infs is None:
            self.__class__._infs = INF * self.__class__._ones

        # -first, compute...
        if self.is_terminal:
            if self.symbol == NUMBER_SYMBOL:
                #'data' is merely a scalar
                #(note: we usually avoid getting here)
                y = scale_w(self.data) * self.__class__._ones
                
            elif self.symbol == VAR_EXPR_SYMBOL:
                #'data' is a tuple of (variable_index, variable_exponent)
                (var_i, var_exp) = self.data

                # -fill in cache as needed
                y_per_var = self.__class__._y_per_var
                if not y_per_var.has_key(var_i):
                    y_per_var[var_i] = {}
                if not y_per_var[var_i].has_key(var_exp):
                    Xi = X[var_i,:]

                    #safeguard against: sqrt() on negative numbers; 1/x on 0.0
                    ok = True
                    if var_exp == 0:
                        pass
                    elif (min(Xi)<0.0) and (abs(var_exp) in [0.5, 1.5, 2.5]):
                        ok = False
                    elif (0.0 in Xi) and (var_exp < 0):
                        ok = False

                    if ok:
                        if var_exp == 0:   
                            y_per_var[var_i][var_exp] = self.__class__._ones
                        elif var_exp == 1: 
                            y_per_var[var_i][var_exp] = Xi
                        else:
                            y_per_var[var_i][var_exp] = Xi ** var_exp
                    else:
                        y_per_var[var_i][var_exp] = self.__class__._infs

                # -retrieve from cache
                y = y_per_var[var_i][var_exp]
                    
            else:
                raise AssertionError('Unknown terminal symbol')
            
        else: #nonterminal symbol
            if self.symbol == SUM_SYMBOL:
                #'data' is (set of PRODUCT_SYMBOL trees, Placeholder)
                children = self.children()
                assert len(children) >= 1
                y = numpy.sum([child.simulate(X) for child in children])
                
            elif self.symbol == PRODUCT_SYMBOL:
                #'data' is (set of trees, RationalAllocation)
                # >=0 NUMBER_SYMBOL trees, >=0 VAR_EXPR_SYMBOL trees,
                # >=0 NONLIN_EXPR_SYMBOL trees

                # -determine number_product, and children2
                number_product = 1.0
                children2 = [] # we'll be recursing (simulating) on just these
                for child in self.children():
                    if child.symbol == NUMBER_SYMBOL:
                        number_product *= scale_w(child.data)
                    elif (child.symbol == VAR_EXPR_SYMBOL) and \
                             (child.data[1] == 0.0):
                        pass #ignore variables with exponents of zero
                    elif (child.symbol == NONLIN_EXPR_SYMBOL) and \
                             (child.data[2] == 0.0):
                        pass #ignore nonlin expr's with exponents of zero
                    elif (child.symbol == NONLIN_EXPR_SYMBOL) and \
                             (child.data[0] == OP_NOP):
                        pass #ignore nonlin expr's that have nop
                    else:
                        #do want to simulate this child
                        children2.append(child)

                if len(children2) == 0:
                    y = number_product * self.__class__._ones
                elif len(children2) == 1:
                    y = number_product * children2[0].simulate(X)
                else:
                    y = number_product * numpy.product(
                        [child.simulate(X) for child in children2])
                
            elif self.symbol == NONLIN_EXPR_SYMBOL:
                #'data' is a tuple of (nonlin_op, SUM_SYMBOL tree, nonlin_exp)
                (op, sum_tree, op_exp) = self.data
                y_sum_tree = sum_tree.simulate(X)
                    
                if (op == OP_NOP) or (op_exp == 0):
                    y = self.__class__._ones

                elif op in [OP_ABS, OP_UNITY, OP_MAX0, OP_MIN0, OP_LOG10]:
                    ok = True
                    if op == OP_ABS:
                        ya = abs(y_sum_tree)
                    elif op == OP_UNITY:
                        ya = y_sum_tree
                    elif op == OP_MAX0:
                        ya = numpy.array([max(0.0, yi) for yi in y_sum_tree])
                    elif op == OP_MIN0:
                        ya = numpy.array([min(0.0, yi) for yi in y_sum_tree])
                    elif op == OP_LOG10:
                        #safeguard against: log() on values <= 0.0
                        if ((min(y_sum_tree) <= 0.0) or mathutil.hasNanOrInf(y_sum_tree)):
                            ok = False
                        else:
                            ya = numpy.array([math.log10(yi) for yi in y_sum_tree])

                    #safeguard against: sqrt() on negative numbers; 1/x on 0.0
                    if ok:
                        if (min(ya)<0.0) and (abs(op_exp) in [0.5, 1.5, 2.5]):
                            ok = False
                        elif (0.0 in ya) and (op_exp < 0):
                            ok = False

                    if ok: #could always do ** exp, but faster ways if exp is 0,1
                        if op_exp == 0:
                            y = self.__class__._ones #already handled
                        elif op_exp == 1:
                            y = ya
                        else:
                            y = ya ** op_exp
                    else:
                        y = self.__class__._infs

                else:
                    raise 'Unknown op %d' % op
            else:
                raise AssertionError('Unknown nonterminal symbol')

        #  -cache y
        self.cached_simdata = (y, id(X))

        #  -postconditions
        assert len(y) == X.shape[1]
        
        #  -return y
        return y

    def hasSpace(self, ss, depth):
        """Does this tree have space for more children?
        """
        num_children = len(self.children())
        if self.symbol == SUM_SYMBOL:
            if depth == 0:
                return num_children < ss.max_num_shallow_sums
            else:
                return num_children < ss.max_num_deep_sums
            
        elif self.symbol == PRODUCT_SYMBOL:
            if depth == 1:
                return num_children < ss.max_num_shallow_product_exprs
            else:
                return num_children < ss.max_num_deep_product_exprs

        elif self.symbol in [NONLIN_EXPR_SYMBOL, NUMBER_SYMBOL,
                             VAR_EXPR_SYMBOL]:
            return False

        else:
            raise AssertionError('unknown symbol')
                

    def randomGrow(self, ss, nvars, cur_depth, X=None):
        """Randomly fill in data (of children), if needed"""
        if self.is_terminal:
            if self.symbol == NUMBER_SYMBOL:
                #'data' is merely a scalar
                self.data = random.uniform(ss.min_w, ss.max_w)
                
            elif self.symbol == VAR_EXPR_SYMBOL:
                #'data' is a tuple of (variable_index, variable_exponent)
                if (random.random() < ss.prob_using_useful_expr) and \
                   (ss.useful_var_exprs):
                    self.data = mathutil.randIndexFromDict(
                        ss.useful_var_exprs).data
                else:
                    self.data = (random.randint(0, nvars - 1),
                                 random.choice(ss.op_exponents))
            else:
                raise AssertionError('Unknown terminal symbol')

        else:
            if self.symbol == SUM_SYMBOL:
                #'data' is (set of PRODUCT_SYMBOL trees, Placeholder)
                children = set([])
                
                #Therefore: make >= 1 children a PRODUCT_SYMBOL tree
                if cur_depth == 0:
                    num_children = random.randint(1, ss.max_num_shallow_sums)
                elif cur_depth == 3:
                    num_children = random.randint(1, ss.max_num_deep_sums)
                else:
                    raise AssertionError('unexpected depth: %d' % cur_depth)
                
                for child_i in range(num_children):
                    if X is None:
                        child = Tree(PRODUCT_SYMBOL, None)
                        child.randomGrow(ss, nvars, cur_depth + 1)
                    else:
                        #Repeat building a child tree until it is ok.  X is
                        # non-None typically on depth==0 SUM symbols; therefore
                        # we repeat building each base until it's good;
                        # therefore the probability of building a good list
                        # of bases is _way_ higher than before
                        while True:
                            child = Tree(PRODUCT_SYMBOL, None)
                            child.randomGrow(ss, nvars, cur_depth + 1)
                            yhat_child = child.simulate(X)
                            if not yIsPoor(yhat_child):
                                break
                        
                    children.add(child)

                #to enhance evolvability, add in a '+ 0.0'
                if (cur_depth == 3) and (len(children) < ss.max_num_deep_sums):
                    children.add(productTreeWithZeroValue())

                assert len(children) >= 1
                self.data = (children, Placeholder())
                
            elif self.symbol == PRODUCT_SYMBOL:
                #'data' is (set of trees, RationalAllocation)
                # >=0 NUMBER_SYMBOL trees, >=0 VAR_EXPR_SYMBOL trees,
                # >=0 NONLIN_EXPR_SYMBOL trees
                children = set([])
                
                #Therefore:
                if cur_depth == 4:
                    #-make one child a NUMBER_SYMBOL tree
                    child = Tree(NUMBER_SYMBOL, None)
                    child.randomGrow(ss, nvars, cur_depth + 1)
                    children.add(child)

                #-make >= 0 children a VAR_EXPR_SYMBOL tree
                if cur_depth == 1:
                    num_var_exp = mathutil.randIndex(
                        ss.num_var_exprs_in_shallow_product_biases)
                else:
                    num_var_exp = mathutil.randIndex(
                        ss.num_var_exprs_in_deep_product_biases)
                for child_i in range(num_var_exp):
                    child = Tree(VAR_EXPR_SYMBOL, None)
                    child.randomGrow(ss, nvars, cur_depth + 1)
                    children.add(child)

                #-make >= 0 children a NONLIN_EXPR_SYMBOL tree
                if cur_depth == 1:
                    num_nonlin_expr = mathutil.randIndex(
                        ss.num_nonlin_exprs_in_shallow_product_biases)
                else:
                    #not a magic number below; this keeps trees interpretable
                    # and stops infinite recursion
                    num_nonlin_expr = 0 
                    
                for child_i in range(num_nonlin_expr):
                    if (random.random() < ss.prob_using_useful_expr) and \
                           (ss.useful_nonlin_exprs):
                        child = mathutil.randIndexFromDict(
                            ss.useful_nonlin_exprs)
                    else:
                        child = Tree(NONLIN_EXPR_SYMBOL, None)
                        child.randomGrow(ss, nvars, cur_depth + 1)
                    children.add(child)
                    
                self.data = (children, randomRationalAllocation())

            elif self.symbol == NONLIN_EXPR_SYMBOL:
                #'data' is a tuple of (nonlin_op, sum_tree, nonlin_exp)
                op = random.choice(OPS)
                op_exp = random.choice(ss.op_exponents)
                child = Tree(SUM_SYMBOL, None)
                child.randomGrow(ss, nvars, cur_depth + 1)
                
                self.data = (op, child, op_exp)
                
            else:
                raise AssertionError('Unknown nonterminal symbol')


def productTreeWithZeroValue():
    """Returns a product Tree that returns 0.0
    """
    depth2_data = 0.0 #'data' is merely a scalar
    depth2_tree = Tree(NUMBER_SYMBOL, depth2_data)
    
    depth1_data = (set([depth2_tree]), RationalAllocation(True, False))
    depth1_tree = Tree(PRODUCT_SYMBOL, depth1_data)

    depth1_tree.validate(1)

    return depth1_tree

class CaffInd(object):
    def __init__(self, tree, varnames):
        assert isinstance(tree, Tree), tree.__class__
        tree.validate(0)

        #root tree node.  All other nodes are attached to it.
        self.tree = tree

        self.varnames = varnames

        #if an ind has been evaluated:
        # -nmse, complexity, bases are always non-None
        # -lin_model is non-None or None, depending on lin learning success
        self.nmse = None
        self.complexity = None
        self.lin_model = None

    def __cost1(self):
        return self.nmse
    cost1 = property(__cost1)
    
    def __cost2(self):
        return self.complexity
    cost2 = property(__cost2)

    def calcComplexity(self, X):
        #corner case
        if self.lin_model is None:
            log.warning('Since lin_model was None, gave complexity of HUGE')
            return 10000000.0

        #main case...
        complexity = 0.0
        (numerator_ids, denominator_ids) = self.tree.rationalAllocations()
        child_depth = 1
        have_some_numerators = False
        have_some_denominators = False
        
        for (i, id) in enumerate(numerator_ids):
            coef = self.lin_model.coefs[1 + i]
            if coef != 0.0:
                child = self.tree.childOfId(id)
                complexity += child.complexity(X, child_depth)
                have_some_numerators = True

        for (i, id) in enumerate(denominator_ids):
            coef = self.lin_model.coefs[1 + len(numerator_ids) + i]
            if coef != 0.0:
                child = self.tree.childOfId(id)
                complexity += child.complexity(X, child_depth)
                have_some_denominators = True
                
        if have_some_numerators:
            complexity += 1.5
        if have_some_denominators:
            complexity += 3.0

        #done
        return complexity


    def isPoor(self):
        return (self.nmse == INF)

    def dominates(self, other_ind):
        """Returns True if self dominates ind2;
        i.e. if self is at least as good as ind2 in all the goals,
        and if self is better in at least one goal."""
        #preconditions
        self_cost1, self_cost2 = self.cost1, self.cost2
        other_cost1, other_cost2 = other_ind.cost1, other_ind.cost2
        assert (self_cost1 is not None)
        assert (self_cost2 is not None)
        assert (other_cost1 is not None)
        assert (other_cost2 is not None)

        #corner case
        if self.isPoor():
            return False

        #main work
        found_better = (self_cost1 < other_cost1) or \
                       (self_cost2 < other_cost2)
        
        at_least_equal = (self_cost1 <= other_cost1) and \
                         (self_cost2 <= other_cost2)
        
        return (found_better and at_least_equal)

    def simulate(self, X):

        #corner case: couldn't build a linear model
        if self.lin_model is None:
            log.warning('Ind.lin_model was None, so returning y-output of Infs')
            import pdb; pdb.set_trace()
            yhat = INF * numpy.ones(X.shape[1], dtype=float)

        #main case
        else:
            #compute each tree's data
            (numerator_ys, denominator_ys) = self.tree.simulateBases(X)
            (numerator_ids, denominator_ids) = self.tree.rationalAllocations()

            N = X.shape[1]

            #compute numerator_yhat
            numerator_yhat = numpy.zeros(N, dtype=float)
            numerator_yhat += self.lin_model.coefs[0]
            for (i, id) in enumerate(numerator_ids):
                coef = self.lin_model.coefs[1 + i]
                numerator_yhat += coef * numerator_ys[id]

            #compute denominator_yhat
            denominator_yhat = numpy.zeros(N, dtype=float)
            denominator_yhat += 1.0
            for (i, id) in enumerate(denominator_ids):
                coef = self.lin_model.coefs[1 + len(numerator_ids) + i]
                denominator_yhat += coef * denominator_ys[id]

            #compute yhat (including avoiding divide-by-zero)
            tiny_number = 1.0e-20
            for i, val in enumerate(denominator_yhat):
                if abs(val) < tiny_number:
                    denominator_yhat[i] = tiny_number

            yhat = numerator_yhat / denominator_yhat

        #done
        return yhat

    def clearCachedData(self):
        """Clears each tree and sub-tree's cached data, and also
        other data that is set when evaluating an ind: nmse, lin_model, bases
        """
        self.tree.clearCachedData()
        self.nmse = None
        self.complexity = None
        self.lin_model = None
    
    def __str__(self):
        return self.str2(True)
        
    def str2(self, use_true_varnames):
        s = 'CaffInd: '

        s += 'nmse=%s' % self.nmse
        s += '; complexity=%s' % self.complexity
        if self.nmse is None:
            s += '; no lin learning done yet'
            coefs = None
        elif self.lin_model is None:
            s += '; was unsuccessful in linear learning'
            coefs = None
        else:
            coefs = self.lin_model.coefs
            
        s += '; model=\n'

        (numerator_ids, denominator_ids) = self.tree.rationalAllocations()

        #print numerator
        s += '(\n'
        if coefs: coef = coefs[0]
        else:     coef = None
        s += ' ' + coefStr(coef)
        s += '\n'
        for (i, id) in enumerate(numerator_ids):
            if coefs: coef = coefs[1 + i]
            else:     coef = None
            s += ' ' + coefStr(coef)
            s += ' * %s' % self.tree.childOfId(id)
            s += '\n'
        s += ')\n'

        #print denominator
        if denominator_ids:
            s += '-------------------------------------------\n'
            s += '(\n'
            s += ' + 1.0'
            s += '\n'
            for (i, id) in enumerate(denominator_ids):
                if coefs: coef = self.lin_model.coefs[1 + len(numerator_ids) + i]
                else:     coef = None
                s += ' ' + coefStr(coef)
                s += ' * %s' % self.tree.childOfId(id)
                s += '\n'
            s += ')\n'

        #replace the 'x1', 'x2', ... ?
        if use_true_varnames:
            for var_i in range(len(self.varnames)-1, -1, -1):
                old_name = 'x%d' % var_i
                new_name = self.varnames[var_i]
                s = s.replace(old_name, new_name)
            
        return s


def coefStr(coef):
    if coef is None:
        return 'None'
    elif coef >= 0.0:
        return '+ %g' % coef
    else:
        return '- %g' % abs(coef)

class CaffBuildStrategy(object):
    """Holds magic numbers related to building a caffeine model.
    """
    
    def __init__(self,
                 do_mobj = False,
                 max_num_nonlinear_bases = 15,
                 popsize = 1000,
                 init_generation_multiplier = 2, 
                 max_numgen = 100,
                 target_nmse = 0.00):
        self.do_mobj = do_mobj
        
        self.popsize = popsize

        #initial population has size: popsize * this value
        self.init_generation_multiplier = init_generation_multiplier

        #upper bound on the num nondominated inds.
        # Ideally this would be inf, but smaller sizes can give big time savings
        self.max_num_nondom = 200
        self.max_numgen = max_numgen
        self.target_nmse = target_nmse

        self.show_all = False #show all zero-weighted parts when printing

        self.op_exponents = [-2.0, -1.5, -1.0, -0.5, 0.0,
                             +0.5, +1.0, +1.5, +2.0]
        
        self.min_w = -2.0 * WEIGHT_BASE_RANGE
        self.max_w = +2.0 * WEIGHT_BASE_RANGE

        self.max_num_shallow_sums = max_num_nonlinear_bases
        self.max_num_deep_sums = 3

        self.num_var_exprs_in_shallow_product_biases = [5.0, 10.0, 5.0, 1.0]
        self.num_var_exprs_in_deep_product_biases = [10.0, 10.0, 1.0]
        
        self.num_nonlin_exprs_in_shallow_product_biases = [10.0, 3.0, 1.0]
        #max_num_nonlin_exprs_in_deep_product == 0 always

        #max_num_numbers_in_shallow_product = 0
        #max_num_numbers_in_deep_product = 1

        self.max_num_shallow_product_exprs = len(self.num_var_exprs_in_shallow_product_biases) + len(self.num_nonlin_exprs_in_shallow_product_biases) + 0
        self.max_num_deep_product_exprs = len(self.num_var_exprs_in_deep_product_biases) + 0 + 1

        self.prob_mutate = 0.5

        #If non-empty, these are used to bias random tree creation
        self.useful_var_exprs = {}  #dict of bias : Tree        
        self.useful_nonlin_exprs = {} # dict of bias : Tree
        self.prob_using_useful_expr = 0.80

        #for all linear learning
        self.lin_ss = LinearBuildStrategy(
            y_transforms=['lin'], target_nmse=target_nmse, regularize=True)
        
        self.lin_ss.reg.thr = 0.90 #prunes more aggressively as it approaches 1

    def probCrossover(self):
        return 1.0 - self.prob_mutate

    def __str__(self):
        s = "CaffBuildStrategy={"
        s +=  'do_mobj=%s' % self.do_mobj
        s += ' popsize=%d' % self.popsize
        s += '; init_generation_multiplier=%d' % self.init_generation_multiplier
        s += '; max_num_nondom=%d' % self.max_num_nondom
        s += '; max_numgen=%d' % self.max_numgen
        s += '; target_nmse=%.2e' % self.target_nmse
        
        s += '; op_exponents=%s' % self.op_exponents
        s += '; min_w = %.2f' % self.min_w
        s += '; max_w = %.2f' % self.max_w
        
        s += '; max_num_shallow_sums = %d (= max num nonlinear bases)' % \
             self.max_num_shallow_sums
        s += '; max_num_deep_sums = %d' % \
             self.max_num_deep_sums
        s += '; num_var_exprs_in_shallow_product_biases = %s' % \
             self.num_var_exprs_in_shallow_product_biases
        s += '; num_var_exprs_in_deep_product_biases = %s' % \
             self.num_var_exprs_in_deep_product_biases
        s += '; num_nonlin_exprs_in_shallow_product_biases = %s' % \
             self.num_nonlin_exprs_in_shallow_product_biases

        s += '; prob_mutate = %.2f' % self.prob_mutate
        s += '; prob_crossover = %.2f' % self.probCrossover()
        
        s += '; prob_using_useful_expr = %.2f' % self.prob_using_useful_expr
        
        s += '; lin_ss = %s' % self.lin_ss
        s += ' /CaffBuildStrategy}'
        return s



class CaffState:
    def __init__(self, ss):
        self.ss = ss #CaffBuildStrategy
        self.R = None #list of inds.  len = 2*popsize.
         
        self.best_ind = None #ind
        self.nondominated_inds = None #list of inds
        
        self.gen = 0
        
        self.do_stop = False #bool
        
    def updateBest(self, inds):
        """Updates self.best_ind, and self.state.nondominated_inds.

        self.best_ind is the one with the lowest cost in goal1.
        """
        #update self.nondominated_ind
        # -have a good starting state
        if self.nondominated_inds is None:
            self.nondominated_inds = []

        # -add newest to nondom
        self.nondominated_inds = uniqueNondominatedInds(
            self.nondominated_inds + inds)

        # -cluster down nondom
        self.nondominated_inds = self._clusteredNondominatedInds(
            self.ss.max_num_nondom)

        #update self.best_ind
        self.best_ind = self.nondominated_inds[0]

    def _clusteredNondominatedInds(self, target_num_inds):
        """Returns a subset of self.nondominated_inds:
        -size is <= target_num_inds
        -always return ind with lowest cost
        -other inds are returned based on clustering on (cost1, cost2)

        Assumes that the inds are already sorted in ascending order
        of cost1 values

        Note: if target_num_inds = 2, 3, or 4, it chooses inds very specially:
         -if 2: return best-cost1, median
         -if 3: return best-cost1, next-best-cost1, median
         -if 4: return best-cost1, next-best-cost1, median, worst-cost-1
        """
        #base data
        I = numpy.argsort([ind.cost1 for ind in self.nondominated_inds])
        nondom_inds = [self.nondominated_inds[i] for i in I]
        N = len(nondom_inds)
        
        #preconditions
        assert nondom_inds is not None
        assert target_num_inds > 0

        #main work...

        #calc I = indices of inds to return
        if len(nondom_inds) <= target_num_inds:
            I = range(N)
        else:

            #Cluster down the nondom points, according to the ranking of
            # the inds.  This can be reduced to the strategy: choose the
            # columns similar to RegressorUtils.generateTestColumns, which
            # is precise, deterministic, and fast.
            #Notes:
            #-Don't aim for 0th,1st, or (N-1)th ind; we guarantee that we
            # get them by adding them after.
            #-The challenge is to avoid roundoff error of stepsize;
            # therefore we we are not allowed to round stepsize to an int
            # during the iteration.
            perc_test = (target_num_inds - 3) / float(N - 3)
            stepsize = perc_test * (N - 3)
            choices_of_I = []
            for j in range(N - 3):
                transition_occurred = (int(j*perc_test) != int((j+1)*perc_test))
                if transition_occurred:
                    choices_of_I.append(j)
            I = [0, 1] + [(i + 2) for i in choices_of_I] + [N - 1]

            #return nondom_inds based on I
            I.sort()
        
        return [nondom_inds[i] for i in I]
        
    
class CaffFactory:
        
    def __init__(self):
        self.unnormalized_X = None
        self.y = None
        
        self.ss = None
        self.varnames = None


    #=============================================================
    # Main
    def build(self, unnormalized_X, y, varnames, ss):
        """
        High-level interface of CAFFEINE model builder.  
         unnormalized_X -- 2d array of [# input vars][# samples] -- training inputs
         y              -- 2d array  training outputs [1..N samples]
         varnames   - list of varnames
         ss         - a CaffBuildStrategy
         test_cols  - colmns to use as test data (use rest as train data)
         X_validate, y_validate - any X & y, used _only_ to calculate
                      a validation error
        """        
        logging.getLogger('lin').setLevel(logging.WARNING)

        (n,N) = unnormalized_X.shape

        if N < 10:
            raise AssertionError('need reasonable # samples (got %d)'%N)
        assert n == len(varnames), "# rows in X needs to equal # var names"
        assert n > 0
        assert max(y) > min(y)
        assert N == len(y)

        self.y = y
        self.unnormalized_X = unnormalized_X
        self.ss = ss

        self.varnames = varnames
       
        log.info('Build start; # input vars=%d, #samples=%d' % (n,N))
        log.info('Max y=%5.3e, min y=%5.3e' % (max(y), min(y)))
        log.info('Strategy=%s',self.ss)

        #============Alg begins here==========================
        #
        self._setUsefulExpressions()

        #initialize state
        self.state = CaffState(self.ss)

        #gen '2 * popsize * mult' random inds, then keep '2 * popsize' of them
        rnd_inds = self._genRandGoodInds(2 * self.ss.popsize *
                                         self.ss.init_generation_multiplier)
        nmses = [ind.nmse for ind in rnd_inds]
        I = numpy.argsort(nmses)
        self.state.R = [rnd_inds[I[i]] for i in I[:2 * self.ss.popsize]]
        
        #set minmax_metrics too = (min_cost1, max_cost1, min_cost2, max_cost2)
        max_cost2 = minMaxMetrics(self.state.R)[3]
        self.state.minmax_metrics = (0.0, 0.2, 0.0, max_cost2)

        #generational loop
        while not self.state.do_stop:
            self._runOneGeneration()

        #done
        log.info('Done opt. %d nondom inds; Lowest-cost1 ind: %s' %
                 (len(self.state.nondominated_inds), self.state.best_ind))

        return (self.state.best_ind, self.state.nondominated_inds)
        
    def _runOneGeneration(self):
        self.state.gen += 1
        
        #cand_F = F[0] + F[1] + ... = nondominated layers
        cand_F = Deb_fastNondominatedSort(self.state.R)

        #update and report best
        self.state.updateBest(cand_F[0])
        gen = self.state.gen
        if True:#(gen<5) or ((gen % 5 == 0) or (gen == self.ss.max_numgen-1)):
            s = 'Gen=%2d' % gen
            pareto_front = [[ind.cost1,ind.cost2]
                            for ind in self.state.nondominated_inds]
            s += ', hypervolume=%.4e' % hypervolumeMaximize(pareto_front, [0.0, 0.0])
            s += ', # nondom=%d' % len(self.state.nondominated_inds)
            s += ', Best: %s' % self.state.best_ind
            log.info(s)

        #maybe stop
        if (gen + 1) > self.ss.max_numgen:
            log.info("Stop because max # generations exceeded")
            self.state.do_stop = True
            return
        elif self.state.best_ind.nmse <= self.ss.target_nmse:
            log.info('Stop because nmse < target')
            self.state.do_stop = True
            return

        if self.ss.do_mobj:
            #fill parent population P
            P = self._nsgaSelectInds(cand_F, self.ss.popsize,
                                     self.state.minmax_metrics)

            #create children Q (plus elitism)
            Q = self._makeNewPop_Mobj(P)
            
            #combine parent and offspring population
            self.state.R = P + Q
        else:
            #fill parent population P
            P = [self.selectInd(self.state.R)
                 for i in range(self.ss.popsize)]
            
            #create children Q
            Q = []
            assert self.ss.popsize % 2 == 0, "need even-sized pop for next op"
            for new_i in range(self.ss.popsize/2):
                par1, par2 = P[new_i*2], P[new_i*2 + 1]
                new_ind1, new_ind2 = self.createNewChildInds(par1, par2, P)
                Q.extend([new_ind1, new_ind2])
            self.evalInds(Q)

            #combine parent and offspring population (plus elitism)
            self.state.R = P + Q + [self.state.best_ind]
        
    #=====================================================================
    #select, and create new pop -- single-objective
    def selectInd(self, pop):
        """k=2 tournament selection; avoid inds with infinite nmse"""
        inf = INF

        #find ind1 with non-inf nmse
        ind1 = random.choice(pop)
        num_loops = 0
        while (ind1.nmse == inf):
            ind1 = random.choice(pop)
            num_loops += 1
            if num_loops > 10000: break

        #find ind2 with non-inf nmse, and different than ind1
        ind2 = random.choice(pop)
        num_loops = 0
        while (ind1 == ind2) or (ind2.nmse == inf):
            ind2 = random.choice(pop)
            num_loops += 1
            if num_loops > 10000: break

        #k=2 tournament selection
        if ind1.nmse < ind2.nmse:
            return ind1
        else:
            return ind2


    #=====================================================================
    #select, and create new pop -- multiobjective
    def _nsgaSelectInds(self, F, target_num_inds, minmax_metrics):
        """
        @description

          Selects 'target_num_inds' using nondominated-layered_inds 'F'
          according to NSGA-II's selection algorithm (which basically
          says take the 50% of inds in the top nondominated layers).
        
        @arguments

          F -- list of nondom_inds_layer where a nondom_inds_layer is a list
            of inds.  E.g. the output of fastNondominatedSort().
          target_num_inds -- int -- number of inds to select.  
        
        @return

          P -- list of inds (parents)
    
        @exceptions

          target_num_inds must be <= total number of inds in F.
    
        @notes
        """
        assert target_num_inds <= numIndsInNestedPop(F)
        
        N = target_num_inds
        P, i = [], 0

        #force elitism when popsize < len(nondom_front)
        if N < len(F[0]):
            P.append(self.state.best_ind)

        #bias for even more elitism
        P.append(self.state.best_ind)
        
        while True:
            #set 'distance' value to each ind in F[i]
            self.crowdingDistanceAssignment(F[i], minmax_metrics)

            #stop if this next layer would overfill 
            if len(P) + len(F[i]) > N: break

            #include ith nondominated front in the parent pop P
            P += F[i]

            #stop if we're full
            if len(P) >= N:
                P = P[:N]
                break

            #check the next front for inclusion
            i += 1

        #fill up the rest of P with elements of F[i], going
        # for highest-distance inds first
        if len(P) < N:
            I = numpy.argsort([-ind.distance for ind in F[i]])
            F[i] = list(numpy.take(F[i], I))

            P += F[i][:(N-len(P))]

        #ensure that references to parents in other layers don't hurt us
        # (don't deepcopy because we don't want to copy each 'S' attribute)
        P = [copy.copy(ind) for ind in P]

        return P

    def crowdingDistanceAssignment(self, layer_inds, minmax_metrics):
        """
        @description

          Assign a crowding distance to each individual in list of inds
          at a layer of F.
        
        @arguments

          layer_inds -- list of Ind
          minmax_metrics -- dict of metric_name : (min_val, max_val) as
            computed across _all_ inds, not just across layer_inds
        
        @return

          <<none>> but alters the 'distance' attribute of each individual
           in layer_inds
    
        @exceptions
    
        @notes
        """
        #corner case
        if len(layer_inds) == 0:
            return

        #initialize distance for each ind to 0.0
        for ind in layer_inds:
            ind.distance = 0.0

        (min_cost1, max_cost1, min_cost2, max_cost2) = minmax_metrics
            
        #increment distance for each ind on a metric-by-metric basis
        for goal_i in [1,2]:
            #retrieve max and min; if max==min then this metric won't
            # affect distance calcs
            if goal_i == 1:
                (met_min, met_max) = (min_cost1, max_cost1)
            else:
                (met_min, met_max) = (min_cost2, max_cost2)
                
            if met_min == met_max:
                continue

            #sort layer_inds and metvals, according to metvals
            if goal_i == 1:
                metvals = [ind.cost1 for ind in layer_inds]
            else:
                metvals = [ind.cost2 for ind in layer_inds]
            I = numpy.argsort(metvals)
            layer_inds = numpy.take(layer_inds, I)
            metvals = numpy.take(metvals, I)

            #ensure that boundary points are always selected via dist = Inf
            layer_inds[0].distance = INF
            layer_inds[-1].distance = INF

            #all other points get distance set based on nearness to
            # ind on both sides (scaled by the max and min seen on metric)
            for i in range(1,len(layer_inds)-1):
                d = abs(metvals[i+1] - metvals[i-1]) / (met_max - met_min)
                layer_inds[i].distance += d

    def _makeNewPop_Mobj(self, P):
        """
        @description

          Use selection, mutation, and crossover to create a new child pop Q.
          Via simulation, ensure that each ind is good (if not, generate
          new children until good).
          
          Selection is based on the 'crowded comparison operator' as follows:
           -if possible, first choose based on domination (ie ind.rank)
           -but if they are of same ind.rank, then choose
            the ind with the largest (crowding) distance

          Each new ind will have a unique performance vector.
          (i.e. not one that is in P or any other new inds)
        
        @arguments

          P -- list of Ind -- parent pop
        
        @return

          Q -- list of Ind -- child pop
    
        @exceptions
    
        @notes

          Each ind in P needs to have the attributes of 'rank' and
          'distance' set prior to calling this routine.

          Parent popsize should be self.ss.popsize.
          Child popsize will be self.ss.popsize.
        """
        N = self.ss.popsize
        assert len(P) == N, (len(P), N)
        
        #select parents
        parents = []
        while len(parents) < len(P):
            ind_a, ind_b = random.choice(P), random.choice(P)

            #first try selecting based on domination
            if ind_a.rank < ind_b.rank:   parents.append(ind_a)
            elif ind_b.rank < ind_a.rank: parents.append(ind_b)

            #if needed, select based on distance
            elif ind_a.distance > ind_b.distance:  parents.append(ind_a)
            else:                                  parents.append(ind_b)

        #with parents, generate children via variation
        Q = []
        parent_index = 0
        while len(Q) < self.ss.popsize:
            parA = parents[parent_index]
            if parent_index+1 == len(P): #happens whenever pop size is odd
                parB = random.choice(parents)
            else:
                parB = parents[parent_index+1]

            childA, childB = self.createNewChildInds(parA, parB, P)
            
            #add children in a fashion that can handles an odd-numbered value
            # for popsize
            Q.append(childA)
            parent_index += 1
            if len(Q) < len(P): 
                Q.append(childB) 
                parent_index += 1

        #also evaluate
        self.evalInds(Q)
        
        #done
        assert len(Q) == N, (len(Q), N)
        return Q


    #================================================================
    #create new inds via mutation or crossover
    def createNewChildInds(self, par1, par2, pop):
        #set new_tree1, new_tree2
        r = random.random()
        if r < self.ss.prob_mutate:
            while True:
                par1_tree_str = str(par1.tree)
                new_tree1 = self.mutateTree(par1.tree, 0)
                if new_tree1 is not None:
                    break
                par1 = random.choice(pop)

            while True:
                par2_tree_str = str(par2.tree)
                new_tree2 = self.mutateTree(par2.tree, 0)
                if new_tree2 is not None:
                    break
                par2 = random.choice(pop)
                
        else:
            while True:
                par1_tree_str = str(par1.tree)
                par2_tree_str = str(par2.tree)
                success, new_tree1, new_tree2 = \
                         self.crossOverTrees(par1.tree, par2.tree)
                if success:
                    break
                if random.random() < 0.5:
                    par1 = random.choice(pop)
                else:
                    par2 = random.choice(pop)

        #did we accidentally change the parent?
        assert par1_tree_str == str(par1.tree) 
        assert par2_tree_str == str(par2.tree)

        #to turn on the following for non-test is a H ACK (big slowdown)
        #self.validateInd(par1) 
        #self.validateInd(par2)

        #postconditions and return
        new_ind1 = CaffInd(new_tree1, self.varnames)
        new_ind2 = CaffInd(new_tree2, self.varnames)
        return new_ind1, new_ind2

    def bestInd(self, pop):
        best_ind, best_nmse = 0, INF
        nmses = []
        for ind in pop:
            assert ind.nmse is not None
            if ind.nmse < best_nmse:
                best_ind, best_nmse = ind, ind.nmse
        
        return best_ind
    

    #=============================================================
    # Sim inds
    def evalInds(self, inds):
        for i,ind in enumerate(inds):
            self.evalInd(inds[i])
                
    def evalInd(self, ind):
        """Fill in ind.nmse, complexity, lin_model, etc"""
        inf = INF
        
        #corner case: already calculated it
        if ind.nmse is not None:
            return

        #simulate ind
        (numerator_ys, denominator_ys) = \
                       ind.tree.simulateBases(self.unnormalized_X)
        
        #corner case: terrible result
        # -therefore set ind.nmse, complexity to Inf
        for y_at_base in (numerator_ys.values() + denominator_ys.values()):
            if yIsPoor(y_at_base):
                ind.nmse, ind.complexity = inf, inf
                return
        
        #main case: use linear learning
        # -ind.nmse, complexity, lin_model all get set
        numerator_ids = sorted(numerator_ys.keys())
        denominator_ids = sorted(denominator_ys.keys())

        regress_n = len(numerator_ys) + len(denominator_ys)
        N = len(self.y)
        regress_X = numpy.zeros((regress_n, N), dtype=float)

        for (numerator_index, id) in enumerate(numerator_ids):
            row = numerator_index
            regress_X[row,:] = numerator_ys[id]

        for (denominator_index, id) in enumerate(denominator_ids):
            row = len(numerator_ids) + denominator_index
            regress_X[row,:] = -1.0 * denominator_ys[id] * self.y
        
        (minX, maxX) = RegressorUtils.minMaxX(regress_X)

        ind.lin_model = LinearModelFactory().quickRegularizedBuild(
            regress_X, self.y, minX, maxX, self.ss.lin_ss)
        
        yhat = ind.simulate(self.unnormalized_X)
        
        if yIsPoor(yhat):
            ind.nmse, ind.complexity = inf, inf
            ind.lin_model = None
            return

        ind.nmse = mathutil.nmse(yhat, self.y, min(self.y), max(self.y))
        ind.complexity = ind.calcComplexity(self.unnormalized_X)
        
        #postconditions
        assert ind.nmse >= 0.0, ind.nmse
        assert ind.complexity >= 0.0, ind.complexity
        assert ind.lin_model is not None

    def validateInd(self, ind):
        """Only turn this on for testing"""
        ind.tree.validate(0)
        
        self.evalInd(ind)
        nmse1, complexity1 = ind.nmse, ind.complexity
        
        ind.clearCachedData()
        assert ind.nmse is None
        assert ind.complexity is None
        self.evalInd(ind)
        nmse2, complexity2 = ind.nmse, ind.complexity

        assert nmse1 == nmse2
        assert complexity1 == complexity2
                
    #==============================================================
    # Specific tree construction
    def _setUsefulExpressions(self):
        """Sets ss.useful_var_exprs and useful_nonlin_exprs
        """
        (n, N) = self.unnormalized_X.shape
        
        #build up each combination of all {var_i} x {op_j}, except for
        # when a combination is unsuccessful
        bases = []
        for var_i in range(self.unnormalized_X.shape[0]):
            for exponent in self.ss.op_exponents:
                #'lin' version of base
                lin_base = Tree(VAR_EXPR_SYMBOL, (var_i, exponent))
                lin_yhat = lin_base.simulate(self.unnormalized_X)
                if not yIsPoor(lin_yhat):
                    bases.append(lin_base)

                    #'log' version of base (only try if lin version ok)
                    log_base = self._logExprTree(lin_base)
                    log_yhat = log_base.simulate(self.unnormalized_X)
                    if not yIsPoor(log_yhat):
                        bases.append(log_base)

        #now, prune them down via regularized lin learning
        # -set up inputs: regress_X, minX, maxX, ss
        regress_X = numpy.zeros((len(bases)*2, N), dtype=float)
        for (base_i, base) in enumerate(bases):
            base_y = base.simulate(self.unnormalized_X)
            regress_X[base_i,:] = base_y #numerators
            regress_X[len(bases)+base_i,:] = -1.0 * base_y * self.y #denomin.
        (minX, maxX) = RegressorUtils.minMaxX(regress_X)

        min_num_bases = min(n*2, N) #magic number alert
        cand_thrs = [0.99, 0.95, 0.90, 0.80, 0.65, 0.50,
                     0.35, 0.20, 0.10, 0.05, 0.01] #magic number alert
        num_nonzero = -1
        for thr in cand_thrs:
            #set lin_ss
            lin_ss = LinearBuildStrategy(
                y_transforms=['lin'], target_nmse=0.0, regularize=True)
            lin_ss.reg.thr = thr

            # -build the model
            lin_model = LinearModelFactory().quickRegularizedBuild(
                regress_X, self.y, minX, maxX, lin_ss)

            # -keep only bases with nonzero coefficients
            #  (can be nonzero in either numerator or denominator)
            num_nonzero = len([base
                               for (base_i, base) in enumerate(bases)
                               if (lin_model.coefs[base_i+1] != 0) or \
                               (lin_model.coefs[len(bases)+base_i+1] != 0)])
            
            #we can stop if we have enough bases, otherwise we need
            # to build less aggressively
            if num_nonzero > min_num_bases:
                break

        #break apart pruned_bases into linear and log components
        infls = lin_model.influencePerVar()
        self.ss.useful_var_exprs, self.ss.useful_nonlin_exprs = {}, {} #reset
        min_infl = 0.25 #magic number alert
        for infl, base in zip(infls, bases):
            if infl > 0.0:
                if base.symbol == VAR_EXPR_SYMBOL:
                    self.ss.useful_var_exprs[base] = max(min_infl, infl)
                elif base.symbol == NONLIN_EXPR_SYMBOL:
                    self.ss.useful_nonlin_exprs[base] = max(min_infl, infl)
                else:
                    raise ValueError

        #output results
        log.info("Have %d / %d nonzero single-variable bases (%d linear and %d log10)" % \
                 (num_nonzero, len(bases), len(self.ss.useful_var_exprs),
                  len(self.ss.useful_nonlin_exprs)))
        
        yhat = lin_model.simulate(regress_X)
        nmse = mathutil.nmse(yhat, self.y, min(self.y), max(self.y))
        log.info('Nmse when using single-variable bases is %.5f' % nmse)

    def _logExprTree(self, var_expr_tree):
        """Return a nonlin_expr_tree that is log10(var_expr_tree)
        """
        assert var_expr_tree.symbol == VAR_EXPR_SYMBOL
        
        depth5_tree = var_expr_tree

        depth4_data = (set([depth5_tree]), RationalAllocation(True, False))
        depth4_tree = Tree(PRODUCT_SYMBOL, depth4_data)

        depth3_data = (set([depth4_tree]), Placeholder())
        depth3_tree = Tree(SUM_SYMBOL, depth3_data)

        depth2_data = (OP_LOG10, depth3_tree, 1.0)
        depth2_tree = Tree(NONLIN_EXPR_SYMBOL, depth2_data)

        nonlin_expr_tree = depth2_tree
        return nonlin_expr_tree
        
    
    #==============================================================
    # Random inds    
    def _genRandGoodInds(self, target_num_good):
        """Randomly generate and eval enough good inds."""
        inf = INF
        num_tries = 0
        new_inds = []
        for next_new_i in range(target_num_good):
            if next_new_i % 10 == 0:
                log.info('Generated %d / %d rand good inds' %
                         (next_new_i, target_num_good))
                
            while True:
                log.debug('Gen good rand ind #%d / %d; tot tries=%d'%
                          (next_new_i, target_num_good, num_tries))
                num_tries += 1
                new_ind = self.randomInd()
                self.evalInd(new_ind)
                if (new_ind.nmse < inf) and (new_ind.complexity < inf):
                    new_inds.append(new_ind)
                    break
                if num_tries > 1000000: #magic number
                    log.warning('Surpasses 1 million tries when gen. rnd inds')
        log.info('Generated %d / %d rand good inds' %
                 (target_num_good, target_num_good))
    
        return new_inds

    def randomInd(self):
        """Returns a randomly-generated ind"""
        tree = Tree(SUM_SYMBOL, None)
        numvars = self.unnormalized_X.shape[0]
        tree.randomGrow(self.ss, numvars, 0, self.unnormalized_X)
        log.debug('New random tree: \n%s\n' % tree)
        ind = CaffInd(tree, self.varnames)
        
        return ind
    
    #=============================================================
    # High-level Crossover op on trees
    def crossOverTrees(self, tree1, tree2):
        """Return 2 crossed-over child trees of tree1 and tree2
        """
        r = random.random()
        if r < 0.50: #magic number alert
            return self._twoPointCrossover(tree1, tree2, 'by_node')
        elif r < 0.75:
            return self._twoPointCrossover(tree1, tree2, 'by_depth')
        else:
            return self._copyBasedCrossover(tree1, tree2)
        
    #=============================================================
    # Low-level Crossover op on trees
    def _copyBasedCrossover(self, tree1, tree2):
        """Copy a subtree of tree2 into an open spot on tree1,
        and vice versa."""
        new_tree1 = self._copyFromTree2IntoTree1(tree1, tree2, 0)
        if new_tree1 is None:
            return (False, None, None)
        
        new_tree2 = self._copyFromTree2IntoTree1(tree2, tree1, 0)
        if new_tree2 is None:
            return (False, None, None)
        
        return (True, new_tree1, new_tree2)

    def _copyFromTree2IntoTree1(self, tree1, tree2, tree1_depth):
        """Copies a subtree of tree2 into an open spot on tree1"""
        #from tree1, choose a subtree with space (add to this)
        cands = [(subtree1, depth1)
                 for (subtree1, depth1) in tree1.subtreeInfo(0)
                 if (subtree1.symbol in [SUM_SYMBOL, PRODUCT_SYMBOL]) and
                 subtree1.hasSpace(self.ss, depth1)]
        if not cands:
            return None
        (subtree_with_space, depth) = random.choice(cands)
        if depth in [0,1]:
            tall_child_ok = True
        else:
            tall_child_ok = False
            
        #choose a subtree in tree2 to copy into tree1
        if subtree_with_space.symbol == SUM_SYMBOL:
            target_symbols = [PRODUCT_SYMBOL]
        elif subtree_with_space.symbol == PRODUCT_SYMBOL:
            target_symbols = [VAR_EXPR_SYMBOL, NONLIN_EXPR_SYMBOL]
        else:
            raise AssertionError
                
        cands = [(subtree2, depth2)
                 for (subtree2, depth2) in tree2.subtreeInfo(0)
                 if (subtree2.symbol in target_symbols)
                 and (tall_child_ok or
                      (not tall_child_ok and not subtree2.isTall()))
                 ]
        if not cands:
            return None
        (additional_child, dummy) = random.choice(cands)

        #insert tree
        paths = tree1.pathsToTarget(subtree_with_space, depth+1, tree1_depth)
        path = random.choice(paths)
        new_tree = self.buildTreeWithAddedSubtree(path, additional_child)
        
        return new_tree
        
    def _twoPointCrossover(self, tree1, tree2, bias):
        """Return 2 crossed-over child trees of tree1 and tree2

        Argument 'bias' tellsl where to apply uniform probability of
        selection of nodes: 'by_depth' or 'by_node'.  Note that we always
        swap subtrees of the same depth (with exception of var expressions).
        -If 'by_depth': small deeper-level nodes have a lower probability of
         being selected and basis functions have a higher chance of being
         swapped around.
        -If 'by_node' then small deeper-level nodes have a higher probability
         of being selected

        We never select at depth 0 because that would be pointless (it's
        the whole tree).

        """
        assert bias in ['by_depth','by_node']
        
        #choose subtree1, subtree2
        sd2 = []
        num_loops = 0
        while (len(sd2) == 0):
            #set sd1 according to bias.  sd1 = list of (tree,depth) tuples.
            if bias == 'by_depth':
                target_depth = random.randint(1, 5)
                sd1 = [(subtree1, depth1)
                       for (subtree1, depth1) in tree1.subtreeInfo(0)
                       if depth1 == target_depth]
            else:
                sd1 = [(subtree1, depth1)
                       for (subtree1, depth1) in tree1.subtreeInfo(0)
                       if depth1 > 0]

            #set subtree1, sd2
            if len(sd1) > 0:
                (subtree1,  depth1) = random.choice(sd1) #uniform selection

                same_depth = (subtree1.symbol != VAR_EXPR_SYMBOL) or \
                             (subtree1.symbol == VAR_EXPR_SYMBOL and \
                              random.random() < 0.5) #magic number alert
                if same_depth:
                    #any symbols of same depth can swap; due to the grammar
                    # this means that only "like symbols swap with like",
                    # except NUM, VAR_EXPR, and NONLIN_EXPR symbols can swap
                    # at the target depth
                    sd2 = \
                        [(subtree2, depth2)
                         for (subtree2, depth2) in tree2.subtreeInfo(0)
                         if (depth2 == depth1) and (subtree2 != subtree1)
                         ]
                else:
                    #swap VAR_EXPR trees; it's ok to jump between the depths
                    target_depths = [2, 5]
                    sd2 = \
                        [(subtree2, depth2)
                         for (subtree2, depth2) in tree2.subtreeInfo(0)
                         if (depth2 in target_depths) and \
                         (subtree2.symbol == VAR_EXPR_SYMBOL) and \
                         (subtree2 != subtree1)
                         ]

            #prevent infinite looping
            num_loops += 1
            if num_loops > 10000: #magic number
                return False, None, None
            
        (subtree2, depth2) = random.choice(sd2)

        #stability tests
        for (tree, depth) in sd1: assert isinstance(tree, Tree)
        for (tree, depth) in sd2: assert isinstance(tree, Tree)
        assert isinstance(subtree1, Tree)
        assert isinstance(subtree2, Tree)

        #
        old_depth = 0
        path1 = random.choice(tree1.pathsToTarget(subtree1, depth1, old_depth))
        path2 = random.choice(tree2.pathsToTarget(subtree2, depth2, old_depth))
        assert path1[0] == tree1 and path1[-1] == subtree1
        assert path2[0] == tree2 and path2[-1] == subtree2

        #swap
        new_tree1 = self.buildTreeWithReplacedSubtree(path1, subtree2)
        new_tree2 = self.buildTreeWithReplacedSubtree(path2, subtree1)
            
        #done
        return (True, new_tree1, new_tree2)
        
    def buildTreeWithReplacedSubtree(self, old_path, new_subtree):
        """Create a new tree, in which an old_subtree (at old_path[-1]) has been
        replaced by a new subtree.  Only copies as needed, and no more."""
        for tree in old_path:
            assert isinstance(tree, Tree)
            
        old_root = old_path[0]
        new_root = copy.copy(old_root) #note: Tree's copy is overridden

        #loop, creating another link in the chain
        # -we eventually want to link new_root to new_subtree
        new_parent = new_root
        for i in range(len(old_path) - 1):
            old_child = old_path[i+1]

            at_intermediate_node = (i < (len(old_path) - 1 - 1))
            if at_intermediate_node:
                new_child = copy.copy(old_child)
            else:
                new_child = new_subtree
                
            new_parent.replaceChild(old_child, new_child)
            
            #update for next loop
            new_parent = new_child

        return new_root
        
    def buildTreeWithAddedSubtree(self, old_path, additional_child):
        """Create a new tree, in which old_path[-1] gets an
        additional child.  Only copies as needed, and no more."""
        for tree in old_path:
            assert isinstance(tree, Tree)
        assert old_path[-1].symbol in [SUM_SYMBOL, PRODUCT_SYMBOL]
        assert isinstance(additional_child, Tree)
        assert additional_child.symbol in \
               SUM_CHILD_SYMBOLS + PRODUCT_CHILD_SYMBOLS

        #copy only as needed (in path); maintain all other references
        new_path = [copy.copy(tree) for tree in old_path]
        for (i, new_parent) in enumerate(new_path[:-1]):
            new_parent.replaceChild(old_path[i+1], new_path[i+1])
        new_root = new_path[0]

        #add the new subtree
        new_path[-1].addChild(additional_child)

        return new_root
        
    #=============================================================
    # High-Level Mutation ops on Ind
    def mutateTree(self, parent_tree, parent_depth):
        """Return a mutated child ind of par (or None if not success)
        """
        assert isinstance(parent_tree, Tree)
        
        #choose operator
        if random.random() < 0.5: #magic number alert
            op = self.mutateTreeViaMutateNumber
        else:
            ops = [self.mutateTreeViaAddTinyBasisFunction,
                   self.mutateTreeViaAddRandomSubtree,
                   self.mutateTreeViaDeleteSubtree,
                   ]
            if parent_depth == 0:
                ops.append(self.mutateTreeViaCopySubtreeAndMutate)
                ops.append(self.mutateTreeViaChangeRationalAlloc)
                
            op = random.choice(ops) #magic number alert (equal biases)

        #apply operator and return
        return op(parent_tree, parent_depth)
    
    #=============================================================
    # Low-Level Mutation ops on tree
    def mutateTreeViaAddTinyBasisFunction(self, parent_tree, parent_depth):
        """Add a new basis function, chosen from
        """
        assert isinstance(parent_tree, Tree)
        
        if not parent_tree.hasSpace(self.ss, parent_depth):
            return None

        #create additional_child
        # -magic number alert
        do_lin = (self.ss.useful_var_exprs and (random.random() < 0.5)) or \
                 (not self.ss.useful_nonlin_exprs)
        if do_lin:
            depth2_tree = random.choice(self.ss.useful_var_exprs.keys())
        else:
            depth2_tree = random.choice(self.ss.useful_nonlin_exprs.keys())

        depth1_data = (set([depth2_tree]), randomRationalAllocation())
        depth1_tree = Tree(PRODUCT_SYMBOL, depth1_data)
        additional_child = depth1_tree
        
        #create new_tree and insert additional_child
        new_tree = copy.copy(parent_tree)
        new_tree.children().add(additional_child)

        return 
    
    def mutateTreeViaAddRandomSubtree(self, old_tree, old_depth):
        """Replace a subtree with a randomly-generated subtree
        (or None if not success)"""
        assert isinstance(old_tree, Tree)
        
        #choose a subtree to replace
        cands = [(old_subtree, depth)
                 for (old_subtree, depth) in old_tree.subtreeInfo(old_depth)
                 if depth > 0]
        (old_subtree, depth) = random.choice(cands)

        #create a new random subtree
        new_subtree = Tree(old_subtree.symbol, None)
        new_subtree.randomGrow(self.ss, self.unnormalized_X.shape[0], depth)

        #swap
        cand_paths = old_tree.pathsToTarget(old_subtree, depth, old_depth)
        path = random.choice(cand_paths)
        new_tree = self.buildTreeWithReplacedSubtree(path, new_subtree)

        #done
        return new_tree

    def mutateTreeViaDeleteSubtree(self, old_tree, old_depth):
        """Delete a randomly-selected subtree
        (or None if not success)"""
        assert isinstance(old_tree, Tree)

        #randomly choose a parent tree: must be a sum or product, with >1 child
        cands = [old_subtree
                 for (old_subtree, depth) in old_tree.subtreeInfo(old_depth)
                 if (old_subtree.symbol in [SUM_SYMBOL, PRODUCT_SYMBOL]) and \
                 (len(old_subtree.children()) > 1)
                 ]
        if len(cands) == 0:
            return None
        old_subtree = random.choice(cands)

        # create a new copy of old_subtree,
        # randomly choose a child from new_subtree's children (ie data),
        # and remove the child
        new_subtree = copy.copy(old_subtree)
        child_tree = random.choice(list(new_subtree.children()))
        new_subtree.removeChild(child_tree)

        # now link up the rest of the old_tree with a new_tree
        cand_paths = old_tree.pathsToTarget(old_subtree, depth,  old_depth)
        if len(cand_paths) == 0: 
            return None
        path = random.choice(cand_paths)
        new_tree = self.buildTreeWithReplacedSubtree(path, new_subtree)

        #done
        return new_tree

    def mutateTreeViaCopySubtreeAndMutate(self, old_tree, old_depth):
        """Copies a basis function of 'old_tree', and mutates the copy."""
        #preconditions
        assert isinstance(old_tree, Tree)
        assert old_tree.symbol == SUM_SYMBOL
        assert old_depth == 0
        assert old_tree.children()

        #corner case
        if not old_tree.hasSpace(self.ss, old_depth):
            return None

        #main case
        subtree = random.choice(list(old_tree.children()))
        new_subtree = copy.copy(subtree)
        mutated_subtree = self.mutateTree(new_subtree, old_depth+1)
        if mutated_subtree is None:
            return None

        new_tree = copy.copy(old_tree)
        new_tree.children().add(mutated_subtree)

        #done!
        return new_tree
    
    def mutateTreeViaChangeRationalAlloc(self, old_tree, old_depth):
        """Return a tree where one of the parent's rational allocations
        has been changed (or None if not success)
        """
        assert isinstance(old_tree, Tree)
        assert old_tree.symbol == SUM_SYMBOL
        assert old_depth == 0
        
        #choose a subtree to replace
        old_subtree = random.choice(list(old_tree.children()))

        #create a new random subtree
        new_subtree = copy.copy(old_subtree)
        new_subtree.data[1].mutate()

        #swap
        path = [old_tree, old_subtree]
        new_tree = self.buildTreeWithReplacedSubtree(path, new_subtree)

        #done
        return new_tree
    
    def mutateTreeViaMutateNumber(self, old_tree, old_depth):
        assert isinstance(old_tree, Tree)
        
        #randomly choose a sub tree: must be a sum or product, with >1 child
        cands = [(old_subtree, depth)
                 for (old_subtree, depth) in old_tree.subtreeInfo(old_depth)
                 if old_subtree.symbol == NUMBER_SYMBOL]

        #select subtree, depth
        if len(cands) > 0:
            #main case
            (old_subtree, depth) = random.choice(cands)

            # create a new copy of old_subtree,
            # and vary its number (ie data)
            new_subtree = copy.copy(old_subtree)
            new_subtree.data = self.helperVaryFloat(new_subtree.data)

            # now link up the rest of the old_tree with a new_tree
            cand_paths = old_tree.pathsToTarget(old_subtree, depth, old_depth)
            path = random.choice(cand_paths)
            new_tree = self.buildTreeWithReplacedSubtree(path, new_subtree)

        else:
            #corner case: no numbers, so insert one (helps evolvability anyway)
            cands = [(old_subtree, depth)
                     for (old_subtree, depth) in old_tree.subtreeInfo(old_depth)
                     if old_subtree.symbol == PRODUCT_SYMBOL]
            (old_subtree, depth) = random.choice(cands)
            
            # create a new copy of old_subtree,
            # and vary its number (ie data)
            new_subtree = copy.copy(old_subtree)
            
            new_number_tree = Tree(NUMBER_SYMBOL, None)
            new_number_tree.randomGrow(
                self.ss, self.unnormalized_X.shape[0], depth+1)
            new_subtree.children().add(new_number_tree)

            # now link up the rest of the old_tree with a new_tree
            cand_paths = old_tree.pathsToTarget(old_subtree, depth, old_depth)
            path = random.choice(cand_paths)
            new_tree = self.buildTreeWithReplacedSubtree(path, new_subtree)

        #done
        return new_tree
         
    def helperVaryFloat(self, value):
        #magic number alert for this function
        assert mathutil.isNumber(value)
        
        if value == 0.0:
            if random.random() < 0.9: #smooth version
                return self.randomFloat()*0.01
            else:                     #non-smooth
                return self.randomFloat()
        else:
            if random.random() < 0.95: #smooth version
                sigma = 0.1* (self.ss.max_w - self.ss.min_w)
                c = value * (1.0 + cauchy(sigma))
                c = min(max(c, self.ss.min_w), self.ss.max_w)
                return c
            else:                     #non-smooth
                return self.randomFloat()
            
    def randomFloat(self):#magic numbers
        if random.random() < 0.25:
            return 0.0
        else:
            return random.random()*(self.ss.max_w-self.ss.min_w) + self.ss.min_w

        
def cauchy(sigma):
    z = random.uniform(0.0, 1.0)
    return sigma * math.tan(math.pi * (z-0.5) )


def scale_w(w):
    """transform the 'gene' to true constant value, just like in CAFFEINE"""
    a = WEIGHT_BASE_RANGE
    if w < 0.0:
        return -10**(-w-a)
    elif w == 0.0:
        return 0.0
    else:
        return +10**(w-a)


def yIsPoor(y):
    """Returns True if y is not usable"""
    miny, maxy = min(y), max(y)
    return ((maxy - miny) < 1e-10) or \
           (miny == float('-Inf')) or \
           (maxy == float('+Inf')) or \
           (mathutil.hasNan(y)) #or \
           #(maxy - miny > 1e3*(max(self.y) - min(self.y)))


def uniqueNondominatedInds(inds):
    """Returns the subset of 'inds' that are nondominated (and unique)."""
    unique_nondom_inds = []
    unique_nondom_strs = []
    for p in inds:
        is_dominated = False
        for q in inds:
            if q.dominates(p):
                is_dominated = True
                break

        if (not is_dominated) and (not p.isPoor()):
            unique = True
            p_str = str(p)
            for nondom_str in unique_nondom_strs:
                if p_str == nondom_str:
                    unique = False
                    break
            if unique:
                unique_nondom_inds.append(p)
                unique_nondom_strs.append(p_str)

    return unique_nondom_inds

def Deb_fastNondominatedSort(P):
    """
    @description

      Uses Deb's algorithm in NSGA-II to build up nondominated 'layers'.
      -Does not check for uniqueness of inds.

    @arguments

      P -- list of Ind -- inds to sort

    @return

      F -- list of nondom_inds_layer where a nondom_inds_layer is a
        list of inds and all inds_layers together make up R.
        E.g. F[2] may have [P[15], P[17], P[4]]

      ALSO: each the inds in P and F have three attributes modified:
        n -- int -- number of inds which dominate the ind
        S -- list of ind -- the inds that this ind dominates
        rank -- int -- 0 means in 0th nondom layer, 1 in 1st layer, etc.

    @exceptions

    @notes
    """
        
    F = [[]]
    
    #corner case
    if not P:
        return F

    #main case...
    for p in P:
        p.n = 0 #n is domination count of ind 'p',ie # inds which dominate p
        p.S = [] #S is the set of solutions that 'p' dominates

        for q in P:
            if p.dominates(q):
                p.S += [q]
            elif q.dominates(p):
                p.n += 1

        #if p belongs to 0th front, remember that
        if p.n == 0:
            p.rank = 0
            F[0] += [p]

    i = 0
    while len(F[i]) > 0:        
        Q = [] #stores members of the next front
        for p in F[i]:
            for q in p.S:
                q.n -= 1
                if q.n == 0:
                    q.rank = i + 1
                    Q += [q]
        i += 1
        F.append(Q)

    #if the last list_of_inds in F is empty, remove it
    if len(F[-1]) == 0:
        F = F[:-1]

    return F

def numIndsInNestedPop(F):
    """
    @description

      Returns the total # inds in a list of list_of_inds

    @arguments

      F -- list of list_of_inds

    @return

      num_inds -- int -- == len(list_of_inds[0]) + len(list_of_inds[1]) + ... 

    @exceptions

    @notes
    """
    return sum([len(Fi) for Fi in F])


def minMaxMetrics(inds):
    """
    @description

      Compute min metric value encountered, and max metric value encountered,
      in 'inds'

    @arguments

      inds -- list of ind

    @return

      min_cost1 - float
      max_cost1 -
      min_cost2 -
      max_cost2 -

    @exceptions

    @notes
    """
    #compute values
    min_cost1, min_cost2 = INF, INF
    max_cost1, max_cost2 = -INF, -INF

    for ind in inds:
        min_cost1 = min(min_cost1, ind.cost1)
        max_cost1 = max(max_cost1, ind.cost1)
        min_cost2 = min(min_cost2, ind.cost2)
        max_cost2 = max(max_cost2, ind.cost2)

    #final data structure
    minmax_metrics = (min_cost1, max_cost1, min_cost2, max_cost2)

    #done
    return minmax_metrics
