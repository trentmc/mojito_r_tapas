"""
IsleBase.py

"""

import logging
import types

import numpy

from util import mathutil
from ConstantModel import ConstantModel
from LinearModel import LinearModel
from Sgb import SgbModel

log = logging.getLogger('isle')

class IsleBase:
    """
    @description

        Abstract class for basis functions.
        
        Iterated Sample Learning Ensembles (ISLE), Friedman
        
    """
    def __init__(self):
        pass 
    
    def __str__(self):
        raise NotImplementedError
    
    def simulate(self, X11):
        raise NotImplementedError

    def hasVar(self, var):
        """Does this base contain variable 'var' ?, i.e. is it influenced
        by that var?"""
        raise NotImplementedError

    def isTrivial(self):
        """Returns 'True' if this base always returns a 1.0
        because it's trivial.
        Default when not known is to assume it's not trivial."""
        return False
    
    def order(self):
        raise NotImplementedError
        
    def influencedOnlyBy(self, vars, tot_numvars):
        """Is this base only influenced by 'vars' or a subset thereof?"""
        for var in range(tot_numvars):
            if self.hasVar(var) and var not in vars: return False
        return True


class LinearBase(IsleBase):
    
    def __init__(self, var_index, mean, stddev):
        IsleBase.__init__(self)
        self.var_index = var_index
        self.mean = mean #mean of this var over training data
        self.stddev = stddev #stddev of this var over the training data

    def setMeanAndStddev(self, X11):
        y = self.simulate(X11)                           
        self.mean = mathutil.average(y)
        self.stddev = mathutil.stddev(y)

    def __str__(self):
        return 'scaled_x%d' % self.var_index

    def __eq__(self, other):
        if isinstance(other, PolyBase):
            return self.equalsPolyBase(other)
        elif isinstance(other, LinearBase):
            #warning: equality is not dependent on self.mean or self.stddev!!
            return self.var_index == var_index 
        elif isinstance(other, ProductBase):
            return self.equalsProductBase(other)
        else:
            raise AssertionError("Base type not handled:%s" % other.__class__)

    def equalsPolyBase(self, poly_base):
        assert isinstance(poly_base, PolyBase)
        return poly_base.equalsLinearBase(self)

    def equalsProductBase(self, prod_base):
        assert isinstance(prod_base, ProductBase)
        return prod_base.equalsLinearBase(self)

    def simulate(self, X11):
        return X11[self.var_index,:]

    def hasVar(self, var):
        return var == self.var_index

    def order(self):
        return 1

    def numKnots(self):
        """Mars needs this function"""
        return 0

    
        
class PolyBase(IsleBase):
    def __init__(self, explist): 
        IsleBase.__init__(self)
        self.explist = explist #explist[var_i] is exponent for var #i

        #Mean and stddev need to be non-None only for IsleModel influence calcs
        #So, they're calculated within the IsleModel constructor.
        self.mean = None
        self.stddev = None

    def setMeanAndStddev(self, X11):
        y = self.simulate(X11)                           
        self.mean = mathutil.average(y)
        self.stddev = mathutil.stddev(y)

    def __eq__(self, other):
        if isinstance(other, PolyBase):
            return self.explist == other.explist
        elif isinstance(other, LinearBase):
            return self.equalsLinearBase(other)
        elif isinstance(other, ProductBase):
            return self.equalsProductBase(other)
        else:
            raise AssertionError("Base type not handled:%s" % other.__class__)

    def equalsLinearBase(self, linear_base):
        """Returns true if 'self' is the equivalent of linear_base"""
        assert isinstance(linear_base, LinearBase)
        for var, exp in enumerate(self.explist):
            if var == linear_base.var_index:
                if exp != 1: return False
            else:
                if exp != 0: return False
        return True
    
    def equalsProductBase(self, product_base):
        """Returns true if 'self' is the equivalent of a product_base"""
        assert isinstance(product_base, ProductBase)
        n = len(self.explist)
        product_base_explist = numpy.zeros(n)
        for product_term in product_base.bases11:
            if not isinstance(product_term, LinearBase):
                return False
            if product_term.var_index >= n:
                return False
            product_base_explist[product_term.var_index] += 1

        return product_base_explist == self.explist
    
    def __str__(self):        
        s = ''
        found_nonzero = False
        for var, exp in enumerate(self.explist):
            if exp>0.0:
                if found_nonzero: s += '*'
                found_nonzero = True
                s += 'scaled_x%d' % var
                if exp != 1.0: s += '^%g' % exp
        if found_nonzero: return s
        else: return '1'     

    def simulate(self, X11):
        y = numpy.ones(X11.shape[1])*1.0
        for var,exp in enumerate(self.explist):
            if exp != 0.0: y *= X11[var,:]**exp
        return y        

    def hasVar(self, var):
        return self.explist[var] != 0

    def order(self):
        return sum(abs(e) for e in self.explist)
    
    def isTrivial(self):
        return max(self.explist) == min(self.explist) == 0

    def higherOrderBases(self, tabu_bases11, tabu_vars, ss):
        """Returns all possible order+1 expansions of this base,
        but ignoring (1) tabu bases, (2) bases where order > max,
        (3) bases which expand with a tabu_var"""
        tot_numvars = len(self.explist)
        if self.influencedOnlyBy(tabu_vars, tot_numvars): return []
        if self.order() >= ss.max_order: return []
        bases11 = []
        tabu_explists = [base.explist for base in tabu_bases11]
        for var in range(len(self.explist)):
            if var in tabu_vars: continue
            cand_explist = self.explist[:]
            cand_explist[var] += 1
            if cand_explist in tabu_explists: continue
            bases11.append(PolyBase(cand_explist))
        return bases11

class QuadraticBase(IsleBase):
    """(scaled_x[var_index] - center_value11)^2
    Example: (x_i - 3.0)^2
    """
    def __init__(self, var_index, numvars, center_value11): 
        IsleBase.__init__(self)
        self.var_index = var_index
        self.numvars = numvars
        self.center_value11 = center_value11 #must correspond to var_index

        #Mean and stddev need to be non-None only for IsleModel influence calcs
        #So, they're calculated within the IsleModel constructor.
        self.mean = None
        self.stddev = None

    def setMeanAndStddev(self, X11):
        y = self.simulate(X11)                           
        self.mean = mathutil.average(y)
        self.stddev = mathutil.stddev(y)

    def __eq__(self, other):
        return self.var_index == other.var_index and \
               self.center_value11 == other.center_value11
    
    def __str__(self):        
        s = '(scaled_x%d - %g)^2' % (self.var_index, self.center_value11)
        return s

    def simulate(self, X11):
        y = (X11[self.var_index] - self.center_value11)**2
        return y

    def hasVar(self, var):
        return self.var_index == var

    def order(self):
        return 2
    
    def isTrivial(self):
        return False

    def higherOrderBases(self, tabu_bases11, tabu_vars, ss):
        raise NotImplementedError

class HSBase(IsleBase):
    """This is a 'hockey stick' (HS) function:
    -it's zero at one side of its knee value,
    -and on the other side it's a linear function of one variable.
    It's usually each item in the list of bases in a ProductBase
    """
    def __init__(self, sign, var, splitval11, q=1):
        IsleBase.__init__(self)

        #+1 or -1
        self.sign = sign

        #input variable
        self.var = var

        #the knee of this 'hockey stick' function
        self.splitval11 = splitval11

        #q==1 means first-order continuous (PWL)
        #q==2 means second-order continuous (ie derivatives are PWL)
        #(recommend q=1 unless you really need second-order continuous)
        self.q = q

    def __eq__(self, other):
        return self.sign == other.sign and \
               self.var == other.var and \
               self.splitval11 == other.splitval11 and \
               self.q == other.q 

    def __str__(self):
        s = ''
        t = self.splitval11
        if self.sign==-1:   s += 'HS(%.3e - scaled_x%d)' % (t, self.var)
        else:               s += 'HS(scaled_x%d - %.3e)' % (self.var, t)
        if self.q == 1:     pass
        elif self.q == 2:   s += '^2'
        else: raise
        return s

    def simulate(self, X11):
        return self.simulateAtVar(X11[self.var,:])
    
    def simulateAtVar(self, x11_at_var):
        """x11_at_var _must_ be at the variable of self.var, i.e.
        x11_at_var == X11[self.var]"""
        inf = float('Inf')
        t = self.splitval11
        y = self.sign * numpy.clip(x11_at_var - t, 0.0, inf)**self.q
        return y
    
    def hasVar(self, var):
        return self.var == var

    def order(self):
        return 1 #not self.q (although it could arguably be that)


class CartBase(IsleBase):
    def __init__(self, tuples, dvars, rel_influence):
        """Each tuple in 'tuples' list contains (splitvar,lte,splitval)"""
        IsleBase.__init__(self)
        self.tuples = tuples
        self.dvars = dvars #'decision variables' == all the variables
                           # that show up in 'splitvar' values of tuples
        self.rel_influence = rel_influence #'support' in Friedman refs

    def __str__(self):
        s = ''
        
        for tup_i, tup in enumerate(self.tuples):
            splitvar, lte, splitval = tup
            if lte:
                s += '(scaled_x%s <= %g)' % (splitvar, splitval)
                #if x11[splitvar] > splitval: return 0.0
            else:
                s += '(scaled_x%s > %g)' % (splitvar, splitval)
                #if x11[splitvar] <= splitval: return 0.0
            if tup_i < len(self.tuples)-1:
                s += '*'

        return s

    def simulate(self, X11):
        y = numpy.zeros(X11.shape[1], dtype=float)
        for i in range(X11.shape[1]):
            y[i] = self.simulate1(X11[:,i])
        return y

    def simulate1(self, x11):
        """Returns 0 or 1, depending if x is in range of all tuples"""
        for tup in self.tuples:
            splitvar, lte, splitval = tup
            if lte:
                if x11[splitvar] > splitval: return 0.0
            else:
                if x11[splitvar] <= splitval: return 0.0

        #passed _all_ tuples, so return 1 (True)
        return 1.0

    def hasVar(self, var):
        return var in self.dvars
    
    def order(self):
        return len(self.tuples)*2

    def isTrivial(self):
        return len(self.tuples) == 0
        
class ProductBase(IsleBase):
    """This is base that multiplies together other isle bases11."""
    def __init__(self, bases11):
        self.bases11 = bases11

        #Mean and stddev need to be non-None only for IsleModel influence calcs
        #So, they're calculated within the IsleModel constructor.
        self.mean = None
        self.stddev = None

    def setMeanAndStddev(self, X11):
        y = self.simulate(X11)                           
        self.mean = mathutil.average(y)
        self.stddev = mathutil.stddev(y)
        
    def __eq__(self, other):
        if isinstance(other, PolyBase):
            return self.explist == other.explist
        elif isinstance(other, LinearBase):
            return self.equalsLinearBase(other)
        elif isinstance(other, ProductBase):
            return self.equalsProductBase(other)
        else:
            raise AssertionError("Base type not handled:%s" % other.__class__)

    def equalsLinearBase(self, lin_base):
        assert isinstance(lin_base, LinearBase)
        if len(self.bases11) != 1:
            return False
        else:
            return self.bases11[0].equalsLinearBase(lin_base)

    def equalsPolyBase(self, poly_base):
        assert isinstance(poly_base, PolyBase)
        return poly_base.equalsProductBase(self)

    def __str__(self):
        #Corner case
        if len(self.bases11)==0:
            return '1'

        #Main case        
        s = []
        for i,base11 in enumerate(self.bases11):
            s += [str(base11)]
            if i < (len(self.bases11)-1):
                s += ['*']
        return ''.join(s)

    def isTrivial(self):
        return len(self.bases11) == 0

    def numKnots(self):
        """This is a complexity measure, needed for Mars model building"""
        count = 0
        for base11 in self.bases11:
            if isinstance(base11, HSBase):
                count += 1
            elif isinstance(base11, LinearBase):
                count += 0
            else:
                raise AssertionError("this base type needs to be supported")
        return count

    def simulate(self, X11):        
        y = numpy.ones(X11.shape[1], dtype=float)
        for base11 in self.bases11:
            next_y = base11.simulate(X11)
            y *= next_y
        return y

    def firingIndicesAtVar(self, x11_at_var, var):
        """Which indices i of x11_at_var == X[var,:] have an output > 0.0
        when simulating on 'this'?"""
        N = len(x11_at_var)
        firing_I = range(N)
        firing = numpy.ones(N)
        for base11 in self.bases11:
            if not isinstance(base11, HSBase): continue #let lin bases11 pass all
            if not base11.hasVar(var): continue
            x11_at_var_left = numpy.take(x11_at_var, firing_I)
            y_left = base11.simulateAtVar(x11_at_var_left)
            firing_I = [i for y_at_i,i in zip(y_left, firing_I)
                        if y_at_i > 0.0]
            
        return firing_I
    
    
    def hasVar(self, var):
        for base11 in self.bases11:
            if base11.hasVar(var): return True
        return False

    def order(self):
        #=='d' or 'depth'
        return sum(base11.order() for base11 in self.bases11)

def baseDiff(large_bases11, bases11_to_remove):
    """Return set-theory diff(large_bases11, bases11_to_remove)"""            
    return [large_base11
            for large_base11 in large_bases11
            if not large_base11 in bases11_to_remove]

def pruneRedundantBases(bases11):
    """
    WARNING: assumes that no bases here are equal!!
    
    -but checks for semantically-equivalent bases:
       -prune LinearBases with poly equivalent
       -prune ProductBases with poly equivalent
    -removes trivial bases too
    """
    return [base11
            for base11 in bases11
            if not (base11.isTrivial() or 
                    _linearBaseHasPolyEquivalent(base11, bases11) or
                    _productBaseHasPolyEquivalent(base11, bases11))]

def _linearBaseHasPolyEquivalent(base11, bases11):
    """If 'base' is a LinearBase and it has a PolyBase equivalent in 'bases'
    then return True.  Else return False."""
    if not isinstance(base11, LinearBase):
        return False
    for cand_base11 in bases11:
        if isinstance(cand_base11, PolyBase) and \
               cand_base11.equalsLinearBase(base11):
            return True
    return False
    
def _productBaseHasPolyEquivalent(base11, bases11):
    """If 'base' is a LinearBase and it has a PolyBase equivalent in 'bases'
    then return True.  Else return False."""
    if not isinstance(base11, ProductBase):
        return False
    for cand_base11 in bases11:
        if isinstance(cand_base11, PolyBase) and \
               cand_base11.equalsProductBase(base11):
            return True
    return False

class IsleBasesFactory:
    """For building a list of IsleBase objects from an arbitrary regressor.
    
    Note that only buildFromRegressor() is public; this ensures
    that fewer assumptions need to be made about the regressor
    coming in, and therefore fewer accidents.
    """

    def buildFromRegressor(self, model, X11):
        if isinstance(model, LinearModel):
            return self._buildFromLinearModel(model, X11)
        else:
            raise AssertionError, \
                  "unsupported regressor type: %s" % str(model.__class__)
            
    def _buildFromLinearModel(self, linear_model, X11):
        m = linear_model
        return [LinearBase(var_i, mathutil.average(X11[var_i,:]),
                           mathutil.stddev(X11[var_i,:]))
                for var_i, coef in enumerate(m.coefs[1:])
                if coef != 0.0]


class IsleBasesFactory:
    """For building a list of IsleBase objects from an arbitrary regressor.
    
    Note that only buildFromRegressor() is public; this ensures
    that fewer assumptions need to be made about the regressor
    coming in, and therefore fewer accidents.
    Example of accident: linearModelFactory() produces a ConstantModel
    then a buildFromLinearModel() is called.  We avoid that.
    """

    def buildFromRegressor(self, model, X11):
        if isinstance(model, ConstantModel):
            return self._buildFromConstantModel(model, X11)
        elif isinstance(model, SgbModel) or isinstance(model, RforModel):
            return self._buildFromCarts(model.carts)
        elif isinstance(model, LinearModel):
            return self._buildFromLinearModel(model, X11)
        elif isinstance(model, CartModel):
            return self._buildFromCart(model)
        else:
            raise AssertionError, \
                  "unsupported regressor type: %s" % str(model.__class__)
            
    def _buildFromConstantModel(self, constant_model, X11):
        return []
            
    def _buildFromLinearModel(self, linear_model, X11):
        if isinstance(linear_model, ConstantModel):
            return 
        m = linear_model
        return [LinearBase(var_i, mathutil.average(X11[var_i,:]),
                           mathutil.stddev(X11[var_i,:]))
                for var_i, coef in enumerate(m.coefs[1:])
                if coef != 0.0]
    
    def _buildFromCarts(self, carts):
        return [base
                for cart in carts
                for base in self._buildFromCart(cart)]

    def _buildFromCart(self, cart):    
        """Convert a cart to a list of IsleBase objects.  
        Note: Unlike nodes in a cart tree, CartBase objects are all independent
          of each other.
        """
        rules = [] #build this

        nodenum_stack = [0]
        nodetuples_stack = [[]]
        dvars_stack = [[]]

        while nodenum_stack:
            node = cart.nodes[nodenum_stack.pop(0)]
            nodetuples = nodetuples_stack.pop(0)
            dvars = dvars_stack.pop(0)

            rules.append(CartBase(nodetuples, dvars, node.rel_influence))

            if ((node.Lchild is None) or
                ((node.Lchild == 0) and
                (node.Rchild == 0))):
                pass
            else:
                new_dvars = dvars + [node.splitvar]
                
                right_noderule= nodetuples + [(node.splitvar,
                                               False,
                                               node.splitval)]
                
                left_noderule = nodetuples + [(node.splitvar,
                                               True,
                                               node.splitval)]

                nodenum_stack.append(node.Rchild)
                nodetuples_stack.append(right_noderule)
                dvars_stack.append(new_dvars)

                nodenum_stack.append(node.Lchild)
                nodetuples_stack.append(left_noderule)
                dvars_stack.append(new_dvars)

        return rules
    
