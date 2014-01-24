""" Cart.py: Builds and simulates CART models

Ref: Breiman, Friedman, Olshen, and Stone.  Classification and Regression Trees.  1980.
"""
import copy
import random
import math
import os
import types

import numpy

from util import mathutil
import RegressorUtils
from util import constants

class CartModel:
    """
    @description
    
    A CartModel is a regressor based upon the Classification and Regression
    Tree described in the following reference:
        
    One tree, a 'CART' tree, i.e. 'Classification and Regression Tree'
    Ref L. Breiman & J. Freidman, 1982
    
    @attributes
    
    nodes -- list of TreeNode objects, which collectively define the tree
    """
    
    def __init__(self, nodes, numvars):
        self.nodes = nodes
        self.numvars = numvars

    def simulate(self, X):
        """
        @description
          
          Returns a vector containing the output of the CartModel simulated at
          the inputs given in the array X.

          While it has many speedups, some special optimizations that simulateFast
          has which requires different input arguments.

        @arguments
        
          X -- 2d array -- inputs [var #][sample #]
    
        @return
        
          yhat -- 1d array -- outputs [sample #], see Description
        
        @exceptions

        @notes
        """
        """
        This is an alternate implementation of simulate, used for validation.
        For each column in X
            Run the column through the cart model
            Add the leaf constval to the y array
        Return the y array
        """
        rows, cols = X.shape
        y = numpy.zeros(cols, dtype=float)
        
        for col in range(cols):
            current_col = X[:,col]
            node_index = 0
            while (self.nodes[node_index].Lchild != 0):
                # not yet at a leaf node
                splitvar = self.nodes[node_index].splitvar
                splitval = self.nodes[node_index].splitval
                tmp = current_col[splitvar]
                if (tmp <= splitval):
                    node_index = self.nodes[node_index].Lchild
                else:
                    node_index = self.nodes[node_index].Rchild
            # at a leaf node
            y[col] = self.nodes[node_index].constval
            
        return y
        
    def __str__(self):
        s = "CartModel={\n" 
        for i in range(len(self.nodes)):
            s += str(i) + " : " + str(self.nodes[i]) + "\n"
        s += "}"
        return s
    

class CartBuildStrategy:
    
    #minimize_memory = bool(os.environ.get('SAYO_CART_MINIMIZE_MEMORY', False))

    def __init__(self):

        #don't split nodes in regions that have with num samples < min_node_N
        self.min_node_N = 3 #[3,3..7]

        #max subtree depth (==max # vars interact)
        self.max_depth = 3  #[3,2..6] 

        #with each split, how many unique x's allowed?  numvars:max_unique
        self.max_unique_xs_per_split = {1:200, 2:50, 3:25, 4:15, 'default':10}

        #search for the use of this to see its effect.
        self.num_train_vars_R = 1.0

        #This affects the relative chance of each input variable being
        # selected during chooseSplit.  If None, they all have equal bias.
        # Else it is a dict of varname:relative_bias
        self.var_biases = None
        
        #in sse calcs, error of sample i is proportional to weight of sample i
        #If None, all samples are treated equally
        self.weight_per_sample = None


    def __str__(self):
        s = "CartBuildStrategy={"
        s += ' min_node_N=%d' % self.min_node_N
        s += '; max_depth=%d' % self.max_depth
        s += '; max_unique_xs_per_split=%s' % str(self.max_unique_xs_per_split)
        s += '; num_train_vars_R=%s' % self.num_train_vars_R

        s += '; var_biases='
        if self.var_biases is None: s += 'None'
        else: s += str(self.var_biases[:2]) + '... '
        
        s += '; weight_per_sample='
        if self.weight_per_sample is None: s += 'None'
        else: s += str(self.weight_per_sample[:3]) + '... '
        
        s += ' /CartBuildStrategy}'
        return s


class CartFactory:
    """
    @description
    
    @attributes
    
      nodes -- list -- nodes built so far
      stack -- list -- maintains state of building process
      ss -- CartBuildStrategy object --
    """

    def __init__(self):
        self.num_nodes = None
        self.nodes_dict = {} #node_number : TreeNode
        self.stack = [] 
        self.ss = None
        
    def trivialBuild(self, node, numvars):
        """
        @description
        
          Build a CartModel that only has one node, therefore a constant
        
        @arguments
        
          node -- TreeNode object -- single, constant TreeNode in the CartModel
          numvars -- int -- num input vars
        
        @return
        
          model -- CartModel object --
        
        @exceptions
        
        @notes
        """
        return CartModel([node], numvars)


    def build(self, X, y, ss):
        """
        @description
        
          Returns a CartModel built using the training samples in array X and 
          training values in vector y. The solution strategy ss is used to guide
          the build.

        @arguments
        
          X -- 2d array -- input points [1..n var #][1..N sample #]
          y -- 1d array -- target outputs [1..N sample #]
          ss -- CartBuildStrategy object -- 
    
        @return
        
          model -- CartModel object --

        @exceptions

        @notes
        """
        self.ss = ss
        n, allN = X.shape
        allX, ally = X, y
        n_try = self.numSplitVariablesToTry(ss.num_train_vars_R, n)
        range_all_n = range(n)

        self.num_samples = allN
        
        # initialize weights (per sample)
        allweights = self.ss.weight_per_sample

        # initialize nodes: start with a (trival) root TreeNode
        self.num_nodes = 0
        self.nodes_dict = {}
        self.nodes_dict[self.num_nodes] = TreeNode()
        self.nodes_dict[self.num_nodes].constval = numpy.average(y)
        self.num_nodes += 1
        
        # initialize the stack, which aids construction by keeping
        # track of node number, depth, and indices of training samples 'I'
        self.stack = [self.StackEntry(node_num=0, depth=1, I=range(allN))]

        while len(self.stack) > 0:           
            # pop the current entry from the stacks
            entry = self.stack.pop(0)
            node_num, cur_depth, I = entry.data()

            # the number of samples considered by the current node
            N = len(I)
            
            # subsample the appropriate input, target output, and weights
            X, y = numpy.take(allX, I, 1), numpy.take(ally, I)
            if allweights is None: weights = None
            else:                  weights = numpy.take(allweights, I)
            
            # update the const value (used for regression) of the current node 
            self.nodes_dict[node_num].constval = numpy.average(y)

            if min(y) == max(y): #degenerate case
                parent = self.nodes_dict[node_num].parent
                continue
  
            if N <= self.ss.min_node_N: #no further splitting
                parent = self.nodes_dict[node_num].parent
                continue
            
            else: #split
                # choose the variable and value to split on
                [bestvar, bestval] = self._chooseSplit(X, y, weights, n_try,
                                                       range_all_n)

                # the values of the samples at the best variable
                xs_of_bestvar = X[bestvar,:]
                
                # update the masks based on the bestval/bestvar
                right_mask = numpy.where(xs_of_bestvar > bestval, 1,0)
                left_mask = 1 - right_mask
                
                # update the number of samples in the right and left branches
                N_right = sum(right_mask)
                N_left = N - N_right

                # no splitting is required if everything is in the left/right
                # branch or if we have explored to a predefined maximum depth 
                if ((N_right == 0) or 
                    (N_left == 0) or 
                    (cur_depth >= self.ss.max_depth)):
                    parent = self.nodes_dict[node_num].parent

                # otherwise it is is possible to add further braches to 
                # improve the accuracy of the model
                else:
                    # set the splitting parameters of the current node to the
                    # best possible choice 
                    self.nodes_dict[node_num].splitval = bestval
                    self.nodes_dict[node_num].splitvar = bestvar
                        
                    # add left/right braches to the regression tree. this 
                    # method also updated the stacks with the newly added
                    # children, so that exploration can continue
                    self._addChildren(node_num, I, cur_depth,  right_mask, left_mask, y)

        nodes_list = [self.nodes_dict[i] for i in range(len(self.nodes_dict))]
        cart = CartModel(nodes_list, n)

        return cart

    def numSplitVariablesToTry(self, num_train_vars_R, num_vars):
        """When choosing a subset of nodes for splitting, we use a subset"""
        #Ref Breiman
        n_try = max(1, int(num_train_vars_R * math.sqrt(num_vars))) 
        return n_try
    
    def _addChildren(self, parent_node_num, I, cur_depth, right_mask, left_mask, y):
        """Modifies self.nodes_dict, self.stack
        """
        # do the right branch
        r_tmp = numpy.compress(right_mask, y)
        if (r_tmp.shape[0] > 0): 
            # then there is a reason to add a right child
            r_node_num = self.num_nodes
            r_child = TreeNode()
            r_child.parent = parent_node_num
            r_child.constval = numpy.average(r_tmp)
            self.nodes_dict[parent_node_num].Rchild = r_node_num
            self.nodes_dict[r_node_num] = r_child
            self.stack.append( self.StackEntry(r_node_num, cur_depth+1,\
                                               numpy.compress(right_mask, I)))
            self.num_nodes += 1
       
        # do the left branch
        l_tmp = numpy.compress(left_mask, y)
        if (l_tmp.shape[0] > 0): 
            l_node_num = self.num_nodes
            l_child = TreeNode()
            l_child.parent = parent_node_num
            l_child.constval = numpy.average(l_tmp)
            self.nodes_dict[parent_node_num].Lchild = l_node_num
            self.nodes_dict[l_node_num] = l_child
            self.stack.append( self.StackEntry(l_node_num, cur_depth+1,\
                                               numpy.compress(left_mask, I)))
            self.num_nodes += 1

    def _chooseSplit(self, X, y, weights, n_try, range_all_n):
        """
        Choose one of the vars of X to split on, and a corresponding
        split value.  Don't look at all options, but of the options
        examined, return the one that gives the lowest nmse.

        Speed of this routine is ultra-important, so many steps have been taken
        to improve that, even at the expense of understandability.
        
        A simple, slow version of the routine is:

        vars_to_consider = subset_of_all_vars
        best_nmse = Inf
        best_splitvar = None
        best_splitval = None
        for candidate_splitvar in vars_to_consider:
          for candidate_splitval in each unique value of X[var]:
            cand_nmse = nmse if we split at (candidate_splitvar, cand_splitval)
            if cand_nmse < best_nmse:
              best_nmse, best_splitvar, best_splitval =
                cand_nmse, cand_splitvar, cand_splitval
        return (best_splitvar, best_splitval).

        Note that the nmse calculations are modified by weights
        (weight per sample). 
        """
        all_n, N = X.shape

        assert self.ss.var_biases is not None
        nonzero_vars = [var
                        for var,bias in enumerate(self.ss.var_biases) if bias>0]
        n_nonzero = len(nonzero_vars)
        max_unique_xs = self._maxUniqueXs(n_nonzero)

        if weights is None:
            weights_times_y = y
            sumsq = numpy.dot(y, numpy.transpose(y))
        else:
            weights_times_y = weights * y
            sumsq = numpy.dot(weights_times_y,
                                numpy.transpose(weights_times_y))
        sumy = sum(weights_times_y)
  
        vars_to_consider = None
        if n_nonzero == 0:
            vars_to_consider = [random.choice(range_all_n)]
        elif n_nonzero <= n_try:
            vars_to_consider = nonzero_vars
        else:
            #there are many values of var_biases that can break;
            # let the check-point be randIndices instead of here
            try:
                vars_to_consider = mathutil.randIndices(self.ss.var_biases, n_try, False)
            except SystemExit: raise
            except Exception:
                vars_to_consider = random.sample(range_all_n, n_try)

        #make function access faster
        numpy_argsort = numpy.argsort
        numpy_take = numpy.take
        random_sample = random.sample
        mathutil_drawFromPyramidDistribution = mathutil.drawFromPyramidDistribution
            
        bestsse, bestvar, bestval = float('inf'), 0, X[0,0]
        for var_to_consider in vars_to_consider:
            xs = X[var_to_consider,:]
            I = numpy_argsort(xs)
            ys = numpy_take(weights_times_y, I)
            xs = numpy_take(xs, I)

            #take steps to cut down num x's
            #1. ideally, x's were discretized via optPointMeta.discretize()

            #2. unique-ify
            unique_xs = list(set(xs))

            #3. random subset of samples
            maxl = min(max_unique_xs, len(unique_xs))
            unique_xs = random_sample(unique_xs, maxl)

            #now, go through xs and unique_xs
            j = -1
            tot_lt = 0.0
            for thr_x in sorted(unique_xs):
                if j==N: continue

                #find the j such that x[j] is <= thr_x but x[j+1] is not
                while xs[j+1] <= thr_x:
                    j+=1
                    tot_lt += ys[j]
                    if (j+1) >= N:
                        break

                num_lt = j+1
                num_gt = N - num_lt
                
                if num_lt==0 or num_gt==0:
                    continue

                tot_gt = sumy - tot_lt
                sse = sumsq - \
                      (tot_lt * tot_lt / num_lt) - \
                      (tot_gt * tot_gt / num_gt)                
                if sse < bestsse:
                    bestsse = sse
                    bestvar = var_to_consider
                    d = xs[j+1] - thr_x

                    #if there was no smoothing:
                    # halfway between entries
                    #bestval = thr_x + d/2.0
                    
                    #simple way to smooth: unif. bias in [bestval,bestval+offset]
                    #bestval = thr_x + random.uniform(0.0,0.99*d)

                    #better way to smooth: bias towards the center
                    offset = mathutil_drawFromPyramidDistribution(d)*0.9999
                    bestval = thr_x + offset

        assert bestvar is not None
        return [bestvar, bestval]

    def _maxUniqueXs(self, numvars_allowed):
        """Returns max num x's possible to consider splitting at, given
        the number of variables that we're considering splitting at"""
        if self.ss.max_unique_xs_per_split.has_key(numvars_allowed):
            return self.ss.max_unique_xs_per_split[numvars_allowed]
        else:
            return self.ss.max_unique_xs_per_split['default']

    class StackEntry:
        """An entry in self.stack, which maintains the overall state
        of the building process"""
        def __init__(self, node_num, depth, I):
            self.node_num = node_num
            self.depth = depth
            self.I = I 

        def data(self):
            return [self.node_num, self.depth, self.I]       
            
class TreeNode:
    """
    @description

    A TreeNode represents a decision point in a classification and regression
    tree (CART). 
        
    @attributes
    
    parent -- TreeNode -- The TreeNode's parent in the CART. All nodes except 
                          for the root node have parent != None
                          
    Lchild -- TreeNode -- The TreeNode's child in the CART. A node with
                          Lchild == None and Rchild == None is referred to as
                          a 'leaf' node.
    splitval -- AUDIT
    
    splitvar -- AUDIT
    
    constval -- float -- The TreeNode's contant value when used as a regression 
                         tree.
                         
    """ 
    
    def __init__(self):
        # Default values for indices are now "0", the root node, rather
        # than "None", which is equivalent to uninitialized.
        self.Lchild = 0
        self.Rchild = 0
        self.parent = 0
        self.splitval = None
        self.splitvar = None
        self.constval = None
    
    def __str__(self):
        s = "{ "
        s += "parent=" + str(self.parent) + ", "
        if self.Lchild is not None:
            s += "splitval=" + str(self.splitval) + ", "
            s += "splitvar=" + str(self.splitvar) + ", "
            s += "constval=" + str(self.constval) + ", "
            s += "Lchild=" + str(self.Lchild) + ", "
            s += "Rchild=" + str(self.Rchild) + ", "
        else:
            s += "constval=" + str(self.constval)
        s += "} "
        return s
