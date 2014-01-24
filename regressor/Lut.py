"""Lut.py = Look-Up Table

Holds:
-LutStrategy
-LutModel
-LutFactory

"""

import logging
import math
import random
import types

import numpy

from adts import *
from util import mathutil
from util.ascii import *

#KDTree was removed in r168
#from util.KDTree import *

log = logging.getLogger('lut')

class LutStrategy:
    """
    @description

      Holds parameters specific to the strategy of building LutModel objects.
      
    @attributes

      bandwidth -- float in (0.0, 1.0] -- the bigger this number is,
        the more 'smoothing' it does, i.e. the more that faraway
        training points matter.
      use_kd_tree -- boolean -- use k-d tree for faster simulation (at the
        cost of maybe some accuracy)
    """ 
    def __init__(self):

        #set data
        self.bandwidth = 0.05
        self.use_kd_tree = False #currently not supported use_kd_tree
        
    def __str__(self):
        s = 'LutStrategy={'
        s += ' bandwidth=%.3f' % self.bandwidth
        s += ' use_kd_tree=%s' % self.use_kd_tree
        s += ' /LutStrategy}'
        return s


class LutFactory:
    """
      Builds a LutModel
    """ 
    def __init__(self):
        pass

    def build(self, X, y, ss):
        """
        @description

          Builds a LutModel, given a target mapping of X=>y and a strategy 'ss'.
        
        @arguments

          X -- 2d array [input variable #][sample #] -- training input data
          y -- 1d array [sample #] -- training output data
          ss -- LutStrategy --
        
        @return

          lut_model -- LutModel object
    
        @exceptions
    
        @notes

          In constrast to most regressor factories, 
          this factory is trivial because all the work for a lookup table
          model is during simulation.
        """
        log.info("Build LutModel: begin")
        model = LutModel(X, y, ss.bandwidth, ss.use_kd_tree)
        log.info("Build LutModel: done")
        return model
            

class LutModel:
    """
    @description

      Simulatable model.
      
    @attributes

      keep_I -- list of int -- indices of input vars that actually vary
      min_x -- list of float -- min val for each _varying_ input var
      max_x -- list of float -- max val for each _varying_ input var    
      training_X01 -- 2d array [0 .. # varying input variables-1][sample #] --
        the training input data, but each value is scaled to be in
        [0,1] based on the max and min value found for that input variable
      training_y -- 1d array [sample #] -- all the training output data
      bandwidth -- float -- kernel width
      
    @notes

    """ 
    def __init__(self, X, y, bandwidth, use_kd_tree):
        """
        @description

            Constructor.
        
        @arguments

          X -- 2d array [input variable #][sample #] -- the training input data
          y -- 1d array [sample #] -- all the training output data
          bandwidth -- float in (0.0, 1.0]
          use_kd_tree -- bool
        
        @return

          lut_model -- LutModel object -- a simulatable model
    
        @exceptions
    
        @notes
          
        """
        #identify the input variables that vary (ie min < max)
        full_min_x = mathutil.minPerRow(X)
        full_max_x = mathutil.maxPerRow(X)
        self.keep_I = [i
                       for i,(mn,mx) in enumerate(zip(full_min_x, full_max_x))
                       if mn < mx]
        
        #only work with input variables that vary as we
        # save min_x, max_x, training_X01/y
        self.min_x = list(numpy.take(full_min_x, self.keep_I))
        self.max_x = list(numpy.take(full_max_x, self.keep_I))
        
        keep_X = numpy.take(X, self.keep_I, 0)
        self.training_X01 = mathutil.scaleTo01(keep_X, self.min_x, self.max_x)
        self.training_y = y
        
        self.bandwidth = bandwidth
        
        if use_kd_tree:
            self.have_kd_tree = True
            self.training_tree = KDTree(self.training_X01.shape[0], 1)
            tt = numpy.transpose(self.training_X01).astype("f")
            self.training_tree.set_coords(tt)
            self.bandwidth_increase = bandwidth * 0.01
            self.min_nb_lut_indices = 20
        else:
            self.have_kd_tree = False
            self.training_tree = None
            self.bandwidth_increase = None
            self.min_nb_lut_indices = None
        
    def __str__(self):
        s = 'LutModel={'
        s += ' bandwidth=%.3f' % self.bandwidth
        s += '; # varying input variables=%d' % self.training_X01.shape[0]
        s += '; # training samples=%d' % self.full_X.shape[1]
        s += '; have_kd_tree = %s' % self.have_kd_tree
        s += ' /LutModel}'
        return s

    def simulate(self, X):
        """
        @description

          For each input point (column) in X, compute the response
          of this model.
        
        @arguments
        
          X -- 2d array [input variable #][sample #] -- inputs 
        
        @return

          yhat -- 1d array [sample #] -- simulated outputs
    
        @exceptions
    
        @notes
        """
        keep_X = numpy.take(X, self.keep_I, 0)
        X01 = mathutil.scaleTo01(keep_X, self.min_x, self.max_x)
        N = X01.shape[1]
        yhat = numpy.zeros(X.shape[1], dtype=float)
        
        for sample_i in range(N):
            if (sample_i % 100) == 0:
                log.debug('Simulate sample %d / %d' % (sample_i+1, N))
            if self.have_kd_tree:
                yhat[sample_i] = self.simulate1_scaled01_kdtree(X01[:,sample_i])
            else:
                yhat[sample_i] = self.simulate1_scaled01_slow(X01[:, sample_i])

        return yhat

    def simulate1(self, x):
        X = numpy.reshape(x, (len(x),1))
        return self.simulate(X)[0]

    def simulate1_scaled01_slow(self, x01):
        """Simulate for an input point x01 (it's already been scaled
        such that each variable is scaled to be in [0,1] range).
        Simulate by measuring distance to all training samples (i.e. slow!)"""
        target_num_interpolants = max(2, len(x01))
        sum_w, sum_output, num_interpolants = 0.0, 0.0, 0
        cur_bw = self.bandwidth

        N_trn = self.training_X01.shape[1]
        dists = [mathutil.distance(x01, self.training_X01[:,trn_sample_j])
                 for trn_sample_j in range(N_trn)]
        
        while num_interpolants < target_num_interpolants and cur_bw <= 1.5:
            sum_w, sum_output, num_interpolants = 0.0, 0.0, 0
            for trn_sample_j, dist in enumerate(dists):
                w = mathutil.epanechnikovQuadraticKernel(dist, cur_bw)
                #if dist == 0:
                #    w = 1e10
                #else:
                #    w = 1.0/(dist**2)
                    
                sum_output += w * self.training_y[trn_sample_j]
                sum_w += w
                if w > 0:
                    num_interpolants += 1

            #increase bandwidth in the case of re-loop
            cur_bw *= 1.1

        if sum_w > 0:
            y1 = sum_output / sum_w
        else:
            y1 = 0.0
            
        return y1

    def simulate1_scaled01_kdtree(self, x01):
        """Simulate for an input point x01 (it's already been scaled
        such that each variable is scaled to be in [0,1] range).
        Avoid measuring distance to all training samples, via k-d tree"""

        sum_w, sum_output = 0.0, 0.0
        point = numpy.transpose(X01[:, sample_i]).astype("f")
        bw = self.bandwidth

        # search neighbors
        self.training_tree.search(point, bw)

        # get indices & radii of points
        indices=self.training_tree.get_indices()

        if len(indices) < self.min_nb_lut_indices:
            log.warning('Not enough LUT indices found, increasing bandwidth...')
            
            while len(indices) < self.min_nb_lut_indices:              
                bw=bw + self.bandwidth_increase

                # search neighbors
                self.training_tree.search(point, bw)

                # get indices & radii of points
                indices=self.training_tree.get_indices()
                if bw > 1:
                    log.warning(' Bandwith too high, out of range')
                    bw=1
                    break

            log.warning(' Increased bandwidth to %f, found %d indices' %\
                    (bw, len(indices)))

        for trn_sample_j in indices:
            dist = mathutil.distance(X01[:, sample_i],
                                     self.training_X01[:,trn_sample_j])
            w = mathutil.epanechnikovQuadraticKernel(dist, bw)
            sum_output += w * self.training_y[trn_sample_j]
            sum_w += w


        if sum_w > 0.0:
            y1 = sum_output / sum_w
        else:
            y1 = 0.0

        return y1

class LutDataPruner:
    """Prunes data in a lookup table"""
    def __init__(self):
        pass
        
    def prune(self, X, y, Xy, thr_error, min_N, pruned_filebase, all_varnames):
        """
        @description

          Given a dataset of X=>y, prune away samples until
          we either get error > thr or # samples < min_N.  Returns
          results both as a list and into files.
        
        @arguments

          X -- 2d array -- has _all_ input data
          y -- 1d array -- has _all_ output data
          Xy -- 2d array -- has an extra row for y.  Note that the ordering
            of the rows may be different than that for X.  It will output
            the pruned_filebase using the pruned samples from this dataset.
          thr_error -- float -- 
          min_N -- int -- stop pruning if len(pruned_y) < min_N
          pruned_filebase -- string -- during pruning, periodically
            save results to pruned_filebase.hdr/.val
          all_varnames -- list of string -- list of varnames, needed
            for saving pruned_filebase.hdr
            
        @return
        
          keep_I -- list of int -- the indices of the samples of X or
            y that we want to keep.  Rest have been pruned away.
          AND
          <<pruned_filebase.hdr, pruned_filebase.val>>
    
        @exceptions
    
        @notes
        """
        assert X.shape[0]+1 == Xy.shape[0] == len(all_varnames) 
        assert X.shape[1] == Xy.shape[1] == len(y)
        N = len(y)
        keep_I = range(N)

        max_error = 0
        for prune_iter in range(100000):
            log.info('=======================================================')
            log.info('LutDataPruner iteration #%d; #samples init=%d, now=%d' %
                     (prune_iter, N, len(keep_I)))
            max_error, next_keep_I = self._prune1(X, y, keep_I, thr_error)
            if len(next_keep_I) <= min_N:
                log.info('Stop pruning because we have <= min num samples')
                break
            elif max_error > thr_error:
                log.info('Stop pruning because it would exceed error threshold')
                break
            else:
                keep_I = next_keep_I

            # -periodically store data to file
            if prune_iter > 10 and prune_iter%10 == 0:
                keep_Xy = numpy.take(Xy, keep_I, 1)
                trainingDataToHdrValFiles(pruned_filebase, all_varnames,keep_Xy)
                log.info("Updated pruned output in %s.hdr, %s.val" %
                         (pruned_filebase, pruned_filebase))
                
                
        log.info('=======================================================')
        return keep_I

    def _prune1(self, X, y, keep_I, thr_error):
        """
        @description

          Strategy: keep randomly choosing samples from keep_I until we find
          one whose removal still has error < thr_error.

          If we can't find such a sample, prune next-best sample.
        
        @arguments

          X -- 2d array -- has _all_ data
          y -- 1d array -- has _all_ data
          keep_I -- list of int -- the indices of the samples of X or
            y that we want to keep.  Rest have been pruned away.
          thr_error -- float -- 
            
        @return

          new_keep_I -- list of int -- remove_I, with one more
            sample removed from
    
        @exceptions
    
        @notes
        """
        num_cands = 10000 #magic number alert

        #
        lut_factory = LutFactory()
        lut_ss = LutStrategy()

        #choose which samples we consider pruning away
        N = len(keep_I)
        num_cands = min(N, max(1, num_cands))
        cand_sample_I = random.sample(keep_I, num_cands)

        #'best' here is the sample which returns the lowest error
        best_error, best_keep_I = float('inf'), None
        for j, cand_sample_i in enumerate(cand_sample_I):
            cand_keep_I = [i for i in keep_I if i != cand_sample_i]
            model = lut_factory.build(numpy.take(X, cand_keep_I, 1),
                                      numpy.take(y, cand_keep_I), lut_ss)
            error = abs(model.simulate1(X[:,cand_sample_i]) - y[cand_sample_i])
            error = error / (max(y) - min(y)) #normalize
            
            if error < best_error:
                best_error = error
                best_keep_I = cand_keep_I

            if (j % 10) == 0 or cand_sample_i == cand_sample_I[-1]:
                log.info('  LutDataPruner cand #%i/%i; error=%8g, lowest=%8g' % 
                         (j+1, num_cands, error, best_error))

            if error < thr_error:
                log.info('  Found candidate with error < %g (cand #%d)' %
                         (thr_error, j+1))
                break

        return best_error, best_keep_I

