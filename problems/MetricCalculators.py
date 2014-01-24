"""MetricCalculators.py

Has fancier calculation methods for metrics.

Each of these objects can be used as 'metrics_calculator' objects
within a Simulator.  To do so, each must supply the following routines:
-compute(2d array of waveforms), returning dict of metric_name : metric_value
-metricNames(), returning list of string
"""
from itertools import izip
import logging
import types
import math
import util.mathutil as mathutil

import numpy

from util import mathutil
from regressor.Pwl import PwlFactory, PwlBuildStrategy, PWL_APPROACHES
from util.octavecall import plotAndPause

log = logging.getLogger('metric_calc')

class NmseOnTargetWaveformCalculator:
    def __init__(self, target_waveform, index_of_cand_waveform):
        """
        @description
        
        @arguments
        
          target_waveform -- 1d array --  'y'
          index_of_cand_waveform -- int -- 'yhat' = waveforms[index]
        
        @return
    
        @exceptions
    
        @notes
        """
        #preconditions
        assert isinstance(target_waveform, types.ListType) or \
               len(target_waveform.shape) == 1
        assert isinstance(index_of_cand_waveform, types.IntType)

        #work...
        self.y = target_waveform
        self.index_of_cand_waveform = index_of_cand_waveform
        self.yrange = max(target_waveform) - min(target_waveform)
        self.yrange = max(self.yrange, 1.0e-15) #make it >0 by small amount

    def metricNames(self):
        """Returns a list of the metric names that this class computes
        """
        return ['nmse']

    def compute(self, waveforms_array):
        """Returns a dict of metric_name : metric_value,
        which are computed from 'waveforms_array'
        """
        yhat = waveforms_array[self.index_of_cand_waveform]
        return {'nmse' : nmseFromDenom(self.y, yhat, self.yrange)}


def nmseFromDenom(waveform1, waveform2, denom):
    """Returns normalized sum of squared differences between
    waveform1 and waveform2.  Normalizes via (waveform1-waveform2).
    """
    assert denom > 0.0
    if len(waveform1) != len(waveform2):
        log.warning('Waveforms have different lengths: %d and %d' %
                    (len(waveform1), len(waveform2)))
        return BAD_METRIC_VALUE
    return (sum( ((w1 - w2)/denom)**2
                 for (w1,w2) in zip(waveform1, waveform2)) )


class NmseOnTargetShapeCalculator:
    
    def __init__(self, target_shape, input_var_index, output_var_index):
        """
        @description

          1. Fits a regression model (constrained by target_shape)
          that maps x to yhat with minimum nmse
          2. Returns the minimum nmse
        
        @arguments
        
          target_shape -- one of regressor.Pwl.PWL_APPROACHES, such as 'hockey'
          input_var_index -- int -- X[0,:] = waveforms[index,:]
          output_var_index -- int -- 'yhat' = waveforms[index,:]
        
        @return
    
        @exceptions
    
        @notes

          The input vector currently only has one variable, but
          we could have more if we wished.
        """
        #preconditions
        assert target_shape in PWL_APPROACHES
        assert isinstance(output_var_index, types.IntType)
        assert isinstance(input_var_index, types.IntType)

        #main work...
        self.shape = target_shape
        self.input_var_index = input_var_index
        self.output_var_index = output_var_index

    def metricNames(self):
        """Returns a list of the metric names that this class computes
        """
        return ['nmse']

    def compute(self, waveforms_array):
        """Returns a dict of metric_name : metric_value,
        which are computed from 'waveforms_array'
        """
        x = waveforms_array[self.input_var_index]
        y = waveforms_array[self.output_var_index]
        
        #fit the lowest-nmse regressor possible to yhat, where
        # the regressor structure is governed by self.shape
        ss = PwlBuildStrategy(self.shape)
        model = PwlFactory().build(x, y, ss)

        yhat = model.simulate(x)
        nmse = mathutil.nmse(yhat, y, min(y), max(y))

        ##uncomment the next two lines for manual testing
        #from util.octavecall import plotAndPause
        #plotAndPause(x, y, x, yhat,
        #             title='red=sim waveform, blue=fitted waveform')
        
        return {'nmse' : nmse}

class TransientWaveformCalculator:
    
    def __init__(self, input_var_index, output_var_index,
                 chop_initial):
        """
        @description

          uses a heuristic to calculate DR, SR, offset etc from
          the transient data. very tied to the problem setup.
          
        @arguments
        
          input_var_index -- int -- X[0,:] = waveforms[index,:]
            (i.e. index of the 'time' variable)
          output_var_index -- int -- 'yhat' = waveforms[index,:]
          chop_initial -- bool -- chop off some of initial waveform?
            (the specific heuristics are in the code below)
        
        @return
    
        @exceptions
    
        @notes

          The input vector currently only has one variable, but
          we could have more if we wished.
        """
        #preconditions
        assert isinstance(output_var_index, types.IntType)
        assert isinstance(input_var_index, types.IntType)
        assert isinstance(chop_initial, types.BooleanType)

        #main work...
        self.input_var_index = input_var_index
        self.output_var_index = output_var_index
        self.chop_initial = chop_initial

    def metricNames(self):
        """Returns a list of the metric names that this class computes.
        Order is not important.
        """
        return ['dynamic_range', 'slewrate', 'slewrate_log', 'offset', 'pulse_width']

    def compute(self, waveforms_array):
        """Returns a dict of metric_name : metric_value,
        which are computed from 'waveforms_array'
        """
        x = waveforms_array[self.input_var_index]
        y = waveforms_array[self.output_var_index]

        #corner case
        if (not x) or (not y):
            log.info('Missing x or y, so return bad results')
            return self._badResults()

        #main case...
        I = numpy.argsort(x)
        x = [x[i] for i in I]
        y = [y[i] for i in I]

        #chop out the initial y's that are >> nearly_min_y (plus
        # some extra neighboring data for cleanness)
        # -another strategy would be: keep chopping initial x's until delta_y>=0
        if self.chop_initial:
            #plotAndPause(x, y)
            miny = min(y)
            maxy = max(y)
            nearly_miny = miny + 0.15 * (maxy - miny) #within 10% of min
            for (i, (xi, yi)) in enumerate(izip(x, y)):
                if yi <= nearly_miny:
                    break
            extra_i = int(0.05 * len(x)) #5% of neighboring data
            x[:i+extra_i] = []
            y[:i+extra_i] = []
            #plotAndPause(x, y)

        I = numpy.argsort(x)
        x = [x[i] for i in I]
        y = [y[i] for i in I]

        # heuristic

        xvals = numpy.array(x)
        yvals = numpy.array(y)
        
        yn = (yvals - min(yvals)) - 0.5 * (max(yvals) - min(yvals))
        
        crossings = mathutil.findZeroCrossings(yn)
        
        if len(crossings) < 2:
            log.info("not enough crossings detected")
            return self._badResults()
        
        dy_rel_err = 0.8
        (rising_center_idx, rising_idxs) = self.search_for_best_line(xvals, yn, crossings[0], dy_rel_err)
        (falling_center_idx, falling_idxs) = self.search_for_best_line(xvals, yn, crossings[1], dy_rel_err)
        
        # calculate all metrics
        start_rising = min(rising_idxs)
        stop_rising = max(rising_idxs)
        start_falling = min(falling_idxs)
        stop_falling = max(falling_idxs)
        
        vout_at_start_rising = yvals[start_rising]
        vout_at_stop_rising = yvals[stop_rising]
        vout_at_start_falling = yvals[start_falling]
        vout_at_stop_falling = yvals[stop_falling]
        
        vout_min = max(vout_at_start_rising, vout_at_stop_falling)
        vout_max = min(vout_at_start_falling, vout_at_stop_rising)
        
        dr = vout_max - vout_min
        if dr < 0:
            log.info("bad dynamic range: %g (%g - %g)" % (dr, vout_max, vout_min))
            return self._badResults()
        offset = (vout_max + vout_min)/2.0
        if offset < 0:
            log.info("bad offset: %g (%g - %g)" % (offset, vout_max, vout_min))
            return self._badResults()
        
        if stop_rising==start_rising or xvals[stop_rising] == xvals[start_rising]:
            log.info("bad rising edge")
            return self._badResults()
        if stop_falling==start_falling or xvals[stop_falling] == xvals[start_falling]:
            log.info("bad falling edge")
            return self._badResults()

        sr_pos = (yvals[stop_rising]-yvals[start_rising])/(xvals[stop_rising]-xvals[start_rising])
        sr_neg = (yvals[stop_falling]-yvals[start_falling])/(xvals[stop_falling]-xvals[start_falling])
        
        if sr_neg >= sr_pos:
            log.info("bad slewrate: %g (%g/%g)" % (sr,sr_pos,sr_neg))
            return self._badResults()
        if sr_neg > 0:
            log.info("bad slewrate: %g (%g/%g)" % (sr,sr_pos,sr_neg))
            return self._badResults()
        if sr_pos < 0:
            log.info("bad slewrate: %g (%g/%g)" % (sr,sr_pos,sr_neg))
            return self._badResults()

        sr = max(0, min(sr_pos, -sr_neg))
        if sr == 0.0:
            log.info("bad slewrate: %g (%g/%g)" % (sr,sr_pos,sr_neg))
            return self._badResults()
        
        x_diff = xvals[falling_center_idx] - xvals[rising_center_idx]
        log.info("dr: %g; offset: %g; sr: %g (%g/%g); tdiff: %g" % (dr,offset,sr,sr_pos,sr_neg,x_diff))

        # -final calculations of metric values
        r = {}
        r['dynamic_range'] = dr
        r['slewrate'] = max(0.1, sr)
        r['slewrate_log'] = math.log10(max(0.1, sr))
        r['offset'] = offset
        r['pulse_width'] = x_diff

        ##uncomment the next two lines for manual testing
        #plotAndPause(x, y, x, yhat,
        #             title='red=sim waveform, blue=fitted waveform')
        
        #postconditions
        assert sorted(r.keys()) == sorted(self.metricNames())

        #done
        return r

    def _badResults(self):
        r = {}
        r['dynamic_range'] = 0
        r['slewrate'] = 0
        r['slewrate_log'] = -100
        r['offset'] = 0
        r['pulse_width'] = 0
        return r

    def find_line(self, x, yn, initial, max_rel_err):
    
        idxs = []
    
        if initial >= len(yn) - 1:
            return numpy.array(idxs)
        
        if x[initial+1] == x[initial]:
            log.info("warning: divide by zero on time axis")
            return numpy.array(idxs)

        dy_at_initial = abs(yn[initial+1]-yn[initial])/(x[initial+1] - x[initial])
        dy_min = dy_at_initial * (1.0-max_rel_err)
        
        # trace backwards
        for i in range(initial, 0, -1):
            if x[i+1] == x[i]:
                log.info("warning: divide by zero on time axis")
                return numpy.array(idxs)
            
            dy_here = abs(yn[i+1]-yn[i])/(x[i+1] - x[i]);
            if dy_here > dy_min:
                idxs.append(i)
            else:
                break
        # trace forwards
        for i in range(initial, len(yn)-1):
            if x[i+1] == x[i]:
                log.info("warning: divide by zero on time axis")
                return numpy.array(idxs)

            dy_here = abs(yn[i+1]-yn[i])/(x[i+1] - x[i]);
            if dy_here > dy_min:
                idxs.append(i)
            else:
                break
        return numpy.array(sorted(idxs))
    
    def search_for_best_line(self, x, yn, initial, max_rel_err, max_iter = 10):
        old_init_point = -1
        init_point = initial
    
        nbiter = 0
        while init_point != old_init_point and init_point > 0 and nbiter < max_iter:
            idxs = self.find_line(x, yn, init_point, max_rel_err)
            old_init_point = init_point
            init_point = int(numpy.average(idxs))
            nbiter += 1
        return (init_point, idxs)

class TransientWaveformCalculatorTrent:
    
    def __init__(self, input_var_index, output_var_index,
                 chop_initial):
        """
        @description

          1. Fits a 1d regression model (constrained by 'bump' shape)
          that maps x to yhat with minimum nmse
          2. Returns many metrics based on the regression model
        
        @arguments
        
          input_var_index -- int -- X[0,:] = waveforms[index,:]
            (i.e. index of the 'time' variable)
          output_var_index -- int -- 'yhat' = waveforms[index,:]
          chop_initial -- bool -- chop off some of initial waveform?
            (the specific heuristics are in the code below)
        
        @return
    
        @exceptions
    
        @notes

          The input vector currently only has one variable, but
          we could have more if we wished.
        """
        #preconditions
        assert isinstance(output_var_index, types.IntType)
        assert isinstance(input_var_index, types.IntType)
        assert isinstance(chop_initial, types.BooleanType)

        #main work...
        self.input_var_index = input_var_index
        self.output_var_index = output_var_index
        self.chop_initial = chop_initial

    def metricNames(self):
        """Returns a list of the metric names that this class computes.
        Order is not important.
        """
        return ['dynamic_range', 'slewrate', 'slewrate_log', 'nmse', 'correlation',
                'ymin_before_ymax', 'ymin_after_ymax']

    def compute(self, waveforms_array):
        """Returns a dict of metric_name : metric_value,
        which are computed from 'waveforms_array'
        """
        x = waveforms_array[self.input_var_index]
        y = waveforms_array[self.output_var_index]

        #corner case
        if (not x) or (not y):
            log.info('Missing x or y, so return bad results')
            return self._badResults()

        #main case...
        I = numpy.argsort(x)
        x = [x[i] for i in I]
        y = [y[i] for i in I]

        #chop out the initial y's that are >> nearly_min_y (plus
        # some extra neighboring data for cleanness)
        # -another strategy would be: keep chopping initial x's until delta_y>=0
        if self.chop_initial:
            #plotAndPause(x, y)
            miny = min(y)
            maxy = max(y)
            nearly_miny = miny + 0.15 * (maxy - miny) #within 10% of min
            for (i, (xi, yi)) in enumerate(izip(x, y)):
                if yi <= nearly_miny:
                    break
            extra_i = int(0.05 * len(x)) #5% of neighboring data
            x[:i+extra_i] = []
            y[:i+extra_i] = []
            #plotAndPause(x, y)

        I = numpy.argsort(x)
        x = [x[i] for i in I]
        y = [y[i] for i in I]
        
        #fit the lowest-nmse regressor possible to yhat, where
        # the regressor structure is governed by self.shape
        log.info('Begin transient waveform fitting on %d points' % len(x))
        ss = PwlBuildStrategy('bump')
        ss.num_yt_reps = 2
        ss.cyt_ss.max_num_opt_points_without_improve = 200
        ss.cyt_ss.max_num_opt_points = 3000
        model = PwlFactory().build(x, y, ss)
        log.info('Done transient waveform fitting.')

        if (len(model.xs) < 4) or (len(model.ys) < 4):
            log.info('Model was incomplete, so return bad results')
            return self._badResults()

        #compute all the metric values
        # -intermediate calculations
        yhat = model.simulate(x)
        min_range = 1e-15 #magic number to avoid division-by-zero
        rangep = max(min_range, model.xs[2] - model.xs[1])
        rangen = max(min_range, model.xs[4] - model.xs[3])
        srp = +(model.ys[2] - model.ys[1]) / rangep
        srn = -(model.ys[4] - model.ys[3]) / rangen
        sr = min(srp, srn)
        miny = min(yhat)
        maxy = max(yhat)
        nearly_miny = miny + 0.05 * (maxy - miny) #within 5% of min

        ys_before_maxy, ys_after_maxy = [], []
        hit_maxy = False
        for yhati in yhat:
            if yhati == maxy:
                hit_maxy = True
            elif not hit_maxy:
                ys_before_maxy.append(yhati)
            else:
                ys_after_maxy.append(yhati)
        hit_near_miny_before_ymax = ((len(ys_before_maxy) > 0) and
                                     (min(ys_before_maxy) <= nearly_miny))
        hit_near_miny_after_ymax = ((len(ys_after_maxy) > 0) and
                                    (min(ys_after_maxy) <= nearly_miny))
            
        # -final calculations of metric values
        r = {}
        r['dynamic_range'] = maxy - miny
        r['slewrate'] = max(0.1, sr)
        r['slewrate_log'] = math.log10(max(0.1, sr))
        r['nmse'] = mathutil.nmse(yhat, y, min(y), max(y))
        r['correlation'] = mathutil.correlation(yhat, y)
        r['ymin_before_ymax'] = float(hit_near_miny_before_ymax)
        r['ymin_after_ymax'] = float(hit_near_miny_after_ymax)

        ##uncomment the next two lines for manual testing
        #plotAndPause(x, y, x, yhat,
        #             title='red=sim waveform, blue=fitted waveform')
        
        #postconditions
        assert sorted(r.keys()) == sorted(self.metricNames())

        #done
        return r

    def _badResults(self):
        r = {}
        r['dynamic_range'] = 0
        r['slewrate'] = 0
        r['slewrate_log'] = -100
        r['nmse'] = 1.0
        r['correlation'] = -1.0
        r['ymin_before_ymax'] = 0.0
        r['ymin_after_ymax'] = 0.0
        return r
