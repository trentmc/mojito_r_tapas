""" DytOptimizer

DytOptimizer = Discrete YureT optimizer.

Yuret=Fast hillclimber based on Deniz Yuret's Master's thesis,
 1995, p. 33, Alg 4-1

Differences from Yuret paper:
-step-taking is all based on unscaled integer stepvalues rather than
 continuous values.
-code architecture is more like a state machine than weirdly-nested if-thens

Minimizes weighted_cost, where weighted_cost = cost_tup[0] + goal2_weight * cost_tup[1].
"""

import copy
import logging
import random
import types

import numpy

from adts import *
from engine.EngineUtils import coststr
from engine.YtAdts import *
from util import mathutil 

log = logging.getLogger('dyt')

#===============================================================
#start magic numbers
class DytSolutionStrategy:

    def __init__(self):
        self.max_num_opt_points = 1500

        #maximum number of neutral wanders
        self.max_neutral = 100

    def setMaxNumPoints(self, max_num_opt_points):
        self.max_num_opt_points = max_num_opt_points

    def __str__(self):
        s = "DytSolutionStrategy={"
        s += " max_num_opt_points=%s" % self.max_num_opt_points
        s += ", max_neutral wanders=%s" % self.max_neutral
        s += " /DytSolutionStrategy}"
        return s
        
#end magic numbers
#===============================================================

class TemplateDytOptimizer:
    """This class is a simple example of how to embed a DytState
    into an optimizer"""
    
    def __init__(self, ps, ss):
        self.ps = ps
        self.ss = ss
        self.state = None

    def stop(self):
        """Call this in order to externally force a stop"""
        self.state._stop = True

    def optimize(self, init_opt_point):
        log.info("Begin opt")
        
        self.state = DytState(self.ps.opt_point_meta, self.ss, init_opt_point, 0.0)

        while True:
            #[at this point: simulate 'inds' and set each ind's 'cost']
            #an example is here, where the cost function is merely quadratic
            inds = self.state.indsWithoutCosts()
            for ind in inds:
                cost = sum(v**2 for v in ind.opt_point.values())
                ind.cost_tup = (cost, )
            
            if self.state.doStop():
                break

            log.info(str(self.state))
            self.state.update()

        log.info("Done opt")
            

class DytInd:
    """Merely an association of an opt point with a cost."""
    def __init__(self, x, opt_point):
        self.x = x                  #unscaled dyt_opm opt point
        self.opt_point = opt_point  #ps_opm opt point
        self.cost_tup = None        #(cost1, cost2, ...)

    def cost(self):
        assert len(self.cost_tup) == 1
        return self.cost_tup[0]

    def weightedCost(self, goal2_weight):
        if len(self.cost_tup) == 1:
            return self.cost_tup[0]
        elif self.cost_tup[0] >= 0.0:
            return self.cost_tup[0]
        else:
            return self.cost_tup[0] + goal2_weight * self.cost_tup[1]


class DytState:
    """Stores the state of a Dyt search.  Each call to update()
    will generate new candidate designs, which can then be simulated
    externally."""
    
    def __init__(self, original_ps_opm, ss, init_opt_point, goal2_weight=None):
        #condition inputs
        if goal2_weight is None:
            goal2_weight = 0.0

        #preconditions
        assert isinstance(original_ps_opm, PointMeta)
        assert isinstance(ss, DytSolutionStrategy)
        assert isinstance(init_opt_point, Point)
        assert isinstance(ss, DytSolutionStrategy)
        assert mathutil.isNumber(goal2_weight)

        #main work...
        ps_opm = original_ps_opm
        self.yt_opm = buildDytPointMeta(ps_opm)
        assert self.yt_opm.discreteVaryingVarNames()
        
        self.ss = ss
        self.goal2_weight = goal2_weight
        
        self.num_opt_points = 0
        
        self.num_neutral = 0
    
        self._stop = False

        self.recent_inds =  [] #cache recent inds so that we can reuse cost info if possible

        self.action = 'TRY_XV'

        assert init_opt_point.is_scaled, "need init point scaled"
        init_x = self.yt_opm.unscale(YTPoint(True, dict(init_opt_point)))
        
        self.x = init_x                           #design point
        self.v = self._randomInitVelocity(self.x) #velocity vector
        self.u = self._zeroVelocity()
        
        self.steps_list = self._newStepsList()
        
        #Rule: always keep ind_x and ind_xv up to date based on what
        # the values of x and v are.
        (self.ind_x, dummy) = self._createInd(self.x)
        (self.ind_xv, self.railed_xv) = self._createInd(self.x + self.v)
        self.ind_xuv = None

        #postconditions
        assert self.ind_x.x == init_x
        for var in init_opt_point.keys():
            #test equality, but account for floating point roundoff
            #HACK off  assert ('%.8e' % self.ind_x.opt_point[var]) == ('%.8e' % init_opt_point[var])
            pass

        #use supplied init_opt_point, rather than the one transormed from x
        # within createInd, in order to maintain ID
        self.ind_x.opt_point = init_opt_point
            
    def __str__(self):
        s = []
        if self.ind_x.cost_tup is None:
            s += ["cost(x)=None"]
        elif len(self.ind_x.cost_tup) == 1:
            s += ["cost(x)=%s" % coststr(self.ind_x.cost_tup[0])]
        else:
            s += ["weightedCost(x)=%s" % coststr(self.ind_x.weightedCost(self.goal2_weight))]
            s += ["; ind_x_cost_tup=%s" % coststr(self.ind_x.cost_tup)]
            s += ["; goal2_weight=%.1f" % self.goal2_weight]
        s += ["; # opt points=%d" % self.num_opt_points]
        s += ["; stepsize=%d" % self.v.discreteStepsize()]
        s += ["; # neutral=%d" % self.num_neutral]
        s += ["; label=%s" % self.action]
        s += ["; len(self.steps_list)=%d" % len(self.steps_list)]

        #output optimization values too
        if len(self.yt_opm) == 1:
            s += ["; x=%.5f; xv=%.5f; v=%.5f, u=%.5f" %
                  (self.ind_x.opt_point.values()[0],
                   self.ind_xv.opt_point.values()[0],
                   self.v.values()[0], self.u.values()[0])]
        return "".join(s)

    def detailedStr(self):
        s = [str(self)]
        if len(self.x) <= 3:
            s += ["; x=%s" % self.x]
            s += ["; v=%s" % self.v]
        return "".join(s)

    def detailedIndStr(self):
        """This string is good for debugging re opt point IDs, costs, etc"""
        s = ["detailedIndStr:\n"]
        for attr in ["ind_x", "ind_xv", "ind_xuv"]:
            ind = getattr(self, attr)
            s += ["  DytInd: %s: " % attr]
            if ind is None:
                s += ["None"]
            else:
                s += ["opt_point_ID=%d, cost=%s" % (ind.opt_point.ID, coststr(ind.cost_tup))]
            s += ["\n"]
        return "".join(s)

    def indsWithoutCosts(self):
        """Returns the subset of {ind_x, ind_xv, ind_xuv} DytInds that need their
        costs filled in."""
        return [ind
                for ind in [self.ind_x, self.ind_xv, self.ind_xuv]
                if (ind is not None) and (ind.cost_tup is None)]

    def indsWithCosts(self):
        return [ind
                for ind in [self.ind_x, self.ind_xv, self.ind_xuv]
                if (ind is not None) and (ind.cost_tup is not None)]

    def activeInds(self):
        """Returns the subset of {ind_x, ind_xv, ind_xuv} DytInds that are non-None"""
        return [ind
                for ind in [self.ind_x, self.ind_xv, self.ind_xuv]
                if (ind is not None)]

    def bestInd(self):
        """Returns ind with lowest weighted cost"""
        inds = [ind
                for ind in [self.ind_x, self.ind_xv, self.ind_xuv]
                if (ind is not None) and (ind.cost_tup is not None)]
        costs = [ind.weightedCost(self.goal2_weight)
                 for ind in inds]
        return inds[numpy.argmin(costs)]
    
    def doStop(self):
        """Returns True is this state wants to stop"""
        if self._stop:
            log.info('Stop because user soft stop requested')
            return True

        if self.num_opt_points >= self.ss.max_num_opt_points:
            log.info('Stop because # opt_points >= maximum allowed (=%d)'%
                     (self.ss.max_num_opt_points))
            return True

        if self.action == 'STOP_CONVERGED':
            log.info('Stop because converged')
            return True
            
        return False
            
    #=========================================================================
    #update state!
    def update(self):
        #preconditions (aggressive for now)
        assert not self.indsWithoutCosts()
        assert self.action != 'STOP_CONVERGED'
        
        assert self.x == self.ind_x.x
        if self.action == 'TRY_XV' and not self.railed_xv:
            assert (self.x + self.v) == self.ind_xv.x
            assert self.ind_x.x != self.ind_xv.x
        assert self.v.discreteStepsize() > 0

        assert isinstance(self.ind_x.cost_tup, types.TupleType)
        assert (self.ind_xv.cost_tup is None) or isinstance(self.ind_xv.cost_tup, types.TupleType)
        assert (self.ind_xuv is None) or (self.ind_xuv.cost_tup is None) or \
               isinstance(self.ind_xuv.cost_tup, types.TupleType)

        #main work...
        
        last_action = self.action

        #update neutral count
        if last_action == 'TRY_XV':
            if self._xvNeutral():
                self.num_neutral += 1
            elif not self._onMinStepsize():
                self.num_neutral = 0
                
        #main update
        if last_action == 'TRY_XV':
            if self.num_neutral >= self.ss.max_neutral:
                    log.info("Too many neutral without improve; stop")
                    new_action = 'STOP_CONVERGED'
            
            elif self._xvWorsened() and self.steps_list:
                log.info("xv worsened; but options left at this stepsize")
                self.v = self.steps_list.pop()
                self.ind_xv, self.railed_xv = self._createInd(self.x + self.v)
                new_action = 'TRY_XV'

            elif self._xvWorsened() and self._onMinStepsize():
                log.info("xv worsened; no options left")
                new_action = 'STOP_CONVERGED'

            elif self._xvWorsened():
                log.info("xv worsened; but options left at smaller stepsize")
                self.v = self._halveStepsize(self.v, self.x)
                self.steps_list = self._newStepsList()
                self.v = self.steps_list.pop()
                self.ind_xv, self.railed_xv = self._createInd(self.x + self.v)
                new_action = 'TRY_XV'
                
            elif self.railed_xv: 
                log.info("xv improved or neutral; but railed; so take the new x, but change v")
                old_x = self.x
                self.x = self.ind_xv.x; self.ind_x = self.ind_xv
                self.v = self._changeDirection(self.v, old_x)
                self.u = self._zeroVelocity()
                self.ind_xv, self.railed_xv = self._createInd(self.x + self.v)
                self.steps_list = self._newStepsList()
                new_action = 'TRY_XV'

            elif not self.steps_list:
                log.info("xv improved or neutral, without a spin; build off u")
                self.x = self.ind_xv.x; self.ind_x = self.ind_xv
                self.u = self.u + self.v
                self.v = self.v * 2
                self.ind_xv, self.railed_xv = self._createInd(self.x + self.v)
                self.steps_list = []
                new_action = 'TRY_XV'

            else:
                log.info("xv improved or neutral, but had to spin to get here; test u")
                self.ind_xuv, dummy = self._createInd(self.x + self.u + self.v)
                new_action = 'TRY_XUV'

        elif last_action == 'TRY_XUV':
            if self._xuvWorsened():
                log.info("xuv worsened, so just go back to x+v")
                self.x = self.ind_xv.x; self.ind_x = self.ind_xv
                self.u = self.v
                self.v = self.v * 2
                self.ind_xv, self.railed_xv = self._createInd(self.x + self.v)
                self.ind_xuv = None
                self.steps_list = self._newStepsList()
                new_action = 'TRY_XV'

            else: 
                log.info("xuv improved or neutral, so incorporate u into v and keep going")
                self.x = self.ind_xuv.x; self.ind_x = self.ind_xuv
                self.u = self.u + self.v
                if self.u.discreteStepsize() > 0:
                    self.v = self.u * 2
                else:
                    self.v = self.v * 2 
                self.ind_xv, self.railed_xv = self._createInd(self.x + self.v)
                self.ind_xuv = None
                self.steps_list = []
                new_action = 'TRY_XV'

        else:
            raise AssertionError('Unknown label "%s"' % last_action)

        self.action = new_action
        

        #postconditions (aggressive for now)
        assert self.x == self.ind_x.x
        if last_action == 'TRY_XV' and not self.railed_xv:
            assert (self.x + self.v) == self.ind_xv.x
            assert self.ind_x.x != self.ind_xv.x
        assert self.v.discreteStepsize() > 0
    
    #=========================================================================
    #state helper functions
    def _xvImproved(self):
        """Did ind.xv improve on ind.x ?"""
        xv_cost = self.ind_xv.weightedCost(self.goal2_weight)
        x_cost = self.ind_x.weightedCost(self.goal2_weight)
        return xv_cost < x_cost

    def _xvWorsened(self):
        """Did ind.xv worsen wrt on ind.x ?"""
        xv_cost = self.ind_xv.weightedCost(self.goal2_weight)
        x_cost = self.ind_x.weightedCost(self.goal2_weight)
        return xv_cost > x_cost

    def _xvNeutral(self):
        return (not self._xvImproved()) and (not self._xvWorsened())
    
    def _xuvWorsened(self):
        """Did ind.xuv worsen on ind.x ?"""
        xuv_cost = self.ind_xuv.weightedCost(self.goal2_weight)
        x_cost = self.ind_x.weightedCost(self.goal2_weight)
        return xuv_cost > x_cost

    def _onMinStepsize(self):
        """Returns true if stepsize is minimum value"""
        return (self.v.discreteStepsize() == 1)

    def _newStepsList(self):
        if self._onMinStepsize():
            steps_list = self._allSmallestSteps(self.x)
            random.shuffle(steps_list)
            
        else:
            steps_list = [self._changeDirection(self.v, self.x),
                          self._changeDirection(self.v, self.x)]
        return steps_list

    def _maxIter(self):
        """Returns number of spins allowed at the current stepsize"""
        if self._onMinStepsize():
            return len(self._allSmallestSteps(self.x))
        else:
            return 2
    
    def _copyPoint(self, opt_point):
        return copy.deepcopy(opt_point)
    
    def _randomInitVelocity(self, unscaled_point):
        """1 variable that has nonzero range gets a direction of +1 or -1"""
        vb = self._zeroVelocity()
        var = random.choice(self.yt_opm.discreteVaryingVarNames())
        
        if unscaled_point[var] == self.yt_opm[var].min_unscaled_value:
            vb[var] = +1
        elif unscaled_point[var] == self.yt_opm[var].max_unscaled_value:
            vb[var] = -1
        elif random.random() < 0.5:
            vb[ var ] = -1
        else:
            vb[ var ] = +1
        return vb

    def _fullRandomInitVelocity(self, unscaled_point):
        """all variables that has nonzero range gets a direction of +1 or -1"""
        vb = self._zeroVelocity()
        for var in self.yt_opm.discreteVaryingVarNames():
            if unscaled_point[var] == self.yt_opm[var].min_unscaled_value:
                vb[var] = +1
            elif unscaled_point[var] == self.yt_opm[var].max_unscaled_value:
                vb[var] = -1
            elif random.random() < 0.5:
                vb[ var ] = -1
            else:
                vb[ var ] = +1
        return vb
        
    def _zeroVelocity(self):
        """Returns an unscaled Point of varname :  0"""
        vb = {}
        for var in self.yt_opm.discreteVarNames():
            vb[var] = 0.0
            
        vb = YTPoint(False, vb)
        return vb

    def _halveStepsize(self, va, unscaled_point):
        """Return a version of 'va' that points the
        same direction but is half the size.
        """
        assert not va.is_scaled
        assert va.discreteStepsize() > 0
        
        vb = self._zeroVelocity()
        vars_which_could_add_one = {}
        for var, old_step in va.items():
            #choose absolute value of new step
            # -if old step was an odd number, 50/50 chance of rounding up vs down
            new_abs_step = int(abs(old_step) / 2)
            if old_step % 2 != 0:
                if random.random() < 0.5:
                    new_abs_step += 1
                else:
                    vars_which_could_add_one[var] = 1

            #give vb[var] = new_step the same direction as old step
            if old_step < 0:
                vb[var] = -1 * new_abs_step
            else:
                vb[var] = new_abs_step

        #try to ensure at least half the stepsize
        while vb.discreteStepsize() < va.discreteStepsize()/2 and \
                  len(vars_which_could_add_one) > 0:
            #choose the var
            var = random.choice(vars_which_could_add_one.keys())
            del vars_which_could_add_one[var]

            #new abs stepsize = prev stepsize, plus one
            new_abs_step = abs(vb[var])  + 1
            
            #give vb[var] = new_step the same direction as va[var] = old_step
            if va[var] < 0:
                vb[var] = -1 * new_abs_step
            else:
                vb[var] = new_abs_step

        #make sure that the new velocity changes _somewhere_
        if vb.discreteStepsize() == 0:
            var = random.choice( self.yt_opm.discreteVaryingVarNames() )
            if unscaled_point[var] == self.yt_opm[var].min_unscaled_value:
                vb[var] = +1
            elif unscaled_point[var] == self.yt_opm[var].max_unscaled_value:
                vb[var] = -1
            elif random.random() < 0.5:
                vb[ var ] = -1
            else:
                vb[ var ] = +1

        log.debug('Done _halveStepsize; old size=%d, new=%d' %
                  (va.discreteStepsize(), vb.discreteStepsize()))
            
        return vb

    def _allSmallestSteps(self, x):
        """Returns a list of all the smallest-possible velocity vectors that
        can go from this unscaled point 'x'"""
        vs = []
        for var in self.yt_opm.discreteVaryingVarNames():
            if x[var] < self.yt_opm[var].max_unscaled_value:
                v = self._zeroVelocity()
                v[var] = +1
                vs.append(v)

            if x[var] > self.yt_opm[var].min_unscaled_value:
                v = self._zeroVelocity()
                v[var] = -1
                vs.append(v)
                
        return vs
        
    def _changeDirection(self, va, unscaled_point):
        """Return a new velocity vector with same size as 'va'.
        Use 'unscaled_point' to as the reference point so we know
        which directions would lead to railing, and to avoid those when
        we are choosing our directions"""
       
        target_stepsize = va.discreteStepsize()
        log.debug('Begin _changeDirection; target_stepsize=%d' % target_stepsize)
        assert target_stepsize > 0
        
        vb = self._zeroVelocity()
        cand_vars = self.yt_opm.discreteVaryingVarNames()
        new_point = self._copyPoint(unscaled_point)

        iters = 0
        num_loops = 0
        while (vb.discreteStepsize() < target_stepsize):
            if (num_loops > 5000) and vb.discreteStepsize() > 0: #avoid infinite loop
                break
            
            var = random.choice(cand_vars)
            
            #force increase
            if new_point[var] == self.yt_opm[var].min_unscaled_value:
                vb[var] += 1
                new_point[var] += 1

            #force decrease
            elif new_point[var] == self.yt_opm[var].max_unscaled_value:
                vb[var] -= 1
                new_point[var] -= 1

            #randomly either way
            elif random.random() < 0.5:
                vb[ var ] -= 1
                new_point[var] -= 1
            else:
                vb[ var ] += 1
                new_point[var] += 1

            num_loops += 1
                
        log.debug('Done _changeDirection; old size=%d, new=%d' %
                   (va.discreteStepsize(), vb.discreteStepsize()))

        return vb

            
    def _createInd(self, x):
        x = self._copyPoint(x)
        assert not x.is_scaled
        x2 = self.yt_opm.railbin(x)
        railed = False
        for var in x.keys():
            if x[var] != x2[var]:
                railed = True
                break
        opt_point2 = self.yt_opm.scale(x2)
        new_ind = None

        new_ind = DytInd(x2, opt_point2)
        self.num_opt_points += 1 
        
        #reuse cost info from a previous ind if possible
        # (also reuse x and opt_point so that IDs stay the same)
        for prev_ind in self.recent_inds:
            if (prev_ind.x == x2) and (prev_ind.cost_tup is not None):
                new_ind.cost_tup = prev_ind.cost_tup
                new_ind.opt_point = prev_ind.opt_point
                new_ind.x = prev_ind.x

        #update recent inds.  Keep its size an upper bound.
        self.recent_inds.append(new_ind)
        if len(self.recent_inds) > 20:  #magic number alert
            self.recent_inds[:1] = []
                
        return (new_ind, railed)
    
