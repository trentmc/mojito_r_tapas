import logging
import unittest

from adts import *
from engine.GtoOptimizer import GtoSolutionStrategy, GtoState, TemplateGtoOptimizer, TestPS

class GtoOptimizerTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

    def testSimpleConvergence(self):
        if self.just1: return
        
        opt_point_meta = PointMeta([ContinuousVarMeta(False, -100.0, +100.0, 'x0'),
                                    ContinuousVarMeta(False,  -20.0,  +40.0, 'x1')])
        ps = TestPS(opt_point_meta)

        start_opt_point = Point(True, {'x0':-25.0, 'x1':+15.0})
        ss = GtoSolutionStrategy([])
        ss.probe_ss.max_rank = 1
        ss.setMaxNumPoints(50)
        ss.setTargetCost(0.0)

        optimizer = TemplateGtoOptimizer(ps, ss, start_opt_point)
        optimizer.optimize()

        expected_o = Point(True, {'x0':0.0, 'x1':0.0})
        best_o = optimizer.state.bestOptPoint()
        
        found_it = True
        for (var_name, var_meta) in ps.opt_point_meta.items():
            tol = 0.10 * (var_meta.max_unscaled_value - var_meta.min_unscaled_value)
            found_it = found_it and (abs(expected_o[var_name] - best_o[var_name]) < tol)

        self.assertTrue(found_it)

        #test str's
        self.assertTrue(len(str(ss)) > 0)
        self.assertTrue(len(str(optimizer)) > 0)
        
    def testVariableScaling(self):
        if self.just1: return

        #the point meta has one var -- a capacitance that follows a log-scale.
        # e.g. if an unscaled value of C is -8, then the scaled value is 10^-8
        opt_point_meta = PointMeta([ContinuousVarMeta(True, -9, -6, 'C')])
        ps = TestPS(opt_point_meta)

        start_opt_point = Point(True, {'C':1.0e-8})
        ss = GtoSolutionStrategy([])
        ss.probe_ss.max_rank = 1
        ss.setMaxNumPoints(20)
        ss.setTargetCost(0.0)

        #test State
        temp_state = GtoState(ps, ss, start_opt_point)
        mn, mx = -9.0, -6.0
        x01 = temp_state.center_ind.x01
        self.assertEqual(x01[0], (-8.0 - mn)/(mx - mn))
        self.assertEqual(temp_state.x01ToScaledPoint(x01)['C'], 1.0e-8)

        #test that it minimizes.  It should get to 1.0e-9, because cost=(1.0e-9)^2, not (-6)^2
        optimizer = TemplateGtoOptimizer(ps, ss, start_opt_point)
        optimizer.optimize()

        best_o = optimizer.state.bestOptPoint()
        self.assertTrue(best_o.is_scaled)
        self.assertTrue(1.0e-9 <= best_o['C'] <= 1.0e-6)
    
        best_o = opt_point_meta.unscale(best_o)
        self.assertTrue(-9 <= best_o['C'] <= -6)
        
        expected_o = Point(True, {'C':1.0e-9})
        expected_o = opt_point_meta.unscale(expected_o)
        
        found_it = True
        for (var_name, var_meta) in ps.opt_point_meta.items():
            tol = 0.10 * (var_meta.max_unscaled_value - var_meta.min_unscaled_value)
            found_it = found_it and (abs(expected_o[var_name] - best_o[var_name]) < tol)
            print tol, expected_o[var_name], best_o[var_name]

        self.assertTrue(found_it)

        #test str's
        self.assertTrue(len(str(ss)) > 0)
        self.assertTrue(len(str(optimizer)) > 0)
        
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    import logging

    logging.basicConfig()
    logging.gegtogger('gto').setLevel(logging.INFO)
    
    unittest.main()
