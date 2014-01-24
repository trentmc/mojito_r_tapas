import types
import random
import math
import unittest

import numpy
        
from util.Hypervolume import \
     hypervolumeMaximize, hypervolumeMinimize, \
     _isNondominated, _dominates, _inBounds
import util.mathutil as mathutil

class HypervolumeTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

    #===========================================================
    def test__problem_2d_A__fast2d(self):
        if self.just1: return
        self._test__problem_2d_A__helper(True)
        
    def test__problem_2d_A__importanceSampling(self):
        if self.just1: return 
        self._test__problem_2d_A__helper(False)
        
    def _test__problem_2d_A__helper(self, use_fast_2d):
        y_A = [1,40]
        y_B = [2,20]
        y_C = [3,10]
        pareto_front = [y_A, y_B, y_C]
        r = [0,0]

        volume = hypervolumeMaximize(pareto_front, r, use_fast_2d)

        target_vol = 40 + 20 + 10
        if use_fast_2d: tol = 0.0
        else:           tol = 2.5
        self.assertTrue((target_vol-tol) <= volume <= (target_vol+tol),
                        (target_vol, volume))

    #===========================================================
    def test__problem_2d_B__fast2d(self):
        if self.just1: return
        self._test__problem_2d_B__helper(True)
        
    def test__problem_2d_B__importanceSampling(self):
        if self.just1: return
        self._test__problem_2d_B__helper(False)
        
    def _test__problem_2d_B__helper(self, use_fast_2d):
        PF = [[1,100],[6,40],[8,20],[7,30],[10,10]]
        r = [0,0]

        volume = hypervolumeMaximize(PF, r, use_fast_2d)
        if use_fast_2d: tol = 0.0
        else:           tol = 20.0
        target_vol = 370
        self.assertTrue((target_vol-tol) <= volume <= (target_vol+tol),
                        (target_vol, volume))
        
    #=========================================================== 
    def _test__problem_2d_C__fast2d(self):
        if self.just1: return
        self._test__problem_2d_C__helper(True)
        
    def test__problem_2d_C__importanceSampling(self):
        if self.just1: return
        self._test__problem_2d_C__helper(False)
        
    def _test__problem_2d_C__helper(self, use_fast_2d):
        """Extremely simple problems: just 1 point in PF."""                
        self.assertAlmostEqual(
            hypervolumeMaximize([[1,1]], [0,0], use_fast_2d), 1.0, 1)
        self.assertAlmostEqual(
            hypervolumeMaximize([[100,1]], [0,0], use_fast_2d), 100.0, 1)
        self.assertAlmostEqual(
            hypervolumeMaximize([[10,10]], [0,0], use_fast_2d), 100.0, 1)
        self.assertAlmostEqual(
            hypervolumeMaximize([[0.5,0.5]], [0,0], use_fast_2d), 0.25, 1)
        
        self.assertAlmostEqual(
            hypervolumeMinimize([[1,1]], [2,2], use_fast_2d), 1.0, 1)
        self.assertAlmostEqual(
            hypervolumeMinimize([[0,1]], [100,2], use_fast_2d), 100.0, 1)
        self.assertAlmostEqual(
            hypervolumeMinimize([[10,10]], [20,20], use_fast_2d), 100.0, 1)
        self.assertAlmostEqual(
            hypervolumeMinimize([[0.75,0.75]], [1,1], use_fast_2d), 0.25**2, 1)
        
  
    #===========================================================  
    def test__problem_2d_D__fast2d(self):
        """Simple problem: just one point in PF contributes to the volume"""
        if self.just1: return
        self._test__problem_2d_D__helper(True)
        
    def test__problem_2d_D__importanceSampling(self):
        if self.just1: return #FAILING
        self._test__problem_2d_D__helper(False)
        
    def _test__problem_2d_D__helper(self, use_fast_2d):
        PF = [[0.5,0.5], [0,1], [1,0]]
        volume = hypervolumeMaximize(PF, [0,0], use_fast_2d)
        self.assertAlmostEqual(volume, 0.25, 1)
        volume = hypervolumeMinimize(PF, [1,1], use_fast_2d)
        self.assertAlmostEqual(volume, 0.25, 1)
        
        PF = [[1,1], [0,2], [2,0]]
        target_vol = 1.0
        if use_fast_2d: tol = 0.0
        else:           tol = 0.1
        volume = hypervolumeMaximize(PF, [0,0], use_fast_2d)
        self.assertTrue((target_vol-tol) <= volume <= (target_vol+tol),
                        (target_vol, volume))
        volume = hypervolumeMinimize(PF, [2,2], use_fast_2d)
        self.assertTrue((target_vol-tol) <= volume <= (target_vol+tol),
                        (target_vol, volume))
        
        PF = [[10,10], [0,20], [20,0]]
        target_vol = 100
        if use_fast_2d: tol = 0.0
        else:           tol = 9.9
        volume = hypervolumeMaximize(PF, [0,0], use_fast_2d)
        self.assertTrue((target_vol-tol) <= volume <= (target_vol+tol),
                        (target_vol, volume))
        volume = hypervolumeMinimize(PF, [20,20], use_fast_2d)
        self.assertTrue((target_vol-tol) <= volume <= (target_vol+tol),
                        (target_vol, volume))
        
    #===========================================================
    def test__problem_2d_E__fast2d(self):
        if self.just1: return
        self._test__problem_2d_E__helper(True)
        
    def test__problem_2d_E__importanceSampling(self):
        if self.just1: return #FAILING
        self._test__problem_2d_E__helper(False)
        
    def _test__problem_2d_E__helper(self, use_fast_2d):
        """The vast fraction of the space has no nondominated points,
        they are all very close to bottom left corner.  Can
        it figure that out and only sample in the sub-region?
        """
        PF = [[8,20],[10,10],[1,100],[6,40],[7,30]]
        r = [1e4, 1e4]

        target_vol = 99889490
        if use_fast_2d:
            tol = 0.0
        else:
            tol = 20.0 #this can NOT be a big tolerance, otherwise it would
                       #destroy the point of the test
                           
        volume = hypervolumeMinimize(PF, r, use_fast_2d)
        
        self.assertTrue((target_vol-tol) <= volume <= (target_vol+tol),
                        (target_vol, volume, abs(target_vol-volume)))
        
    #===========================================================
    def test__problem_2d_F__fast2d(self):
        if self.just1: return
        self._test__problem_2d_F__helper(True)
        
    def test__problem_2d_F__importanceSampling(self):
        if self.just1: return
        self._test__problem_2d_F__helper(False)
        
    def _test__problem_2d_F__helper(self, use_fast_2d):
        """The vast fraction of the space has no nondominated points,
        they are all very close to top right corner.  Can
        it figure that out and only sample in the sub-region?
        """
        b = 1e4
        PF = [[b+8,b+20],[b+10,b+10],[b+1,b+100],[b+6,b+40],[b+7,b+30]]
        r = [0, 0]

        target_vol = 101100370.0
        if use_fast_2d:
            tol = 0.0
        else:
            tol = 20.0 #this can NOT be a big tolerance, otherwise it would
                       #destroy the point of the test
        volume = hypervolumeMaximize(PF, r, use_fast_2d)
        
        self.assertTrue((target_vol-tol) <= volume <= (target_vol+tol),
                        (target_vol, volume, abs(target_vol-volume)))
        
    #===========================================================
    def test__problem_2d_G__fast2d(self):
        if self.just1: return
        self._test__problem_2d_G__helper(True)
        
    def test__problem_2d_G__importanceSampling(self):
        if self.just1: return
        self._test__problem_2d_G__helper(False)
        
    def _test__problem_2d_G__helper(self, use_fast_2d):
        """This is just like problem_2d_B, except that r is different
        which means that the first and last pareto points are right
        on the edge and do not contribute to volume.
        """
        PF = [[1,100],[6,40],[8,20],[7,30],[10,10]]
        r = [1,10]

        volume = hypervolumeMaximize(PF, r, use_fast_2d)
        if use_fast_2d: tol = 0.0
        else:           tol = 20.0
        target_vol = 180
        self.assertTrue((target_vol-tol) <= volume <= (target_vol+tol),
                        (target_vol, volume))
        
    #===========================================================
    def test__problem_3d_A(self):
        return #FIXME: the pareto front below is not valid -- not nondom.
        if self.just1: return

        y_A = [1,40,7]
        y_B = [2,20,7]
        y_C = [3,10,7]
        y_D = [3,30,11]
        y_E = [4,20,11]
        y_F = [4,50,19]
        y_G = [5,40,19]
        pareto_front = [y_A, y_B, y_C, y_D, y_E, y_F, y_G]
        r = [6, 60, 22]

        target_volume = (2-1)*(60-40)*(22-7) + \
                        (3-2)*(60-20)*(22-7) + \
                        (6-3)*(60-10)*(22-7) + \
                        (4-3)*(60-30)*(22-11) + \
                        (6-4)*(60-20)*(22-11) + \
                        (5-4)*(60-50)*(22-19) + \
                        (6-5)*(60-40)*(22-19)
        target_volume2 = 300 + 600 + 2250 + \
                         330 + 880 + \
                         30+ 60
        #validate hand calculations 
        self.assertEqual(target_volume, target_volume2)

        volume = hypervolumeMinimize(pareto_front, r)

        self.assertAlmostEqual(volume, target_volume, 1)
        
    #===========================================================
    def test__problem_3d_B(self):
        """points are not differentiated on last dimension """
        if self.just1: return
        PF = [[1, 40, 7], [2, 20, 7]]
        r = [0, 0, 0]

        target_volume = 7 * (40 + 20)
        
        target_volume2 = 7 * hypervolumeMaximize([[1,40],[2,20]], [0,0], True)
        self.assertEqual(target_volume, target_volume2)

        volume = hypervolumeMaximize(PF, r)
        self.assertAlmostEqual(volume, target_volume, 1)

    #===========================================================
    def test__problem_3d_C(self):
        """Extremely simple problems: just 1 point in PF."""          
        if self.just1: return

        self.assertAlmostEqual(
            hypervolumeMaximize([[1,1,1]], [0,0,0]), 1.0, 1)
        self.assertAlmostEqual(
            hypervolumeMaximize([[100,1,1]], [0,0,0]), 100.0, 1)
        self.assertAlmostEqual(
            hypervolumeMaximize([[10,10,1]], [0,0,0]), 100.0, 1)
        self.assertAlmostEqual(
            hypervolumeMaximize([[0.5,0.5,1]], [0,0,0]), 0.25, 1)
        
        self.assertAlmostEqual(
            hypervolumeMinimize([[1,1,1]], [2,2,2]), 1.0, 1)
        self.assertAlmostEqual(
            hypervolumeMinimize([[0,1,1]], [100,2,2]), 100.0, 1)
        self.assertAlmostEqual(
            hypervolumeMinimize([[10,10,10]], [20,20,11]), 100.0, 1)
        self.assertAlmostEqual(
            hypervolumeMinimize([[0.75,0.75,1]], [1,1,2]), 0.25**2, 1)
        
    #===========================================================
    def test__problem_5d_A(self):
        """points are not differentiated on 3 of 5 dimensions"""
        if self.just1: return
        PF = [[1, 40, 7, 7, 7], [2, 20, 7, 7, 7]]
        r = [0, 0, 0, 0, 0]

        target_volume = (7**3) * (40 + 20)
        
        target_volume2 = (7**3) * \
                         hypervolumeMaximize([[1,40],[2,20]], [0,0], True)
        self.assertEqual(target_volume, target_volume2)

        volume = hypervolumeMaximize(PF, r, True)
        self.assertAlmostEqual(volume, target_volume, 1)
        
        volume = hypervolumeMaximize(PF, r, False)
        self.assertAlmostEqual(volume, target_volume, 1)
        
    def test__problem_5d_B(self):
        """points are not differentiated on 4 of 5 dimensions"""
        if self.just1: return
        PF = [[1, 7, 7, 7, 7], [10, 7, 7, 7, 7]]
        r = [0, 0, 0, 0, 0]

        target_volume = 9 * (7**4)

        volume = hypervolumeMaximize(PF, r, True)
        self.assertAlmostEqual(volume, target_volume, 1)
        
        volume = hypervolumeMaximize(PF, r, False)
        self.assertAlmostEqual(volume, target_volume, 1)
        
    def test__problem_5d_B(self):
        """points are not differentiated on 5 of 5 dimensions"""
        if self.just1: return
        PF = [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7]]
        r = [0, 0, 0, 0, 0]

        target_volume = 7**5

        volume = hypervolumeMaximize(PF, r, True)
        self.assertAlmostEqual(volume, target_volume, 1)
        
        volume = hypervolumeMaximize(PF, r, False)
        self.assertAlmostEqual(volume, target_volume, 1)

    #===========================================================
    def testDominates(self):     
        if self.just1: return

        self.assertFalse(_dominates([1,1],[2,2]))
        self.assertFalse(_dominates([2,2],[2,2]))
        self.assertTrue(_dominates([2,2],[1,1]))
        self.assertTrue(_dominates([2,2],[1,2]))

        self.assertTrue( _isNondominated([2,2],[[0,0]]))
        self.assertTrue( _isNondominated([2,2],[[0,0],[2,0],[1,1],[0,2]]))
        self.assertTrue(_isNondominated([2,2],[[0,0],[2,0],[1,1],[0,2],[2,2]]))
        self.assertFalse(_isNondominated([2,2],[[0,0],[2,0],[1,1],[0,2],[3,2]]))

        #hypervolumeMaximize should complain that this is not a PF
        PF = [[1, 2, 7], [40, 20, 7]]
        r = [0, 0, 0]
        self.assertRaises(AssertionError, hypervolumeMaximize, PF, r)

    def testInBounds(self):
        if self.just1: return

        self.assertTrue(_inBounds([],[],[]))
        self.assertTrue(_inBounds([1],[0],[2]))
        self.assertTrue(_inBounds([1],[1],[2]))
        self.assertTrue(_inBounds([1],[0],[1]))
        self.assertFalse(_inBounds([-1],[0],[1]))
        self.assertFalse(_inBounds([2],[0],[1]))

#     def OLD_test3d_B(self):
#         if self.just1: return

#         #points are not differentiated on last dimension (sweep dimension)
#         points = [[1, 40, 7], [2, 20, 7]]
#         region = [[1, 10, 7], [3, 60, 22]]
        
#         r = [6, 60, 22]

#         calculator = HypervolumeCalculator(points, r)
#         split_var = 0
#         cover_val = 22
        
#         volume = calculator.volumeOY(region, points, split_var, cover_val, 0)

#         self.assertAlmostEqual(volume, 60*(22-7), 1)


if __name__ == '__main__':
    unittest.main()
    
    logging.basicConfig()
    logging.getLogger('hypervolume').setLevel(logging.DEBUG)
    
    unittest.main()
