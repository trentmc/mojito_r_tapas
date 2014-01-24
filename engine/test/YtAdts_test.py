import logging
import unittest

import numpy

from adts import *
from engine.YtAdts import *

class YtAdtsTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

    def testYTContinuousPointMeta(self):
        if self.just1: return

        yt_opm = PointMeta([YTContinuousVarMeta(-100.0, +100.0, 'x0'),
                            YTContinuousVarMeta(   0.0,   40.0, 'x1')])

        self.assertEqual(yt_opm['x0'].min_scaled_value, -100.0)
        self.assertEqual(yt_opm['x0'].max_scaled_value, +100.0)
        self.assertEqual(yt_opm['x1'].min_scaled_value, 0.0)
        self.assertEqual(yt_opm['x1'].max_scaled_value, +40.0)
        
        self.assertEqual(yt_opm['x0'].min_unscaled_value, 0.0)
        self.assertEqual(yt_opm['x0'].max_unscaled_value, 1.0)
        self.assertEqual(yt_opm['x1'].min_unscaled_value, 0.0)
        self.assertEqual(yt_opm['x1'].max_unscaled_value, 1.0)

        vm = yt_opm['x0']
        self.assertEqual(vm.railbinUnscaled(-12.0), 0.0)
        self.assertEqual(vm.railbinUnscaled(0.0), 0.0)
        self.assertEqual(vm.railbinUnscaled(0.56), 0.56)
        self.assertEqual(vm.railbinUnscaled(1.0), 1.0)
        self.assertEqual(vm.railbinUnscaled(12.0), 1.0)

        self.assertEqual(vm.scale(-12.0), -100.0)
        self.assertEqual(vm.scale(0.0), -100.0)
        self.assertAlmostEqual(vm.scale(0.5), 0.0, 5)
        self.assertEqual(vm.scale(1.0), +100.0)
        self.assertEqual(vm.scale(12.0), +100.0)

        self.assertEqual(vm.unscale(-120.0), +0.0)
        self.assertEqual(vm.unscale(-100.0), +0.0)
        self.assertAlmostEqual(vm.unscale(0.0), +0.5, 5)
        self.assertEqual(vm.unscale(100.0), +1.0)
        self.assertEqual(vm.unscale(120.0), +1.0)

    def testConvertContinuous(self):
        ps_opm = PointMeta([ContinuousVarMeta(False, -100.0, +100.0, 'x0'),
                            ContinuousVarMeta(False,    0.0,  +40.0, 'x1')])
        yt_opm = buildCytPointMeta(ps_opm)

        self.assertEqual(yt_opm['x0'].min_scaled_value, -100.0)
        self.assertEqual(yt_opm['x0'].max_scaled_value, +100.0)
        self.assertEqual(yt_opm['x1'].min_scaled_value, 0.0)
        self.assertEqual(yt_opm['x1'].max_scaled_value, +40.0)
                
        for i in range(100):
            rnd_opt_point = ps_opm.createRandomScaledPoint(False)
            rnd_x = yt_opm.unscale(YTPoint(True, rnd_opt_point))
            assert yt_opm.scale(rnd_x) == rnd_opt_point, "yt_opm scale/unscale is broken"
            
    def testYTDiscretePointMeta(self):
        if self.just1: return


        yt_opm = PointMeta([YTDiscreteVarMeta(numpy.arange(-100.0, +101.0, 1.0), 'x0'),
                            YTDiscreteVarMeta(numpy.arange(   0.0,  +41.0, 1.0), 'x1'),
                            YTDiscreteVarMeta(numpy.arange(   0.0,  +0.0, 1.0), 'x2'),
                            ])

        self.assertEqual(yt_opm['x0'].min_scaled_value, -100.0)
        self.assertEqual(yt_opm['x0'].max_scaled_value, +100.0)
        self.assertEqual(yt_opm['x1'].min_scaled_value, 0.0)
        self.assertEqual(yt_opm['x1'].max_scaled_value, +40.0)
        
        self.assertEqual(yt_opm['x0'].min_unscaled_value, 0)
        self.assertEqual(yt_opm['x0'].max_unscaled_value, len(yt_opm['x0'].possible_values)-1)
        self.assertEqual(yt_opm['x1'].min_unscaled_value, 0)
        self.assertEqual(yt_opm['x1'].max_unscaled_value, len(yt_opm['x1'].possible_values)-1)

        vm = yt_opm['x0']
        num_vals = len(vm.possible_values)
        self.assertEqual(vm.railbinUnscaled(-12), 0)
        self.assertEqual(vm.railbinUnscaled(0), 0)
        self.assertEqual(vm.railbinUnscaled(3), 3)
        self.assertEqual(vm.railbinUnscaled(1), 1)
        self.assertEqual(vm.railbinUnscaled(num_vals-1), num_vals-1)
        self.assertEqual(vm.railbinUnscaled(num_vals), num_vals-1)
        self.assertEqual(vm.railbinUnscaled(num_vals+10), num_vals-1)
        
        self.assertEqual(vm.scale(-12), -100.0)
        self.assertEqual(vm.scale(0), -100.0)
        self.assertEqual(vm.scale(num_vals/2), 0.0)
        self.assertEqual(vm.scale(num_vals-1), +100.0)
        self.assertEqual(vm.scale(num_vals+10), +100.0)

        self.assertEqual(vm.unscale(-120.0), 0)
        self.assertEqual(vm.unscale(-100.0), 0)
        self.assertEqual(vm.unscale(0.0), num_vals/2)
        self.assertEqual(vm.unscale(100.0), num_vals-1)
        self.assertEqual(vm.unscale(120.0), num_vals-1)
        
    def testConvertDiscrete(self):
        ps_opm = PointMeta([DiscreteVarMeta(numpy.arange(-100.0, +101.0, +1.0), 'x0'), 
                            DiscreteVarMeta(numpy.arange(0.0, +41.0, +1.0), 'x1')])
        yt_opm = buildDytPointMeta(ps_opm)

        self.assertEqual(yt_opm['x0'].min_scaled_value, -100.0)
        self.assertEqual(yt_opm['x0'].max_scaled_value, +100.0)
        self.assertEqual(yt_opm['x1'].min_scaled_value, 0.0)
        self.assertEqual(yt_opm['x1'].max_scaled_value, +40.0)

        for i in range(100):
            rnd_opt_point = ps_opm.createRandomScaledPoint(False)
            rnd_x = yt_opm.unscale(YTPoint(True, rnd_opt_point))
            assert yt_opm.scale(rnd_x) == rnd_opt_point, "yt_opm scale/unscale is broken"

        
    def tearDown(self):
        pass

if __name__ == '__main__':
    import logging
    import sayo.util.sayolog

    logging.basicConfig()
    sayo.util.sayolog.getLogger('yt').setLevel(logging.INFO)
    
    unittest.main()
