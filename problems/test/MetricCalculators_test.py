import unittest

import numpy

from adts import *
from problems.MetricCalculators import *
from regressor.Pwl import PwlModel
from util.octavecall import plotAndPause
from util import mathutil

class MetricCalculatorsTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK         

    def testNmseOnTargetWaveform(self):
        if self.just1: return
        
        #set base data
        y = numpy.array([1.0, 2.0, 3.0, 2.0, 1.0])

        yhat = numpy.ones(5, dtype=float)
        yhat[2] = 9.9
        
        waveforms_array = numpy.ones((3,5), dtype=float)
        waveforms_array[1,:] = yhat

        #calc expected nmse 'nmse1'
        yrange = max(y) - min(y)
        nmse1 = nmseFromDenom(y, yhat, yrange)

        #calc nmse using the callable class, 'nmse2'
        calculator = NmseOnTargetWaveformCalculator(y, 1)
        self.assertEqual(calculator.metricNames(), ['nmse'])
        sim_results = calculator.compute(waveforms_array)
        self.assertEqual(sim_results.keys(), ['nmse'])
        nmse2 = sim_results['nmse']

        self.assertEqual(nmse1, nmse2)

        #try way worse waveform
        waveforms_array[1,:] = 100.0 * waveforms_array[1,:]
        nmse3 = calculator.compute(waveforms_array)['nmse']
        self.assertTrue(nmse3 > nmse1)

    def testNmseOnTargetShape_Hockey(self):
        if self.just1: return
        self._testNmseOnTargetShape('hockey')
        
    def testNmseOnTargetShape_Bump(self):
        if self.just1: return
        self._testNmseOnTargetShape('bump')
        
    def _testNmseOnTargetShape(self, shape):
        #set base data
        # -both the x and y take a step jump for 7th, 8th etc datapoints
        waveforms_array = numpy.ones((4,20), dtype=float)

        # -output rises from 20 to 30, then flattens there (at the value of 30)
        waveforms_array[1,:] = numpy.arange(20.0, 40.0, 1.0)
        for i in range(20):
            waveforms_array[1,i] = min(waveforms_array[1,i], 30.0)

        # -input is merely 1.0, 2.0, ..., 20.0
        waveforms_array[3,:] = numpy.arange(0.0, 20.0, 1.0)
        
        #calc nmse using the calculator
        calculator = NmseOnTargetShapeCalculator(shape, 3, 1)
        self.assertEqual(calculator.metricNames(), ['nmse'])
        sim_results = calculator.compute(waveforms_array)
        self.assertEqual(sim_results.keys(), ['nmse'])
        nmse2 = sim_results['nmse']

        #should be a perfect fit if hockey
        if shape == 'hockey':
            self.assertAlmostEqual(nmse2, 0.0, 2)

        
    def testTransientWaveformCalculator_NoResampling(self):
        if self.just1: return
        self._testTransientWaveformCalculator(30)
        
    def testTransientWaveformCalculator_Resampling(self):
        if self.just1: return
        self._testTransientWaveformCalculator(3000)
        
    def testTransientWaveformCalculator_NoSamples(self):
        if self.just1: return
        self._testTransientWaveformCalculator(0)
        
        
    def _testTransientWaveformCalculator(self, num_samples):
        
        #construct waveforms by building a pwl model and simulating it
        if num_samples > 0:
            xs = [0.0,  0.1,  1.0, 2.0, 5.0,  6.0, 7.0]
            ys = [1.0, -1.0, -1.0, 2.0, 2.0, -1.0, -1.01]
            target_pwl_model = PwlModel(xs, ys)
        
            stepsize = (max(xs) - min(xs)) / float(num_samples)
            x = numpy.arange(min(xs), max(xs), stepsize)
            y = target_pwl_model.simulate(x)
        else:
            x, y = [], []
        waveforms = numpy.array([x,y])

        #plotAndPause(x, y) #uncomment to test
        
        #run the calculator
        calculator = TransientWaveformCalculator(0, 1, True)
        self.assertEqual(
            sorted(calculator.metricNames()),
            sorted(['dynamic_range', 'slewrate', 'nmse', 'correlation',
                    'ymin_before_ymax', 'ymin_after_ymax']))
        
        r = calculator.compute(waveforms)
        self.assertEqual(sorted(r.keys()), sorted(calculator.metricNames()))

        if num_samples > 0:
            self.assertTrue((3.01 - 0.10) <= r['dynamic_range'] <= (3.01 + 0.10),
                            r['dynamic_range'])
            srp = + (2.0 - (-1.0)) / (2.0 - 1.0)
            srn = - (-1.0 - 2.0) / (6.0 - 5.0)
            sr = min(srp, srn)
            self.assertTrue((sr - 0.20) <= r['slewrate'] <= (sr + 0.30), (r['slewrate'], sr))
            self.assertTrue(r['nmse'] < 0.02)
            self.assertTrue(r['correlation'] > 0.95)
            self.assertEqual(r['ymin_before_ymax'], 1.0)
            self.assertEqual(r['ymin_after_ymax'], 1.0)
        else:
            self.assertEqual(r['dynamic_range'], 0)
            self.assertEqual(r['slewrate'], -100)
            self.assertEqual(r['nmse'], 1.0)
            self.assertEqual(r['correlation'], -1.0)
            self.assertEqual(r['ymin_before_ymax'], 0.0)
            self.assertEqual(r['ymin_after_ymax'], 0.0)


    def testTransientWaveformCalculator_EmptyX(self):
        if self.just1: return

        waveforms = numpy.ones((2,0), dtype=float)
        
        #run the calculator
        calculator = TransientWaveformCalculator(0, 1, True)

        #just make sure it doesn't die
        r = calculator.compute(waveforms)

        
    def tearDown(self):
        pass

if __name__ == '__main__':
    #if desired, this is where logging would be set up
    
    unittest.main()
