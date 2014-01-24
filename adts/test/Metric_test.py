import unittest

from adts.Metric import *
from util.constants import BAD_METRIC_VALUE, INF

class MetricTest(unittest.TestCase):

    def setUp(self):
        self.infs_with_bad = [-INF, INF, BAD_METRIC_VALUE]

    def testSimpleMetric(self):
        #test a basic, simple metric constraint
        metric = Metric('gain', 10, INF, False, 0, 20)
        self.assertEqual(metric.name, 'gain')
        self.assertEqual(metric.min_threshold, 10)
        self.assertEqual(metric.max_threshold, INF)
        self.assertFalse(metric.improve_past_feasible)
        self.assertEqual(metric.rough_minval, 0)
        self.assertEqual(metric.rough_maxval, 20)
        self.assertTrue(len(str(metric)) > 0)

    def testThresholdCombinations(self):
        #support a variety of thresholds: int or float or 'Inf', and most
        # combinations thereof
        self.assertEqual(Metric('gain', 10.1, INF, False, 0, 10).min_threshold, 10.1)
        self.assertEqual(Metric('gain', 10, 120, False, 0, 10).max_threshold, 120)
        self.assertEqual(Metric('gain', 10, 120.1, False, 0, 10).max_threshold, 120.1)
        self.assertEqual(Metric('gain', 10.1, 120, False, 0, 10).min_threshold,10.1)
        self.assertEqual(Metric('gain', -INF, 120, False, 0, 10).max_threshold, 120)
        self.assertEqual(Metric('gain', -INF, 120.1, False, 0, 10).max_threshold, 120.1)

        # support equality constraints
        eq_metric = Metric('gain', 10, 10, False, 0, 10)
        self.assertEqual(eq_metric.min_threshold, 10)
        self.assertEqual(eq_metric.max_threshold, 10)
        eq_metric = Metric('gain', 10.1, 10.1, False, 0, 10)
        self.assertEqual(eq_metric.min_threshold, 10.1)
        self.assertEqual(eq_metric.max_threshold, 10.1)
        
        #but there are some disallowed combinations of threshold values
        # -no: equality constraint of -Inf or +Inf
        self.assertRaises(ValueError, Metric,'gain',-INF,-INF, False, 0, 1)
        self.assertRaises(ValueError, Metric,'gain',INF,INF, False, 0, 1)
        
        # -no: equality objectives (nowhere to improve!)
        self.assertRaises(ValueError, Metric, 'gain', 10, 10, True, 0, 10)
        self.assertRaises(ValueError, Metric, 'gain', 10.1, 10.1, True, 0, 10)
        
        # -no: min>max
        self.assertRaises(ValueError, Metric, 'gain', 11, 10, False, 0, 10)
        self.assertRaises(ValueError, Metric, 'gain', 11.1, 10, False, 0, 10)
        
        # -no: -inf/+inf
        self.assertRaises(ValueError, Metric, 'gain', -INF,INF, False, 0, 10)

        # -no: non-number values
        self.assertRaises(ValueError, Metric, 'gain', None, 1, False, 0, 10)
        self.assertRaises(ValueError, Metric, 'gain', 1, None, False, 0, 10)
        

    #the next few tests enumerate through:
    # {MINIMIZE, MAXIMIZE, IN_RANGE} x {do_not_improve_past_feasible, do_improve}
    def testMinimizeDontImproveMetric(self):
        metric = Metric('gain', -INF, 10, False, 0, 10)
        self.assertFalse(metric.improve_past_feasible)
        self.assertEqual(metric._aim, MINIMIZE)

        self.assertRaises(AssertionError, metric.margin, -20) #error b/c only on objectives
        
        self.assertEqual(metric.worstCaseValue([-10, -12.2, 3000]), 3000)
        self.assertEqual(metric.worstCaseValue([10, 12.2, -3000]), 12.2)
        self.assertEqual(metric.worstCaseValue([10, 12.2, -INF]),12.2)
        self.assertEqual(metric.worstCaseValue(self.infs_with_bad), BAD_METRIC_VALUE)

        self.assertTrue(metric.isFeasible(9))
        self.assertTrue(metric.isFeasible(10))
        self.assertTrue(metric.isFeasible(10.0))
        self.assertFalse(metric.isFeasible(10.1))
        self.assertFalse(metric.isFeasible(11))
        self.assertFalse(metric.isFeasible(BAD_METRIC_VALUE))
        
        self.assertTrue(metric.isBetter(8, 12))
        self.assertFalse(metric.isBetter(12, 8))
        self.assertFalse(metric.isBetter(8,9))
        self.assertFalse(metric.isBetter(8,10))
        self.assertTrue(metric.isBetter(11, 12))
        self.assertFalse(metric.isBetter(12, 11))
        self.assertTrue(metric.isBetter(11, INF))
        self.assertFalse(metric.isBetter(INF, 11))
        
        self.assertFalse(metric.isBetter(BAD_METRIC_VALUE, BAD_METRIC_VALUE))
        self.assertTrue(metric.isBetter(INF, BAD_METRIC_VALUE))
        self.assertTrue(metric.isBetter(+10000, BAD_METRIC_VALUE))
        self.assertFalse(metric.isBetter(BAD_METRIC_VALUE, INF))
        self.assertFalse(metric.isBetter(BAD_METRIC_VALUE, +10000))

        self.assertEqual(metric.constraintViolation(BAD_METRIC_VALUE), INF)
        self.assertEqual(metric.constraintViolation(10), 0)
        self.assertEqual(metric.constraintViolation(9), 0)
        self.assertEqual(metric.constraintViolation(9.0), 0.0)
        self.assertEqual(metric.constraintViolation(11), 1)
        self.assertEqual(metric.constraintViolation(11.0), 1.0)
        self.assertEqual(metric.constraintViolation(11.5), 1.5)
        

    def testMinimizeImproveMetric(self):
        metric = Metric('gain', -INF, 10, True, 0, 10)
        self.assertTrue(metric.improve_past_feasible)
        self.assertEqual(metric._aim, MINIMIZE)
        
        self.assertRaises(AssertionError, metric.margin, 20) #error b/c metric value is infeasible
        self.assertRaises(AssertionError, metric.margin, INF) # ""
        self.assertRaises(AssertionError, metric.margin, BAD_METRIC_VALUE) # ""
        self.assertEqual(metric.margin(10), 0)
        self.assertEqual(metric.margin(8), 2)
        self.assertEqual(metric.margin(7.5), 2.5)
        self.assertEqual(metric.margin(-INF), INF)
        
        self.assertEqual(metric.poorValue(), 10 + 10)
        
        self.assertTrue(metric.isBetter(8, 12))
        self.assertFalse(metric.isBetter(12, 8))
        self.assertTrue(metric.isBetter(8,9)) # where 'improve' diff shows up
        self.assertTrue(metric.isBetter(8,10)) # "
        self.assertTrue(metric.isBetter(11, 12))
        self.assertFalse(metric.isBetter(12, 11))
        self.assertTrue(metric.isBetter(11, INF))
        self.assertFalse(metric.isBetter(INF, 11))
    
    def testMaximizeDontImproveMetric(self):
        metric = Metric('gain', 10, INF, False, 0, 10)
        self.assertFalse(metric.improve_past_feasible)
        self.assertEqual(metric._aim, MAXIMIZE)
        
        self.assertRaises(AssertionError, metric.margin, 20) #error b/c only on objectives
        
        self.assertEqual(metric.poorValue(), 10 - 10)
        
        self.assertEqual(metric.worstCaseValue([10, 12.2, -3000]), -3000)
        self.assertEqual(metric.worstCaseValue([-10, -12.2, 3000]), -12.2)
        self.assertEqual(metric.worstCaseValue([-10, -12.2, INF]),-12.2)
        self.assertEqual(metric.worstCaseValue(self.infs_with_bad), BAD_METRIC_VALUE)

        self.assertFalse(metric.isFeasible(9))
        self.assertTrue(metric.isFeasible(10))
        self.assertTrue(metric.isFeasible(10.0))
        self.assertTrue(metric.isFeasible(10.1))
        self.assertTrue(metric.isFeasible(11))
        self.assertFalse(metric.isFeasible(BAD_METRIC_VALUE))
        
        self.assertFalse(metric.isBetter(BAD_METRIC_VALUE, BAD_METRIC_VALUE))
        self.assertTrue(metric.isBetter(-INF, BAD_METRIC_VALUE))
        self.assertTrue(metric.isBetter(-10000, BAD_METRIC_VALUE))
        self.assertFalse(metric.isBetter(BAD_METRIC_VALUE, -INF))
        self.assertFalse(metric.isBetter(BAD_METRIC_VALUE, -10000))
        
        self.assertTrue(metric.isBetter(12, 8))
        self.assertFalse(metric.isBetter(8, 12))
        self.assertFalse(metric.isBetter(12,11))
        self.assertFalse(metric.isBetter(12,10))
        self.assertTrue(metric.isBetter(9, 8))
        self.assertFalse(metric.isBetter(8, 9))
        self.assertTrue(metric.isBetter(9, -INF))
        self.assertFalse(metric.isBetter(-INF, 9))

        self.assertEqual(metric.constraintViolation(BAD_METRIC_VALUE), INF)
        self.assertEqual(metric.constraintViolation(10), 0)
        self.assertEqual(metric.constraintViolation(11), 0)
        self.assertEqual(metric.constraintViolation(11.0), 0.0)
        self.assertEqual(metric.constraintViolation(9), 1)
        self.assertEqual(metric.constraintViolation(9.0), 1.0)
        self.assertEqual(metric.constraintViolation(8.5), 1.5)
    
    def testMaximizeImproveMetric(self):
        metric = Metric('gain', 10, INF, True, 0, 10)
        self.assertTrue(metric.improve_past_feasible)
        self.assertEqual(metric._aim, MAXIMIZE)
        
        self.assertRaises(AssertionError, metric.margin, -20) #error b/c metric value is infeasible
        self.assertRaises(AssertionError, metric.margin, -INF) # ""
        self.assertRaises(AssertionError, metric.margin, BAD_METRIC_VALUE) # ""
        self.assertEqual(metric.margin(10), 0)
        self.assertEqual(metric.margin(12), 2)
        self.assertEqual(metric.margin(12.5), 2.5)
        self.assertEqual(metric.margin(INF), INF)
        
        self.assertTrue(metric.isBetter(12, 8))
        self.assertFalse(metric.isBetter(8, 12))
        self.assertTrue(metric.isBetter(12,11)) # where 'improve' diff shows up
        self.assertTrue(metric.isBetter(12,10)) # "
        self.assertTrue(metric.isBetter(9, 8))
        self.assertFalse(metric.isBetter(8, 9))
        self.assertTrue(metric.isBetter(9, -INF))
        self.assertFalse(metric.isBetter(-INF, 9))

    def testInrangeDontImproveMetric(self):
        metric = Metric('gain', 10, 20, False, 0, 10)
        self.assertFalse(metric.improve_past_feasible)
        self.assertEqual(metric._aim, IN_RANGE)
        
        self.assertEqual(metric.poorValue(), 20 + 10)
        
        self.assertRaises(AssertionError, metric.margin, 15) #error b/c only on objectives
        
        self.assertEqual(metric.worstCaseValue([10, 12.2, -3000, 25]), -3000)
        self.assertEqual(metric.worstCaseValue([10, 12.2, +3000, 25]), +3000)
        self.assertEqual(metric.worstCaseValue([10, 3000, 25, INF]), INF)
        self.assertEqual(metric.worstCaseValue([10, 14, 15, 19]), 10)
        self.assertEqual(metric.worstCaseValue([11, 20, 14, 15, 19]), 20)
        self.assertEqual(metric.worstCaseValue([14, 15, 19]), 19)
        self.assertEqual(metric.worstCaseValue(self.infs_with_bad), BAD_METRIC_VALUE)

        self.assertFalse(metric.isFeasible(9))
        self.assertTrue(metric.isFeasible(10))
        self.assertTrue(metric.isFeasible(10.0))
        self.assertTrue(metric.isFeasible(11))
        self.assertTrue(metric.isFeasible(19.99))
        self.assertTrue(metric.isFeasible(20.0))
        self.assertTrue(metric.isFeasible(20))
        self.assertFalse(metric.isFeasible(20.001))
        self.assertFalse(metric.isFeasible(21))
        self.assertFalse(metric.isFeasible(BAD_METRIC_VALUE))

        #test lower bound
        self.assertTrue(metric.isBetter(12, 8))
        self.assertFalse(metric.isBetter(8, 12))
        self.assertFalse(metric.isBetter(12,11))
        self.assertFalse(metric.isBetter(12,10))
        self.assertTrue(metric.isBetter(9, 8))
        self.assertFalse(metric.isBetter(8, 9))
        self.assertTrue(metric.isBetter(9, -INF))
        self.assertFalse(metric.isBetter(-INF, 9))

        self.assertEqual(metric.constraintViolation(BAD_METRIC_VALUE), INF)
        self.assertEqual(metric.constraintViolation(10), 0)
        self.assertEqual(metric.constraintViolation(11), 0)
        self.assertEqual(metric.constraintViolation(11.0), 0.0)
        self.assertEqual(metric.constraintViolation(9), 1)
        self.assertEqual(metric.constraintViolation(9.0), 1.0)
        self.assertEqual(metric.constraintViolation(8.5), 1.5)

        #test upper bound
        self.assertTrue(metric.isBetter(18, 22))
        self.assertFalse(metric.isBetter(22, 18))
        self.assertFalse(metric.isBetter(15,16))
        self.assertFalse(metric.isBetter(15,20))
        self.assertTrue(metric.isBetter(22, 24))
        self.assertFalse(metric.isBetter(24, 22))
        self.assertTrue(metric.isBetter(24, INF))
        self.assertFalse(metric.isBetter(INF, 24))

        self.assertEqual(metric.constraintViolation(20), 0)
        self.assertEqual(metric.constraintViolation(19), 0)
        self.assertEqual(metric.constraintViolation(19.0), 0.0)
        self.assertEqual(metric.constraintViolation(21), 1)
        self.assertEqual(metric.constraintViolation(21.0), 1.0)
        self.assertEqual(metric.constraintViolation(21.5), 1.5)

    def testInrangeImproveMetric(self):
        metric = Metric('gain', 10, 20, True, 0, 10)
        self.assertTrue(metric.improve_past_feasible)
        self.assertEqual(metric._aim, IN_RANGE)

        self.assertRaises(AssertionError, metric.margin, 5) #error b/c metric value is infeasible
        self.assertRaises(AssertionError, metric.margin, -INF) # ""
        self.assertRaises(AssertionError, metric.margin, +INF) # ""
        self.assertRaises(AssertionError, metric.margin, BAD_METRIC_VALUE) # ""
        self.assertEqual(metric.margin(10), 0)
        self.assertEqual(metric.margin(12), 2)
        self.assertEqual(metric.margin(12.5), 2.5)
        self.assertEqual(metric.margin(15), 5)
        self.assertEqual(metric.margin(19.5), 0.5)
        self.assertEqual(metric.margin(20), 0)

        #test lower bound
        self.assertTrue(metric.isBetter(12, 8))
        self.assertFalse(metric.isBetter(8, 12))
        self.assertTrue(metric.isBetter(12,11)) # where 'improve' diff shows up
        self.assertTrue(metric.isBetter(12,10)) # ''
        self.assertTrue(metric.isBetter(9, 8))
        self.assertFalse(metric.isBetter(8, 9))
        self.assertTrue(metric.isBetter(9, -INF))
        self.assertFalse(metric.isBetter(-INF, 9))

        #test upper bound
        self.assertTrue(metric.isBetter(18, 22))
        self.assertFalse(metric.isBetter(22, 18))
        self.assertTrue(metric.isBetter(15,16)) # where 'improve' diff shows up
        self.assertTrue(metric.isBetter(15,20)) # ''
        self.assertTrue(metric.isBetter(22, 24))
        self.assertFalse(metric.isBetter(24, 22))
        self.assertTrue(metric.isBetter(24, INF))
        self.assertFalse(metric.isBetter(INF, 24))

    def testIntegerEqualityMetric(self):
        metric = Metric('gain', 10, 10, False, 0, 10)
        
        self.assertFalse(metric.improve_past_feasible)
        self.assertEqual(metric._aim, IN_RANGE)
        
        self.assertEqual(metric.worstCaseValue([10, 12.2, -3000, 25]), -3000)
        self.assertEqual(metric.worstCaseValue([10, 12.2, +3000, 25]), +3000)
        self.assertEqual(metric.worstCaseValue([10, 3000, 25, INF]), INF)
        self.assertEqual(metric.worstCaseValue([0, 4, 5, 9]), 0)
        self.assertEqual(metric.worstCaseValue([11, 20, 14, 15, 19]), 20)
        self.assertEqual(metric.worstCaseValue([14, 15, 19]), 19)
        self.assertEqual(metric.worstCaseValue(self.infs_with_bad), BAD_METRIC_VALUE)

        self.assertFalse(metric.isFeasible(9))
        self.assertFalse(metric.isFeasible(9.999999))
        self.assertTrue(metric.isFeasible(10))
        self.assertTrue(metric.isFeasible(10.0))
        self.assertFalse(metric.isFeasible(10.00001))
        self.assertFalse(metric.isFeasible(11))
        self.assertFalse(metric.isFeasible(19.99))
        self.assertFalse(metric.isFeasible(BAD_METRIC_VALUE))

        self.assertEqual(metric.constraintViolation(10), 0)
        self.assertEqual(metric.constraintViolation(10.0), 0)
                         
        #test lower bound
        self.assertTrue(metric.isBetter(10, 8))
        self.assertFalse(metric.isBetter(8, 10))
        self.assertTrue(metric.isBetter(9, 8))
        self.assertFalse(metric.isBetter(8, 9))
        self.assertTrue(metric.isBetter(9, -INF))
        self.assertFalse(metric.isBetter(-INF, 9))
        self.assertEqual(metric.constraintViolation(BAD_METRIC_VALUE),
                         INF)
        self.assertEqual(metric.constraintViolation(9), 1)
        self.assertEqual(metric.constraintViolation(9.0), 1.0)
        self.assertEqual(metric.constraintViolation(8.5), 1.5)

        #test upper bound
        self.assertTrue(metric.isBetter(10, 12))
        self.assertFalse(metric.isBetter(12, 10))
        self.assertFalse(metric.isBetter(24, 22))
        self.assertTrue(metric.isBetter(24, INF))
        self.assertFalse(metric.isBetter(INF, 24))

        self.assertEqual(metric.constraintViolation(11), 1)
        self.assertEqual(metric.constraintViolation(11.0), 1.0)
        self.assertEqual(metric.constraintViolation(11.5), 1.5)

    def testFloatEqualityMetric(self):
        metric = Metric('gain', 10.1, 10.1, False, 0, 10)

        self.assertFalse(metric.improve_past_feasible)
        self.assertEqual(metric._aim, IN_RANGE)
        
        self.assertEqual(metric.worstCaseValue([10, 12.2, -3000, 25]), -3000)
        self.assertEqual(metric.worstCaseValue([10, 12.2, +3000, 25]), +3000)
        self.assertEqual(metric.worstCaseValue([10, 3000, 25, INF]),
                         INF)
        self.assertEqual(metric.worstCaseValue([0, 4, 5, 9]), 0)
        self.assertEqual(metric.worstCaseValue([11, 20, 14, 15, 19]), 20)
        self.assertEqual(metric.worstCaseValue([14, 15, 19]), 19)
        self.assertEqual(metric.worstCaseValue(self.infs_with_bad),
                         BAD_METRIC_VALUE)

        self.assertFalse(metric.isFeasible(9))
        self.assertFalse(metric.isFeasible(10.0999999))
        self.assertFalse(metric.isFeasible(10))
        self.assertTrue(metric.isFeasible(10.1))
        self.assertFalse(metric.isFeasible(10.100001))
        self.assertFalse(metric.isFeasible(11))
        self.assertFalse(metric.isFeasible(19.99))
        self.assertFalse(metric.isFeasible(BAD_METRIC_VALUE))

        self.assertEqual(metric.constraintViolation(BAD_METRIC_VALUE), INF)
        self.assertTrue(metric.constraintViolation(10.0999999) > 0.0)
        self.assertEqual(metric.constraintViolation(10.1), 0.0)
        self.assertTrue(metric.constraintViolation(10.1000001) > 0.0)
                         
        #test lower bound
        self.assertTrue(metric.isBetter(10.1, 8))
        self.assertFalse(metric.isBetter(8, 10.1))
        self.assertTrue(metric.isBetter(9, 8))
        self.assertFalse(metric.isBetter(8, 9))
        self.assertTrue(metric.isBetter(9, -INF))
        self.assertFalse(metric.isBetter(-INF, 9))
        self.assertAlmostEqual(metric.constraintViolation(9), 1.1)
        self.assertAlmostEqual(metric.constraintViolation(9.0), 1.1)
        self.assertAlmostEqual(metric.constraintViolation(8.5), 1.6)

        #test upper bound
        self.assertTrue(metric.isBetter(10.1, 12))
        self.assertFalse(metric.isBetter(12, 10.1))
        self.assertFalse(metric.isBetter(24, 22))
        self.assertTrue(metric.isBetter(24, INF))
        self.assertFalse(metric.isBetter(INF, 24))

        self.assertAlmostEqual(metric.constraintViolation(11), 1-0.1)
        self.assertAlmostEqual(metric.constraintViolation(11.0), 1.0-0.1)
        self.assertAlmostEqual(metric.constraintViolation(11.5), 1.5-0.1)


    def tearDown(self):
        pass

if __name__ == '__main__':
    #if desired, this is where logging would be set up
    
    unittest.main()
