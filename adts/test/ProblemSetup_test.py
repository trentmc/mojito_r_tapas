import unittest

from adts import *
from problems.Problems import ProblemFactory

def some_function(x):
    return x+2

def function2(x):
    return x+2

def dummyPart():
    return AtomicPart('R',['1','2'],PointMeta({}))

class ProblemSetupTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False    #to make True is a HACK
        
    def testBasic(self):
        if self.just1: return
        an = FunctionAnalysis(some_function, [EnvPoint(True)], 10, 20, False, 10, 20)
        an2 = FunctionAnalysis(function2, [EnvPoint(True)], 10, 20, False, 10, 20)
        dummy_part = dummyPart()
        emb_part = EmbeddedPart(dummy_part, dummy_part.unityPortMap(), {})
        dummy_lib = None
        ps = ProblemSetup(emb_part, [an, an2], dummy_lib)
        self.assertEqual(len(ps.analyses), 2)
        
        fm = ps.flattenedMetrics()
        fm_names = [metric.name for metric in fm]
        self.assertEqual(len(fm), 2)
        self.assertTrue(an.metric.name in fm_names)
        self.assertTrue(an2.metric.name in fm_names)
        
        empty_list = []
        self.assertRaises(ValueError, ProblemSetup, emb_part, empty_list, None)

        self.assertEqual(ps.metric(an2.metric.name).name, an2.metric.name)
        self.assertRaises(ValueError, ps.metric, 'nonexistent_metric_name')

        r1 = ps.devices_setup.nominalRndPoint()
        r2 = ps.devices_setup.nominal_rnd_point
        r3 = ps.nominalRndPoint()
        r4 = ps.nominalRndPoint()
        self.assertTrue(r1 == r2 == r3 == r4)

        ps.devices_setup.makeRobust()
        r5 = ps.devices_setup.all_rnd_points[1]
        r6 = ps.devices_setup.all_rnd_points[2]
        self.assertNotEqual(r5.ID, r6.ID)
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    #if desired, this is where logging would be set up
    
    unittest.main()
