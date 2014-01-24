import unittest

from engine.SynthSolutionStrategy import SynthSolutionStrategy

class SynthSolutionStrategyTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

    def test1(self):
        if self.just1: return

        for num_inds_per_age_layer in [3,6]:
            for do_novelty_gen in [False, True]:
                ss = SynthSolutionStrategy(do_novelty_gen, num_inds_per_age_layer)

                self.assertEqual(ss.do_novelty_gen, do_novelty_gen)
                self.assertEqual(ss.num_inds_per_age_layer, num_inds_per_age_layer)
                self.assertEqual(ss.age_gap, 20) #default age_gap should be 20

                ss2 = SynthSolutionStrategy(do_novelty_gen, num_inds_per_age_layer, 3)
                self.assertEqual(ss2.age_gap, 3)
        
        ss.setMaxNumGenerations(14)
        self.assertEqual(ss.max_num_gens, 14)

        self.assertTrue("SynthSolutionStrategy" in str(ss))
        
    def tearDown(self):
        pass

if __name__ == '__main__':

    import logging
    logging.basicConfig()
    
    unittest.main()
