import unittest
import random

from adts.Var import *

class VarTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK
        pass

    def testContinuousVarMetaNoLogScale(self):
        if self.just1: return
        vm = ContinuousVarMeta(False, -5, 3.0)
        
        self.assertTrue('av' in vm.name)
        self.assertTrue(vm.use_eq_in_netlist)
        
        rb = vm.railbinUnscaled
        self.assertEqual( rb(2), 2)
        self.assertEqual( rb(-6), -5)
        self.assertEqual( rb(-6.1), -5)
        self.assertEqual( rb(30), 3.0)
        self.assertEqual( rb(30.1), 3.0)

        for i in range(100):
            self.assertTrue( -5 <= vm.createRandomUnscaledVar(False) <= 3.0 )
        
        self.assertEqual( vm.scale(4), 4)
        self.assertEqual( vm.scale(4.1), 4.1)
        self.assertEqual( vm.scale(-1), -1)
        self.assertEqual( vm.scale(-10), -10) #note no railbining

        #self.assertEqual( vm.railbinThenScale(-10), -5)
        #self.assertEqual( vm.railbinThenScale(5), 3.0)

        self.assertEqual( vm.unscale(5), 5)
        self.assertEqual( vm.unscale(-6.0), -6.0)

        self.assertEqual( vm.min_unscaled_value, -5)
        self.assertEqual( vm.max_unscaled_value, 3.0)

        self.assertTrue( len(str(vm)) > 0)

        self.assertRaises(ValueError, ContinuousVarMeta, 'not_bool', -5, 3.0)
        self.assertRaises(ValueError, ContinuousVarMeta, False,'not number', 3.0)
        self.assertRaises(ValueError, ContinuousVarMeta, False, -5, 'not number')
        self.assertRaises(ValueError, ContinuousVarMeta, False, 20, 10)
        
        NOT_STRING = 3.0
        self.assertRaises(ValueError, ContinuousVarMeta, False, -5,3, NOT_STRING)
        
    def testContinuousVarMetaLogScale(self):
        if self.just1: return
        vm = ContinuousVarMeta(True, -5, 3, 'myname')
        rb = vm.railbinUnscaled
                
        self.assertEqual( rb(2), 2)
        self.assertEqual( rb(-6), -5)
        self.assertEqual( rb(-6.1), -5)
        self.assertEqual( rb(30), 3.0)
        self.assertEqual( rb(30.1), 3.0)

        for i in range(100):
            self.assertTrue( -5 <= vm.createRandomUnscaledVar(False) <= 3.0 )
        
        self.assertEqual( vm.scale(4), 10**4)
        self.assertEqual( vm.scale(4.1), 10**4.1)
        self.assertEqual( vm.scale(-1), 10**(-1))
        self.assertEqual( vm.scale(-10), 10**(-10)) #note no railing

        #self.assertEqual( vm.railbinThenScale(-10), 10**(-5))
        #self.assertEqual( vm.railbinThenScale(5), 10**(3.0))

        self.assertAlmostEqual( vm.unscale(10**(-6)), -6.0, 3)
        self.assertAlmostEqual( vm.unscale(10**(4)), 4.0, 3)
        
        self.assertEqual( vm.min_unscaled_value, -5)
        self.assertEqual( vm.max_unscaled_value, 3)

        self.assertTrue( len(str(vm)) > 0)

    def testMutateContinuousVarMeta(self):
        if self.just1: return
        vm = ContinuousVarMeta(True, -5, 3, 'myname')
        for i in range(1000):
            self.assertTrue(
                -5 <= vm.mutate(random.uniform(-6,4), 0.05, False) <= 3)
            self.assertTrue(
                -5 <= vm.mutate(random.uniform(-6,4), 1.0, False) <= 3)
        self.assertRaises(ValueError, vm.mutate, 3.0, -0.5, False)
        self.assertRaises(ValueError, vm.mutate, 3.0, 1.1, False)

    def testContinuousVarMetaNoRange_Logscale(self):
        if self.just1: return
        self.helper_testContinuousVarMetaNoRange(True)

    def testContinuousVarMetaNoRange_NoLogscale(self):
        if self.just1: return
        self.helper_testContinuousVarMetaNoRange(False)
        
    def helper_testContinuousVarMetaNoRange(self, logscale):
        vm =  ContinuousVarMeta(logscale, -5, -5)
        for i in range(20):
            r = (random.random()-0.5) * 10.0
            self.assertEqual( vm.railbinUnscaled(r), -5 )

        for i in range(100):
            self.assertTrue( -5 <= vm.createRandomUnscaledVar(False) <= -5 )

        self.assertEqual( vm.min_unscaled_value, -5)
        self.assertEqual( vm.max_unscaled_value, -5)
        
    def testDiscreteVarMeta(self):
        if self.just1: return
        possvals = [-8.0, 3000.0, 3010.1]
        vm = DiscreteVarMeta(possvals)
        
        self.assertTrue('av' in vm.name)
        self.assertTrue(vm.use_eq_in_netlist)

        poss_unscaled_vals = [0, 1, 2]
        for i in range(100):
            self.assertTrue( vm.createRandomUnscaledVar(False)
                             in poss_unscaled_vals)
            
        rb = vm.railbinUnscaled
        
        self.assertEqual( rb(-3.1), 0)
        self.assertEqual( rb(-0.1), 0)
        self.assertEqual( rb(0.0), 0)
        self.assertEqual( rb(0), 0)
        self.assertEqual( rb(0.4), 0)
        self.assertEqual( rb(0.6), 1)
        self.assertEqual( rb(1), 1)
        self.assertEqual( rb(1.0), 1)
        self.assertEqual( rb(1.6), 2)
        self.assertEqual( rb(2), 2)
        self.assertEqual( rb(2.1), 2)
        self.assertEqual( rb(2.6), 2)
        self.assertEqual( rb(3), 2)
        self.assertEqual( rb(30), 2)

        self.assertEqual( vm.scale(-3.1), possvals[0])
        self.assertEqual( vm.scale(0),    possvals[0])
        self.assertEqual( vm.scale(50),   possvals[2])

        for i in range(20):
            r = (random.random()-0.3) * 10.0
            self.assertEqual( vm.scale(r), vm._railbinThenScale(r) )

        self.assertEqual( vm.unscale(-5000.0), 0)
        self.assertEqual( vm.unscale(-8.1), 0)
        self.assertEqual( vm.unscale(-8.0), 0)
        self.assertEqual( vm.unscale(-10), 0)
        self.assertEqual( vm.unscale(1000.0), 0)#still closer to -8.0 than 3000.0
        self.assertEqual( vm.unscale(2000), 1)
        self.assertEqual( vm.unscale(2000.0), 1)
        self.assertEqual( vm.unscale(2999), 1)
        self.assertEqual( vm.unscale(3006), 2)
        self.assertEqual( vm.unscale(1e8), 2)
        
        self.assertEqual( vm.min_unscaled_value, 0)
        self.assertEqual( vm.max_unscaled_value, 2)

        vm.addNewPossibleValue(-2.0)
        self.assertEqual( vm.possible_values, [-8.0, -2.0, 3000.0, 3010.1] )
        self.assertEqual( vm.max_unscaled_value, 3)
        self.assertEqual( vm.unscale(-2.1), 1)
        self.assertEqual( vm.unscale(2999), 2)
        
        self.assertTrue( len(str(vm)) > 0)
            

        self.assertRaises(ValueError, DiscreteVarMeta, [1.0,3.0,2.0,'not_numbr'])
        self.assertRaises(ValueError, DiscreteVarMeta, [1.0,3.0,2.0])
        NOT_STRING = 3.0
        self.assertRaises(ValueError, DiscreteVarMeta, [1.0,2.0], NOT_STRING)

    def testMutateDiscreteVarMeta(self):
        if self.just1: return
        #                     0     1       2       3
        vm = DiscreteVarMeta([-8.0, 3000.0, 3010.1, 40000])
        choice_vm = DiscreteVarMeta([0,1,2,3])
        one_possval_vm = DiscreteVarMeta([3.1])

        # -also test isChoiceVar()
        self.assertEqual(vm._is_choice_var, None)
        self.assertFalse(vm.isChoiceVar())
        self.assertEqual(vm._is_choice_var, False)
        
        self.assertEqual(choice_vm._is_choice_var, None)
        self.assertTrue(choice_vm.isChoiceVar())
        self.assertEqual(choice_vm._is_choice_var, True)
        
        self.assertFalse(one_possval_vm.isChoiceVar())
        self.assertFalse(DiscreteVarMeta([0,2]).isChoiceVar())
        
        for i in range(100):
            self.assertTrue(
                vm.mutate(random.randint(-1,6), 0.25, False) in [0,1,2,3])
            self.assertEqual(vm.mutate(-3, 0.0, False), 1) #0, plus 1
            self.assertEqual(vm.mutate(0, 0.0, False), 1)  #0, plus 1
            self.assertEqual(vm.mutate(3, 0.0, False), 2)  #3, minus 1
            self.assertEqual(vm.mutate(10, 0.0, False), 2) #3, minus 1
            self.assertTrue(
                choice_vm.mutate(random.randint(-1,5), 0.05, False) in [0,1,2,3])
            self.assertEqual(
                one_possval_vm.mutate(random.randint(-1,5),0.25, False),0)

    def testDiscreteVarMetaNoRange(self):
        if self.just1: return
        possvals = [-5.0]
        vm = DiscreteVarMeta(possvals)

        poss_unscaled_vals = [0]
        for i in range(100):
            self.assertTrue( vm.createRandomUnscaledVar(False)
                             in poss_unscaled_vals)

        for i in range(20):
            r = (random.random()-0.3) * 10.0
            self.assertEqual( vm.railbinUnscaled(r), 0 )
            
        self.assertEqual( vm.min_unscaled_value, 0)
        self.assertEqual( vm.max_unscaled_value, 0)

    def testEqualityOverride(self):
        if self.just1: return
        
        #test override of == on cont.
        #
        self.assertTrue(ContinuousVarMeta(False, -5, 3.0,'name_a') == \
                        ContinuousVarMeta(False, -5, 3.0,'name_a'))
        self.assertTrue(ContinuousVarMeta(True, -5, 3.0,'name_a') == \
                        ContinuousVarMeta(True, -5, 3.0,'name_a'))
        self.assertFalse(ContinuousVarMeta(False, -5, 3.0,'name_a') == \
                         ContinuousVarMeta(True, -5, 3.0,'name_a'))
        self.assertFalse(ContinuousVarMeta(False, -4, 3.0,'name_a') == \
                         ContinuousVarMeta(False, -5, 3.0,'name_a'))
        self.assertFalse(ContinuousVarMeta(True, -5, 4.0,'name_a') == \
                         ContinuousVarMeta(True, -5, 3.0,'name_a'))

        #test override of != on cont.
        self.assertTrue(ContinuousVarMeta(False, -5, 3.0,'name_a') != \
                        ContinuousVarMeta(True, -5, 3.0,'name_a'))
        self.assertTrue(ContinuousVarMeta(False, -4, 3.0,'name_a') != \
                        ContinuousVarMeta(False, -5, 3.0,'name_a'))
        self.assertTrue(ContinuousVarMeta(True, -5, 4.0,'name_a') != \
                        ContinuousVarMeta(True, -5, 3.0,'name_a'))
        self.assertTrue(ContinuousVarMeta(False, -5, 3.0,'name_a') != \
                        ContinuousVarMeta(False, -5, 3.0,'name_b'))

        #test override of == on discrete
        self.assertTrue( DiscreteVarMeta([-8.0, 3000.0],'name_a') == \
                         DiscreteVarMeta([-8.0, 3000.0],'name_a') )
        self.assertFalse( DiscreteVarMeta([-8.0, 3001.0],'name_a') == \
                          DiscreteVarMeta([-8.0, 3000.0],'name_a') )
        self.assertFalse( DiscreteVarMeta([-8.0],'name_a') == \
                          DiscreteVarMeta([-8.0, 3000.0],'name_a') )

        #test override of != on discrete
        self.assertFalse( DiscreteVarMeta([-8.0, 3000.0],'name_a') != \
                         DiscreteVarMeta([-8.0, 3000.0],'name_a') )
        self.assertTrue( DiscreteVarMeta([-8.0, 3001.0],'name_a') != \
                         DiscreteVarMeta([-8.0, 3000.0],'name_a') )
        self.assertTrue( DiscreteVarMeta([-8.0],'name_a') != \
                         DiscreteVarMeta([-8.0, 3000.0],'name_a') )

        #test override of == on discrete-vs-continuous
        self.assertFalse(DiscreteVarMeta([-8.0, 3000.0],'name_a') == \
                         ContinuousVarMeta(True, -5, 3.0,'name_a'))
        self.assertFalse(ContinuousVarMeta(True, -5, 3.0,'name_a') == \
                         DiscreteVarMeta([-8.0, 3000.0],'name_a'))
        
        
    def testDiscreteVarMetaNoPossibleValues(self):
        if self.just1: return
        pass
        #DO NOT have the following check, because we want to be able
        # to add possible values after creation of the DiscreteVarMeta
        #self.assertRaises(ValueError, DiscreteVarMeta, [])
        
    def testChoiceVarMeta(self):
        if self.just1: return
        possvals = [0, 1, 2]
        vm = ChoiceVarMeta(possvals)
        
        self.assertTrue('av' in vm.name)
        self.assertTrue(vm.use_eq_in_netlist)

        poss_unscaled_vals = [0, 1, 2]
        for i in range(100):
            self.assertTrue( vm.createRandomUnscaledVar(False)
                             in poss_unscaled_vals)
        
    def testChoiceVarMetaProbability(self):
        if self.just1: return

        # the nominal value is uniform probability
        possvals = [0, 1]
        weights = [1, 1]
        vm = ChoiceVarMeta(possvals)
        self._testChoiceVarMetaProbabilityHelper(vm, possvals, weights, True)        
        # novelty should not matter
        self._testChoiceVarMetaProbabilityHelper(vm, possvals, weights, True)        

        # add a nonnovel choice
        vm.addNewPossibleValue(2, None, False)
        possvals = [0, 1, 2]
        weights = [1, 1, 1]
        self._testChoiceVarMetaProbabilityHelper(vm, possvals, weights, True)        
        # novelty should not matter
        self._testChoiceVarMetaProbabilityHelper(vm, possvals, weights, True)        

        # add a novel choice
        vm.addNewPossibleValue(3, None, True)
        possvals = [0, 1, 2, 3]
        weights_nov = [1, 1, 1, 1]
        self._testChoiceVarMetaProbabilityHelper(vm, possvals, weights_nov, True)        
        # now novelty should matter
        weights_nonnov = [1, 1, 1, 0]
        self._testChoiceVarMetaProbabilityHelper(vm, possvals, weights_nonnov, False)
        
        # adding a weighted item when no weigthing is used is not allowed
        self.assertRaises(AssertionError, vm.addNewPossibleValue, 2, 2, False)
            
        # some other tests
        possvals = [0, 1]
        weights = [4, 1]
        vm = ChoiceVarMeta(possvals, weights)
        self._testChoiceVarMetaProbabilityHelper(vm, possvals, weights, True)        

        # adding an unweighted item when weigthing is used is not allowed
        self.assertRaises(AssertionError, vm.addNewPossibleValue, 2, None, False)

        possvals = [0, 1, 2, 3]
        weights = [4, 1, 1, 1]
        vm = ChoiceVarMeta(possvals, weights)
        self._testChoiceVarMetaProbabilityHelper(vm, possvals, weights, True)        

    def _testChoiceVarMetaProbabilityHelper(self, vm, possvals, weights, allow_novel_choices,
                                            nb_testpoints = 10000, max_deviation_pct = 10.0):
        # gather some test points
        measured_occurance = [0 for i in range(0, len(possvals) )]
        for i in range(0, nb_testpoints):
            randval = vm.createRandomUnscaledVar(allow_novel_choices)
            measured_occurance[possvals.index(randval)] += 1

        # calculate the expected occurance
        sum_weight = 0.0
        for i in range(0, len(possvals)):
            sum_weight += float(weights[i])
        expected_occurance=[]
        for i in range(0, len(possvals)):
            v = float(weights[i]) / sum_weight
            v *= nb_testpoints
            expected_occurance.append(round(v,0))

        # check whether the diff between maximum and
        # actual occurance is not too large
        for i in range(0, len(possvals)):
            abs_diff = abs(expected_occurance[i] - measured_occurance[i])
            if abs_diff < 1e-20:
                rel_diff = 0
            else:
                rel_diff = 2.0 * abs_diff / abs(expected_occurance[i] + measured_occurance[i])
            
            self.assertTrue( rel_diff * 100.0 < max_deviation_pct )  
        
    def testVarMetaNoName(self):
        if self.just1: return
        vm1 = ContinuousVarMeta(True, -5, 3)
        vm2 = ContinuousVarMeta(True, -5, 3)
        self.assertTrue( len(vm1.name) > 0 )
        self.assertNotEqual( vm1.name, vm2.name )
        
        vm3 = DiscreteVarMeta([1,2,3])
        vm4 = DiscreteVarMeta([1,2,3])
        self.assertTrue( len(vm3.name) > 0 )
        self.assertNotEqual( vm3.name, vm4.name )
        
    def testSpiceStr(self):
        if self.just1: return
        #the testing here also ensures that spiceNetlistStr is not
        # responsible for scaling and binning
        resistance_vm = ContinuousVarMeta(True, 1, 7, 'R', True)
        self.assertEqual( resistance_vm.spiceNetlistStr(1e8), 'R=%g' % 1e8)
        
        dc_current_vm = ContinuousVarMeta(False, 1e-3, 10e-3, 'DC', False)
        self.assertEqual( dc_current_vm.spiceNetlistStr(22), 'DC %g' % 22)
        
        resistance_vm = DiscreteVarMeta([1e3, 1e4, 1e5, 1e7], 'R', True)
        self.assertEqual( resistance_vm.spiceNetlistStr(2e7), 'R=%g' % 2e7)
        
        dc_current_vm = DiscreteVarMeta([1e-3, 2e-3, 10e-3], 'DC', False)
        self.assertEqual( dc_current_vm.spiceNetlistStr(3e-3), 'DC %g' % 3e-3)
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    #if desired, this is where logging would be set up
    
    unittest.main()
