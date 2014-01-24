import types
import random
import math

import unittest
        
from util.mathutil import *

class mathutilTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

    def testStddev(self):
        if self.just1: return

        self.assertAlmostEqual(stddev([3,1]), 1.4142136, 4)
        self.assertEqual(stddev([1,1]), 0.0)
        self.assertAlmostEqual(stddev(numpy.array([3,1])), 1.4142136, 4)

        self.assertRaises(ValueError, stddev, [])
        self.assertRaises(ValueError, stddev, [3.0])
        self.assertRaises(ValueError, stddev, numpy.array([[1,2],[3,4]]))

    def testAverage(self):
        if self.just1: return

        self.assertEqual(average([3.0]), 3.0)
        self.assertEqual(average([1,2]), 1.5)
        self.assertEqual(average([1,3]), 2.0)

    def testMedian(self):
        if self.just1: return

        #test median directly
        self.assertEqual(median([3.0]), 3.0)
        self.assertEqual(median([3,4]), 3.0)
        self.assertEqual(median([4,3]), 3.0)
        self.assertEqual(median([1,2,4,5,3]), 3)
        self.assertEqual(median([3,1,2,4,5]), 3)
        self.assertEqual(median([3,1,2,4,5,3]), 3)
        self.assertEqual(median([1,2,4,5,3,3]), 3)
        self.assertEqual(median([3,3,1,2,4,5]), 3)
        self.assertEqual(median([3,3,1,2,4,5,1,5]), 3)


    def testRandIndex(self):
        if self.just1: return

        for i in range(50):
            self.assertTrue(randIndex([5.0, 1.0]) in [0,1])
            self.assertTrue(randIndex([5]) in [0])
            self.assertTrue(randIndex([1, 5.0]) in [0,1])
            self.assertTrue(randIndex([1, 5, 1]) in [0,1,2])
            self.assertTrue(randIndex([1, 0, 2]) in [0,2])
            self.assertTrue(randIndex([0]) in [0])
            self.assertTrue(randIndex([0,0]) in [0,1])

        self.assertRaises(ValueError, randIndex, [])
        self.assertRaises(ValueError, randIndex, [1,-1])

    def testNiceValuesStr(self):
        if self.just1: return

        d = {'gain':3.19999999999999999999, 'power':2.0}
        self.assertEqual(niceValuesStr(d), '{gain:3.2,power:2}')

        
        self.assertEqual(niceValuesStr({'gain':BAD_METRIC_VALUE}),
                         '{gain:BAD_METRIC_VALUE}')
        
        #all values in dict must be a number
        self.assertRaises(TypeError, niceValuesStr, {'gain':3.19, 'power':'2.0'})

    def testAllEntriesAreUnique(self):
        if self.just1: return
        self.assertEqual( allEntriesAreUnique( [] ), True )
        self.assertEqual( allEntriesAreUnique( [1,2,3,0] ), True )
        self.assertEqual( allEntriesAreUnique( [1,2,3,1] ), False )
        self.assertEqual( allEntriesAreUnique( [1,2,3,'0'] ), True )
        self.assertEqual( allEntriesAreUnique( [1,2,3,'1'] ), True )
        self.assertEqual( allEntriesAreUnique( ['1','2','3','1'] ), False )

    def testListDiff(self):
        if self.just1: return
        self.assertEqual( listDiff( [], [] ), [] )
        self.assertEqual( listDiff( [1], [3]), [1] )
        self.assertEqual( listDiff( [4,1,2,7,1], [1] ), [4,2,7] )
        self.assertEqual( listDiff( [], range(10) ), [] )

    def testListsOverlap(self):
        if self.just1: return
        self.assertFalse( listsOverlap([],[]) )
        self.assertFalse( listsOverlap([1],[]) )
        self.assertFalse( listsOverlap([],[1]) )
        self.assertFalse( listsOverlap([],[1,1,1]) )
        self.assertFalse( listsOverlap([1],[2,3]) )
        self.assertFalse( listsOverlap([1,4,'a'],[2,3]) )
        self.assertTrue( listsOverlap([1,2],[2,3]) )
        self.assertTrue( listsOverlap([0,0,0,2],[2,3]) )
        self.assertTrue( listsOverlap([0,0,0,2,4],[7,2,2,2,3]) )

    def testIsNumberType(self):
        if self.just1: return
        self.assertTrue( isNumber( -3.0 ) )
        self.assertTrue( isNumber( -3 ) )
        #self.assertTrue( isNumber( long(10) ) ) #turned off the 'long' check for speed reasons

        #we shouldn't be seeing complex, so catch it here for now
        # (if we decide to use it, then we can change isNumber then)
        # self.assertTrue( isNumber( complex(2) ) ) #turned off the 'complex' check for speed reasons

        #don't work for arrays (can change target functionality if we want)
        a = numpy.array([1,2])
        self.assertFalse( isNumber(a) )

        self.assertFalse( isNumber( [] ) )
        self.assertFalse( isNumber( [] ) )
        self.assertFalse( isNumber( 'asd' ) )
        self.assertFalse( isNumber( {} ) )
        
        self.assertFalse( isNumber( types.FloatType ) )
        self.assertFalse( isNumber( types.IntType ) )
        self.assertFalse( isNumber( types.LongType ) )

        nan = float('NaN')
        self.assertTrue(isNan(nan))     #guarantee it's nan
        self.assertFalse(isNumber(nan)) #a nan is not a number

    def testAllEntriesAreNumbers(self):
        if self.just1: return
        self.assertTrue( allEntriesAreNumbers( [] ) )
        self.assertTrue( allEntriesAreNumbers( [1, 3.0, float(30)] ) )
        self.assertTrue( allEntriesAreNumbers( set([1,2] ) ) )
        self.assertTrue( allEntriesAreNumbers( numpy.arange(1,10) ) )

        #don't work for matrices (can change if we decide)
        self.assertFalse( allEntriesAreNumbers( numpy.array([[1,2],[3,4]]) ) )

        self.assertFalse( allEntriesAreNumbers( [1, 2, 'blah']  ) )

    def testPermutations(self):
        if self.just1: return

        self.assertEqual(permutations([]),[[]])

        self.assertEqual(permutations([2,0,3]),
                         [ [0,0,0],
                           [0,0,1],
                           [0,0,2],
                           [1,0,0],
                           [1,0,1],
                           [1,0,2] ] )


    def testBaseIncrement(self):
        if self.just1: return
        self.assertEqual(baseIncrement([0],[]), (None,True))
        self.assertEqual(baseIncrement([0],[0]), (None,True))
        self.assertEqual(baseIncrement([0],[1]), (None,True))
        self.assertEqual(baseIncrement([0],[2]), ([1],False))
        self.assertEqual(baseIncrement([1],[2]), (None,True))

        self.assertEqual(baseIncrement([0,0],[2,2]), ([0,1],False))
        self.assertEqual(baseIncrement([0,1],[2,2]), ([1,0],False))
        self.assertEqual(baseIncrement([1,0],[2,2]), ([1,1],False))
        self.assertEqual(baseIncrement([1,1],[2,2]), (None,True))
        
        self.assertEqual(baseIncrement([0,0,0],[2,1,3]), ([0,0,1],False))
        self.assertEqual(baseIncrement([0,0,2],[2,1,3]), ([1,0,0],False))
        self.assertEqual(baseIncrement([1,0,0],[2,1,3]), ([1,0,1],False))
        self.assertEqual(baseIncrement([1,0,2],[2,1,3]), (None,True))


    def testUniqueStringIndices(self):
        if self.just1: return
        self.assertEqual(uniqueStringIndices([]), [])
        self.assertEqual(uniqueStringIndices(['a']), [0])
        self.assertEqual(uniqueStringIndices(['']), [0])
        self.assertEqual(sorted(uniqueStringIndices(['a','b'])), [0,1])

        self.assertEqual(sorted(uniqueStringIndices(['a','b','b'])),[0,1])
        self.assertEqual(sorted(uniqueStringIndices(['b','a','b'])),[0,1])
        self.assertEqual(sorted(uniqueStringIndices(['b','b','b','a'])),[0,3])
        self.assertEqual(sorted(uniqueStringIndices(['','b','b','a'])),[0,1,3])
        self.assertEqual(sorted(uniqueStringIndices(['','b','b','a','a'])),
                         [0,1,3])

        self.assertRaises(ValueError, uniqueStringIndices, 'not list')
        not_string = 3
        self.assertRaises(ValueError, uniqueStringIndices, ['a',not_string])

        #this has 10 strings; all are unique except last string
        ind_strs = ['{metric_transistorArea:1.01995e-10,metric_numAtomicParts:18}', '{metric_transistorArea:1.34424e-10,metric_numAtomicParts:20}', '{metric_transistorArea:2.0399e-10,metric_numAtomicParts:20}', '{metric_transistorArea:9.54889e-11,metric_numAtomicParts:13}', '{metric_transistorArea:9.06167e-11,metric_numAtomicParts:13}', '{metric_transistorArea:1.15687e-10,metric_numAtomicParts:16}', '{metric_transistorArea:7.42621e-11,metric_numAtomicParts:11}', '{metric_transistorArea:9.30186e-11,metric_numAtomicParts:18}', '{metric_transistorArea:5.12247e-11,metric_numAtomicParts:11}', '{metric_transistorArea:9.54889e-11,metric_numAtomicParts:13}']
        self.assertEqual(sorted(uniqueStringIndices(ind_strs)), range(9))
        
    def testUniquifyVector(self):
        if self.just1: return
        
        self._testUniquifyVector( \
            numpy.array([1,2,3,2,2,1,6,6,5,4,1,2,3,3,3]),
            numpy.array([1,2,3,4,5,6]))
        
        self._testUniquifyVector( \
            numpy.array([1]),
            numpy.array([1]))
        
        self._testUniquifyVector( \
            numpy.array([1.0,1.0,2.0,5,6,7,2.0,1]),
            numpy.array([1.0,2.0,5,6,7]))
        
        self._testUniquifyVector( \
            numpy.array([ 1.,  1.,  4.,  4.,  2.,  2.],'f'),
            numpy.array([ 1., 2., 4.],'f'))
        
    def _testUniquifyVector(self,array,target):
        clean = uniquifyVector(array)
        
        self.assertEqual(clean.shape,target.shape)
        
        clean = sorted(clean) #easier to compare for testing
        self.assertEqual(numpy.sum(clean!=target),0)

    def testPermutations(self):
        if self.just1: return

        perms = permutations([])
        self.assertEqual(perms, [[]])
        
        perms = permutations([3,1])
        self.assertEqual(perms[0], [0,0])
        self.assertEqual(perms[1], [1,0])
        self.assertEqual(perms[2], [2,0])
        
        perms = permutations([3,2])
        self.assertEqual(perms[0], [0,0])
        self.assertEqual(perms[1], [0,1])
        self.assertEqual(perms[2], [1,0])
        self.assertEqual(perms[3], [1,1])
        self.assertEqual(perms[4], [2,0])
        self.assertEqual(perms[5], [2,1])

    def testIntegerValue_binaryValue(self):
        if self.just1: return

        self.assertEqual(integerValue([1,0]), 2)
        
        self.assertEqual(integerValue([0]), 0)
        self.assertEqual(integerValue([1]), 1)
        self.assertEqual(integerValue([0,0]), 0)
        self.assertEqual(integerValue([0,1]), 1)
        self.assertEqual(integerValue([1,0]), 2)
        self.assertEqual(integerValue([1,1]), 3)
        self.assertEqual(integerValue([1,0,0]), 4)

        self.assertEqual(binaryValue(4, 3), [1,0,0])
        self.assertEqual(binaryValue(3, 2), [1,1])
        self.assertEqual(binaryValue(2, 2), [1,0])
        self.assertEqual(binaryValue(1, 2), [0,1])
        self.assertEqual(binaryValue(0, 2), [0,0])
        self.assertEqual(binaryValue(1, 1), [1])
        self.assertEqual(binaryValue(0, 1), [0])
        
        self.assertRaises(AssertionError, binaryValue, 1413, 3)

        for i in range(500):
            v = random.randint(0, 2**4 - 1)
            self.assertEqual(integerValue(binaryValue(v, 5)), v)

    def testAlmostEqual(self):
        if self.just1: return

        dict1 = {'ampmos_fracAmp': 0.56287113656109244, 'Ibias': 0.00023547127659604881, 'Ibias2': 0.0071758298222309065, 'inputcascode_L': 2.4318038742045221e-06, 'inputcascode_Vgs': 0.71233965574777647, 'ampmos_L': 3.783505573957562e-05, 'degen_fracDeg': 0.5655708342978375, 'ampmos_Vgs': 0.5}

        dict2 = {'ampmos_fracAmp': 0.56287113656109244, 'Ibias': 0.00023547127659604878, 'Ibias2': 0.0071758298222309065, 'inputcascode_L': 2.4318038742045221e-06, 'inputcascode_Vgs': 0.71233965574777647, 'ampmos_L': 3.783505573957562e-05, 'degen_fracDeg': 0.5655708342978375, 'ampmos_Vgs': 0.5}

        #typical case
        self.assertTrue(dictsAlmostEqual(dict1, dict2))

        #handle a case where not equal
        dict1['ampmos_fracAmp'] = 1.0
        self.assertFalse(dictsAlmostEqual(dict1, dict2))

        #handle denominator of zero
        # -not equal
        dict2['ampmos_fracAmp'] = 0.0
        self.assertFalse(dictsAlmostEqual(dict1, dict2))

        # -exactly equal
        dict1['ampmos_fracAmp'] = 0.0
        self.assertTrue(dictsAlmostEqual(dict1, dict2))

        # -equal within a tolerance
        dict1['ampmos_fracAmp'] = 1.0e-8
        self.assertTrue(dictsAlmostEqual(dict1, dict2))

        #test precondition
        self.assertRaises(AssertionError, dictsAlmostEqual, {}, dict2)

if __name__ == '__main__':
    unittest.main()
