import unittest

from adts import *


A, B, C, D, E, F = 'A', 'B', 'C', 'D', 'E', 'F'
        
class SchemaTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

    def testCheckConsistency(self):
        if self.just1: return

        #the following should _not_ raise a ValueError
        Schema({A: [0, 1], B: [0]}).checkConsistency()
        Schema({A: [0, 3, 2], B: [0]}).checkConsistency()
        Schemas([Schema({A: [0, 3, 2], B: [0]})]).checkConsistency()
        Schemas([Schema({A: [0, 3, 2], B: [0]}),
                 Schema({C: [0,1]})]).checkConsistency()

        #the following _should_ raise a ValueError
        self.assertRaises( ValueError,
                           Schema, {A: [0, 1, 1], B: [0]})

    def testStr(self):
        if self.just1: return
        schema1 = Schema({'foo': [0, 12], 'bar': [0]})
        s = str(schema1)
        self.assertTrue( 'foo' in s )
        self.assertTrue( '12' in s )
        self.assertTrue( 'bar' in s )

        schema2 = Schema({'foo': [0, 12], 'bar': [1]})
        schemas = Schemas([schema1, schema2])
        s = str(schemas)
        self.assertTrue( 'foo' in s )
        self.assertTrue( '12' in s )
        self.assertTrue( 'bar' in s )

        s = schemas.compactStr()
        self.assertTrue( 'foo' in s )
        self.assertTrue( '12' in s )
        self.assertTrue( 'bar' in s )
        self.assertTrue(
            'Schema values shared across all schemas: {foo: [0, 12]}' in s)
        
    def testOverlap(self):
        return #we don't do this yet
        if self.just1: return

        #shouldn't be able to build the following, because
        # some schemas overlap with others (e.g. a=1,b=0)
        self.assertRaises(ValueError, Schemas, [
            Schema({A: [0, 1], B: [0]}),
            Schema({A: [1], C: [0, 1]}),
            ])
        
        self.assertRaises(ValueError, Schemas, [ 
            Schema({A: [0, 1], B: [0]}),
            Schema({A: [1], C: [0, 1]}),
            Schema({B: [0], C: [0, 1]}),
            ])

        #also test when using append
        schemas = Schemas()
        schemas.append({A: [0, 1], B: [0]})
        self.assertRaises(ValueError, schemas.append, {A: [1], C: [0, 1]})

        
    def testMerge1(self):
        if self.just1: return

        #4 schemas, each occupying a corner of the a=(0,1) and b=(0,1) square
        # can merge into one schema that covers the whole square
        schemas = Schemas([ 
            Schema({A: [0], B: [0]}),
            Schema({A: [0], B: [1]}),
            Schema({A: [1], B: [0]}),
            Schema({A: [1], B: [1]}),
            ])
        target = Schemas([ 
            Schema({A: [0,1], B: [0,1]}),            
            ])
        schemas.merge()
        self.assertEqual( schemas, target )
        
    def testMerge2(self):
        if self.just1: return

        #-the first 2 schemas are identical except for A, so that merges.
        #-the last schema should remain independent because it has different
        # variables
        schemas = Schemas([ 
            Schema({A: [0], B: [0,1,2], C: [0, 1], D: [0, 1]}),
            Schema({A: [1], B: [0,1,2], C: [0, 1], D: [0, 1]}),
            Schema({A: [2], 'blah':[1]}),
            ])

        target = Schemas([ 
            Schema({A: [0,1], B: [0,1,2], C: [0, 1], D: [0, 1]}),
            Schema({A: [2], 'blah':[1]}),
            ])
        schemas.merge()
        self.assertEqual( schemas, target )
        
    def testMerge3(self):
        if self.just1: return

        #ensure that we don't get {A: [0,1], B: [0,1,1]}
        schemas = Schemas([ 
            Schema({A: [0,1], B: [0]}),
            Schema({A: [0,1], B: [1]}),
            Schema({A: [0,1], B: [1]}),
            ])
        target = Schemas([ 
            Schema({A: [0,1], B: [0,1]}),            
            ])
        schemas.merge()
        self.assertEqual( schemas, target )
        
    def testMerge4(self):
        if self.just1: return

        #the second schema is a subset of the first
        schemas = Schemas([ 
            Schema({A: [0,1], B: [0,1]}),
            Schema({A: [0,1], B: [1]}),
            ])
        target = Schemas([ 
            Schema({A: [0,1], B: [0,1]}),            
            ])
        schemas.merge()
        self.assertEqual( schemas, target )
        
    def testMerge5(self):
        if self.just1: return

        #variable A doesn't matter, it will disappear during merge
        schemas = Schemas([ 
            Schema({A: [0], B: [0,1]}),
            ])
        target = Schemas([ 
            Schema({B: [0,1]}),            
            ])
        schemas.merge()
        self.assertEqual( schemas, target )
        
        #combine var-deletion with a typical merge
        schemas = Schemas([ 
            Schema({A: [0], B: [0,1]}),
            Schema({A: [0], B: [1]}),
            ])
        target = Schemas([ 
            Schema({B: [0,1]}),            
            ])
        schemas.merge()
        self.assertEqual( schemas, target )
        
        #bigger example (vars A, B, E do not matter)
        schemas = Schemas([ 
            Schema({B: [0], A: [0], D: [0, 1], C: [0],    E: [0]}),
            Schema({B: [0], A: [0], D: [0, 1], C: [0, 1], E: [0]}),
            ])
        target = Schemas([  
            Schema({D: [0, 1], C: [0, 1]}),
            ])
        schemas.merge()
        self.assertEqual( schemas, target )
        
    def testMerge6(self):
        """Test cases taken from inputCascode_Stacked"""
        if self.just1: return

        #pilot merge
        schemas = Schemas([ 
            Schema({B: [0], A: [0], D: [0, 1], C: [0],    E: [0], F: [0]}),
            Schema({B: [0], A: [0], D: [0, 1], C: [0, 1], E: [0]}),
            ])
        target = Schemas([  
            Schema({D: [0, 1], C: [0, 1]}),
            ])
        schemas.merge()
        self.assertEqual( schemas, target )
        
        #pilot merge
        schemas = Schemas([ 
            Schema({B: [0], A: [0], D: [0, 1], C: [0],    E: [0], F: [0]}),
            Schema({B: [0], A: [0], D: [0, 1], C: [1]}),
            ])
        target = Schemas([  
            Schema({D: [0, 1], C: [0, 1]}),
            ])
        schemas.merge()
        self.assertEqual( schemas, target )

        #pilot merge
        schemas = Schemas([ 
            Schema({B: [0], A: [0], D: [0, 1], C: [0],    E: [0], F: [0]}),
            Schema({B: [0], A: [0], D: [0, 1], C: [0, 1], E: [0]}),
            Schema({B: [0], A: [0], D: [0, 1], C: [0, 1], F: [0]}),
            Schema({B: [0], A: [0], D: [0, 1], C: [1]}),
            ])
        target = Schemas([  
            Schema({D: [0, 1], C: [0, 1]}),
            ])
        schemas.merge()
        self.assertEqual( schemas, target )

        #pilot merge
        schemas = Schemas([             
            Schema({B: [0], D: [0, 1], C: [0], F: [0]}),
            Schema({B: [0], D: [0, 1], C: [1]}),
            ])
        target = Schemas([  
            Schema({D: [0, 1], C: [0, 1]}),
            ])
        schemas.merge()
        self.assertEqual( schemas, target )
        
        #big merge -- what inputCascode_Stacked has to do
        schemas = Schemas([ 
            Schema({B: [0], A: [0], D: [0, 1], C: [0],    E: [0], F: [0]}),
            Schema({B: [0], A: [0], D: [0, 1], C: [0, 1], E: [0]}),
            Schema({B: [0], A: [0], D: [0, 1], C: [0, 1], F: [0]}),
            Schema({B: [0], A: [0], D: [0, 1], C: [1]}),
            
            Schema({B: [1], D: [0, 1], C: [0], F: [0]}),
            Schema({B: [1], D: [0, 1], C: [1]}),
            ])
        target = Schemas([  
            Schema({B: [0,1], D: [0, 1], C: [0, 1]}),
            ])
        schemas.merge()
        self.assertEqual(schemas, target)

    def testCoversSpace(self):
        if self.just1: return

        self.assertTrue(Schemas([Schema({})]).coversSpace())
        self.assertTrue(Schemas([Schema({A:[0]})]).coversSpace())
        self.assertTrue(Schemas([Schema({A:[0, 1]})]).coversSpace())
        self.assertTrue(Schemas([Schema({A:[0, 1, 3, 2, 5, 4]})]).coversSpace())
        self.assertTrue(Schemas([Schema({A:[0, 1], B:[0]})]).coversSpace())
        self.assertTrue(Schemas([Schema({A:[1, 0], B:[0]})]).coversSpace())
        self.assertTrue(Schemas([Schema({A:[0, 1], B:[0]}), Schema({B:[1]})]).coversSpace())
        self.assertTrue(Schemas([Schema({A:[0, 1], B:[1]}), Schema({B:[0]})]).coversSpace())
        self.assertTrue(Schemas([Schema({A:[0, 3]}), Schema({A:[1,2]})]).coversSpace())
        
        self.assertFalse(Schemas([Schema({A:[1]})]).coversSpace())
        self.assertFalse(Schemas([Schema({A:[0, 1, 5]})]).coversSpace())
        self.assertFalse(Schemas([Schema({A:[0, 1], B:[1]})]).coversSpace())
        self.assertFalse(Schemas([Schema({A:[0, 1], B:[1]}), Schema({B:[1]})]).coversSpace())
        self.assertFalse(Schemas([Schema({A:[0, 1], B:[2]}), Schema({B:[0]})]).coversSpace())
        self.assertFalse(Schemas([Schema({A:[0, 4]}), Schema({A:[1,2]})]).coversSpace())
        
        
    def testCombine0(self):
        """Test preconditions, etc"""
        if self.just1: return

        self.assertRaises(AssertionError, combineSchemasList, 'foo')      #must be list
        self.assertRaises(AssertionError, combineSchemasList, [])         #must be non-empty list
        self.assertRaises(AssertionError, combineSchemasList, ['foo'])    #must hold Schemas
        self.assertRaises(AssertionError, combineSchemasList, [Schema()]) #must hold Schemas, not Schema
        self.assertRaises(AssertionError, combineSchemasList, [Schemas(), 'foo']) #all entries Schemas?
                          
        schemas1 = Schemas([
            Schema({A : [0,1], B : [0, 1]}),
            ])

        #'schemas2' does not fully cover the variable space (because C == 0 not covered)
        schemas2 = Schemas([
            Schema({C : [1]}),
            ])
        
        self.assertRaises(AssertionError, combineSchemasList, [schemas1, schemas2])
        self.assertRaises(AssertionError, combineSchemasList, [schemas2])
                          
    def testCombine1(self):
        if self.just1: return

        schemas1 = Schemas([
            Schema({A : [0,1], B : [0,1]}),
            ])
        schemas2 = Schemas([
            Schema({C : [0,1]}),
            ])
        target = Schemas([
            Schema({A : [0,1], B : [0,1], C : [0,1]}),
            ])

        #main test
        self.assertEqual(combineSchemasList([schemas1, schemas2]), target)

        #secondary tests
        self.assertEqual(combineTwoSchemas(schemas1, schemas2), target)
        self.assertEqual(combineTwoSchemas(schemas2, schemas1), target)
        
        self.assertEqual(combineSchemasList([schemas1]), schemas1)
        self.assertEqual(combineSchemasList([schemas1, schemas1]), schemas1)
        self.assertEqual(combineSchemasList([schemas1, schemas1, schemas1]), schemas1)
        self.assertEqual(combineSchemasList([schemas2, schemas2]), schemas2)
        
        self.assertEqual(combineSchemasList([schemas1, schemas1, schemas2]), target)
        self.assertEqual(combineSchemasList([schemas2, schemas1, schemas2]), target)
        
        
    def testCombine2(self):
        if self.just1: return

        schemas1 = Schemas([
            Schema({A : [0,1], B : [0], C : [0,1]}),
            Schema({B : [1]}),
            ])
        schemas2 = Schemas([
            Schema({C : [0,1]}),
            ])
        target = Schemas([
            Schema({A : [0,1], B : [0], C : [0,1]}),
            Schema({B : [1], C : [0,1]}),
            ])

        #main test
        self.assertEqual(combineSchemasList([schemas1, schemas2]), target)
        
    def testCombine3(self):
        """Real-world example: how an inputCascode_Stacked has to combine
        its lower-level schemas; ignoring the source_degen schemas

        """
        if self.just1: return

        schemas1 = Schemas([
            Schema({'cascode_recurse' : [0], 'cascode_is_wire' : [0], 'input_is_pmos' : [0,1]}),
            Schema({'cascode_is_wire' : [1]}),
            ])
        self.assertEqual(schemas1.numPermutations(), 3)
        schemas2 = Schemas([
            Schema({'input_is_pmos' : [0,1]}),
            ])
        self.assertEqual(schemas2.numPermutations(), 2)

        #note how variables which only take one value will get deleted during the merge
        target = Schemas([
            Schema({'cascode_is_wire' : [0,1], 'input_is_pmos' : [0,1]}),
            ])

        #main test
        self.assertEqual(combineSchemasList([schemas1, schemas2]), target)
        self.assertEqual(combineSchemasList([schemas1, schemas2]).numPermutations(), 4)
        
    def testCombine4(self):
        """Like testCombine3 except add 'degen_choice' too; therefore making it
        exactly like inputCascode_Stacked.
        """
        if self.just1: return

        schemas1 = Schemas([
            Schema({'cascode_recurse' : [0], 'cascode_is_wire' : [0], 'input_is_pmos' : [0,1]}),
            Schema({'cascode_is_wire' : [1]}),
            ])
        schemas2 = Schemas([
            Schema({'input_is_pmos' : [0,1]}),
            ])
        schemas3 = Schemas([
            Schema({'degen_choice' : [0,1]})
            ])
        
        target = Schemas([
            Schema({'cascode_is_wire' : [0,1], 'degen_choice' : [0,1], 'input_is_pmos' : [0,1] }),
            ])

        #main test
        self.assertEqual(combineSchemasList([schemas1, schemas2, schemas3]), target)
        self.assertEqual(combineSchemasList([schemas1, schemas2, schemas3]).numPermutations(), 8)

    def testNumPermutations(self):
        if self.just1: return

        #simplest possible: 1 schema, 1 variable and 1 choice for that variable
        self.assertEqual(Schemas([Schema({A:[0]})]).numPermutations(), 1)

        #1 schema, 1 variable and 3 choices for that variable
        self.assertEqual(Schemas([Schema({A:[0,1,5]})]).numPermutations(), 3)

        #1 schema, 2 variables, but only 1 variable varies
        s1 = Schema({A:[0],B:[0,1,2]})
        self.assertEqual(Schemas([s1]).numPermutations(), 3)

        #1 schema, 2 variables where both vary, so effect is multiplicative
        s2 = Schema({A:[1,2],B:[3,4,5]})
        self.assertEqual(Schemas([s2]).numPermutations(), 2*3)

        #2 schemas that are wholly independent; add each contribution
        self.assertEqual(Schemas([s1, s2]).numPermutations(), 3 + 2*3)

        #2 schemas, but s3 is a subset of s1 so it has no effect
        s3 = Schema({A:[0],B:[0]})
        ss = Schemas([s1, s3])
        ss.merge()
        self.assertEqual(ss.numPermutations(), 3)

        #schemas with different variable groups
        s4 = Schema({A:[0],C:[0,1,2,3]}) 
        self.assertEqual(Schemas([s1,s4]).numPermutations(), 3+4)
        
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    #if desired, this is where logging would be set up
    
    unittest.main()
