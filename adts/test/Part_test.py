import unittest

from adts import *
from adts.Part import NodeNameFactory, flattenedTupleList, validateFunctions,\
     replaceAutoNodesWithXXX, evalFunction, recursiveEval, functionUsesVar
from problems.OpLibrary import ApproxMosModels, OpLibraryStrategy, OpLibrary
from problems.SizesLibrary import Point18SizesLibrary
from util.constants import *

#make this global for testing so we only have one disk access
DEVICES_SETUP = DevicesSetup('UMC180')
_GLOBAL_approx_mos_models = DEVICES_SETUP.approxMosModels()

def getOpLibrary():
    ss = OpLibraryStrategy(DEVICES_SETUP)
    return OpLibrary(ss)

def getSizesLibrary():
    return Point18SizesLibrary()

class PartTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False #to make True is a HACK

        #a couple atomic parts (res and cap)
        self.resistance_varmeta = ContinuousVarMeta(True, 1, 7, 'R')
        self.res_part = AtomicPart('R', ['node_a', 'node_b'],
                                   PointMeta([self.resistance_varmeta]),
                                   name = 'resistor')
    
        self.cap_part = AtomicPart('C', ['cnode_a', 'cnode_b'],
                                   PointMeta([]),
                                   name = 'capacitor')

        #DOCs
        self.sim_doc1 = SimulationDOC(
            Metric('operating_region', REGION_SATURATION, REGION_SATURATION, False, 0, 1),
            'region',
            REGION_SATURATION,
            REGION_SATURATION)
        self.sim_doc2 = SimulationDOC(
            Metric('v_od', 0.2, 5.0, False, 0, 5),
            'vgs-vt',
            0.0,
            5.0)

        self.min_R = 10**self.resistance_varmeta.min_unscaled_value
        self.func_doc1 = FunctionDOC(Metric('R_metric',self.min_R*2,
                                            100.0, False, 0, 100),'(R)')
        
        self.func_doc2 = FunctionDOC(Metric('R_metric',self.min_R*2,
                                            100.0, False, 0, 100),'(r)')
        
        #a FlexPart
        flex = FlexPart( ['flex_ext1','flex_ext2'],
                         PointMeta([self.resistance_varmeta]),
                         'res_or_cap')
        self.assertEqual(flex.part_choices, [])
        flex.addPartChoice( self.res_part,
                            {'node_a':'flex_ext1', 'node_b':'flex_ext2'},
                            {'R':'R'} )
        flex.addPartChoice( self.cap_part,
                            {'cnode_a':'flex_ext1','cnode_b':'flex_ext2'},
                            {} )
        self.flex_part = flex

        #a CompoundPart
        r1a_varmeta = ContinuousVarMeta(True, 1, 7, 'res1a')
        r1b_varmeta = ContinuousVarMeta(True, 1, 7, 'res1b')
        r2_varmeta = ContinuousVarMeta(True, 1, 7, 'res2')
        pointmeta = PointMeta([r1a_varmeta, r1b_varmeta, r2_varmeta])

        comp_part = CompoundPart(['A', 'B'], pointmeta, 'series_Rs')
        self.assertEqual(comp_part.internalNodenames(), [])
        
        internal_nodename = comp_part.addInternalNode()
        comp_part.addPart( self.res_part,
                           {'node_a':'A','node_b':internal_nodename},
                           {'R':'res1a + res1b'} )
        comp_part.addPart( self.res_part,
                           {'node_a':internal_nodename,'node_b':'B'},
                           {'R':'res2'} )
        self.compound_part = comp_part
        self.compound_internal_nodename = internal_nodename

        
    #====================================================================================
    #Basic tests of AtomicPart, CompoundPart, FlexPart
    def testAtomicPart(self):
        if self.just1: return
        part = self.res_part
        self.assertEqual(part.spice_symbol, 'R')
        self.assertEqual(part.point_meta['R'].railbinUnscaled(10), 7)
        self.assertEqual(part.point_meta['R'].spiceNetlistStr(1e5),'R=%g' % 1e5)
        
        self.assertEqual( part.externalPortnames(), ['node_a', 'node_b'])
        self.assertEqual( part.internalNodenames(), [])
        self.assertEqual( part.embeddedParts({}), [] )
        self.assertEqual( part.portNames(), ['node_a', 'node_b'])
        self.assertEqual( part.unityVarMap(), {'R':'R'} )
        self.assertEqual( part.unityPortMap(),
                          {'node_a':'node_a','node_b':'node_b'} )
        self.assertTrue(len(str(part))>0)
        self.assertTrue(len(part.str2(1))>0)
        
        part.addSimulationDOC(self.sim_doc1)
        part.addSimulationDOC(self.sim_doc2)
        self.assertEqual(part.simulation_DOCs[1].metric.name, 'v_od')
        
        part.addFunctionDOC(self.func_doc1)
        self.assertEqual(part.function_DOCs[0].metric.name, 'R_metric')

    def testFlexPart(self):
        if self.just1: return

        part = self.flex_part

        #test functionality specific to FlexPart
        self.assertEqual(len(part.part_choices), 2)
        self.assertTrue(isinstance(part.part_choices[0], EmbeddedPart))
        self.assertEqual(part.part_choices[1].part.ID, self.cap_part.ID)
        self.assertEqual(sorted(part.point_meta.keys()),
                         sorted(['chosen_part_index', 'R']))
        scaled_point = Point(True, {'chosen_part_index':1, 'R':10})
        self.assertEqual(part.chosenPart(scaled_point).part.ID, self.cap_part.ID)

        #test functionality common to all Part types
        # (though FlexPart has interesting behavior with that, so test it!)
        self.assertEqual( part.externalPortnames(), ['flex_ext1', 'flex_ext2'])
        emb_parts = part.embeddedParts(scaled_point)
        names = part.internalNodenames()
        self.assertEqual(len(emb_parts), 1)
        self.assertEqual(emb_parts[0].part.ID, self.cap_part.ID)
        self.assertEqual(names, [])
        self.assertEqual( part.portNames(), ['flex_ext1', 'flex_ext2'])
        self.assertEqual( part.unityVarMap(),
                          {'R':'R','chosen_part_index':'chosen_part_index'} )
        self.assertEqual( part.unityPortMap(),
                          {'flex_ext1':'flex_ext1','flex_ext2':'flex_ext2'} )
        self.assertTrue(len(str(part))>0)
        self.assertTrue(len(part.str2(1))>0)

        part.addSimulationDOC(self.sim_doc1)
        part.addSimulationDOC(self.sim_doc2)
        self.assertEqual(part.simulation_DOCs[1].metric.name, 'v_od')
        
        part.addFunctionDOC(self.func_doc1)
        self.assertEqual(part.function_DOCs[0].metric.name, 'R_metric')

    def testCompoundPart_and_EmbeddedPart(self):
        if self.just1: return

        part = self.compound_part
        internal_nodename = self.compound_internal_nodename

        #test functionality common to all Part types
        self.assertEqual(part.point_meta['res1a'].railbinUnscaled(10), 7)
        self.assertEqual( part.externalPortnames(), ['A', 'B'])
        self.assertEqual( part.internalNodenames(), [internal_nodename] )
        self.assertTrue( len(part.embeddedParts({})) > 0 )
        self.assertEqual( part.portNames(), ['A', 'B', internal_nodename])
        self.assertEqual( part.unityVarMap(),
                          {'res1a':'res1a', 'res1b':'res1b', 'res2':'res2'} )
        self.assertEqual( part.unityPortMap(), {'A':'A','B':'B'} )
        self.assertTrue(len(str(part))>0)
        self.assertTrue(len(part.str2(1))>0)

        part.addSimulationDOC(self.sim_doc1)
        part.addSimulationDOC(self.sim_doc2)
        self.assertEqual(part.simulation_DOCs[1].metric.name, 'v_od')
        
        part.addFunctionDOC(self.func_doc1)
        self.assertEqual(part.function_DOCs[0].metric.name, 'R_metric')

        #test functionality specific to CompoundPart
        self.assertEqual(len(part.embedded_parts), 2)
        
        emb1 = part.embedded_parts[1]
        self.assertEqual(emb1.part.name, 'resistor')
        self.assertEqual(emb1.connections['node_b'], 'B')
        self.assertEqual(emb1.functions['R'], 'res2')
        self.assertTrue( len(str(emb1)) > 0)

        #test EmbeddedPart, including SPICE netlisting
        bigemb = EmbeddedPart(part,
                              {'A':'a', 'B':'b'},
                              {'res1a':10e3, 'res1b':11e3,
                               'res2':20e3})
        self.assertEqual(bigemb.part.name, 'series_Rs')
        self.assertEqual(bigemb.connections['B'], 'b')
        self.assertEqual(len(bigemb.connections), 2)
        self.assertEqual(bigemb.functions['res1a'], 10e3)
        self.assertEqual(len(bigemb.functions), 3)
        self.assertTrue(len(str(bigemb)) > 0)

        """We want a netlist that looks like the following; the number
        following 'n_auto_' could be any number, however.
        'R0 a n_auto_2  R=21000\n'
        'R1 n_auto_2 b  R=20000\n'
        Our strategy is to replace all 'n_auto_NUM' with 'XXX'.
        """
        netlist = replaceAutoNodesWithXXX(bigemb.spiceNetlistStr())
        self.assertEqual(netlist,
                         'R0 a XXX  R=21000\n'
                         'R1 XXX b  R=20000\n')

        bb_netlist = bigemb.spiceNetlistStr(True)

    #====================================================================================
    #tests of some simple functionality
    def testFlattenedTupleList(self):
        if self.just1: return
        self.assertEqual(flattenedTupleList([]),[])
        self.assertEqual(flattenedTupleList([(1,2),(3,4)]),[1,3,2,4])

    def testNodeNameFactory(self):
        if self.just1: return
        for i in range(500):
            name1 = NodeNameFactory().build()
            name2 = NodeNameFactory().build()
            self.assertNotEqual(name1, name2)
                        
    #====================================================================================
    #Test DOCs (function and simulation)
    def testFunctionDoc(self):
        if self.just1: return

        scaled_point = {'R':self.min_R}

        self.assertFalse(self.func_doc1.resultsAreFeasible(scaled_point))
        
        scaled_point = {'R':self.min_R*5}
        self.assertTrue(self.func_doc1.resultsAreFeasible(scaled_point))

        self.assertEqual(self.func_doc1.metric.name, 'R_metric')
        self.assertEqual(self.func_doc1.function_str, '(R)')


        #
        self.assertRaises(ValueError, FunctionDOC, 'not_metric', '(r)')

        #
        metric_with_objective = Metric('R_metric', 0.0, 100.0, True, 0, 100)
        self.assertRaises(ValueError, FunctionDOC,
                          metric_with_objective, '(r)')

        #
        self.assertRaises(ValueError, FunctionDOC,
                          Metric('R_metric', 0.0, 100.0, False, 0, 100), [])


    def testSimulationDoc(self):
        if self.just1: return

        lis_results = {'lis__m42__region':0,
                       'lis__m12__vgs':5.0,
                       'lis__m12__vt':2.0,
                       'lis__m12__blah':500.0,
                       'lis__blah__vgs':10000,
                       'lis__m14__region':REGION_LINEAR,
                       'lis__m15__region':REGION_SATURATION}

        self.assertEqual(self.sim_doc1.evaluateFunction(lis_results, 'm14'),
                         REGION_LINEAR)
        self.assertFalse(self.sim_doc1.resultsAreFeasible(lis_results, 'm14'))
        
        self.assertEqual(self.sim_doc1.evaluateFunction(lis_results, 'm15'),
                         REGION_SATURATION)
        self.assertTrue(self.sim_doc1.resultsAreFeasible(lis_results, 'm15'))

        self.assertEqual(self.sim_doc2.metric.name, 'v_od')
        self.assertEqual(self.sim_doc2.function_str, 'vgs-vt')
        self.assertEqual(self.sim_doc2.evaluateFunction(lis_results, 'm12'), 5.0-2.0)
        self.assertTrue(self.sim_doc2.resultsAreFeasible(lis_results, 'm12'))
        
        lis_results['lis__m12__vt'] = 4.95
        self.assertFalse(self.sim_doc2.resultsAreFeasible(lis_results, 'm12'))

        #it can handle uppercase or lowercase device_names
        self.assertEqual(self.sim_doc1.evaluateFunction(lis_results, 'M14'),
                         REGION_LINEAR)

        #
        self.assertRaises(ValueError, SimulationDOC, 'not_metric', 'vgs', 0.0, 5.0)

        #
        metric_with_objective = Metric('vgs', 1.0, 2.0, True, 0, 2)
        self.assertRaises(ValueError, SimulationDOC,
                          metric_with_objective, 'vgs-vt', 0.0, 5.0)

        #
        self.assertRaises(ValueError, SimulationDOC,
                          Metric('vgs', 1.0, 2.0, False, 0, 2), [], 0.0, 5.0)

        #
        self.assertRaises(ValueError, SimulationDOC,
                          Metric('vgs', 1.0, 2.0, False, 0, 2), 'VGS', 0.0, 5.0)

    #====================================================================================
    #test function/var relations: evalFunction, switchAndEval, recursiveEval,
    # isNumberFunc, functionUsesVar, varsGoverningChildVar, varsUsed, choiceVarsUsed, subPartsInfo
    def testSwitchAndEval(self):
        if self.just1: return
        case2result = {3:'4.2', 'yo':'7+2', 'p':'1/0', 'default':'400/9'}
        self.assertEqual(switchAndEval(3, case2result), 4.2)
        self.assertEqual(switchAndEval('yo', case2result), 7+2)
        self.assertRaises(ZeroDivisionError, switchAndEval, 'p', case2result)
        self.assertEqual(switchAndEval('blah', case2result), 400/9)
        self.assertEqual(switchAndEval('default', case2result), 400/9)
        
        no_default = {3:'4.2', 'yo':'7+2', 'p':'1/0'}
        self.assertRaises(KeyError, switchAndEval, 'blah', no_default)
        
    def testValidateFunctions(self):
        if self.just1: return
        unscaled_point = Point(False, {'W':10,'L':2})
        self.assertRaises(ValueError, validateFunctions,
                          {'x':'W/L'}, unscaled_point)
        
        scaled_point = Point(False, {'W':10,'L':2})
        self.assertRaises(ValueError, validateFunctions,
                          {'x':'blah W/L'}, scaled_point)
        
    def testEvalFunction(self):
        if self.just1: return 
        
        self.assertEqual(evalFunction({'W':10,'L':2}, 'W / L'), 5)

        #make sure that we don't accidentally substitite 'var'
        # into 'bigvarname'
        self.assertEqual(evalFunction({'var':10,'bigvarname':2},
                                      'var / bigvarname'), 5)

        #make sure that the fix for this doesn't just wrap () around
        # alphanumeric characteris
        self.assertEqual(evalFunction({'var':10,'bigvarname':2},
                                      'max(var, bigvarname)'), 10)

        #more tests
        p = {'x':1,'xx':10,'xxx':100,'xx_xx':1000,'_':12}
        self.assertEqual(evalFunction(p,'x+xx+xxx*2+xx_xx'), 1211)
        self.assertEqual(evalFunction(p,' x+xx+xxx*2+xx_xx'), 1211)
        self.assertEqual(evalFunction(p,'x+xx+xxx*2+xx_xx '), 1211)
        self.assertEqual(evalFunction(p,'x+    xx+xxx  *2+xx_xx '), 1211)
        self.assertEqual(evalFunction(p,'x+xx + xxx*2+xx_xx '), 1211)
        self.assertEqual(evalFunction(p,'xx_xx * x + xx_xx '), 2000)
        self.assertEqual(evalFunction(p,' _'), 12) #yes it supports just '_'
        self.assertEqual(evalFunction(p,'3'), 3)
        self.assertEqual(evalFunction(p,' 32'), 32)
        self.assertEqual(evalFunction(p,'32 '), 32)
        self.assertAlmostEqual(evalFunction(p,'3.2e5'), 3.2e5)
        self.assertAlmostEqual(evalFunction(p,'3.2e5 + 1e5'), 4.2e5)

        #'raise' tests
        self.assertRaises(ValueError, evalFunction, {'W':10,'L':2}, 'bad W / L')
        
        #special case: return a '' if the function is ''
        self.assertEqual(evalFunction({'W':10,'L':2}, ''), '')

    def testEvalInternalFunctions(self):
        if self.just1: return
        point={}
        
        self.assertAlmostEqual(recursiveEval(point,'0'), 0)
        self.assertAlmostEqual(recursiveEval(point,'0+1.2'), 1.2)
        self.assertAlmostEqual(recursiveEval(point,'<$0+1.2$>'), 1.2)
        self.assertAlmostEqual(recursiveEval(point,'<$1$>+<$1.2$>'), 2.2)
        self.assertAlmostEqual(recursiveEval(point,'<$1+1$> +<$1.2$>'), 3.2)
        self.assertAlmostEqual(recursiveEval(point,'<$ 2 + 5 $>+ <$1.2$>'), 8.2)
        self.assertAlmostEqual(recursiveEval(point,'<$6/3$> + <$1.2*2$>'), 4.4)
        self.assertAlmostEqual(recursiveEval(point,' <$0$>+<$1.2$> '), 1.2)
        self.assertAlmostEqual(recursiveEval(point,' <$0$>+<$1.2$> '), 1.2)
        
        test_string='switchAndEval((0!=1),{0:1, 1:2})'
        self.assertEqual(recursiveEval(point,test_string), 2)
        test_string='switchAndEval((0==1),{0:1, 1:2})'
        self.assertEqual(recursiveEval(point,test_string), 1)
        test_string='switchAndEval(<$(1!=1)$>,{0:1, 1:2})'
        self.assertEqual(recursiveEval(point,test_string), 1)
        test_string='switchAndEval(<$(1!=1)$>,{0:<$1*2$>, 1:<$2*2$>})'
        self.assertEqual(recursiveEval(point,test_string), 2)
        test_string='switchAndEval(<$(1==1)$>,{0:<$1*2$>, 1:<$2*2$>})'
        self.assertEqual(recursiveEval(point,test_string), 4)

        test_string='<$<$2$>$>'
        self.assertEqual(recursiveEval(point,test_string), 2)
        test_string='<$ <$2$> $>'
        self.assertEqual(recursiveEval(point,test_string), 2)
        test_string='<$1+<$2$>$>'
        self.assertEqual(recursiveEval(point,test_string), 3)
        test_string='<$1+<$2$>+<$ 1 + <$ 3  $> $>$>'
        self.assertEqual(recursiveEval(point,test_string), 7)
        
        # interesting corner cases
        test_string='<$1<$2+3$>$>'
        self.assertEqual(recursiveEval(point,test_string), 15)
        test_string='<$1+<$2$><$3$>$>'
        self.assertEqual(recursiveEval(point,test_string), 24)
        test_string='<$1+<$2$><$3$>$>'
        self.assertEqual(recursiveEval(point,test_string), 24)
        
        test_string='''(switchAndEval((0!=1), {1:<$(0.0 + 0.9 * (1 - 2 * (0==1)))$>, 0:<$(0.0 + 0.9 * (1 - 2 * (1==1)))$>}) - switchAndEval((0!=0), {1:<$(0.0 + (switchAndEval((0!=1), {1:<$(0.0 + 0.9 * 0.1 * (1 - 2 * (0==1)))$>, 0:<$(0.0 + 0.9 * (1 - 2 * (0==1)))$>})-0.0) * 0.4)$>, 0:<$0.0$>}))*(1 - 2 * (0==1))'''
        self.assertAlmostEqual(recursiveEval(point,test_string), 0.9)

    def testIsNumberFunc(self):
        if self.just1: return

        self.assertFalse(None)
        self.assertFalse('')
        self.assertTrue(isNumberFunc('401'))
        self.assertTrue(isNumberFunc('  401  '))
        self.assertTrue(isNumberFunc('(401)'))
        self.assertTrue(isNumberFunc('  (  401 ) '))
        self.assertTrue(isNumberFunc(' ((( (  401 ) )))'))
        self.assertTrue(isNumberFunc('12.34'))
        self.assertTrue(isNumberFunc('+1'))
        self.assertTrue(isNumberFunc('-31.2'))
        self.assertTrue(isNumberFunc('-12.34e-32'))
        self.assertTrue(isNumberFunc('.34e+32'))
        self.assertTrue(isNumberFunc('+.48e4'))
        self.assertTrue(isNumberFunc('0+e1'))
        self.assertFalse(isNumberFunc('hello'))
        self.assertFalse(isNumberFunc(' hello '))
        self.assertFalse(isNumberFunc('e'))
        self.assertFalse(isNumberFunc('ee'))
        self.assertFalse(isNumberFunc('eee'))
        self.assertFalse(isNumberFunc('e1'))
        self.assertFalse(isNumberFunc('e11'))
        self.assertFalse(isNumberFunc('ee1'))
        self.assertFalse(isNumberFunc('eee1'))
        self.assertFalse(isNumberFunc('+e1'))
        self.assertFalse(isNumberFunc('a+e1'))
        self.assertFalse(isNumberFunc('a+e1'))
        
        self.assertTrue(isNumberFunc(' 401'))
        self.assertTrue(isNumberFunc('12.34 '))
        self.assertTrue(isNumberFunc(' +1 '))

        self.assertTrue(isNumberFunc('12e3 + .12e-3'))
        self.assertTrue(isNumberFunc('12e3 * .12e-3'))
        
    def testFunctionUsesVar(self):
        if self.just1: return

        self.assertRaises(AssertionError, functionUsesVar, '', 'x1+x2', ['x1','x2'])
        self.assertRaises(AssertionError, functionUsesVar, 'x1', '', ['x1','x2'])
        self.assertRaises(AssertionError, functionUsesVar, 'x1', 'x1+x2', [])
        self.assertRaises(AssertionError, functionUsesVar, 'x1', 'x1+x2', ['x3'])
        self.assertRaises(AssertionError, functionUsesVar, 'x', 'x1+x2', ['x1','x2'])
        
        self.assertTrue( functionUsesVar('x1', 'x1', ['x1','x2','x3']))
        self.assertTrue( functionUsesVar('x1', 'x1+x3', ['x1','x2','x3']))
        self.assertTrue( functionUsesVar('x1', '(x1)', ['x1','x2','x3']))
        self.assertTrue( functionUsesVar('x1', '((x1))', ['x1','x2','x3']))
        self.assertTrue( functionUsesVar('x1', '1/x1', ['x1','x2','x3']))
        self.assertTrue( functionUsesVar('x1', '1+(x1)', ['x1','x2','x3']))
        self.assertTrue( functionUsesVar('x1', '(((1+(x1))))', ['x1','x2','x3']))
        self.assertFalse(functionUsesVar('x1', 'x2+x3', ['x1','x2','x3']))
        self.assertFalse(functionUsesVar('x1', 'x2+x3', ['x1','x2','x3']))
        self.assertFalse(functionUsesVar('x1', 'x11', ['x1','x2','x3','x11']))
        self.assertFalse(functionUsesVar('x1', 'x2+x11', ['x1','x2','x3','x11']))
        self.assertFalse(functionUsesVar('x11', 'x2+x1', ['x1','x2','x3','x11']))
        self.assertFalse(functionUsesVar('i', 'sin(x)', ['i','x']))
        
    def testVarsGoverningChildVar(self):
        if self.just1: return

        for (R_expr, target_list) in [('r1', ['r1']), 
                                      ('r2', ['r2']), 
                                      ('     r1 ', ['r1']), 
                                      ('     ( r1 ) ', ['r1']), 
                                      ('  ((   ( r1 ) )) ', ['r1']), 
                                      (' 3.2', []), 
                                      (' (   3.2 ) ', []), 
                                      ('1-r1', ['r1']), 
                                      ('(1-r1) ', ['r1']), 
                                      ('(((1-r1))) ', ['r1']), 
                                      (' (  1  - r1   )  ', ['r1']), 
                                      ('r1==r2', ['r1','r2']), 
                                      ('(r1==r2)', ['r1','r2']), 
                                      ('(((r1==r2)))', ['r1','r2']), 
                                      ('(r1==r1)', ['r1']), 
                                      (' (  r1  ==   r2 ) ', ['r1','r2']), 
                                      ('r1<r2', ['r1','r2']), 
                                      ('r1<=r2', ['r1','r2']), 
                                      ('r1>r2', ['r1','r2']), 
                                      ('r1>=r2', ['r1','r2']),
                                      ('  ( ( (  r1>=r2   )))  ', ['r1','r2']),  
                                      ('r1+r2', ['r1','r2']), 
                                      (' r1 + r2 ', ['r1','r2']), 
                                      (' (  r1 + r2   )', ['r1','r2']), 
                                      (' (( (  r1 + r2   )) )  ', ['r1','r2']), 
                                      ('r1+r1', ['r1']), 
                                      (' r1 - r2 ', ['r1','r2']), 
                                      (' r1 * r2 ', ['r1','r2']), 
                                      (' r1 / r2 ', ['r1','r2']),
                                      ]:
            self._testVarsGoverningChildVar(R_expr, target_list)
        
    def _testVarsGoverningChildVar(self, R_expr, target_list):
        """Helper function for testVarsGoverningChildVar_*()"""
        pointmeta = PointMeta([ContinuousVarMeta(True, 1, 7, 'r1'),
                               ContinuousVarMeta(True, 1, 7, 'r2')])

        #comp_part holds a single resistor.  comp_part's variables r1 and r2 map into the
        # resistor's value for R.
        comp_part = CompoundPart(['A', 'B'], pointmeta, 'embedded_R')
        comp_part.addPart( self.res_part, {'node_a':'A','node_b':'B'}, {'R' : R_expr} )
        emb_comp_part = EmbeddedPart(comp_part, {'A':'A','B':'B'}, {'r1':'r1','r2':'r2'})

        #check mapping from emb_comp_part => comp_part
        self.assertEqual(emb_comp_part.varsGoverningChildVar(['r1','r2'],'r1',True), ['r1'])
        self.assertEqual(emb_comp_part.varsGoverningChildVar(['r1','r2'],'r2',True), ['r2'])

        #check mapping from comp_part's child => self.res_part
        self.assertEqual(sorted(comp_part.embedded_parts[0].varsGoverningChildVar(['r1','r2'],'R',True)),
                         target_list)
        
    def testVarsUsed_nmos4(self):
        """A test on an Atomic part"""
        if self.just1: return 
        self._testVarsUsed(getSizesLibrary().nmos4(), ['L','W','M'])

    def testVarsUsed_sphere2dPart(self):
        """A test on an Atomic part that we use for special sphere2d test"""
        if self.just1: return 
        self._testVarsUsed(getSizesLibrary().sphere2dPart(), ['x0', 'x1'])

    def testVarsUsed_mos4(self):
        """A test on a Flex part"""
        if self.just1: return 
        self._testVarsUsed(getSizesLibrary().mos4(), ['L','W','chosen_part_index'])

    def _testVarsUsed(self, part, expected_vars_used):
        """Helper to testVarsUsed_*"""
        connections = part.unityPortMap()
        scaled_point = part.point_meta.createRandomScaledPoint(False)
        functions = scaled_point
        emb_part = EmbeddedPart(part, connections, functions)

        vars_used = emb_part.varsUsed(scaled_point)
        self.assertEqual(sorted(vars_used), sorted(expected_vars_used))

    def testChoiceVarsUsed_nmos4(self):
        if self.just1: return 
        self._testChoiceVarsUsed(getOpLibrary().nmos4(), [])
        
    def testChoiceVarsUsed_mos4(self):
        if self.just1: return
        self._testChoiceVarsUsed(getOpLibrary().mos4(), ['chosen_part_index'])
        
    def testChoiceVarsUsed_mos3(self):
        if self.just1: return 
        self._testChoiceVarsUsed(getOpLibrary().mos3(), ['use_pmos'])
        
    def testChoiceVarsUsed_biasedMos(self):
        if self.just1: return 
        self._testChoiceVarsUsed(getOpLibrary().biasedMos(), ['use_pmos'])
        
    def testChoiceVarsUsed_cascodeDevice(self):
        if self.just1: return

        self._testChoiceVarsUsed(getOpLibrary().cascodeDevice(), ['chosen_part_index','loadrail_is_vdd'])

        #test that different combinations of choices don't affect things
        self._testChoiceVarsUsed(getOpLibrary().cascodeDevice(), ['chosen_part_index','loadrail_is_vdd'],
                                 {'loadrail_is_vdd':0})
        self._testChoiceVarsUsed(getOpLibrary().cascodeDevice(), ['chosen_part_index','loadrail_is_vdd'],
                                 {'loadrail_is_vdd':1})
        
    def testChoiceVarsUsed_wire(self):
        if self.just1: return
        self._testChoiceVarsUsed(getOpLibrary().wire(), [])
        
    def testChoiceVarsUsed_cascodeDeviceOrWire_0(self):
        if self.just1: return

        #'chosen_part_index' == 0 means chose cascodeDevice
        p = {'chosen_part_index' : 0} 
        self._testChoiceVarsUsed(getOpLibrary().cascodeDeviceOrWire(),
                                 ['chosen_part_index','cascode_recurse', 'loadrail_is_vdd'], p)
        
    def testChoiceVarsUsed_cascodeDeviceOrWire_1(self): 
        if self.just1: return
        
        #'chosen_part_index' == 1 means chose wire
        #-therefore only 'chosen_part_index' should show up in varsUsed()
        #-'cascode_recurse' and 'loadrail_is_vdd' should not have impact
        p = {'chosen_part_index' : 1} 
        self._testChoiceVarsUsed(getOpLibrary().cascodeDeviceOrWire(), ['chosen_part_index'], p)
        
    def testChoiceVarsUsed_inputCascode_Stacked_0(self):
        if self.just1: return
        
        #choose a cascode device, not wire
        #-therefore all the available choice vars should show up in varsUsed()
        p = {'cascode_is_wire' : 0} 
        self._testChoiceVarsUsed(getOpLibrary().inputCascode_Stacked(),
                                 ['cascode_is_wire','cascode_recurse','degen_choice','input_is_pmos'], p)
        
    def testChoiceVarsUsed_inputCascode_Stacked_1(self): #BREAKING; has 'cascode_recurse' and shouldn't
        if self.just1: return
        
        #choose a wire, not cascode device
        #-therefore 'cascode_recurse' should not show up in varsUsed(); other choice vars should
        p = {'cascode_is_wire' : 1} 
        self._testChoiceVarsUsed(getOpLibrary().inputCascode_Stacked(),
                                 ['cascode_is_wire','degen_choice','input_is_pmos'], p)
        
    def _testChoiceVarsUsed(self, part, expected_vars, sub_point=None):
        """Helper to testChoiceVarsUsed_*"""
        connections = part.unityPortMap()
        scaled_point = part.point_meta.createRandomScaledPoint(False)
        if sub_point is not None:
            scaled_point.update(sub_point)
        functions = scaled_point
        emb_part = EmbeddedPart(part, connections, functions)

        vars_used = emb_part.varsUsed(scaled_point)
        choice_vars = part.point_meta.choiceVars()
        choice_vars_used = set(vars_used) & set(choice_vars)
        self.assertEqual(sorted(choice_vars_used), sorted(expected_vars))

    #====================================================================================
    #test counting parts using schemas
    def testPartCountingWithSchemas1(self):
        if self.just1: return

        #test the parts that we use in other unit tests
        self.assertEqual(self.res_part.schemas(), Schemas([Schema({})]))
        self.assertEqual(self.res_part.numSubpartPermutations(), 1)
        self.assertEqual(self.cap_part.schemas(), Schemas([Schema({})]))
        self.assertEqual(self.cap_part.numSubpartPermutations(), 1)
        self.assertEqual(self.flex_part.schemas(),
                         Schemas([Schema({'chosen_part_index':[0,1]})]))
        self.assertEqual(self.flex_part.numSubpartPermutations(), 2)
        self.assertEqual(self.compound_part.schemas(),
                         Schemas([Schema({})]))
        self.assertEqual(self.compound_part.numSubpartPermutations(), 1)
        
    def testPartCountingWithSchemas2(self):
        if self.just1: return

        #A, B, C, D
        A = AtomicPart('A', [], PointMeta([]), name = 'A')
        B = AtomicPart('B', [], PointMeta([]), name = 'B')
        C = AtomicPart('C', [], PointMeta([]), name = 'C')
        D = AtomicPart('D', [], PointMeta([]), name = 'D')
        
        #AB = A_or_B
        pm = PointMeta([])
        AB = FlexPart([], pm, 'AB', 'choice1')
        AB.addPartChoice(A, {}, {})
        AB.addPartChoice(B, {}, {})
        
        #BCD = B_or_C_or_D
        pm = PointMeta([])
        BCD = FlexPart([], pm, 'BCD', 'choice1')
        BCD.addPartChoice(B, {}, {})
        BCD.addPartChoice(C, {}, {})
        BCD.addPartChoice(D, {}, {})

        #
        pm = PointMeta([DiscreteVarMeta([0,1],'choiceAB'),
                        DiscreteVarMeta([0,1,2],'choiceBCD'),
                        ])
        comp = CompoundPart([], pm, 'comp')
        comp.addPart(AB, {}, {'choice1':'choiceAB'})
        comp.addPart(BCD, {}, {'choice1':'choiceBCD'})

        #ideally this will accidentally kill one of the two choices
        x = comp.schemas()
        
    def testPartCountingWithSchemas3(self):
        if self.just1: return

        #test using parts we create here; focus is on multiple levels of
        # hierarchy, without extra variables or external nodes to clutter it up
        #-but note that FlexParts add the 'chosen_part_index' variable,
        # which is, of course, the key to counting

        #nmos4
        nmos4 = AtomicPart('M', [], PointMeta([]), name = 'nmos4')
        self.assertEqual(nmos4.schemas(), Schemas([Schema({})]))
        self.assertEqual(nmos4.numSubpartPermutations(), 1)

        #pmos4
        pmos4 = AtomicPart('M', [], PointMeta([]), name = 'pmos4')
        self.assertEqual(pmos4.schemas(), Schemas([Schema({})]))
        self.assertEqual(pmos4.numSubpartPermutations(), 1)

        #mos4 = nmos4 or pmos4
        mos4 = FlexPart([], PointMeta([]), 'mos4')
        mos4.addPartChoice(nmos4, {}, {})
        mos4.addPartChoice(pmos4, {}, {})
        self.assertEqual(mos4.schemas(),
                         Schemas([Schema({'chosen_part_index':[0,1]})]))
        self.assertEqual(mos4.numSubpartPermutations(), 2)

        #mos3 = embeds mos4
        pm_mos3 = PointMeta([DiscreteVarMeta([0,1],'is_pmos')])
        mos3 = CompoundPart([], pm_mos3, 'mos3') 
        mos3.addPart(mos4, {}, {'chosen_part_index':'is_pmos'})
        self.assertEqual(mos3.schemas(),
                         Schemas([Schema({'is_pmos':[0,1]})]))
        self.assertEqual(mos3.numSubpartPermutations(), 2)

        #small_cm = 2 mos3's and 2 choice vars
        pm_small_cm = PointMeta([DiscreteVarMeta([0,1],'is_pmos1'),
                                 DiscreteVarMeta([0,1],'is_pmos2')])
        small_cm = CompoundPart([], pm_small_cm, 'small_cm')
        small_cm.addPart(mos3, {}, {'is_pmos':'is_pmos1'} )
        small_cm.addPart(mos3, {}, {'is_pmos':'is_pmos2'} )
        self.assertEqual(small_cm.schemas(),
                         Schemas([Schema({'is_pmos1':[0,1],'is_pmos2':[0,1]})]))
        self.assertEqual(small_cm.numSubpartPermutations(), 4) # 2*2=4

        #big_cm = 4 mos3's and 3 choice vars (not 4)
        pm_big_cm = PointMeta([DiscreteVarMeta([0,1],'is_pmos2'),
                               DiscreteVarMeta([0,1],'is_pmos3'),
                               DiscreteVarMeta([0,1],'is_pmos4')])
        big_cm = CompoundPart([], pm_big_cm, 'big_cm')
        big_cm.addPart(mos3, {}, {'is_pmos':'is_pmos2'} )
        big_cm.addPart(mos3, {}, {'is_pmos':'is_pmos3'} )
        big_cm.addPart(mos3, {}, {'is_pmos':'1-is_pmos3'} )
        big_cm.addPart(mos3, {}, {'is_pmos':'1-is_pmos4'} )
        s1 = Schema({'is_pmos2':[0,1],'is_pmos3':[0,1],'is_pmos4':[0,1]})
        self.assertEqual(big_cm.schemas(), Schemas([s1]))
        self.assertEqual(big_cm.numSubpartPermutations(), 8) # 2*2*2=8

        #cm = small_cm or big_cm
        #'cm' is the most complicated of all the parts so far.  Some
        # of its variables affect small_cm, and a different subset affects
        # big_cm.  Therefore there are _two_ schemas, each with a different
        # set of variables
        pm_cm = PointMeta([DiscreteVarMeta([0,1],'cm_var1'),
                           DiscreteVarMeta([0,1],'cm_var2'),
                           DiscreteVarMeta([0,1],'cm_var3'),
                           DiscreteVarMeta([0,1],'cm_var4')])
        cm = FlexPart([], pm_cm, 'cm')
        cm.addPartChoice(small_cm, {}, {'is_pmos1':'cm_var1',
                                        'is_pmos2':'cm_var2'})
        cm.addPartChoice(big_cm, {},   {'is_pmos2':'cm_var2',
                                        'is_pmos3':'cm_var3',
                                        'is_pmos4':'cm_var4'})
        self.assertEqual(sorted(cm.point_meta.keys()),
                         sorted(['chosen_part_index','cm_var1','cm_var2',
                                 'cm_var3','cm_var4']))
        s1 = Schema({'chosen_part_index':[0],'cm_var1':[0,1],'cm_var2':[0,1]})
        s2 = Schema({'chosen_part_index':[1],'cm_var2':[0,1],'cm_var3':[0,1],
                     'cm_var4':[0,1]})
        self.assertEqual(cm.schemas(), Schemas([s1,s2]))
        self.assertEqual(cm.numSubpartPermutations(), 12) # 4+8=12
#         print cm.schemas()

#         for emb_part in cm.part_choices:
#             print "schemasWithVarRemap for embedded_part: %s"%emb_part.part.name
#             print cm.schemasWithVarRemap(emb_part)

        #double_cm = two cm's
        #tests if can we handle the merging of two Schemas objects, in which
        # each object has >1 Schema
        pm_double_cm = PointMeta([DiscreteVarMeta([0,1],'chosen_part_index'),
                                  DiscreteVarMeta([0,1],'cm_var2'),
                                  DiscreteVarMeta([0,1],'cm_var3'),
                                  DiscreteVarMeta([0,1],'cm_var4')])
        

    #====================================================================================
    #begin tests of topology counting
    def testNumtop_biasedMos(self):
        if self.just1: return
        self._testCount('biasedMos', 2)

    def testNumtop_cascodeDevice(self):
        if self.just1: return
        self._testCount('cascodeDevice', 2)
        
    def testNumtop_cascodeDeviceOrWire(self):
        if self.just1: return
        self._testCount('cascodeDeviceOrWire', 1+2)
        
    def testNumtop_saturatedMos3(self):
        if self.just1: return
        self._testCount('saturatedMos3', 2)
        
    def testNumtop_sourceDegen(self):
        if self.just1: return
        self._testCount('sourceDegen', 2)
        
    def testNumtop_inputCascode_Stacked(self):
        if self.just1: return
        self._testCount('inputCascode_Stacked', 1*2*2 + 1*2*2) #==8

    def _testCount(self, name, expected_count):
        library = getOpLibrary()
        part = eval('library.' + name + '()')
        schemas = part.schemas()
        self.assertEqual(part.numSubpartPermutations(), expected_count)
        
    #====================================================================================
    #begin tests of Wrap 
    def testWrap_Nmos4_Sized(self):
        if self.just1: return 
        self._testWrap('nmos4_sized')
        
    def testWrap_nmos4(self):
        if self.just1: return 
        self._testWrap('nmos4')
        
    def testWrap_mos4(self):
        if self.just1: return 
        self._testWrap('mos4')
        
    def testWrap_mos3(self):
        if self.just1: return 
        self._testWrap('mos3')
        
    def testWrap_saturatedMos3(self):
        if self.just1: return  
        self._testWrap('saturatedMos3')

    def testWrap_inputCascode_Stacked(self):
        if self.just1: return  
        self._testWrap('inputCascode_Stacked')

    def testWrap_inputCascodeFlex(self):
        if self.just1: return  
        self._testWrap('inputCascodeFlex')

    def testWrap_ssViAmp1(self):
        if self.just1: return 
        self._testWrap('ssViAmp1')
                
    def _testWrap(self, name):
        """Breaks if the # topologies before and after a wrap is different.
        """
        library = getOpLibrary()
        
        child_part = eval('library.' + name + '()')

        #instantiate all parts in library up to level of the child_part below
        if name == 'nmos4_sized':             parent_part = library.nmos4()
        elif name == 'nmos4':                 parent_part = library.mos4()
        elif name == 'mos4':                  parent_part = library.mos3()
        elif name == 'mos3':                  parent_part = library.saturatedMos3()
        elif name == 'saturatedMos3':         parent_part = library.mosDiode()
        elif name == 'inputCascode_Stacked':  parent_part = library.inputCascodeFlex()
        elif name == 'inputCascodeFlex':      parent_part = library.inputCascodeStage()
        elif name == 'ssViAmp1':              parent_part = library.ssViAmp1_VddGndPorts()
        else: raise NotImplementedError(name)

        #find the embedded_part on parent_part that points to the child_part
        embedded_part = None
        for embedded_part in parent_part.possibleEmbeddedParts():
            if embedded_part.part.ID == child_part.ID:
                break
        assert embedded_part is not None, \
               "must have 'child_part' as an embedded part in parent_part for these tests"

        self._validateStability(parent_part)
        num_top_before = embedded_part.part.numSubpartPermutations()
        
        library.wrapEachNonFlexPartWithFlexPart()
        
        self._validateStability(parent_part)
        num_top_after = embedded_part.part.numSubpartPermutations()
        
        self.assertEqual(num_top_before, num_top_after,
                         "# topologies before wrap = %d; "
                         "# topologies after = %d; child_part = '%s'" %
                         (num_top_before, num_top_after, child_part.name))
        
    def _validateStability(self, part):
        """Randomly set all the values for each point in part, and make sure it doesn't break."""
        num_loops = 10
        for i in range(num_loops):
            connections = part.unityPortMap()
            scaled_point = part.point_meta.createRandomScaledPoint(False)
            functions = scaled_point
            emb_part = EmbeddedPart(part, connections, functions)

            #the test: see if the following calls break things
            emb_part.subPartsInfo(scaled_point)
            emb_part.spiceNetlistStr()
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    # If desired, this is where logging would be set up
    
    unittest.main()
