import logging
import unittest

from adts import *
from engine.RandomPart import *
from problems.Library import Library
from problems.OpLibrary import *
from util import mathutil

log = logging.getLogger('slave')

#make this global for testing so we only have one disk access
DEVICES_SETUP = DevicesSetup('UMC180')

def getOpLibrary():
    ss = OpLibraryStrategy(DEVICES_SETUP)
    return OpLibrary(ss)

class DummyToplevelEmbPart:
    """All this needs to do is return a list of parts that is allowable to use"""
    
    def __init__(self, lib):
        self._lib = lib

    def parts(self):
        return [self._lib.resistor(), self._lib.capacitor()]

class DummyPart:
    def __init__(self):
        self.name = 'dummy'

class MyLibrary(Library):
    def __init__(self, res_part, cap_part):
        Library.__init__(self)
        
        assert res_part.name == 'resistor'
        self._parts['resistor'] = res_part

        assert cap_part.name == 'capacitor'
        self._parts['capacitor'] = cap_part
            
    def resistor(self):
        return self._parts['resistor']

    def capacitor(self):
        return self._parts['capacitor']

    #the following can safely return None because they will not be included
    # if DummyToplevelEmbPart.parts() does not include their names
    def wire(self):
        return DummyPart() 

    def openCircuit(self):
        return DummyPart()

    def mosDiode(self):
        return DummyPart()

    def biasedMos(self):
        return DummyPart()

    def RC_series(self):
        return DummyPart()

class RandomPartTest(unittest.TestCase):

    def setUp(self):
        self.do_profiling = False #to make True is a HACK
        if not self.do_profiling:
            self.just1 = False    #to make True is a HACK
        else:
            self.just1 = True

        #res part
        resistance_varmeta = ContinuousVarMeta(True, 1, 7, 'R')
        self.res_part = AtomicPart(
            'R',
            ['r1', 'r2'],
            PointMeta([resistance_varmeta]),
            name = 'resistor')

        #cap part
        capacitance_varmeta = ContinuousVarMeta(True, -12, -8, 'C')
        self.cap_part = AtomicPart(
            'C',
            ['c1', 'c2'],
            PointMeta([capacitance_varmeta]),
            name = 'capacitor')

        #build library
        self.library = MyLibrary(self.res_part, self.cap_part)        

        #build toplevel_emb_part
        self.toplevel_emb_part = DummyToplevelEmbPart(self.library)

    def testAddTwoPortParallel(self):
        if self.just1: return
        for i in range(10):
            self._testAddTwoPort(do_parallel=True)
        
    def testAddTwoPortSeries(self):
        if self.just1: return
        for i in range(10):
            self._testAddTwoPort(do_parallel=False)
        
    def _testAddTwoPort(self, do_parallel):
        assert do_parallel in [True, False]

        res_part = self.res_part
        cap_part = self.cap_part
        library = self.library
        toplevel_emb_part = self.toplevel_emb_part
        
        res_part_ID = res_part.ID
        cap_part_ID = cap_part.ID
        
        #build op
        worker = MutateWorker(library, toplevel_emb_part)

        #build 'mutated_part'.  Before mutation happens:
        # mutated_part is a wholly new part, with new ID and name (test AA)
        # mutated_part.embedded_parts = [emb__res_part] (B)
        # emb_res_part has res_part                     (C)
        #  -resistor is unchanged
        mutated_part = atomicPartToCompoundPart(res_part)     #AA
        self.assertNotEqual(mutated_part.ID, res_part.ID)     #AA
        self.assertNotEqual(mutated_part.name, 'resistor')    #AA
        self.assertTrue('mutated' in mutated_part.name)       #AA
        
        self.assertEqual(len(mutated_part.embedded_parts), 1)               #B
        emb__res_part = mutated_part.embedded_parts[0]        
        self.assertEqual(emb__res_part.part.ID, res_part.ID)                #B
        self.assertEqual(emb__res_part.functions, {'R':'R'})                #B
        self.assertEqual(emb__res_part.connections, {'r1':'r1', 'r2':'r2'}) #B
        
        self.assertEqual(emb__res_part.part.name, 'resistor')        #C
        self.assertEqual(emb__res_part.part.ID, res_part_ID)         #C
        self.assertEqual(emb__res_part.part.point_meta.keys(), ['R'])#C

        #build 'emb__mutated_part'
        #before mutate:
        # emb__mutated_part has mutated_part (test A)
        # mutated_part.embedded_parts = [emb__res_part] (B)
        # emb_res_part has res_part (C)
        emb__mutated_part = EmbeddedPart(
            mutated_part,
            {'r1':'A', 'r2':'B'},
            {'R':10.0e3}
            )
        self.assertEqual(emb__mutated_part.part.ID, mutated_part.ID) #A
        self.assertEqual(emb__mutated_part.functions, {'R':10.0e3})  #A
        self.assertEqual(emb__mutated_part.connections,
                         {'r1':'A', 'r2':'B'})             #A
        
        #apply addTwoPortParallel op to emb_part
        # -its only option of a part to add is a capacitor, because
        # that's all that library.parts() returns
        tabu_ID = res_part.ID
        if do_parallel:
            op = 'addTwoPortParallel'
        else:
            op = 'addTwoPortSeries'
        changed = worker.inPlaceMutate(emb__mutated_part, tabu_ID, op)

        #after_mutate:
        # emb__mutated_part has mutated_part (test A)
        # mutated_part.embedded_parts = [emb__res_part, emb__cap_part] (B)
        #   -did we alter mutated_part to have a capacitor?
        # emb_res_part has res_part, emb__cap_part has cap_part (C)
        #   -did emb__res_part remain unchanged?
        #   -did res_part remain unchanged?
        #   -did emb__cap_part get set appropriately?
        #   -did cap_part get set appropriately?
        
        emb__res_part = mutated_part.embedded_parts[0]
        emb__cap_part = mutated_part.embedded_parts[1]
        
        self.assertEqual(emb__mutated_part.part.ID, mutated_part.ID)          #A
        self.assertEqual(emb__mutated_part.functions, {'R':10.0e3})           #A
        self.assertEqual(emb__mutated_part.connections, {'r1':'A', 'r2':'B'}) #A

        self.assertEqual(len(emb__mutated_part.part.embedded_parts), 2)  #B

        self.assertEqual(emb__res_part.part.ID, res_part.ID)             #B
        self.assertEqual(emb__res_part.functions, {'R':'R'})             #B
        self.assertEqual(emb__cap_part.part.ID, cap_part.ID)             #B
        self.assertFalse(mathutil.isNumber(emb__cap_part.functions['C'])) #B
        self.assertTrue(isNumberFunc(emb__cap_part.functions['C']))       #B
        if do_parallel:
            self.assertEqual(emb__res_part.connections, {'r1':'r1', 'r2':'r2'})#B
            self.assertTrue(
                (emb__cap_part.connections == {'c1':'r1', 'c2':'r2'}) or \
                (emb__cap_part.connections == {'c2':'r1', 'c1':'r2'}))         #B
        else:
            #auto-detect name for new node 'nn'
            self.assertEqual(len(emb__mutated_part.part.internalNodenames()), 1)
            nn = emb__mutated_part.part.internalNodenames()[0]

            #case: had split apart 'r1' node
            if emb__res_part.connections['r1'] == nn:
                self.assertEqual(
                    emb__res_part.connections, {'r1':nn, 'r2':'r2'})          #B
                self.assertTrue(
                    (emb__cap_part.connections == {'c1':nn, 'c2':'r1'}) or \
                    (emb__cap_part.connections == {'c2':nn, 'c1':'r1'}))      #B
                
            #case: had split apart 'r2' node
            else:
                self.assertEqual(
                    emb__res_part.connections, {'r2':nn, 'r1':'r1'})          #B 
                self.assertTrue(
                    (emb__cap_part.connections == {'c1':nn, 'c2':'r2'}) or \
                    (emb__cap_part.connections == {'c2':nn, 'c1':'r2'}))      #B 
            

        self.assertEqual(emb__res_part.part.name, 'resistor')            #C
        self.assertEqual(emb__res_part.part.ID, res_part_ID)             #C
        self.assertEqual(emb__res_part.part.point_meta.keys(), ['R'])    #C
        self.assertTrue(isinstance(emb__res_part.part, AtomicPart))      #C
        
        self.assertEqual(emb__cap_part.part.name, 'capacitor')           #C
        self.assertEqual(emb__cap_part.part.ID, cap_part_ID)             #C
        self.assertEqual(emb__cap_part.part.point_meta.keys(), ['C'])    #C
        self.assertTrue(isinstance(emb__cap_part.part, AtomicPart))      #C

        
    def testMultipleMutatesOnOpLibrary(self):
        if self.just1: return
        self._doMultipleMutatesOnOpLibrary(20, False)

    def _doMultipleMutatesOnOpLibrary(self, num_reps, print_iter):
        library = getOpLibrary()
        part = library.ssViAmp1_VddGndPorts()
        connections = {'Vin':'ninp', 'Iout':'nout',
                       'Vdd':'ndd', 'gnd':'gnd'}
        functions = {}
        for varname in part.point_meta.keys():
            functions[varname] = None #these need to get set ind-by-ind
        embedded_part = EmbeddedPart(part, connections, functions)

        for i in range(num_reps):
            if print_iter:
                print "%2d / %d" % (i+1, num_reps)
            log.info('----------------------------------------------')
            log.info('----------------------------------------------')
            log.info('Rep #%d / %d: begin' % (i+1, num_reps))
            factory = RandomPartFactory(library)
            new_point = embedded_part.part.point_meta.createRandomUnscaledPoint(
                with_novelty = False)
            (var_name, var_value) = factory.build(embedded_part, new_point)
            
    def test_ProfileMutation(self):
        """Only turn this on for special temporary profiling needs"""
        if not self.do_profiling:
            return
        num_reps = 100

        import cProfile
        filename = "/tmp/randompart.cprof"
        
        print "Begin to gather profile data"
        prof = cProfile.runctx(
            "ret_code = self._doMultipleMutatesOnOpLibrary("
            + str(num_reps) + ", True)",
            globals(), locals(), filename)
        print "Done gathering profile data"

        print "Analyze profile data"
        import pstats
        p = pstats.Stats(filename)
        p.strip_dirs() #remove extraneous path from all module names

        print ""
        print "======================================================="
        print "Sort by cumulative time in a function (and children)"
        p.sort_stats('cum').print_stats(40)
        print ""

        print ""
        print "======================================================="
        print "Sort by time in a function (no recursion)"
        p.sort_stats('time').print_stats(40)

        #This line sorts statistics with a primary key of time, and
        # a secondary key of cumulative time, and then prints out
        # some of the statistics. To be specific, the list is first
        # culled down to 50% (re: ".5") of its original size, then
        # only lines containing init are maintained, and that
        # sub-sub-list is printed.
        #p.sort_stats('time', 'cum').print_stats(.5, 'init')

        

if __name__ == '__main__':
    #if desired, this is where logging would be set up
    
    unittest.main()
