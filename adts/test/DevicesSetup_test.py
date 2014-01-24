import cPickle as pickle
import os
import unittest

from adts import *
import util.constants as constants

def some_function(x):
    return x+2

def function2(x):
    return x+2

def dummyPart():
    return AtomicPart('R',['1','2'],PointMeta({}))

class DevicesSetupTest(unittest.TestCase):

    def setUp(self):
        self.just1 = False  #to make True is a HACK

        self.testfile = 'devices_test.db'
        
        #possible cleanup from prev run
        if os.path.exists(self.testfile):
            os.remove(self.testfile)

    def testPicklingRobust(self):
        if self.just1: return

        #build ds
        ds = DevicesSetup('UMC180')
        ds.makeRobust()

        #pickle 
        pickle.dump(ds, open(self.testfile,'w'))

        #now load as a different item
        ds2 = pickle.load(open(self.testfile,'r'))

        #compare
        self.assertFalse(ds2.always_nominal)
        self.assertEqual(ds.all_rnd_points[1].ID, ds2.all_rnd_points[1].ID)

    def testBasic(self):
        if self.just1: return

        #test always-nominal
        ds = DevicesSetup('UMC180')
        self.assertTrue(ds.always_nominal)
        self.assertEqual(ds.process, 'UMC180')
        
        r = ds.nominalRndPoint()
        self.assertTrue(isinstance(r, RndPoint))
        self.assertTrue(r.isNominal())
        r2 = ds.nominal_rnd_point
        self.assertEqual(r.ID, r2.ID) #want IDs of nominal rnd points to be the same!!
        self.assertTrue(len(str(ds)) > 0)

        #test robust (i.e not always-nominal)
        ds.makeRobust()
        self.assertFalse(ds.always_nominal)
        self.assertTrue(len(str(ds)) > 0)
        
        r = ds.nominalRndPoint()
        self.assertTrue(isinstance(r, RndPoint))
        self.assertTrue(r.isNominal())

        for (i, r) in enumerate(ds.all_rnd_points):
            self.assertTrue(isinstance(r, RndPoint))
            if i == 0:
                self.assertTrue(r.isNominal())
            else:
                self.assertFalse(r.isNominal())

        #set back to nominal
        ds.makeNominal()
        self.assertTrue(ds.always_nominal)
        
    def testResistor(self):
        if self.just1: return
        
        ds = DevicesSetup('UMC180')
        ds.makeRobust()

        tiny_R = 0.1 #0.1 ohm
        typical_R = 1.0e3 #1 k ohm
        huge_R = 1.0e6 #1 meg ohm

        #test area
        #-tiny resistor's area should be bounded by technology
        self.assertEqual(ds.resistorArea(tiny_R), ds._techResistorArea())

        #-huge resistor's area should be governed by its resistance
        self.assertEqual(ds.resistorArea(huge_R) , ds._rBasedResistorArea(huge_R))

        #-corner case
        self.assertEqual(ds._rBasedResistorArea(0.0), 0.0)
        self.assertEqual(ds.resistorArea(0.0), ds._techResistorArea())

        #test variation

        #-assertions
        # GOOD:                           ds.varyResistance, typical_R, >=0.0)
        self.assertRaises(AssertionError, ds.varyResistance, 'foo',       0.1)
        self.assertRaises(AssertionError, ds.varyResistance, typical_R,   'foo')
        self.assertRaises(AssertionError, ds.varyResistance, -1.0,        0.1)

        #-corner case
        self.assertEqual(ds.varyResistance(0.0, 0.0), 0.0)
        self.assertEqual(ds.varyResistance(0.0, 0.1), 0.0)
        self.assertEqual(ds.varyResistance(0.0001, -1000.0), 0.0) #force varied R < 0

        #-typical cases
        self.assertEqual(ds.varyResistance(tiny_R, 0.0), tiny_R)
        self.assertTrue(ds.varyResistance(tiny_R,  0.1) > tiny_R)
        self.assertTrue(ds.varyResistance(tiny_R, -0.1) < tiny_R)
        
        self.assertEqual(ds.varyResistance(huge_R, 0.0), huge_R)
        self.assertTrue(ds.varyResistance(huge_R,  0.1) > huge_R)
        self.assertTrue(ds.varyResistance(huge_R, -0.1) < huge_R)

    def testCapacitor(self):
        if self.just1: return
        
        ds = DevicesSetup('UMC180')
        ds.makeRobust()

        tiny_C = 0.001 * 1.0e-15 #0.001 fF
        typical_C = 1.0e-9 #1pF
        huge_C = 1.0e-9 #1 nF

        #test area
        #-tiny capacitor's area should be bounded by technology
        self.assertEqual(ds.capacitorArea(tiny_C), ds._techCapacitorArea())

        #-huge capacitor's area should be governed by its capacitance
        self.assertEqual(ds.capacitorArea(huge_C) , ds._cBasedCapacitorArea(huge_C))

        #-corner case
        self.assertEqual(ds._cBasedCapacitorArea(0.0), 0.0)
        self.assertEqual(ds.capacitorArea(0.0), ds._techCapacitorArea())

        #test variation

        #-assertions
        # GOOD:                           ds.varyCapacitance, typical_C, >=0.0)
        self.assertRaises(AssertionError, ds.varyCapacitance, 'foo',       0.1)
        self.assertRaises(AssertionError, ds.varyCapacitance, typical_C,   'foo')
        self.assertRaises(AssertionError, ds.varyCapacitance, -1.0,        0.1)

        #-corner cases
        self.assertEqual(ds.varyCapacitance(0.0, 0.0), 0.0)
        self.assertEqual(ds.varyCapacitance(0.0, 0.1), 0.0)
        self.assertEqual(ds.varyCapacitance(tiny_C, -1000.0), 0.0) #force varied C < 0

        #-typical cases
        self.assertEqual(ds.varyCapacitance(tiny_C, 0.0), tiny_C)
        self.assertTrue(ds.varyCapacitance(tiny_C,  0.1) > tiny_C)
        self.assertTrue(ds.varyCapacitance(tiny_C, -0.1) < tiny_C)
        
        self.assertEqual(ds.varyCapacitance(huge_C, 0.0), huge_C)
        self.assertTrue(ds.varyCapacitance(huge_C,  0.1) > huge_C)
        self.assertTrue(ds.varyCapacitance(huge_C, -0.1) < huge_C)
        
    def testMos(self):
        if self.just1: return
        
        ds = DevicesSetup('UMC180')
        ds.makeRobust()
        
        #nmos with no variation
        netlist = ds.mosNetlistStr(False, 'm32', 1.0e-6, 1.0e-6, 0.0)
        self.assertTrue('.model Nm32 NMOS' in netlist)
        self.assertTrue('VTH0      =  3.075000e-01' in netlist)
        self.assertTrue('PDIBLC2' in netlist) #arbitrary check of a parameter's existence
        
        #nmos with variation
        netlist = ds.mosNetlistStr(False, 'm32', 1.0e-6, 1.0e-6, 0.1)
        self.assertTrue('.model Nm32 NMOS' in netlist)
        self.assertTrue('VTH0      =  3.081000e-01' in netlist) #should be >nominal

        #pmos with no variation
        netlist = ds.mosNetlistStr(True, 'm32', 1.0e-6, 1.0e-6, 0.0)
        self.assertTrue('.model Pm32 PMOS' in netlist)
        self.assertTrue('VTH0      =  -4.555000e-01' in netlist)

        #pmos with variation
        netlist = ds.mosNetlistStr(True, 'm33', 1.0e-6, 1.0e-6, 0.1)
        self.assertTrue('.model Pm33 PMOS' in netlist)
        self.assertTrue('VTH0      =  -4.549000e-01' in netlist)  #should be >nominal

        #test nominalMosNetlistStr
        netlist = ds.nominalMosNetlistStr()
        self.assertTrue('.model Nnom NMOS' in netlist)
        self.assertTrue('.model Nnom NMOS' in netlist)

        #------------------------------------------------------------------------------
        #test UMC90 model too
        self.assertRaises(AssertionError, ds.setProcess, 'not a process')
        ds.setProcess('UMC90')
        self.assertEqual(ds.process, 'UMC90')

        #nmos with no variation
        netlist = ds.mosNetlistStr(False, 'm32', 1.0e-6, 1.0e-6, 0.0)
        self.assertTrue('.model Nm32 NMOS' in netlist)
        self.assertTrue('VTH0      = -1.000000e-03' in netlist)
        
        #nmos with variation
        netlist = ds.mosNetlistStr(False, 'm32', 1.0e-6, 1.0e-6, 0.1)
        self.assertTrue('.model Nm32 NMOS' in netlist)
        self.assertTrue('VTH0      = 2.000000e-04' in netlist)   #should be >nominal
        netlist = ds.mosNetlistStr(False, 'm32', 1.0e-6, 1.0e-6, -0.1)
        self.assertTrue('VTH0      = -2.200000e-03' in netlist)  #should be <nominal

        #pmos with no variation
        netlist = ds.mosNetlistStr(True, 'm32', 1.0e-6, 1.0e-6, 0.0)
        self.assertTrue('.model Pm32 PMOS' in netlist)
        self.assertTrue('VTH0      = -5.810000e-02' in netlist)

        #pmos with variation
        netlist = ds.mosNetlistStr(True, 'm33', 1.0e-6, 1.0e-6, 0.1)
        self.assertTrue('.model Pm33 PMOS' in netlist)
        self.assertTrue('VTH0      = -5.690000e-02' in netlist)   #should be >nominal 
        netlist = ds.mosNetlistStr(True, 'm33', 1.0e-6, 1.0e-6, -0.1)
        self.assertTrue('VTH0      = -5.930000e-02' in netlist)   #should be <nominal 


        #set back to UMC180
        ds.setProcess('UMC180')
        self.assertEqual(ds.process, 'UMC180')
    
    def tearDown(self):
        pass

if __name__ == '__main__':
    #if desired, this is where logging would be set up
    
    unittest.main()
