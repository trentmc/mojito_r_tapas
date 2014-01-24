import unittest, os
from tests import doctest, importSuite

from Channel_test import ChannelTest
from CytOptimizer_test import CytOptimizerTest
from DytOptimizer_test import DytOptimizerTest
from EvoliteOptimizer_test import EvoliteOptimizerTest
from EngineUtils_test import EngineUtilsTest
from GtoOptimizer_test import GtoOptimizerTest
from Master_test import MasterTest
from Ind_test import IndTest
from RandomPart_test import RandomPartTest #turn on again for novelty work!!
from Slave_test import SlaveTest
from SynthSolutionStrategy_test import SynthSolutionStrategyTest
from TloOptimizer_test import TloOptimizerTest
from YtAdts_test import YtAdtsTest

TestClasses = [
    ChannelTest,
    CytOptimizerTest,
    DytOptimizerTest,
    EvoliteOptimizerTest,
    EngineUtilsTest,
    GtoOptimizerTest,
    IndTest,
    MasterTest, 
    RandomPartTest,
    SlaveTest, 
    SynthSolutionStrategyTest,
    TloOptimizerTest,
    YtAdtsTest,
    ]

def unittest_suite():
    return unittest.TestSuite(
      [unittest.makeSuite(t,'test') for t in TestClasses]
    )    

def doctest_suite():
    return doctest.DocFileSuite(
        package='engine',
        )

allSuites = [
    'engine.test.unittest_suite',
    'engine.test.doctest_suite',
]

def test_suite():
    return importSuite(allSuites, globals())

if __name__=="__main__":
    import logging
    logging.basicConfig()
    
    unittest.main(defaultTest='test_suite')
