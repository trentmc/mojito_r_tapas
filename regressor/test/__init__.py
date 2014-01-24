import unittest, os
from tests import doctest, importSuite

from Bottomup_test import BottomupTest
from Kmeans_test import KmeansTest
from LinearModel_test import LinearModelTest
from Lut_test import LutTest
from Luc_test import LucTest
from Probe_test import ProbeTest
from Pwl_test import PwlTest

TestClasses = [BottomupTest,
               KmeansTest,
               LinearModelTest,
               LutTest,
               LucTest,
               ProbeTest,
               PwlTest,
               ]

def unittest_suite():
    return unittest.TestSuite(
      [unittest.makeSuite(t,'test') for t in TestClasses]
    )    

def doctest_suite():
    return doctest.DocFileSuite(
        package='regressor',
        )

allSuites = [
    'regressor.test.unittest_suite',
    'regressor.test.doctest_suite',
]

def test_suite():
    return importSuite(allSuites, globals())

if __name__=="__main__":
    import logging
    logging.basicConfig()
    logging.getLogger('lut').setLevel(logging.ERROR)
    logging.getLogger('hockey').setLevel(logging.ERROR)
    
    unittest.main(defaultTest='test_suite')
