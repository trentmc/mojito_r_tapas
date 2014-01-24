import unittest, os
from tests import doctest, importSuite

from Library_test import LibraryTest
from MetricCalculators_test import MetricCalculatorsTest
from OpLibrary_test import OpLibraryTest
from OpLibrary2_test import OpLibrary2Test
from OpLibraryEMC_test import OpLibraryEMCTest
from Problems_test import ProblemsTest
from SizesLibrary_test import SizesLibraryTest

TestClasses = [ \
    LibraryTest,
    MetricCalculatorsTest,
    SizesLibraryTest,
    OpLibraryTest,
    OpLibrary2Test,
    OpLibraryEMCTest,
    ProblemsTest,
    ]

def unittest_suite():
    return unittest.TestSuite(
      [unittest.makeSuite(t,'test') for t in TestClasses]
    )    

def doctest_suite():
    return doctest.DocFileSuite(
        package='problems',
        )

allSuites = [
    'problems.test.unittest_suite',
    'problems.test.doctest_suite',
]

def test_suite():
    return importSuite(allSuites, globals())

if __name__=="__main__":
    unittest.main(defaultTest='test_suite')
