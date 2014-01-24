import unittest, os
from tests import doctest, importSuite

from Constants_test import ConstantsTest
from Hypervolume_test import HypervolumeTest
from mathutil_test import mathutilTest

TestClasses = [
    ConstantsTest,
    HypervolumeTest,
    mathutilTest,
    ]

def unittest_suite():
    return unittest.TestSuite(
      [unittest.makeSuite(t,'test') for t in TestClasses]
    )    

def doctest_suite():
    return doctest.DocFileSuite(
        package='util',
        )

allSuites = [
    'util.test.unittest_suite',
    'util.test.doctest_suite',
]

def test_suite():
    return importSuite(allSuites, globals())

if __name__=="__main__":
    unittest.main(defaultTest='test_suite')
