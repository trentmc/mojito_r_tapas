

import unittest
        
from util.constants import *

class ConstantsTest(unittest.TestCase):

    def testBadMetricValue(self):
        a = BAD_METRIC_VALUE
        self.assertEqual(str(BAD_METRIC_VALUE), 'BAD_METRIC_VALUE')
        self.assertFalse(a == 2)
        
        b = BAD_METRIC_VALUE
        self.assertTrue(a == b)
        self.assertFalse(a != b)
                      

if __name__ == '__main__':
    unittest.main()
