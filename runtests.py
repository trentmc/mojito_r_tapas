#!/usr/bin/env python 
##!/usr/bin/env python2.4

import logging
import sys

from tests import unittest, importSuite, importString

if __name__== '__main__':
    from util.constants import setAggressiveTests
    setAggressiveTests()
    
    #set up logging
    logging.basicConfig()
    
    #want to suppress warnings if they show up.
    # -currently just synth has WARNINGs that show up in unit tests
    logging.getLogger('engine_utils').setLevel(logging.ERROR)
    
    #set help message
    help = """
Usage: runtests MODULE_NAME [LOGGER_NAME1 [LOGGER_NAME2 ...]

If MODULE_NAME is specified as 'all', it runs all tests.
Otherwise it just runs the tests for the specified module/
-Example module names: regressor, engine, engine.EngineUtils

For each LOGGER_NAME specified, it sets that logger to INFO.
-Example logger names: cyt, synth, pwl
"""

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args in [1]:
        print help
        sys.exit(0)

    if num_args >= 3:
        logger_names = sys.argv[2:]
        for logger_name in logger_names:
            logging.getLogger(logger_name).setLevel(logging.INFO)
    else:
        pass

    if sys.argv[1] == 'all':
        from tests import test_suite
        test_suite_obj = test_suite()
        unittest.TextTestRunner().run(test_suite_obj)
        
    else:
        module_name = sys.argv[1]
        if '.' in module_name:
            major_module_name = module_name.split('.')[0] #e.g. 'regressor'
            minor_module_name = module_name.split('.')[1] #e.g. 'Pwl'
            test_class_name = major_module_name + '.test.' + minor_module_name + 'Test' #e.g. 'regressor.test.PwlTest
            test_class = importString(test_class_name)    #e.g. eval('import regressor.test.PwlTest')
            unittest_suite = unittest.makeSuite(test_class, 'test')
            test_suite = unittest_suite
        else:
            test_suite_name = module_name + '.test.test_suite'
            test_suite = importSuite(test_suite_name)
        result = unittest.TextTestRunner().run(test_suite)

    

