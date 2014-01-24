import unittest

from adts import *
from problems.Problems import *

class ProblemsTest(unittest.TestCase):

    def setUp(self):
        pass
        

    def testInstantiateEachProblem(self):
        factory = ProblemFactory()
        for problem_choice in [1,2,4,6,8,10,11,12,13,15,
                               31,33,#34, #uncomment when 90nm supported
                               39,
                               41,42,43,44,
                               #61, #uncomment when WL supported
                               62,63, #64, #uncomment when 90nm supported
                               69,
                               #71,72, #not supported 
                               81,82,83,84, 
                               100,
                               101]:
            factory.build(problem_choice)
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    #if desired, this is where logging would be set up
    
    unittest.main()
