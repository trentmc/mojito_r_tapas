"""RegressorUtils.py

Utilities that many regressors use, which aren't really general
enough to be in util.

Currently includes:
-routines for detecting and removing outliers
-generateTestColumns() provides a better-than-random approach to dividing
 up data into training and test data
-minMaxX() provides a convenient routine for calculating min and max per row
 of 2d array X

"""

__all__ = [
    'insertVariableNamesInModelStr',
    'removeOutliers',
    'generateTestColumns',
    'minMaxX',
    ]


import string

import numpy

from util import mathutil

def insertVariableNamesInModelStr(varnames, model_str):
    """Replaces each 'xi' in model_str with varnames[i] and returns that"""
    num_vars = len(varnames)
    for i in range(num_vars-1, -1, -1):
        # note that we put a 'str' around varnames[i] for robustness;
        #  b/c for some reason certain circuit varnames are not
        #  naturally string buffers
        model_str = model_str.replace('x'  + str(i), str(varnames[i]))
    return model_str

def removeOutliers(X, y, test_I):
    """
    @description

      Detect and remove outliers in y.  Gives updated versions of X, y,
      and test_I.

      Detection method: any y-values that are not in +-3 sigmas of the y's avg.

    @arguments

      X -- 2d array -- training inputs [var #][sample #]
      y -- 1d array -- training outputs [sample #]
      test_I -- list of int -- indices of a subset of samples to use for test.
        (can also be None)

    @return

      new_X -- like X, but with outlier samples removed
      new_y -- like y, but with outlier samples removed
      new_test_I -- like test_I, but with outlier samples removed (or None)
      non_outlier_I -- the indices into the original data that weren't outliers

    @exceptions

    @notes

      - If the test columns have been generated deterministically, the
      process of removing outliers will somewhat alter the test column
      sampling (though the sampling is preserved as best possible).  It
      would be better to generate test data _after_ outlier removal has
      been performed if possible, rather than passing test_I into this function.
      - Currently assumes that the resulting test_I needs to be sorted,
      just to be safe.  If performance is an issue and test_I doesn't need
      to be sorted, then the sort can be removed.

    """

    # Detect outliers
    # :NOTE: kcb - make sure to use logical_and for the range comparison
    # when using numpy arrays
    avg, stddev = mathutil.average(y), mathutil.stddev(y)
    non_outlier_mask = numpy.logical_and((avg-3.0*stddev <= y),
                                           (y <= avg+3.0*stddev))
    non_outlier_I = numpy.nonzero(non_outlier_mask)

    # Calculate new_X, new_y
    new_X = numpy.take(X, non_outlier_I, 1)
    new_y = numpy.take(y, non_outlier_I)

    # Calculate new_test_I
    if test_I is None:
        new_test_I = None
    else:
        num_outliers_removed = len(y) - len(non_outlier_I)
        if num_outliers_removed > 0:
            # Get the new indices by finding the cumulative sum of the
            # non outlier mask
            new_indices = numpy.cumsum(non_outlier_mask) - 1

            # Take the new indices using the original indices
            new_test_I = numpy.take(new_indices, test_I)

            # Remove negative indices when first n entries are outliers
            new_test_I = numpy.compress(new_test_I >= 0, new_test_I)

            # Remove duplicates and sort (assumes test_I needs to be sorted)
            new_test_I = sorted(list(set(new_test_I)))

        else:
            new_test_I = test_I[:]

    return new_X, new_y, new_test_I, non_outlier_I

def generateTrainTestData(X, y, perc_test):
    """Uses generateTestColumns to return (train_X, train_y, test_X, test_y)
    """
    #determine train_I, test_I
    test_I = sorted(generateTestColumns(y, perc_test))
    all_I = range(len(y))
    train_I = sorted(mathutil.listDiff(all_I, test_I))

    #determine [train,test] * [X,y]
    train_X = numpy.take(X, train_I, 1)
    test_X = numpy.take(X, test_I, 1)
    train_y = numpy.take(y, train_I, 0)
    test_y = numpy.take(y, test_I, 0)

    #postconditions
    assert train_X.shape[0] == test_X.shape[0]
    assert train_X.shape[1] == len(train_y) == train_y.shape[0]
    assert test_X.shape[1] == len(test_y) == test_y.shape[0]

    #done
    return (train_X, train_y, test_X, test_y)

def generateTestColumns(y, perc_test):
    """
    @description

      Generate test columns _deterministically_ and in such
      a way that we have a nice spread of y-values.

    @arguments

      y -- 1d array of float/int -- target outputs for a regressor
      perc_test -- float in [0,1] -- percentage of data to use for test

    @return

      test_cols -- list of int, each int in [0,len(y)] and unique --
        A subset of all indices into y that this routine suggests to use for test

    @exceptions

    @notes        
    """
    I = numpy.argsort(y)

    #The challenge is to avoid roundoff error of stepsize, so we
    # can't round stepsize to an int!
    N = len(I)
    stepsize = perc_test*N
    choices_of_I = []
    for j in range(N):
        transition_occurred = int(j*perc_test) != int((j+1)*perc_test)
        #print j, j*perc_test, transition_occurred
        
        if transition_occurred:
            choices_of_I.append(j)
            
    test_cols = list(numpy.take(I, choices_of_I))
        
    return test_cols
    
def minMaxX(X):
    """
    @description

      Returns tuple (min_per_row_of_X, max_per_row_of_X)

    @arguments

      X -- 2d array

    @return

      See description.

    @exceptions

    @notes
    """
    assert len(X.shape)==2, "must be a 2d array"
    minX = numpy.array([min(X[i,:]) for i in range(X.shape[0])])
    maxX = numpy.array([max(X[i,:]) for i in range(X.shape[0])])
    return (minX, maxX)    
