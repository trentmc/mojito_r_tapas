""" Provides data for testing regressors
"""

import random
from numpy import arange, sin, reshape, zeros, sqrt, array, take, where

def data1d():
    x = arange(0.0,2*3.14,0.1)
    y = sin(x)
    X = reshape(x,(1,len(x)))
    return [x,X,y]

def data2d(gridsize, funcID=0):
    scale = 10.0
    mn = -0.5 * scale 
    mx = +0.5 * scale
    stepsize = float(mx - mn)/float(gridsize)
    xx0 = arange(mn, mx, stepsize)*1.0
    xx1 = arange(mn, mx, stepsize)*1.0

    if funcID==4:
        xx0 = xx0 + 0.1
        xx1 = xx1 + 0.1

    num_samples = len(xx0)*len(xx1)

    X = zeros((2, num_samples))*1.0
    maty = zeros((gridsize, gridsize))*1.0
    vecy = zeros(num_samples)*1.0

    inc = 0
    for i in range(gridsize):
        for j in range(gridsize):
            x0, x1 =  xx0[i], xx1[j]
	    X[0,inc], X[1,inc] = x0, x1
            
            if funcID==0:
                maty[i,j] = 2.5*sin(x0*1.3) + 1.5*sin(x1*0.4)
                
            elif funcID==1:
                maty[i,j] = 1.0*x0*x1 + \
                            0.5*sin(x0*1.3) + 0.3*sin(x1*0.4)
                
            elif funcID==2:
                maty[i,j] = 1.0*x0 + 2.0*x1

            elif funcID==3:
                maty[i,j] = x0 * x1

            elif funcID==4:
                maty[i,j] = 1.0*x0*x1 + 2.0 * x1/x0

            elif funcID==5:
                maty[i,j] = 2.0 + 3.0*x0 - 1.0*x1
                
            elif funcID==6:
                maty[i,j] = 1.9 + 3.1*x0 - 1.1*x1

            elif funcID==7:
                maty[i,j] = 2.0 + 3.0*x0 - 4.0*x1 + 5.0*x0**2 + 6.0*x1**2 + 7.0*x0*x1
                
            else:
                raise AssertionError('unknown function id: %d' % funcID)
	    vecy[inc] = maty[i,j]
	    inc += 1

    return [xx0, xx1, X, vecy]

    
