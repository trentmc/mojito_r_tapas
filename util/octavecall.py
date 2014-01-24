"""octavecall.py

Supplies these functions:
-octavecall(commands) executes the given commands in octave
-plotAndPause() plots then pauses, without scipy dependency.
-plotScatterAndPause() plots a 2d scatter plot then pauses
"""

import os
import types

import numpy

from ascii import stringsToAscii

def plotAndPause(x1, y1, x2=None, y2=None,
                 title=None, xlabel=None, ylabel=None):
    """
    Calls plot(x1,y2) or plot(x1,y1,'b', x2,y2,'r').
    Optionally gives title, xlabel, and ylabel.
    Then pauses.  (Hit any key to continue).

    Uses a call to octave to do this (therefore doesn't need
    extra python plotting packages).
    """
    #Condition inputs
    if title is None:
        title = ''
    if xlabel is None:
        xlabel = ''
    if ylabel is None:
        ylabel = ''
    
    #Build up 'commands'
    commands = []
    x1, y1 = list(x1), list(y1) 
    commands.append("x1 = %s;\n" % str(x1))
    commands.append("y1 = %s;\n" % str(y1))
    
    if x2 is not None:
        assert y2 is not None
        x2, y2 = list(x2), list(y2)
        commands.append("x2 = %s;\n" % str(x2))
        commands.append("y2 = %s;\n" % str(y2))

    if x2 is None:
        commands.append("plot(x1,y1);\n")
    else:
        commands.append("plot(x1,y1,'b',x2,y2,'r');\n")

    commands.append("title('%s');\n" % title)
    commands.append("xlabel('%s');\n" % xlabel)
    commands.append("ylabel('%s');\n" % ylabel)

    commands.append("fprintf(1, 'Hit a key to continue.');\n")
    commands.append("pause;\n");

    #Make the call to octave
    print ""
    octavecall(commands)
    print ""
        
def plotScatterAndPause(XA, XB=None):
    """
    Calls plot(XA[0,:],XA[1,:],'bx') or plot(XA[0,:],XA[1,:],'bo', XB[0,:],XB[1,:],'rx').
    Then pauses.  (Hit any key to continue).
    """
    #condition inputs
    XA = numpy.asarray(XA)
    XB = numpy.asarray(XB)
    
    #Build up 'commands'
    commands = []
    commands.append("xa0 = %s;\n" % str(list(XA[0,:])))
    commands.append("xa1 = %s;\n" % str(list(XA[1,:])))
    if XB is not None:
        commands.append("xb0 = %s;\n" % str(list(XB[0,:])))
        commands.append("xb1 = %s;\n" % str(list(XB[1,:])))

    if XB is not None:
        commands.append("plot(xa0,xa1,'bo', xb0,xb1,'ro');\n")
        commands.append("mnx = min(min(xa0),min(xb0)); mxx = max(max(xa0),max(xb0)); rx = mxx - mnx;\n")
        commands.append("mny = min(min(xa1),min(xb1)); mxy = max(max(xa1),max(xb1)); ry = mxy - mny;\n")
        commands.append("axis([mnx - 0.2*rx, mxx + 0.2*rx, mny - 0.2*ry, mxy + 0.2*ry]);")
    else:
        commands.append("plot(xa0,xa1,'bo');\n")
        commands.append("mnx = min(xa0); mxx = max(xa0); rx=mxx - mnx;\n")
        commands.append("mny = min(xa1); mxy = max(xa1); ry=mxy - mny;\n")
        commands.append("axis([mnx - 0.2*rx, mxx + 0.2*rx, mny - 0.2*ry, mxy + 0.2*ry]);\n")

    commands.append("fprintf(1, 'Hit a key to continue.');\n")
    commands.append("pause;\n");

    #Make the call to octave
    print ""
    octavecall(commands)
    print ""
        

def octavecall(octave_commands):
    """
    @description

      Runs one or many octave commands.
      
    @arguments

      octave_commands -- a single string, or a whole list of octave commands
      
    @return

      <<nothing>> but the octave commands are executed

    @exceptions

    @notes

      Implemented by creating a temporary octave function,
      and calling it.
    """
    if isinstance(octave_commands, types.StringType):
        octave_commands = [octave_commands]
    elif isinstance(octave_commands, types.ListType):
        pass
    else:
        raise AssertionError(
            "unknown type: %s, %s" %
            (type(octave_commands), octave_commands.__class__))
        
    octave_commands += ["\n"]
    
    stringsToAscii('/tmp/tempfunc.m', octave_commands, False)
    os.system("octave /tmp/tempfunc.m")

    

