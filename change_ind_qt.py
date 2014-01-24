#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import sys
from qt import *
from gui.change_ind_qt_ui import ChangeSettingsUI
import pickle

from adts.Point import Point, RndPoint, EnvPoint
from adts.DevicesSetup import DevicesSetup

ind_out_file = "" #FIXME

def spiceCommentPrefixBlock(in_str):
    lines = in_str.splitlines()
    out_str = ""
    for line in lines:
        # if no spice prefix, add one
        if line[0:2] != '* ':
            out_str += '* '
            
        # add the original line
        out_str += line + "\n"

    return out_str

def rewrapLine(line, length):
    new_line = ""
    is_comment = False
    if len(line) == 0:
        return ""

    if line[0] == "*":
        is_comment = True

    # first run has no indent
    chunk = line[0:length]
    line = line[length:]
    if is_comment:
        new_line += "* " + chunk + "\n"
    else:
        new_line += chunk + "\n"

    while len(line):
        chunk = line[0:length]
        line = line[length:]
        if is_comment:
            new_line += "*  " + chunk + "\n"
        else:
            new_line += " " + chunk + "\n"
            
    return new_line

def spiceSanitize(in_str, maxlen = 1000):
    lines = in_str.splitlines()
    out_str = ""
    for line in lines:
        out_str += rewrapLine(line, maxlen)
    return out_str

class ChangeSettings(ChangeSettingsUI):
    def __init__(self, ind, parent=None):
        ChangeSettingsUI.__init__(self, parent)
        self.ind = ind
        
        self.tblValues.setNumCols(2)
        #import pdb;pdb.set_trace()
        varnames = ind._ps.ordered_optvars
        self.tblValues.setNumRows(len(varnames))
        for name in varnames:
            idx = varnames.index(name)
            self.tblValues.setText(idx, 0, name)
            val = ind.unscaled_optvals[idx]
            self.tblValues.setText(idx, 1, str(val))
    
        for i in range(2):
            self.tblValues.adjustColumn(i)

        QObject.connect(self.btnPrint, SIGNAL("clicked()"), self.summary)
        QObject.connect(self.btnUpdate, SIGNAL("clicked()"), self.update)
        QObject.connect(self.btnSave, SIGNAL("clicked()"), self.save)
        QObject.connect(self.btnNetlist, SIGNAL("clicked()"), self.saveNetlist)
        QObject.connect(self.btnOpenFile, SIGNAL("clicked()"), self.openfile)

    def summary(self):
        print self.ind.pointSummary()    

    def openfile(self):
        netlistfile = QFileDialog.getSaveFileName()
        if netlistfile:
            self.txtFilename.setText(netlistfile)

    def save(self):
        print "saving"
        # remove some bogus stuff
        S_tmp = self.ind.S
        ps_tmp = self.ind._ps

        self.ind.S = None
        self.ind._ps = None
        if ind_out_file:
            filename = ind_out_file
        else:
            filename = QFileDialog.getSaveFileName()

        if filename:
            print "saving to %s" % filename
            # save the modified ind
            fid=open(filename,'w')
            pickle.dump(self.ind, fid)
            fid.close()
        else:
            print "not saving"

        #restore
        self.ind.S = S_tmp
        self.ind._ps = ps_tmp

    def update(self):
        print "updating"
        varnames = self.ind._ps.ordered_optvars
        for name in varnames:
            idx = varnames.index(name)
            val = self.tblValues.text(idx, 1).latin1()
            self.ind.unscaled_optvals[idx] = eval(val)

        self.txtNetlist.setText(self.generateNetlist())

    def generateNetlist(self):
        annotate_points = self.chkAnnPoint.isChecked()
        annotate_bb_info = self.chkAnnBB.isChecked()
    
        make_simulatable = self.chkSimulatable.isChecked()
        if make_simulatable:
            analysis_index = self.spinAnalysis.value()
            env_index = self.spinEnv.value()
            if analysis_index >= len(self.ind._ps.analyses):
                print "Requested analysis_index=%d but only %d analyses available" % (analysis_index, len(self.ind._ps.analyses))
                return
            analysis = self.ind._ps.analyses[analysis_index]
            if env_index >= len(analysis.env_points):
                print "Requested env_index=%d but only %d env_points available" % (env_index, len(analysis.env_points))
                return

        #-we'll be building up 'big_s' (a netlist)
        big_s = ''
    
        # -add design info (and maybe simulation info)
        if not make_simulatable:
            env_point = EnvPoint(is_scaled=True)
            big_s += self.ind.nominalNetlist(annotate_bb_info = annotate_bb_info, add_infostring=True)
            
        else:
            analysis = self.ind._ps.analyses[analysis_index]
            env_point = analysis.env_points[env_index]
            variation_data = (RndPoint([]) , env_point, DevicesSetup('UMC180'))
            big_s += analysis.createFullNetlist(self.ind._ps.embedded_part, self.ind._ps.scaledPoint(self.ind), variation_data)
    
        # -maybe info about unscaled_point, scaled_point
        if annotate_points:
            big_s += "* Netlist=\n%s" % spiceCommentPrefixBlock(self.ind.pointSummary())
    
        #successful, so print netlist
        netlist = spiceSanitize(big_s, 1000)
        return netlist

    def saveNetlist(self):
        print "netlist"
        self.update()
        netlist = self.generateNetlist()
        # save netlist if file is set
        netlistfile = self.txtFilename.text().latin1()
        if netlistfile:
            fid = open(netlistfile,'w')
            fid.write(netlist)
            fid.close()
        

if __name__== '__main__':            
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.DEBUG)
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    #set help message
    help = """
Usage: change_ind PROBLEM_NUM IND_FILE OUT_FILE

    Allows to change a pickled ind. The file is loaded from IND_FILE,
    then the python debugger is called, and the resulting ind is saved in
    OUT_FILE

Details:
 PROBLEM_NUM -- int -- see listproblems.py
 IND_FILE -- string -- the file containing the ind  (saved using get_ind.py)
 OUT_FILE -- string -- the file the ind changed ind is to be saved to
"""

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [4]:
        print help
        sys.exit(0)

    #yank out the args into usable values
    problem_choice = eval(sys.argv[1])
    ind_file = sys.argv[2]
    ind_out_file = sys.argv[3]

    #late imports
    from adts import *
    from problems.Problems import ProblemFactory

    #do the work


    # -load data
    ps = ProblemFactory().build(problem_choice)
    if not os.path.exists(ind_file):
        print "Cannot find file with name %s" % ind_file
        sys.exit(0)
    
    fid = open(ind_file,'r')
    ind = pickle.load(fid)
    fid.close()

    ind._ps = ps

    app = QApplication(sys.argv)
    form = ChangeSettings(ind)
    form.show()
    
    QObject.connect(app, SIGNAL("lastWindowClosed()"), app, SLOT("quit()"))
    
    print "start main loop"
    app.exec_loop()
    
