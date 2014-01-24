#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import sys
import re

from qt import *
from qttable import *
from gui.QtResultBrowseUi import QtResultBrowseUi
import pickle

from change_ind_qt import ChangeSettings

import engine.EngineUtils as EngineUtils
from util import mathutil
from itertools import izip

import numpy

from adts.Metric import MAXIMIZE, MINIMIZE, IN_RANGE
from util.ascii import stringToAscii
from util.constants import BAD_METRIC_VALUE

 
#The following line styles are supported::
 
    #-     # solid line
    #--    # dashed line
    #-.    # dash-dot line
    #:     # dotted line
    #.     # points
    #,     # pixels
    #o     # circle symbols
    #^     # triangle up symbols
    #v     # triangle down symbols
    #<     # triangle left symbols
    #>     # triangle right symbols
    #s     # square symbols
    #+     # plus symbols
    #x     # cross symbols
    #D     # diamond symbols
    #d     # thin diamond symbols
    #1     # tripod down symbols
    #2     # tripod up symbols
    #3     # tripod left symbols
    #4     # tripod right symbols
    #h     # hexagon symbols
    #H     # rotated hexagon symbols
    #p     # pentagon symbols
    #|     # vertical line symbols
    #_     # horizontal line symbols
    #steps # use gnuplot style 'steps' # kwarg only
 
#The following color abbreviations are supported::
 
    #b  # blue
    #g  # green
    #r  # red
    #c  # cyan
    #m  # magenta
    #y  # yellow
    #k  # black
    #w  # white

COLORS = ["b","g","r","c","m","y","k"]
SHAPES = ["o","D","s","x","+","h","H","^","v"]

TOPOPLOT_COLORCODES = ["%s%s" % (color, shape) for shape in SHAPES for color in COLORS ]

TOPO_COLOR_MAP = {}

def maskTopology(topo, topomask):
    # corner case
    if len(topo) == 0:
      return ""

    masked = ""
    idx = 0
    
    while idx < len(topomask):
        mchar = topomask[idx]
        if mchar == "*": # keep all remaining
            masked += topo[idx:]
            return masked
        if mchar == "&": # mask all remaining
            return masked
        if mchar == "#": # keep current char
            masked += topo[idx]
        elif mchar == "$": # skip current char
            pass
        idx += 1
    return masked

def filterTopology(topo, expression):
    rawstr = r"""%s""" % expression

    compile_obj = re.compile(rawstr)
    match_obj = compile_obj.match(topo)

    return match_obj != None

class CustomTableItem(QTableItem):
    def __init__(self, table, et, text = ""):
        QTableItem.__init__(self, table, et, text)

    def key(self):
        try:
            num = eval(self.text().latin1())
            t = type(num)
            if t == float:
                #print "float: %f" % num
                return "%020d" % long(1000000*num)
            elif t == int or t == long:
                #print "int/long %d" % num
                return "%020d" % num
            else:
                print type(num)
                return self.text()
        except:
            #print "other"
            return self.text()

class StateWrapper:
    def __init__(self, state):
        self.state = state
        self.ps = state.ps

        self.inds = []
        self.shortid_idx_map = {}
        self.metric_map = {}

        self.nb_cols = 0
        self.nb_rows = 0

        assert self.state
        assert self.ps

    def nbAgeLayers(self):
        return self.state.R_per_age_layer.numAgeLayers()

    def update(self, all_inds = False, age_layer = None, add_lower = False, feasible_only = True, topo_filter = "*"):
        if all_inds:
            self.inds = self.state.allInds()
        elif self.ps.doRobust():
            print "FIXME"
            self.inds = self.state.yield_nondom_lite_inds
        else:
            self.inds = self.state.nominal_nondom_inds

        # filter by age layer if needed
        if age_layer != None:
            assert age_layer >= 0
            if age_layer < self.state.R_per_age_layer.numAgeLayers():
                cands = set(self.state.R_per_age_layer[age_layer])
                if add_lower and age_layer > 0:
                    cands.update(self.state.R_per_age_layer[age_layer-1])
                if all_inds:
                    self.inds = list(cands)
                else:
                    self.inds = EngineUtils.nondominatedFilter(list(cands), None)
            else:
                print "age layer index too high: %s" % age_layer

        # if necessary, filter on feasibility
        if feasible_only:
            feasible_inds = []
            for ind in self.inds:
                if ind.isFeasibleAtNominal():
                    feasible_inds.append(ind)
            self.inds = feasible_inds

        # filter by topology if needed
        if topo_filter != "" and topo_filter != "*":
            filtered_inds = []
            for ind in self.inds:
                pass_filter = filterTopology("T" + ind.topoSummary(), topo_filter)
                if pass_filter:
                    filtered_inds.append(ind)
            self.inds = filtered_inds

        self.shortid_idx_map = {}
        self.metric_map = {}

        # figure out the number of columns
        # Ind_ID GeneticAge  Feas    TopoSummary  [analysis metrics]
        nb_cols = 4
        for analysis in self.ps.analyses:
            nb_cols += len(analysis.metrics)

        self.nb_cols = nb_cols
        self.nb_rows = len(self.inds)

        # headers
        self.metric_map["ShortID"] = 0
        self.metric_map["Age"] = 1
        self.metric_map["Feasible"] = 2
        self.metric_map["Topo"] = 3

        if len(self.inds):
            ind = self.inds[0]
            col = 4
            for analysis_index, analysis in enumerate(self.ps.analyses):
                for metric in analysis.metrics:
                    self.metric_map[metric.name] = col
                    col += 1

        #   -for each ind, a row of '<ID_value> <metric1_value> <metric2_value> ...'
        for ind_idx, ind in enumerate(self.inds):
            self.shortid_idx_map[str(ind.shortID())] = ind_idx

class QtResultBrowse(QtResultBrowseUi):
    def __init__(self, filename = None):
        QtResultBrowseUi.__init__(self)

        self.tblValues.setNumCols(0)
        self.tblValues.setNumRows(0)
        self.tblDetails.setNumCols(0)
        self.tblDetails.setNumRows(0)

        self.state_wrappers = {}
        self.current_state = None
        
        self.state_plots = []

        QObject.connect(self.btnUpdate, SIGNAL("clicked()"), self.update)
        QObject.connect(self.btnSort, SIGNAL("clicked()"), self.sort)
        QObject.connect(self.btnSave, SIGNAL("clicked()"), self.save)
        QObject.connect(self.btnIndSaveAs, SIGNAL("clicked()"), self.saveInd)
        QObject.connect(self.btnSheetSaveAs, SIGNAL("clicked()"), self.saveSheet)
        QObject.connect(self.btnSaveData, SIGNAL("clicked()"), self.saveData)
        QObject.connect(self.btnIndEdit, SIGNAL("clicked()"), self.editInd)
        QObject.connect(self.btnClose, SIGNAL("clicked()"), self.closeState)
        QObject.connect(self.btnCloseAll, SIGNAL("clicked()"), self.closeAllStates)
        QObject.connect(self.btnLoad, SIGNAL("clicked()"), self.loadState)
        
        QObject.connect(self.btnOpenFile, SIGNAL("clicked()"), self.showOpenFileDialog)
        QObject.connect(self.tblValues, SIGNAL("clicked(int,int,int,const QPoint&)"), self.handleClick)
        QObject.connect(self.tblValues, SIGNAL("currentChanged(int,int)"), self.handleCurrentChanged)
        QObject.connect(self.chkBlockInfo, SIGNAL("stateChanged(int)"), self.refreshDetails)
        QObject.connect(self.chkInfoString, SIGNAL("stateChanged(int)"), self.refreshDetails)

        QObject.connect(self.chkNondom, SIGNAL("stateChanged(int)"), self.update)
        QObject.connect(self.chkAgeLayer, SIGNAL("stateChanged(int)"), self.update)
        QObject.connect(self.chkAddLowerLayer, SIGNAL("stateChanged(int)"), self.update)
        QObject.connect(self.chkFeasible, SIGNAL("stateChanged(int)"), self.update)
        QObject.connect(self.spinAgeLayer, SIGNAL("valueChanged(int)"), self.update)
        QObject.connect(self.btnUpdateListTopoFilter, SIGNAL("clicked()"), self.update)

        QObject.connect(self.btnClearPlot, SIGNAL("clicked()"), self.clearPlot)
        QObject.connect(self.btnUpdatePlot, SIGNAL("clicked()"), self.updatePlot)
        #QObject.connect(self.cmbXAxis, SIGNAL("activated(int)"), self.comboAxisChanged)
        #QObject.connect(self.cmbYAxis, SIGNAL("activated(int)"), self.comboAxisChanged)

        QObject.connect(self.lstOpenStates, SIGNAL("selected(int)"), self.setCurrentState)

        if filename:
            self.txtFilename.setText(filename)
            self.loadState()

    #def comboAxisChanged(self, a0 = 0):
        #self.updatePlot()

    def handleClick( self, row, col, button, mousePos ):
        pass

    def handleCurrentChanged( self, row, col ):
        self.updateDetails(row)

    def refreshDetails(self, a0=None):
        self.updateDetails(self.tblValues.currentRow())

    def updateDetails(self, row):
        curr_state = self.lstOpenStates.currentText().latin1()
        if curr_state == "":
            raise "no current state"

        state_wrapper = self.state_wrappers[curr_state]
        
        shortID = self.tblValues.text(row, 0).latin1()
        try:
            ind_idx = state_wrapper.shortid_idx_map[shortID]
        except:
            print "shortID not found"
            return
        ind = state_wrapper.inds[ind_idx]

        self.showDetails(ind)

    def showDetails(self, ind):
        curr_state = self.lstOpenStates.currentText().latin1()
        if curr_state == "":
            raise "no current state"

        state_wrapper = self.state_wrappers[curr_state]
        
        # figure out the number of columns
        nb_cols = 3 # name, scaled, unscaled
        nb_rows = len(state_wrapper.ps.ordered_optvars)
        self.tblDetails.setNumCols(nb_cols)
        self.tblDetails.setNumRows(nb_rows)

        self.tblDetails.horizontalHeader().setLabel( 0, "OptVar" )
        self.tblDetails.horizontalHeader().setLabel( 1, "Scaled" )
        self.tblDetails.horizontalHeader().setLabel( 2, "Unscaled" )

        scaled_point = state_wrapper.ps.scaledPoint(ind)
        for (i, name) in enumerate(state_wrapper.ps.ordered_optvars):
            unscaled = "%g" % ind.unscaled_optvals[i]
            scaled = "%g" % scaled_point[name]
            self.tblDetails.setText(i, 0, str(name))
            self.tblDetails.setText(i, 1, str(scaled))
            self.tblDetails.setText(i, 2, str(unscaled))
        for i in range(nb_cols):
            self.tblDetails.adjustColumn(i)

        self.txtNetlist.setText( ind.nominalNetlist(
                                   annotate_bb_info=self.chkBlockInfo.isChecked(),
                                   add_infostring=self.chkInfoString.isChecked()) )
        self.txtPointSummary.setText( ind.pointSummary() )
        self.txtIndString.setText( str(ind) )

    def clearDetails(self):
        self.tblDetails.setNumCols(0)
        self.tblDetails.setNumRows(0)
        self.txtNetlist.setText( "" )
        self.txtPointSummary.setText( "" )
        self.txtIndString.setText( "" )

        if not self.chkHoldPlot.isChecked():
            self.clearPlot()

    def preparePlottingOptions(self):
        curr_X = self.cmbXAxis.currentText()
        curr_Y = self.cmbYAxis.currentText()

        self.cmbXAxis.clear()
        self.cmbYAxis.clear()

        curr_state = self.lstOpenStates.currentText().latin1()
        if curr_state == "":
            raise "no current state"

        state_wrapper = self.state_wrappers[curr_state]

        for name in state_wrapper.metric_map.keys():
            self.cmbXAxis.insertItem( name , -1)
            self.cmbYAxis.insertItem( name , -1)

        for i in range(self.cmbXAxis.count()):
            if self.cmbXAxis.text(i) == curr_X:
                self.cmbXAxis.setCurrentItem(i)
        for i in range(self.cmbYAxis.count()):
            if self.cmbYAxis.text(i) == curr_Y:
                self.cmbYAxis.setCurrentItem(i)

        #self.updatePlot()

    def getValuesForMetric(self, ID, no_eval=False):
        print "get values for metric: %s" % ID
        curr_state = self.lstOpenStates.currentText().latin1()
        if curr_state == "":
            raise "no current state"

        state_wrapper = self.state_wrappers[curr_state]

        if ID in state_wrapper.metric_map.keys():
            col = state_wrapper.metric_map[ID]
            nb_rows = self.tblValues.numRows()
            V = []
            for i in range(nb_rows):
                if no_eval:
                    V.append(self.tblValues.text(i, col).latin1())
                else:
                    V.append(eval(self.tblValues.text(i, col).latin1()))
            return V
        else:
            raise "unknown ID"

    def updatePlot(self):
        drw = self.matplotlibWidget1.axes
        if self.chkHoldPlot.isChecked():
            drw.hold(self.chkHoldPlot.isChecked())
        else:
            self.clearPlot()
        drw.grid(True)
        topofilter = self.txtTopoFilter.text().latin1()

        if self.chkClustering.isChecked():
            drw.hold(True)
            curr_state = self.lstOpenStates.currentText().latin1()
            if curr_state == "":
                raise "no current state"
    
            state_wrapper = self.state_wrappers[curr_state]
            
            inds = list(state_wrapper.inds)
            (layer, kicked_inds, layer_cost) = EngineUtils.prepareMOEADLayer(state_wrapper.ps, inds, state_wrapper.state.W,
                                                                             state_wrapper.state.ss.metric_weights,
                                                                             state_wrapper.state.ss.topology_layers_per_weight)
            clusters = EngineUtils.clusterPerTopology(layer, layer_cost, state_wrapper.state.W, state_wrapper.state.indices_of_neighbors)
            X = self.getValuesForMetric(self.cmbXAxis.currentText().latin1())
            Y = self.getValuesForMetric(self.cmbYAxis.currentText().latin1())
            IDs = self.getValuesForMetric("ShortID", no_eval=True)
            
            selected_inds_per_cluster = {}
            for cluster in clusters:
                if self.chkFrontOnly.isChecked() and cluster.in_front == False:
                    continue
                if self.chkNonFront.isChecked() and cluster.in_front == True:
                    continue

                selected_inds_per_cluster[cluster] = []
                cluster_shortids = [str(ind.shortID()) for ind in cluster.getInds()]
                for (idx, ID) in enumerate(IDs):
                    if ID in cluster_shortids:
                        selected_inds_per_cluster[cluster].append(idx)

            color_idx = 0
            for (idx, cluster) in enumerate(clusters):
                if not cluster in selected_inds_per_cluster.keys():
                    continue
                # check if it passes the filter
                if len(topofilter) == 0:
                    pass_filter = True
                else:
                    pass_filter = filterTopology("T" + cluster.topo, topofilter)
                
                if pass_filter:
                    idxs = selected_inds_per_cluster[cluster]
                    drwX = numpy.take(X, idxs)
                    drwY = numpy.take(Y, idxs)
                    drwIDs = numpy.take(IDs, idxs)
                    drwIDs = []
                    for idx in idxs:
                        drwIDs.append(IDs[idx])

                    # keeps track of used colors for certain topo's

                    if color_idx < len(TOPOPLOT_COLORCODES):
                        colorcode = TOPOPLOT_COLORCODES[color_idx]
                        color_idx += 1
                    else:
                        colorcode = "b."

                    print "cluster %4d [%20s] idx %3d color '%5s'" % (cluster.ID, cluster.topo, idx, colorcode)
                    line = drw.plot(drwX, drwY, colorcode, picker=5)
                    self.state_plots.append((self.lstOpenStates.currentText().latin1(), drwIDs))

            drw.hold(self.chkHoldPlot.isChecked())

        elif self.chkTopoPlot.isChecked():
            drw.hold(True)
            T = self.getValuesForMetric("Topo", no_eval=True)
            X = self.getValuesForMetric(self.cmbXAxis.currentText().latin1())
            Y = self.getValuesForMetric(self.cmbYAxis.currentText().latin1())
            IDs = self.getValuesForMetric("ShortID", no_eval=True)

            # the set of unique topologies
            unique_T = set(T)

            masked_topos = set()
            topo_masked_map = {}
            X_by_topo = {}
            Y_by_topo = {}
            ID_by_topo = {}

            topomask = self.txtTopoMask.text().latin1()
            if len(topomask) == 0:
                topomask = "*" # keep remainder by default

            # apply topology mask
            for topo in unique_T:
                masked = maskTopology(topo, topomask)
                topo_masked_map[topo] = masked
                masked_topos.update([masked])

            # prepare the data containers
            for topo in masked_topos:
                X_by_topo[topo] = []
                Y_by_topo[topo] = []
                ID_by_topo[topo] = []

            # collect all data points into the masked topology "bins"
            for idx, topo in enumerate(T):
                masked_topo = topo_masked_map[topo]
                X_by_topo[masked_topo].append(X[idx])
                Y_by_topo[masked_topo].append(Y[idx])
                ID_by_topo[masked_topo].append(IDs[idx])

            # plot for all masked topo bin's
            color_idx=0
            for idx, masked_topo in enumerate(X_by_topo.keys()):
                # check if it passes the filter
                if len(topofilter) == 0:
                    pass_filter = True
                else:
                    pass_filter = filterTopology(masked_topo, topofilter)

                if pass_filter and len(X_by_topo[masked_topo]) > 0:
                    # keeps track of used colors for certain topo's
                    if masked_topo in TOPO_COLOR_MAP.keys():
                        colorcode = TOPO_COLOR_MAP[masked_topo]
                    else:
                        if color_idx < len(TOPOPLOT_COLORCODES):
                            colorcode = TOPOPLOT_COLORCODES[color_idx]
                        else:
                            colorcode = "b."
                        TOPO_COLOR_MAP[masked_topo] = colorcode

                    drw.plot(X_by_topo[masked_topo], Y_by_topo[masked_topo], colorcode, picker=5)
                    self.state_plots.append((self.lstOpenStates.currentText().latin1(), ID_by_topo[masked_topo]))
                    color_idx += 1

            drw.hold(self.chkHoldPlot.isChecked())

        else:
            X = self.getValuesForMetric(self.cmbXAxis.currentText().latin1())
            Y = self.getValuesForMetric(self.cmbYAxis.currentText().latin1())
            IDs = self.getValuesForMetric("ShortID", no_eval=True)
            shortIDs = IDs
            if len(topofilter) != 0:
                T = self.getValuesForMetric("Topo", no_eval=True)
                idxs = []
                for (idx, topo) in enumerate(T):
                    # check if it passes the filter
                    pass_filter = filterTopology(topo, topofilter)
                    if pass_filter:
                        idxs.append(idx)

                X = numpy.take(X, idxs)
                Y = numpy.take(Y, idxs)
                shortIDs = []
                for idx in idxs:
                    shortIDs.append(IDs[idx])

            formatstring = self.txtPlotFormatString.text().latin1()
            line = drw.plot(X, Y, formatstring, picker=5)
            self.state_plots.append((self.lstOpenStates.currentText().latin1(), shortIDs))

        # this is to allow for interactive plots
        self.matplotlibWidget1.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.matplotlibWidget1.draw()
    
    def onpick(self, event):
        """ this will react when we click a point """
        thisline = event.artist
        ax = thisline.axes
        state_idx = ax.lines.index(thisline)
        state = self.state_plots[state_idx][0]
        print "clicked on line: %d for state %s" % (state_idx, state)
        # get id's for the items clicked on

        ind = event.ind
        shortids = set()
        #import pdb;pdb.set_trace()
        for i in ind:
            try:
                shortids.add(self.state_plots[state_idx][1][i])
            except:
                print "failed to get shortid for ind=%s (%s)" % (i, self.state_plots[state_idx][1])
        print " item ShortIDs are %s" % shortids


        #import pdb;pdb.set_trace()
        state_wrapper = self.state_wrappers[self.lstOpenStates.currentText().latin1()]
        col = state_wrapper.metric_map["ShortID"]
        for i in range(self.tblValues.numRows()):
            for shortid in set(shortids):
                if self.tblValues.text(i, col).latin1() == str(shortid):
                    self.tblValues.selectRow(i)
                    self.tblValues.ensureCellVisible(i,0)

        #xdata = thisline.get_xdata()
        #ydata = thisline.get_ydata()
        #print 'onpick points:', zip(xdata[ind], ydata[ind])

    def clearPlot(self):
        drw = self.matplotlibWidget1.axes
        self.state_plots = []
        drw.clear()
        self.matplotlibWidget1.draw()
        TOPO_COLOR_MAP = {}

    def sort(self):
        print "sort"
        self.columnClicked(self.tblValues.currentColumn())

    def showOpenFileDialog(self):
        file = QFileDialog.getOpenFileName()
        if file:
            self.txtFilename.setText(file)

    def save(self):
        print "saving"

    def getMetricFromInd(self, state_wrapper, name, ind):
        col = state_wrapper.metric_map[name]
        if "ShortID" == name:
            return str(ind.shortID())
        elif "Age" == name:
            return str(ind.genetic_age)
        elif "Feasible" == name:
            return str(ind.isFeasible(num_rnd_points))
        elif "Topo" == name:
           return "T" + str(ind.topoSummary())
        else:
            wc_value =  ind.worstCaseMetricValue(1, name)
            if wc_value == BAD_METRIC_VALUE:
                val_str = '%s' % BAD_METRIC_VALUE
            else:
                val_str = '%g' % wc_value
            return str(val_str)
    
    def saveData(self):
        print "saving plot data"
        file = QFileDialog.getSaveFileName()
        if file:
            fid = open(file,'w')
            plot_cnt=0
            for state_plot in self.state_plots:
                plot_cnt += 1
                #import pdb;pdb.set_trace()
                state = self.state_wrappers[state_plot[0]]
                shortids = state_plot[1]
            
                xmetric = self.cmbXAxis.currentText().latin1()
                ymetric = self.cmbYAxis.currentText().latin1()
                
                for sid in shortids:
                    idx = state.shortid_idx_map[sid]
                    ind = state.inds[idx]
                    xval = self.getMetricFromInd(state, xmetric, ind)
                    yval = self.getMetricFromInd(state, ymetric, ind)
                    
                    fid.write("%s %s %s\n" % (plot_cnt, xval, yval))
            fid.close()
                    
    def saveSheet(self):
        print "saving sheet"
        file = QFileDialog.getSaveFileName()
        if file:
            curr_state = self.lstOpenStates.currentText().latin1()
            if curr_state == "":
                print "no current state"
                return
            
            state_wrapper = self.state_wrappers[curr_state]
            
            fid = open(file,'w')
            # figure out the number of columns
            # Ind_ID GeneticAge  Feas    TopoSummary  [analysis metrics]
            nb_cols = 4
            for analysis in state_wrapper.ps.analyses:
                nb_cols += len(analysis.metrics)

            # headers
            s = ""
            for name in state_wrapper.metric_map.keys():
                s += "%s " % name

            fid.write("%s\n" % s)
            
            #   -for each ind, a row of '<ID_value> <metric1_value> <metric2_value> ...'
            for ind_idx, ind in enumerate(state_wrapper.inds):
                num_rnd_points = ind.numRndPointsFullyEvaluated()
                s = ""
                for name in state_wrapper.metric_map.keys():
                    col = state_wrapper.metric_map[name]
                    if "ShortID" == name:
                        s += "%s " % str(ind.shortID())
                    elif "Age" == name:
                        s += "%s " % str(ind.genetic_age)
                    elif "Feasible" == name:
                        s += "%s " % str(ind.isFeasible(num_rnd_points))
                    elif "Topo" == name:
                        s += "%s " % str(ind.topoSummary())
                    else:
                        wc_value =  ind.worstCaseMetricValue(num_rnd_points, name)
                        if wc_value == BAD_METRIC_VALUE:
                            val_str = '%s' % BAD_METRIC_VALUE
                        else:
                            val_str = '%g' % wc_value
                        s += "%s " % str(val_str)
                fid.write("%s\n" % s)
            fid.close()

    def saveInd(self):
        print "saving ind"
        # fetch ind
        curr_state = self.lstOpenStates.currentText().latin1()
        if curr_state == "":
            raise "no current state"

        state_wrapper = self.state_wrappers[curr_state]
        
        shortID = self.tblValues.text(self.tblValues.currentRow(), 0).latin1()
        try:
            ind_idx = state_wrapper.shortid_idx_map[shortID]
        except:
            print "shortID not found"
            return
        ind = state_wrapper.inds[ind_idx]

        # get filename
        file = QFileDialog.getSaveFileName()
        if file:
            ind.S = None
            ind._ps = None
            fid = open(file,'w')
            pickle.dump(ind, fid)
            fid.close()

    def editInd(self):
        print "editing ind"
        # fetch ind
        curr_state = self.lstOpenStates.currentText().latin1()
        if curr_state == "":
            raise "no current state"

        state_wrapper = self.state_wrappers[curr_state]
        
        shortID = self.tblValues.text(self.tblValues.currentRow(), 0).latin1()
        try:
            ind_idx = state_wrapper.shortid_idx_map[shortID]
        except:
            print "shortID not found"
            return
        ind = state_wrapper.inds[ind_idx]

        # show edit dialog
        editform = ChangeSettings(ind, self)
        editform.setModal(False)
        editform.show()

    def updateStateList(self, to_select=None):
        self.lstOpenStates.clear()
        for filename in self.state_wrappers.keys():
            self.lstOpenStates.insertItem(filename, -1)
        if to_select:
            self.lstOpenStates.setCurrentItem(self.lstOpenStates.findItem(to_select))

    def setCurrentState(self, state_idx):
        self.update()

    def update(self):
        print "updating"

        all_inds = not self.chkNondom.isChecked()
        if self.chkAgeLayer.isChecked():
            age_layer = self.spinAgeLayer.value()
            add_lower = self.chkAddLowerLayer.isChecked()
        else:
            age_layer = None
            add_lower = False

        feasible_only = self.chkFeasible.isChecked()
        topofilter = self.txtListTopoFilter.text().latin1()

        curr_state = self.lstOpenStates.currentText().latin1()
        if self.current_state != curr_state:
            self.tblValues.setNumCols(0)
            self.tblValues.setNumRows(0)
            self.clearDetails()

        if curr_state == None:
            return

        state_wrapper = self.state_wrappers[curr_state]
        self.spinAgeLayer.setMinValue(0)
        self.spinAgeLayer.setMaxValue(state_wrapper.nbAgeLayers() - 1)

        state_wrapper.update(all_inds, age_layer, add_lower, feasible_only, topofilter)

        # figure out the number of columns
        # Ind_ID GeneticAge  Feas    TopoSummary  [analysis metrics]
        nb_cols = 4
        for analysis in state_wrapper.ps.analyses:
            nb_cols += len(analysis.metrics)
            
        self.tblValues.setNumCols(nb_cols)
        self.tblValues.setNumRows(len(state_wrapper.inds))
        
        # headers
        for name in state_wrapper.metric_map.keys():
            self.tblValues.horizontalHeader().setLabel( state_wrapper.metric_map[name], name )

        #   -for each ind, a row of '<ID_value> <metric1_value> <metric2_value> ...'
        for ind_idx, ind in enumerate(state_wrapper.inds):
            num_rnd_points = ind.numRndPointsFullyEvaluated()
            for name in state_wrapper.metric_map.keys():
                col = state_wrapper.metric_map[name]
                if "ShortID" == name:
                    self.tblValues.setItem(ind_idx, col, CustomTableItem(self.tblValues, QTableItem.OnTyping, str(ind.shortID())))
                elif "Age" == name:
                    self.tblValues.setItem(ind_idx, col, CustomTableItem(self.tblValues, QTableItem.OnTyping, str(ind.genetic_age)))
                elif "Feasible" == name:
                    self.tblValues.setItem(ind_idx, col, CustomTableItem(self.tblValues, QTableItem.OnTyping, str(ind.isFeasible(num_rnd_points))))
                elif "Topo" == name:
                    self.tblValues.setItem(ind_idx, col, CustomTableItem(self.tblValues, QTableItem.OnTyping, "T" + str(ind.topoSummary())))
                else:
                    wc_value =  ind.worstCaseMetricValue(num_rnd_points, name)
                    if wc_value == BAD_METRIC_VALUE:
                        val_str = '%s' % BAD_METRIC_VALUE
                    else:
                        val_str = '%g' % wc_value
                    self.tblValues.setItem(ind_idx, col, CustomTableItem(self.tblValues, QTableItem.OnTyping, str(val_str)))

        self.preparePlottingOptions()

        for i in range(nb_cols):
            self.tblValues.adjustColumn(i)
        self.ascending = {}

        if len(state_wrapper.inds):
            self.showDetails(state_wrapper.inds[0])

        self.current_state = curr_state

    def loadState(self):
        db_file = self.txtFilename.text().latin1()
        print "load %s" % db_file

        # -load data
        if not os.path.exists(db_file):
            print "Cannot find file with name %s" % db_file
            return
        
        state = EngineUtils.loadSynthState(db_file, None)

        reference_ind = state.R_per_age_layer[-1][0]
        rnd_ID = reference_ind.rnd_IDs[0]
        metric_names_measured = reference_ind.sim_results[rnd_ID].keys()
        old_ps_metric_names = state.ps.flattenedMetricNames()
        state.ps.stripToSpecifiedMetrics(metric_names_measured)
        print "Stripped the following metrics from ps which weren't in inds: %s" % \
            mathutil.listDiff(old_ps_metric_names, metric_names_measured)

        # this will automatically remove a state that is reloaded
        self.state_wrappers[db_file] = StateWrapper(state)

        self.updateStateList(db_file)

        self.update()

    def closeState(self):
        print "closing"

        curr_state = self.lstOpenStates.currentText().latin1()
        if curr_state == None:
            return

        del self.state_wrappers[curr_state]
        self.updateStateList()
        self.update()

    def closeAllStates(self):
        print "closing all states"

        self.state_wrappers = {}
        self.updateStateList()
        self.update()

    def columnClicked(self, col):
        #print "sort on col %s" % col
        if not col in self.ascending.keys():
            self.ascending[col] = None
            
        if self.ascending[col] == None:
            self.ascending[col] = True
        elif self.ascending[col] == True:
            self.ascending[col] = False
        else:
            self.ascending[col] = True

        self.tblValues.sortColumn(col, self.ascending[col], True )

if __name__== '__main__':
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.DEBUG)
    logging.getLogger('analysis').setLevel(logging.DEBUG)

    #set help message
    help = """
Usage: QtResultBrowse [dbfile]

    Provides a Qt UI to explore the result database

Details:
    dbfile -- database to open at startup
    
"""
    num_args = len(sys.argv)

    #yank out the args into usable values
    db_file = None
    if num_args == 2:
        db_file = sys.argv[1]

    app = QApplication(sys.argv)
    form = QtResultBrowse(db_file)
    form.show()
    
    QObject.connect(app, SIGNAL("lastWindowClosed()"), app, SLOT("quit()"))
    
    print "start main loop"
    app.exec_loop()
    
