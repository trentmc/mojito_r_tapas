# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/QtResultBrowseUi.ui'
#
# Created: Thu Sep 11 18:38:28 2008
#      by: The PyQt User Interface Compiler (pyuic) 3.17.3
#
# WARNING! All changes made in this file will be lost!


from qt import *
from qttable import QTable
from mplwidget import *


class QtResultBrowseUi(QDialog):
    def __init__(self,parent = None,name = None,modal = 0,fl = 0):
        QDialog.__init__(self,parent,name,modal,fl)

        if not name:
            self.setName("QtResultBrowseUi")


        QtResultBrowseUiLayout = QGridLayout(self,1,1,11,6,"QtResultBrowseUiLayout")

        self.splitter5 = QSplitter(self,"splitter5")
        self.splitter5.setOrientation(QSplitter.Vertical)

        self.splitter4 = QSplitter(self.splitter5,"splitter4")
        self.splitter4.setOrientation(QSplitter.Horizontal)

        self.tblValues = QTable(self.splitter4,"tblValues")
        self.tblValues.setSizePolicy(QSizePolicy(QSizePolicy.Minimum,QSizePolicy.Expanding,0,0,self.tblValues.sizePolicy().hasHeightForWidth()))
        tblValues_font = QFont(self.tblValues.font())
        tblValues_font.setFamily("Lucida Sans Typewriter")
        self.tblValues.setFont(tblValues_font)
        self.tblValues.setResizePolicy(QTable.Default)
        self.tblValues.setNumRows(23)
        self.tblValues.setNumCols(7)
        self.tblValues.setSorting(0)

        self.tabWidget2 = QTabWidget(self.splitter4,"tabWidget2")

        self.tab = QWidget(self.tabWidget2,"tab")
        tabLayout = QGridLayout(self.tab,1,1,11,6,"tabLayout")

        self.tblDetails = QTable(self.tab,"tblDetails")
        tblDetails_font = QFont(self.tblDetails.font())
        tblDetails_font.setFamily("Lucida Sans Typewriter")
        self.tblDetails.setFont(tblDetails_font)
        self.tblDetails.setNumRows(3)
        self.tblDetails.setNumCols(3)

        tabLayout.addWidget(self.tblDetails,0,0)
        self.tabWidget2.insertTab(self.tab,QString.fromLatin1(""))

        self.TabPage = QWidget(self.tabWidget2,"TabPage")
        TabPageLayout = QGridLayout(self.TabPage,1,1,11,6,"TabPageLayout")

        self.txtPointSummary = QTextEdit(self.TabPage,"txtPointSummary")
        txtPointSummary_font = QFont(self.txtPointSummary.font())
        txtPointSummary_font.setFamily("Lucida Sans Typewriter")
        self.txtPointSummary.setFont(txtPointSummary_font)
        self.txtPointSummary.setWordWrap(QTextEdit.NoWrap)

        TabPageLayout.addWidget(self.txtPointSummary,0,0)
        self.tabWidget2.insertTab(self.TabPage,QString.fromLatin1(""))

        self.tab_2 = QWidget(self.tabWidget2,"tab_2")
        tabLayout_2 = QVBoxLayout(self.tab_2,11,6,"tabLayout_2")

        self.txtNetlist = QTextEdit(self.tab_2,"txtNetlist")
        txtNetlist_font = QFont(self.txtNetlist.font())
        txtNetlist_font.setFamily("Lucida Sans Typewriter")
        self.txtNetlist.setFont(txtNetlist_font)
        self.txtNetlist.setWordWrap(QTextEdit.NoWrap)
        tabLayout_2.addWidget(self.txtNetlist)

        layout3 = QHBoxLayout(None,0,6,"layout3")

        self.chkBlockInfo = QCheckBox(self.tab_2,"chkBlockInfo")
        layout3.addWidget(self.chkBlockInfo)

        self.chkInfoString = QCheckBox(self.tab_2,"chkInfoString")
        layout3.addWidget(self.chkInfoString)
        tabLayout_2.addLayout(layout3)
        self.tabWidget2.insertTab(self.tab_2,QString.fromLatin1(""))

        self.TabPage_2 = QWidget(self.tabWidget2,"TabPage_2")
        TabPageLayout_2 = QVBoxLayout(self.TabPage_2,11,6,"TabPageLayout_2")

        self.txtIndString = QTextEdit(self.TabPage_2,"txtIndString")
        txtIndString_font = QFont(self.txtIndString.font())
        txtIndString_font.setFamily("Lucida Sans Typewriter")
        self.txtIndString.setFont(txtIndString_font)
        self.txtIndString.setWordWrap(QTextEdit.WidgetWidth)
        self.txtIndString.setWrapPolicy(QTextEdit.AtWordBoundary)
        TabPageLayout_2.addWidget(self.txtIndString)
        self.tabWidget2.insertTab(self.TabPage_2,QString.fromLatin1(""))

        self.TabPage_3 = QWidget(self.tabWidget2,"TabPage_3")
        TabPageLayout_3 = QVBoxLayout(self.TabPage_3,11,6,"TabPageLayout_3")

        self.matplotlibWidget1 = MatplotlibWidget(self.TabPage_3,"matplotlibWidget1")
        self.matplotlibWidget1.setSizePolicy(QSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred,0,0,self.matplotlibWidget1.sizePolicy().hasHeightForWidth()))
        TabPageLayout_3.addWidget(self.matplotlibWidget1)

        self.groupBox3 = QGroupBox(self.TabPage_3,"groupBox3")
        self.groupBox3.setSizePolicy(QSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum,0,0,self.groupBox3.sizePolicy().hasHeightForWidth()))
        self.groupBox3.setColumnLayout(0,Qt.Vertical)
        self.groupBox3.layout().setSpacing(6)
        self.groupBox3.layout().setMargin(11)
        groupBox3Layout = QGridLayout(self.groupBox3.layout())
        groupBox3Layout.setAlignment(Qt.AlignTop)

        layout10 = QGridLayout(None,1,1,0,6,"layout10")

        self.textLabel1_2 = QLabel(self.groupBox3,"textLabel1_2")
        self.textLabel1_2.setSizePolicy(QSizePolicy(QSizePolicy.Maximum,QSizePolicy.Preferred,0,0,self.textLabel1_2.sizePolicy().hasHeightForWidth()))

        layout10.addWidget(self.textLabel1_2,1,0)

        self.cmbXAxis = QComboBox(0,self.groupBox3,"cmbXAxis")

        layout10.addWidget(self.cmbXAxis,0,1)

        self.textLabel1 = QLabel(self.groupBox3,"textLabel1")
        self.textLabel1.setSizePolicy(QSizePolicy(QSizePolicy.Maximum,QSizePolicy.Preferred,0,0,self.textLabel1.sizePolicy().hasHeightForWidth()))

        layout10.addWidget(self.textLabel1,0,0)

        self.cmbYAxis = QComboBox(0,self.groupBox3,"cmbYAxis")

        layout10.addWidget(self.cmbYAxis,1,1)

        groupBox3Layout.addMultiCellLayout(layout10,0,0,0,1)

        layout21 = QVBoxLayout(None,0,6,"layout21")

        layout19 = QHBoxLayout(None,0,6,"layout19")

        self.chkHoldPlot = QCheckBox(self.groupBox3,"chkHoldPlot")
        layout19.addWidget(self.chkHoldPlot)

        self.txtPlotFormatString = QLineEdit(self.groupBox3,"txtPlotFormatString")
        layout19.addWidget(self.txtPlotFormatString)
        layout21.addLayout(layout19)

        layout18 = QHBoxLayout(None,0,6,"layout18")

        self.chkTopoPlot = QCheckBox(self.groupBox3,"chkTopoPlot")
        layout18.addWidget(self.chkTopoPlot)

        layout15 = QHBoxLayout(None,0,6,"layout15")

        self.txtTopoMask = QLineEdit(self.groupBox3,"txtTopoMask")
        layout15.addWidget(self.txtTopoMask)

        self.txtTopoFilter = QLineEdit(self.groupBox3,"txtTopoFilter")
        layout15.addWidget(self.txtTopoFilter)
        layout18.addLayout(layout15)
        layout21.addLayout(layout18)

        layout17 = QHBoxLayout(None,0,6,"layout17")

        self.chkClustering = QCheckBox(self.groupBox3,"chkClustering")
        layout17.addWidget(self.chkClustering)

        self.chkFrontOnly = QCheckBox(self.groupBox3,"chkFrontOnly")
        layout17.addWidget(self.chkFrontOnly)

        self.chkNonFront = QCheckBox(self.groupBox3,"chkNonFront")
        layout17.addWidget(self.chkNonFront)
        layout21.addLayout(layout17)

        groupBox3Layout.addLayout(layout21,1,0)

        layout20 = QVBoxLayout(None,0,6,"layout20")

        self.btnUpdatePlot = QPushButton(self.groupBox3,"btnUpdatePlot")
        layout20.addWidget(self.btnUpdatePlot)

        self.btnClearPlot = QPushButton(self.groupBox3,"btnClearPlot")
        layout20.addWidget(self.btnClearPlot)

        self.btnSaveData = QPushButton(self.groupBox3,"btnSaveData")
        layout20.addWidget(self.btnSaveData)

        groupBox3Layout.addLayout(layout20,1,1)
        TabPageLayout_3.addWidget(self.groupBox3)
        self.tabWidget2.insertTab(self.TabPage_3,QString.fromLatin1(""))

        self.splitter4_2 = QSplitter(self.splitter5,"splitter4_2")
        self.splitter4_2.setOrientation(QSplitter.Horizontal)

        LayoutWidget = QWidget(self.splitter4_2,"layout20")
        layout20_2 = QVBoxLayout(LayoutWidget,11,6,"layout20_2")

        layout16 = QHBoxLayout(None,0,6,"layout16")

        self.btnSort = QPushButton(LayoutWidget,"btnSort")
        layout16.addWidget(self.btnSort)
        spacer4 = QSpacerItem(161,21,QSizePolicy.Expanding,QSizePolicy.Minimum)
        layout16.addItem(spacer4)
        layout20_2.addLayout(layout16)

        layout19_2 = QHBoxLayout(None,0,6,"layout19_2")

        self.btnIndEdit = QPushButton(LayoutWidget,"btnIndEdit")
        layout19_2.addWidget(self.btnIndEdit)
        spacer4_2 = QSpacerItem(250,21,QSizePolicy.Expanding,QSizePolicy.Minimum)
        layout19_2.addItem(spacer4_2)

        self.btnSheetSaveAs = QPushButton(LayoutWidget,"btnSheetSaveAs")
        layout19_2.addWidget(self.btnSheetSaveAs)

        self.btnIndSaveAs = QPushButton(LayoutWidget,"btnIndSaveAs")
        layout19_2.addWidget(self.btnIndSaveAs)
        layout20_2.addLayout(layout19_2)
        spacer5 = QSpacerItem(20,106,QSizePolicy.Minimum,QSizePolicy.Expanding)
        layout20_2.addItem(spacer5)

        self.groupBox3_2 = QGroupBox(LayoutWidget,"groupBox3_2")
        self.groupBox3_2.setColumnLayout(0,Qt.Vertical)
        self.groupBox3_2.layout().setSpacing(6)
        self.groupBox3_2.layout().setMargin(11)
        groupBox3_2Layout = QVBoxLayout(self.groupBox3_2.layout())
        groupBox3_2Layout.setAlignment(Qt.AlignTop)

        layout14 = QHBoxLayout(None,0,6,"layout14")

        self.chkNondom = QCheckBox(self.groupBox3_2,"chkNondom")
        self.chkNondom.setChecked(1)
        layout14.addWidget(self.chkNondom)

        self.chkFeasible = QCheckBox(self.groupBox3_2,"chkFeasible")
        self.chkFeasible.setChecked(1)
        layout14.addWidget(self.chkFeasible)
        groupBox3_2Layout.addLayout(layout14)

        layout20_3 = QHBoxLayout(None,0,6,"layout20_3")

        self.chkAgeLayer = QCheckBox(self.groupBox3_2,"chkAgeLayer")
        layout20_3.addWidget(self.chkAgeLayer)

        self.chkAddLowerLayer = QCheckBox(self.groupBox3_2,"chkAddLowerLayer")
        layout20_3.addWidget(self.chkAddLowerLayer)

        self.spinAgeLayer = QSpinBox(self.groupBox3_2,"spinAgeLayer")
        layout20_3.addWidget(self.spinAgeLayer)
        groupBox3_2Layout.addLayout(layout20_3)

        layout20_4 = QHBoxLayout(None,0,6,"layout20_4")

        self.textLabel1_3 = QLabel(self.groupBox3_2,"textLabel1_3")
        layout20_4.addWidget(self.textLabel1_3)

        self.txtListTopoFilter = QLineEdit(self.groupBox3_2,"txtListTopoFilter")
        layout20_4.addWidget(self.txtListTopoFilter)

        self.btnUpdateListTopoFilter = QPushButton(self.groupBox3_2,"btnUpdateListTopoFilter")
        layout20_4.addWidget(self.btnUpdateListTopoFilter)
        groupBox3_2Layout.addLayout(layout20_4)
        layout20_2.addWidget(self.groupBox3_2)

        LayoutWidget_2 = QWidget(self.splitter4_2,"layout17")
        layout17_2 = QHBoxLayout(LayoutWidget_2,11,6,"layout17_2")

        layout16_2 = QVBoxLayout(None,0,6,"layout16_2")

        self.btnUpdate = QPushButton(LayoutWidget_2,"btnUpdate")
        layout16_2.addWidget(self.btnUpdate)
        spacer3 = QSpacerItem(20,110,QSizePolicy.Minimum,QSizePolicy.Expanding)
        layout16_2.addItem(spacer3)

        self.btnCloseAll = QPushButton(LayoutWidget_2,"btnCloseAll")
        layout16_2.addWidget(self.btnCloseAll)

        self.btnClose = QPushButton(LayoutWidget_2,"btnClose")
        layout16_2.addWidget(self.btnClose)

        self.btnSave = QPushButton(LayoutWidget_2,"btnSave")
        layout16_2.addWidget(self.btnSave)

        self.btnLoad = QPushButton(LayoutWidget_2,"btnLoad")
        self.btnLoad.setDefault(1)
        layout16_2.addWidget(self.btnLoad)
        layout17_2.addLayout(layout16_2)

        layout12 = QVBoxLayout(None,0,6,"layout12")

        self.lstOpenStates = QListBox(LayoutWidget_2,"lstOpenStates")
        layout12.addWidget(self.lstOpenStates)

        layout11 = QHBoxLayout(None,0,6,"layout11")

        self.textLabel2 = QLabel(LayoutWidget_2,"textLabel2")
        layout11.addWidget(self.textLabel2)

        self.txtFilename = QLineEdit(LayoutWidget_2,"txtFilename")
        self.txtFilename.setSizePolicy(QSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed,0,0,self.txtFilename.sizePolicy().hasHeightForWidth()))
        layout11.addWidget(self.txtFilename)

        self.btnOpenFile = QPushButton(LayoutWidget_2,"btnOpenFile")
        self.btnOpenFile.setSizePolicy(QSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed,0,0,self.btnOpenFile.sizePolicy().hasHeightForWidth()))
        self.btnOpenFile.setMaximumSize(QSize(30,32767))
        layout11.addWidget(self.btnOpenFile)
        layout12.addLayout(layout11)
        layout17_2.addLayout(layout12)

        QtResultBrowseUiLayout.addWidget(self.splitter5,0,0)

        self.languageChange()

        self.resize(QSize(1299,877).expandedTo(self.minimumSizeHint()))
        self.clearWState(Qt.WState_Polished)


    def languageChange(self):
        self.setCaption(self.__tr("Form1"))
        self.tabWidget2.changeTab(self.tab,self.__tr("Point"))
        self.tabWidget2.changeTab(self.TabPage,self.__tr("PointSummary"))
        self.chkBlockInfo.setText(self.__tr("Block Info"))
        self.chkInfoString.setText(self.__tr("Info String"))
        self.tabWidget2.changeTab(self.tab_2,self.__tr("Netlist"))
        self.tabWidget2.changeTab(self.TabPage_2,self.__tr("Ind"))
        self.groupBox3.setTitle(self.__tr("Options"))
        self.textLabel1_2.setText(self.__tr("Y-Axis"))
        self.textLabel1.setText(self.__tr("X-Axis"))
        self.chkHoldPlot.setText(self.__tr("Hold plots"))
        self.txtPlotFormatString.setText(self.__tr("."))
        QToolTip.add(self.txtPlotFormatString,self.__tr("Format string"))
        self.chkTopoPlot.setText(self.__tr("Topo plot"))
        QToolTip.add(self.txtTopoMask,self.__tr("topo mask"))
        QToolTip.add(self.txtTopoFilter,self.__tr("topo filter"))
        self.chkClustering.setText(self.__tr("Show clustering"))
        self.chkFrontOnly.setText(self.__tr("Front only"))
        self.chkNonFront.setText(self.__tr("Non-front"))
        self.btnUpdatePlot.setText(self.__tr("Plot"))
        self.btnClearPlot.setText(self.__tr("Clear Plot"))
        self.btnSaveData.setText(self.__tr("Save data"))
        self.tabWidget2.changeTab(self.TabPage_3,self.__tr("Plot"))
        self.btnSort.setText(self.__tr("Sort"))
        self.btnIndEdit.setText(self.__tr("Edit Ind"))
        self.btnSheetSaveAs.setText(self.__tr("Save Sheet as"))
        self.btnIndSaveAs.setText(self.__tr("Save Ind As"))
        self.groupBox3_2.setTitle(self.__tr("Set selection"))
        self.chkNondom.setText(self.__tr("Non dominated only"))
        self.chkFeasible.setText(self.__tr("Feasible only"))
        self.chkAgeLayer.setText(self.__tr("Age layer"))
        self.chkAddLowerLayer.setText(self.__tr("Include one lower layer"))
        self.textLabel1_3.setText(self.__tr("Topology filter"))
        QToolTip.add(self.txtListTopoFilter,self.__tr("topo filter"))
        self.btnUpdateListTopoFilter.setText(self.__tr("Update"))
        self.btnUpdate.setText(self.__tr("update"))
        self.btnCloseAll.setText(self.__tr("Close all"))
        self.btnClose.setText(self.__tr("Close"))
        self.btnSave.setText(self.__tr("Save"))
        self.btnLoad.setText(self.__tr("Load"))
        self.textLabel2.setText(self.__tr("Filename:"))
        self.btnOpenFile.setText(self.__tr("..."))


    def __tr(self,s,c = None):
        return qApp.translate("QtResultBrowseUi",s,c)
