# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/change_ind_qt_ui.ui'
#
# Created: Fri Aug 1 14:36:29 2008
#      by: The PyQt User Interface Compiler (pyuic) 3.17.3
#
# WARNING! All changes made in this file will be lost!


from qt import *
from qttable import QTable


class ChangeSettingsUI(QDialog):
    def __init__(self,parent = None,name = None,modal = 0,fl = 0):
        QDialog.__init__(self,parent,name,modal,fl)

        if not name:
            self.setName("Form1")


        Form1Layout = QVBoxLayout(self,11,6,"Form1Layout")

        self.splitter4 = QSplitter(self,"splitter4")
        self.splitter4.setOrientation(QSplitter.Horizontal)

        LayoutWidget = QWidget(self.splitter4,"layout29")
        layout29 = QGridLayout(LayoutWidget,1,1,11,6,"layout29")

        self.tblValues = QTable(LayoutWidget,"tblValues")
        self.tblValues.setResizePolicy(QTable.Default)
        self.tblValues.setNumRows(23)
        self.tblValues.setNumCols(7)
        self.tblValues.setSorting(0)

        layout29.addMultiCellWidget(self.tblValues,0,0,0,2)

        self.groupBox1 = QGroupBox(LayoutWidget,"groupBox1")
        self.groupBox1.setColumnLayout(0,Qt.Vertical)
        self.groupBox1.layout().setSpacing(6)
        self.groupBox1.layout().setMargin(11)
        groupBox1Layout = QGridLayout(self.groupBox1.layout())
        groupBox1Layout.setAlignment(Qt.AlignTop)

        layout5 = QGridLayout(None,1,1,0,6,"layout5")

        self.spinAnalysis = QSpinBox(self.groupBox1,"spinAnalysis")

        layout5.addWidget(self.spinAnalysis,1,1)

        self.textLabel1_2 = QLabel(self.groupBox1,"textLabel1_2")

        layout5.addWidget(self.textLabel1_2,2,0)

        self.chkSimulatable = QCheckBox(self.groupBox1,"chkSimulatable")

        layout5.addWidget(self.chkSimulatable,0,0)

        self.textLabel1 = QLabel(self.groupBox1,"textLabel1")

        layout5.addWidget(self.textLabel1,1,0)

        self.spinEnv = QSpinBox(self.groupBox1,"spinEnv")

        layout5.addWidget(self.spinEnv,2,1)

        groupBox1Layout.addLayout(layout5,0,0)

        layout8 = QVBoxLayout(None,0,6,"layout8")

        layout7 = QHBoxLayout(None,0,6,"layout7")

        self.chkAnnBB = QCheckBox(self.groupBox1,"chkAnnBB")
        layout7.addWidget(self.chkAnnBB)

        self.chkAnnPoint = QCheckBox(self.groupBox1,"chkAnnPoint")
        layout7.addWidget(self.chkAnnPoint)
        layout8.addLayout(layout7)

        layout4 = QHBoxLayout(None,0,6,"layout4")

        self.textLabel2 = QLabel(self.groupBox1,"textLabel2")
        layout4.addWidget(self.textLabel2)

        self.txtFilename = QLineEdit(self.groupBox1,"txtFilename")
        self.txtFilename.setSizePolicy(QSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed,0,0,self.txtFilename.sizePolicy().hasHeightForWidth()))
        layout4.addWidget(self.txtFilename)

        self.btnOpenFile = QPushButton(self.groupBox1,"btnOpenFile")
        self.btnOpenFile.setSizePolicy(QSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed,0,0,self.btnOpenFile.sizePolicy().hasHeightForWidth()))
        self.btnOpenFile.setMaximumSize(QSize(30,32767))
        layout4.addWidget(self.btnOpenFile)
        layout8.addLayout(layout4)

        self.btnNetlist = QPushButton(self.groupBox1,"btnNetlist")
        layout8.addWidget(self.btnNetlist)

        groupBox1Layout.addLayout(layout8,0,1)

        layout29.addMultiCellWidget(self.groupBox1,2,2,0,2)

        self.btnSave = QPushButton(LayoutWidget,"btnSave")

        layout29.addWidget(self.btnSave,1,1)

        self.btnUpdate = QPushButton(LayoutWidget,"btnUpdate")

        layout29.addWidget(self.btnUpdate,1,0)

        self.btnPrint = QPushButton(LayoutWidget,"btnPrint")

        layout29.addWidget(self.btnPrint,1,2)

        self.tabWidget4 = QTabWidget(self.splitter4,"tabWidget4")

        self.tab = QWidget(self.tabWidget4,"tab")
        tabLayout = QVBoxLayout(self.tab,11,6,"tabLayout")

        self.txtNetlist = QTextEdit(self.tab,"txtNetlist")
        self.txtNetlist.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Expanding,0,0,self.txtNetlist.sizePolicy().hasHeightForWidth()))
        self.txtNetlist.setWordWrap(QTextEdit.NoWrap)
        self.txtNetlist.setWrapPolicy(QTextEdit.AtWordBoundary)
        tabLayout.addWidget(self.txtNetlist)
        self.tabWidget4.insertTab(self.tab,QString.fromLatin1(""))
        Form1Layout.addWidget(self.splitter4)

        self.languageChange()

        self.resize(QSize(854,621).expandedTo(self.minimumSizeHint()))
        self.clearWState(Qt.WState_Polished)


    def languageChange(self):
        self.setCaption(self.__tr("Form1"))
        self.groupBox1.setTitle(self.__tr("Netlist"))
        self.textLabel1_2.setText(self.__tr("Env point:"))
        self.chkSimulatable.setText(self.__tr("Simulatable"))
        self.textLabel1.setText(self.__tr("Analysis:"))
        self.chkAnnBB.setText(self.__tr("Annotate BB"))
        self.chkAnnPoint.setText(self.__tr("Annotate Point"))
        self.textLabel2.setText(self.__tr("Filename:"))
        self.btnOpenFile.setText(self.__tr("..."))
        self.btnNetlist.setText(self.__tr("Netlist"))
        self.btnSave.setText(self.__tr("save"))
        self.btnUpdate.setText(self.__tr("update"))
        self.btnPrint.setText(self.__tr("print"))
        self.tabWidget4.changeTab(self.tab,self.__tr("Netlist"))


    def __tr(self,s,c = None):
        return qApp.translate("ChangeSettingsUI",s,c)
