import os
import time

#AGGR_TEST will invoke more validations during a run.  It should be False
# unless thorough testing is needed.  Note that unit tests automatically
# set AGGR_TEST to True, by calling setAggressiveTests().
AGGR_TEST = None
def setAggressiveTests():
    global AGGR_TEST
    AGGR_TEST = True
if AGGR_TEST is None:
    AGGR_TEST = False

#MOS operating regions
REGION_LINEAR     = 0
REGION_SATURATION = 1
REGION_CUTOFF     = 2

#conveniently set infinity
INF = float('Inf')

#this is the metric name that the _simulation_ (not function) DOCs get
DOCS_METRIC_NAME = 'sim_DOCs_cost'

#This used to have a fancier definition, but it was not pickle-able.  This version works!
BAD_METRIC_VALUE = 'BAD_METRIC_VALUE' 

#Where we do simulations -- see Analysis.py
while True:
    SIMFILE_DIR = '/tmp/sim_mojito-' + str(time.time()) + '/'
    if not os.path.exists(SIMFILE_DIR):
        break
os.mkdir(SIMFILE_DIR)

#these are the file extensions that might have waveforms
WAVEFORMS_FILE_EXTENSIONS = ['sw0','tr0']

#these describe the problems defined in problems/Problems.py.  Update
# this list when those change.
PROBLEM_DESCRIPTIONS = """
    Legend: ** = recommended; x = not currently;  WL = SizingLibrary; OP = OpLibrary, OP2 = OpLibrary2; DS = diff-in, single-ended-out, 1 stage amp; DSS = diff-in, single-middle, single-out, 2 stage; DDS = diff-in, diff-middle, single-out, 2 stage; note that we do not count yield objective)

    ----PROBLEM_NUM choices-----
    1       - maximizePartCountProblem, 180
    2 .. 9  - maxPartCount_minArea_Problem (higher means larger), 180
    10 / 11 - func  sim ten-objective test problems
    12 / 13 - func / func+sim DOCs test problem, 180
    15      - 2d sphere problem
    31      - SS, WL,  5 objs, 180nm, no GBW
 ** 33      - SS, OP,  5 objs, 180nm, no GBW **
 ** 34      - SS, OP,  5 objs,  90nm, no GBW **
    39      - SS, OP2, 5 objs, 180nm, no GBW
    41 / 42 - DS,  WL / OP, 6 objs, 180nm, no GBW
    51 / 52 - DSS, WL / OP, 6 objs, 180nm, no GBW
  x 61      - DS+DSS, WL,  6 objs, 180nm, lin GBW
    62      - DS+DSS, OP,  6 objs, 180nm, lin GBW
 ** 63      - DS+DSS, OP,  6 objs, 180nm, log GBW **
 ** 64      - DS+DSS, OP,  6 objs,  90nm, log GBW **
    69      - DS+DSS, OP2, 6 objs, 180nm, lin GBW
  x 71 / 72 - DS+DSS+DDS, (WL / OP), 180nm, lin GBW
    81 / 82 / 83 - minimizeNmseOnTarget (fixed /  hockey / bump waveform), 180
    84      - on bump shape, optimize transient measures (for testing), 180
    100     - current mirror EMC
    101     - SS, OP,  5 objs, 180nm, no GBW, noveltytest

"""

#=============================================================================================
#Tech info.  This is used by DevicesSetup, and possibly elsewhere.
# Supported technologies: UMC180, UMC90

#---------------------------------------------------------------------------------------
#Capacitor tech info (capacitor type: MIM)
CAP_TARGET_MATCHING_SPEC = 1.0 #units: %.  i.e. make C's area big enough to hit 1%, so matching is ok
CAP_SIGMA_MATCHING = {'UMC180' : 4.29, 'UMC90' : 3.38} #units: % * um.   From UMC datasheets.
CAP_DENSITY        = {'UMC180' : 1,    'UMC90' : 1}    #units: fF/um^2.  From UMC datasheets.
DEFAULT_MIN_C = 0.5e-12  #0.5 pF
DEFAULT_MAX_C = 50.0e-12 #50 pF

#---------------------------------------------------------------------------------------
#Resistor tech info (resistor type: high-resistance poly)
RES_TARGET_MATCHING_SPEC = 1.0 #units: %.  i.e. make R's area big enough to hit 1%, so matching is ok
RES_SIGMA_MATCHING   = {'UMC180' : 20.0,  'UMC90' : 20.0}  #units: % * um.  From UMC datasheets.
RES_SHEET_RESISTANCE = {'UMC180' : 1.0,   'UMC90' : 1.0}   #units: kohms / square.  From UMC datasheets.
RES_AREA_PER_SQUARE  = {'UMC180' : (5*180e-9)**2, 'UMC90' : (5*90e-9)**2} #units: m^2/sq
DEFAULT_MIN_R = 1      #1 ohm
DEFAULT_MAX_R = 100e3  #100 kohm

#---------------------------------------------------------------------------------------
#Transistor tech info
NMOS_BASE_VTH0 = {'UMC180' : 3.0750e-1,  'UMC90' : -1.0000e-3} #units: volts.  From pre-template model.
PMOS_BASE_VTH0 = {'UMC180' : -4.5550e-1, 'UMC90' : -5.8100e-2} #units: volts.  From pre-template model.

#By changing minL_compared_to_180, we can estimate variations at 90, 65, 45 nm:
#  Set it to 1.0 for 180nm, 2.0 for 90nm, 2.77 for 65nm, 4.0 for 45nm. 
MINL_COMPARED_TO_180 = {'UMC180': 1.0, 'UMC90' : 2.0} 
AVT_AT_180 = 6e-9 #units: unitless.  From ITRS 2007.

#These are defined below: NMOS_TEMPLATE_MODEL[UMC180 or UMC90], PMOS_TEMPLATE_MODEL[UMC180 or UMC90]
#In each *_TEMPLATE_MODEL[*], replace:
#  -'REPLACEME_MODELNAME' with chosen model name
#  -'REPLACEME_VTH0' with random.gauss({N,P}MOS_BASE_VTH0, VTH_stddev)

FEATURE_SIZE = {'UMC180' : 180.0e-9, 'UMC90' : 90.0e-9} #units: m
VDD = {'UMC180' : 1.8, 'UMC90' : 1.0} #units: volts

# NMOS_APPROX_MOS_MODEL_FILEBASE = {'UMC180' : 'problems/miller2/nmos_data_set2',
#                                   'UMC90' : 'FIXME'} 
# PMOS_APPROX_MOS_MODEL_FILEBASE = {'UMC180' : 'problems/miller2/pmos_data_set2',
#                                   'UMC90' : 'FIXME'} 

NMOS_APPROX_MOS_MODEL_FILEBASE = {'UMC180' : 'problems/miller2/LUC_N_18_MM_1',
                                  'UMC90' : 'FIXME'} 
PMOS_APPROX_MOS_MODEL_FILEBASE = {'UMC180' : 'problems/miller2/LUC_P_18_MM_1',
                                  'UMC90' : 'FIXME'} 

#---------------------------------------------------------------------------------------
#Future tech info (not currently used, but useful for reference)
#-taken from ITRS 2007, "Wireless" document, p. 14 Table RFAMS3a "On Chip Passives Tech Requirements"
#
#                                          Year: 2007 2008 2009 2010 2011 2012 2013 2014 2015
#                           DRAM 1/2 pitch (nm): 65   57   50   45   40   36   32   28   25
#               MIM Capacitor density (fF/um^2): 2    4    4    5    5    5    7    7    7
#           MIM Capacitor sigma matching (%*um): 0.5  0.5  0.5  0.4  0.4  0.4  0.3  0.3  0.3
#     P+ Poly-Si Resistor sigma matching (%*um): 1.7  1.7  1.7  1.7  1.7  1.7  1    1    1
#P+ Poly-Si Resistor sheet resistance (ohms/sq): 200- 200- 200- 200- 200- 200- 200- 200- 200- 
#                                                300  300  300  300  300  300  300  300  300  


#---------------------------------------------------------------------------------------
#Define MOS template models
NMOS_TEMPLATE_MODEL, PMOS_TEMPLATE_MODEL = {}, {} #fill these in

NMOS_TEMPLATE_MODEL['UMC180'] = """

.model REPLACEME_MODELNAME NMOS
 *****Model Selectors/Controllers*********************************
 +  LEVEL     =   4.9000E+01                          VERSION   =   3.2000E+00
 +  BINUNIT   =   1.0000E+00                          MOBMOD    =   1.0000E+00
 +  CAPMOD    =   2.0000E+00                          NQSMOD    =   0.0000E+00


 *****Process Parameters******************************************
 +  TOX       =   4.2000E-09                          TOXM      =   4.2000E-09
 +  XJ        =   1.6000E-07                          NCH       =   3.7446E+17
 +  RSH       =   8.0000E+00                          NGATE     =   1.0000E+23


 *****Basic Model Parameters**************************************
 +  VTH0      =  REPLACEME_VTH0           K1        =   4.5780E-01
 +  K2        =  -2.6380E-02                          K3        =  -1.0880E+01
 +  K3B       =   2.3790E-01                          W0        =  -8.8130E-08
 +  NLX       =   4.2790E-07                          DVT0      =   4.0420E-01
 +  DVT1      =   3.2370E-01                          DVT2      =  -8.6020E-01
 +  DVT0W     =   3.8300E-01                          DVT1W     =   6.0000E+05
 +  DVT2W     =  -2.5000E-02                          LINT      =   1.5870E-08
 +  WINT      =   1.0220E-08                          DWG       =  -3.3960E-09
 +  DWB       =   1.3460E-09                          U0        =  3.1410E+02
 +  UA        =  -9.2010E-10                          UB        =   1.9070E-18
 +  UC        =   4.3550E-11                          VSAT      =   7.1580E+04
 +  A0        =   1.9300E+00                          AGS       =   5.0720E-01
 +  B0        =   1.4860E-06                          B1        =   9.0640E-06
 +  KETA      =   1.7520E-02                          A1        =   0.0000E+00
 +  A2        =   1.0000E+00                          VOFF      =  -1.0880E-01
 +  NFACTOR   =   1.0380E+00                          CIT       =  -1.5110E-03
 +  CDSC      =   2.1750E-03                          CDSCD     =  -5.0000E-04
 +  CDSCB     =   8.2410E-04                          ETA0      =   1.0040E-03
 +  ETAB      =  -1.4590E-03                          DSUB      =   1.5920E-03
 +  PCLM      =   1.0910E+00                          PDIBLC1   =   3.0610E-03
 +  PDIBLC2   =   1.0000E-06                          PDIBLCB   =   0.0000E+00
 +  DROUT     =   1.5920E-03                          PSCBE1    =   4.8660E+08
 +  PSCBE2    =   2.8000E-07                          PVAG      =  -2.9580E-01


 *****Parameters for Asymmetric and Bias-Dependent Rds Model******
 +  RDSW      =   4.9050E+00                          PRWG      =   0.0000E+00
 +  PRWB      =   0.0000E+00                          WR        =   1.0000E+00


 *****Impact Ionization Current Model Parameters******************
 +  ALPHA0    =   0.0000E+00                          ALPHA1    =   0.0000E+00
 +  BETA0     =   3.0000E+01

 *****Charge and Capacitance Model Parameters*********************
 +  XPART     =   1.0000E+00                          CGSO      =   2.3500E-10
 +  CGDO      =   2.3500E-10                          CGBO      =   0.0000E+00
 +  CGSL      =   0.0000E+00                          CGDL      =   0.0000E+00
 +  CKAPPA    =   6.0000E-01                          CF        =   1.5330E-10
 +  CLC       =   1.0000E-07                          CLE       =   6.0000E-01
 +  DLC       =   2.9000E-08                          DWC       =   0.0000E+00
 +  VFBCV     =  -1.0000E+00                          NOFF      =   1.0000E+00
 +  VOFFCV    =   0.0000E+00                          ACDE      =   1.0000E+00
 +  MOIN      =   1.5000E+01

 *****Layout-Dependent Parasitics Model Parameters****************
 +  LMIN      =   1.8000E-07                          LMAX      =   5.0000E-05
 +  WMIN      =   2.4000E-07                          WMAX      =   1.0000E-04
 +  XL        =  -1.0500E-08                          XW        =   0.0000E-00

 *****Asymmetric Source/Drain Junction Diode Model Parameters*****
 +  JS        =   1.0000E-06                          JSW       =   7.0000E-11
 +  CJ        =   1.0300E-03                          MJ        =   4.4300E-01
 +  PB        =   8.1300E-01                          CJSW      =   1.3400E-10
 +  MJSW      =   3.3000E-01

 *****Temperature Dependence Parameters***************************
 +  TNOM      =   2.5000E+01                          UTE       =  -1.2860E+00
 +  KT1       =  -2.2550E-01                          KT1L      =  -4.1750E-09
 +  KT2       =  -2.5270E-02                          UA1       =   2.1530E-09
 +  UB1       =  -2.6730E-18                          UC1       =  -3.8320E-11
 +  AT        =   1.4490E+04                          PRT       =  -1.0180E+01
 +  XTI       =   3.0000E+00


 *****dW and dL Parameters****************************************
 +  WL        =   0.0000E+00                          WLN       =   1.0000E+00
 +  WW        =   7.2620E-16                          WWN       =   1.0000E+00
 +  WWL       =   0.0000E+00                          LL        =  -1.0620E-15
 +  LLN       =   1.0000E+00                          LW        =   2.9960E-15
 +  LWN       =   1.0000E+00                          LWL       =   0.0000E+00
 +  LLC       =  -2.1400E-15                          LWC       =   0.0000E+00
 +  LWLC      =   0.0000E+00                          WLC       =   0.0000E+00
 +  WWC       =   0.0000E+00                          WWLC      =   0.0000E+00

 *****Other Parameters********************************************
 +  LVTH0     =  -1.0000E-03                          WVTH0     =  6.027E-02
 +  PVTH0     =   0.0000E+00                          LNLX      =  -2.8540E-08
 +  WNLX      =   0.0000E+00                          PNLX      =   0.0000E+00
 +  WUA       =  -1.8800E-11                          WU0       =   5.4000E-01
 +  PUB       =   3.8000E-20                          PW0       =   1.3000E-09
 +  WRDSW     =   0.0000E+00                          WETA0     =   0.0000E+00
 +  WETAB     =   0.0000E+00                          LETA0     =   1.5740E-03
 +  LETAB     =   0.0000E+00                          PETA0     =   0.0000E+00
 +  PETAB     =   0.0000E+00                          WPCLM     =   0.0000E+00
 +  WVOFF     =  -4.0780E-04                          LVOFF     =  -4.2080E-03
 +  PVOFF     =  -3.7880E-04                          WA0       =  -4.7310E-02
 +  LA0       =  -4.6670E-01                          PA0       =  -2.6490E-02
 +  WAGS      =   4.2420E-03                          LAGS      =   3.0280E-01
 +  PAGS      =   0.0000E+00                          WKETA     =   0.0000E+00
 +  LKETA     =  -1.9420E-02                          PKETA     =   0.0000E+00
 +  WUTE      =   6.3730E-02                          LUTE      =   0.0000E+00
 +  PUTE      =   0.0000E+00                          WVSAT     =   5.0660E+03
 +  LVSAT     =   0.0000E+00                          PVSAT     =   0.0000E+00
 +  LPDIBLC2  =  -4.7520E-03                          WAT       =   7.0670E+03
 +  WPRT      =   0.0000E+00                          ACM       =   3.0000E+00
 +  LDIF      =   8.0000E-08                          HDIF      =   2.6000E-07
 +  N         =   1.0000E+00                          PHP       =   8.8000E-01
 +  CJGATE    =   5.0000E-10                          CTP       =   9.1400E-04
 +  PTP       =   9.2400E-04                          CTA       =   9.1900E-04
 +  PTA       =   1.5800E-03                          ELM       =   5.0000E+00
 +  TLEVC     =   1.0000E+00
"""


NMOS_TEMPLATE_MODEL['UMC90'] = """
.model REPLACEME_MODELNAME NMOS
*****Model Selectors/Controllers*********************************
+  LEVEL     =   5.4000E+01                          VERSION   =   4.3000E+00                        
+  BINUNIT   =   1.0000E+00                          PARAMCHK  =   1.0000E+00                        
+  MOBMOD    =   0.0000E+00                          CAPMOD    =   2.0000E+00                        
+  IGCMOD    =   1.0000E+00                          IGBMOD    =   1.0000E+00                        
+  GEOMOD    =   0.0000E+00                          DIOMOD    =   2.0000E+00                        
+  RDSMOD    =   0.0000E+00                          RBODYMOD  =   0.0000E+00                        
+  RGATEMOD  =   0.0000E+00                          PERMOD    =   1.0000E+00                        
+  ACNQSMOD  =   0.0000E+00                          TRNQSMOD  =   0.0000E+00                        
+  RGEOMOD   =   1.0000E+00                              FNOIMOD   =   1.0000E+00                   
+  TNOIMOD   =   0.0000E+00                        


*****Process Parameters******************************************
+  TOXE      =   2.2500E-09                          TOXP      =  1.8220E-09
+  TOXM      =   2.2500E-09                          EPSROX    =   3.9000E+00                        
+  XJ        =   1.2000E-07                          NGATE     =   1.3000E+20                        
+  NDEP      =   1.0000E+17                          NSD       =   1.0000E+20                        
+  RSH       =   8.0000E+00                        


*****Basic Model Parameters**************************************
+  WINT      =   2.0210E-08                          LINT      =  -4.0910E-09                        
+  VTH0      = REPLACEME_VTH0                        K1        =   1.5690E-01                        
+  K2        =   4.0000E-03                          K3        =  -1.2880E+00
+  K3B       =   2.9280E+00                          W0        =   9.0000E-08                        
+  DVT0      =   3.9630E+00                          DVT1      =   5.6320E-01                        
+  DVT2      =  -3.3200E-02                          DVT0W     =   5.2510E-01                        
+  DVT1W     =   1.1170E+07                          DVT2W     =  -7.7000E-01                        
+  DSUB      =   3.9000E-02                          MINV      =   7.7040E-01                        
+  VOFFL     =  -4.9270E-09                          DVTP0     =   8.9100E-06                        
+  DVTP1     =  -8.0630E-01                          LPE0      =   1.0000E-10
+  LPEB      =  -1.6990E-07                          PHIN      =   8.7670E-02                        
+  CDSC      =   4.6490E-04                          CDSCB     =   1.5000E-04                        
+  CDSCD     =   0.0000E+00                          CIT       =   1.5520E-03                        
+  VOFF      =  -6.3870E-02                          NFACTOR   =   1.0000E-01                        
+  ETA0      =   5.0000E-05                          ETAB      =  -1.8500E-04                        
+  VFB       =  -1.0000E+00                          U0        =   2.3200E-02                        
+  UA        =  -1.5500E-09                          UB        =   3.4800E-18                        
+  UC        =   1.7330E-10                          VSAT      =   1.6250E+05                        
+  A0        =   8.8340E+00                          AGS       =   1.0020E+00                        
+  A1        =   0.0000E+00                          A2        =   1.0000E+00                        
+  B0        =   0.0000E+00                          B1        =   1.0000E-08                        
+  KETA      =  -4.4080E-02                          DWG       =  -5.4000E-09                        
+  DWB       =   4.8000E-09                          PCLM      =   1.0000E-01                        
+  PDIBLC1   =   1.0000E-07                          PDIBLC2   =   3.9540E-02                        
+  PDIBLCB   =   1.0000E-01                          DROUT     =   5.5990E-01                        
+  PVAG      =   8.6180E+00                          DELTA     =   1.0000E-02                        
+  PSCBE1    =   6.5350E+09                          PSCBE2    =   3.3000E-01                        
+  FPROUT    =   1.0000E-02                          PDITS     =   6.1100E-01                        
+  PDITSD    =   8.8000E-01                          PDITSL    =   1.0000E+05                        


*****Parameters for Asymmetric and Bias-Dependent Rds Model******
+  RDSW      =   5.0000E+01                          RDSWMIN   =   5.0000E+01                        
+  PRWG      =   2.8000E-01                          PRWB      =   4.4700E-01                        
+  WR        =   1.0000E+00                        


*****Impact Ionization Current Model Parameters******************
+  ALPHA0    =   2.0000E-07                          ALPHA1    =   4.0000E+00                        
+  BETA0     =   1.5000E+01                        


*****Gate-Induced Drain Leakage Model Parameters*****************
+  AGIDL     =   1.1080E-08                          BGIDL     =   1.3900E+09                        
+  CGIDL     =   2.9630E-01                          EGIDL     =   9.4400E-01                        


*****Gate Dielectric Tunneling Current Model Parameters**********
+  TOXREF    =   2.2500E-09                          DLCIG     =   1.8000E-08                        
+  AIGBACC   =   1.1980E-02                          BIGBACC   =   8.0130E-03                        
+  CIGBACC   =   6.2560E-01                          NIGBACC   =   4.3970E+00                        
+  AIGBINV   =   1.5300E-02                          BIGBINV   =   4.8520E-03                        
+  CIGBINV   =   1.0000E-03                          EIGBINV   =   1.1000E+00                        
+  NIGBINV   =   1.6000E+00                          AIGC      =   1.1380E-02                        
+  BIGC      =   1.8790E-03                          CIGC      =   1.0000E-04                        
+  AIGSD     =   9.8830E-03                          BIGSD     =   1.2690E-03                        
+  CIGSD     =   1.5540E-01                          NIGC      =   1.0000E+00                        
+  POXEDGE   =   1.0000E+00                          PIGCD     =   2.5000E+00                        
+  NTOX      =   1.0000E+00                        


*****Charge and Capacitance Model Parameters*********************
+  DLC       =   1.6400E-08                          DWC       =  -3.0000E-08                        
+  XPART     =   1.0000E+00                          CGSO      =   5.0000E-11
+  CGDO      =   5.0000E-11                          CGBO      =   0.0000E+00                        
+  CGDL      =   2.2000E-10                          CGSL      =   2.2000E-10
+  CLC       =   1.0000E-07                          CLE       =   6.0000E-01                        
+  CF        =   9.2600E-11                          CKAPPAS   =   3.0000E+00                        
+  VFBCV     =  -1.0000E+00                          ACDE      =   2.8080E-01                        
+  MOIN      =   1.1830E+01                          NOFF      =   2.4860E+00                        
+  VOFFCV    =  -1.3720E-02                        


*****High-Speed/RF Model Parameters******************************


*****Flicker and Thermal Noise Model Parameters******************


+ef=0.9448
+noia=3.8700000E+41
+noib=1.8600000E+25
+noic=6.7000000E+08
+em=6.3600000E+06
+ntnoi=1.0

*****Layout-Dependent Parasitics Model Parameters****************
+  XL        =  -1.0000E-08                          XW        =   0.0000E+00
+  DMCG      =   1.6000E-07                          DMCI      =   1.0000E-07                        
+  DWJ       =   0.0000E+00                        


*****Asymmetric Source/Drain Junction Diode Model Parameters*****
+  JSS       =   2.3350E-07                          JSWS      =   7.0330E-14                        
+  JSWGS     =   3.2986E-14                          IJTHSFWD  =   3.4450E-03                        
+  IJTHSREV  =   1.6910E-03                          BVS       =   1.1470E+01                        
+  XJBVS     =   1.0000E+00                          PBS       =   6.1000E-01                        
+  CJS       =   1.0700E-03                          MJS       =   2.9000E-01                        
+  PBSWS     =   9.9000E-01                          CJSWS     =   1.2600E-10
+  MJSWS     =   1.0000E-01                          PBSWGS    =   6.0000E-01                        
+  CJSWGS    =   2.3100E-10                          MJSWGS    =   9.8900E-01                        


*****Temperature Dependence Parameters***************************
+  TNOM      =   2.5000E+01                          KT1       =  -3.8000E-01                        
+  KT1L      =   1.0000E-09                          KT2       =  -4.0740E-02                        
+  UTE       =  -1.0220E+00                          UA1       =   4.3500E-09                        
+  UB1       =  -4.1040E-18                          UC1       =   2.6360E-10                        
+  PRT       =   0.0000E+00                          AT        =   1.0000E+05                        
+  NJS       =   1.0560E+00                          TPB       =   1.3000E-03                        
+  TCJ       =   9.0000E-04                          TPBSW     =   3.5150E-03                        
+  TCJSW     =   4.0000E-04                          TPBSWG    =   1.2470E-03                        
+  TCJSWG    =   8.2290E-03                          XTIS      =   3.0000E+00                        


*****dW and dL Parameters****************************************
+  LL        =   4.3480E-16                          WL        =  -4.0050E-15                        
+  LLN       =   9.0000E-01                          WLN       =   1.0000E+00                        
+  LW        =   3.2080E-15                          WW        =  -1.5010E-15                        
+  LWN       =   1.0000E+00                          WWN       =   1.0000E+00                        
+  LWL       =  -1.6220E-21                          WWL       =   1.7820E-22
+  LLC       =  -9.0100E-16                          WLC       =   0.0000E+00                        
+  LWC       =   0.0000E+00                          WWC       =   1.0000E-15                        
+  LWLC      =   0.0000E+00                          WWLC      =   0.0000E+00                        


*****Range Parameters for Model Application**********************
+  LMIN      =   8.0000E-08                          LMAX      =   5.0000E-05                        
+  WMIN      =   1.2000E-07                          WMAX      =   1.0000E-04                        


*****Other Parameters********************************************
+  PVTH0     =  -1.2500E-03                          LK3       =   7.2000E-01                        
+  WK3       =  -1.3000E-01                          LK3B      =  -2.0000E-01                        
+  PK3B      =   2.0000E-02                          LDSUB     =  -1.2720E-03                        
+  WDSUB     =   5.0000E-04                          LLPE0     =   3.8910E-08
+  LCIT      =   7.0000E-05                          WVOFF     =  -1.3400E-03                        
+  LETA0     =   1.3000E-05                          WETA0     =   3.7760E-05                        
+  LETAB     =   8.2510E-06                          WU0       =   2.4000E-04                        
+  PU0       =  -6.5000E-05                          LUB       =  -2.5220E-20                        
+  WUB       =  -3.0000E-20                          PUB       =  -6.5270E-21                        
+  WUC       =  -5.5000E-12                          PVSAT     =  -7.3390E+02                        
+  LAGS      =   8.0000E-01                          LKETA     =   4.3920E-03                        
+  PKETA     =  -5.0000E-04                          LDELTA    =   5.5800E-04                        
+  LVOFFCV   =  -5.3220E-03                          PKT1      =   1.0000E-03                        
+  LUTE      =   7.5240E-02                          WUTE      =   2.5000E-02                        
+  PUTE      =   7.4000E-03                          LUB1      =   6.5000E-20                        
+  WUC1      =  -7.2000E-12                          SAREF     =   1.7600E-06                        
+  SBREF     =   1.7600E-06                          WLOD      =   0.0000E+00                        
+  KVTH0     =   5.0000E-08                          LKVTH0    =   3.9000E-06                        
+  WKVTH0    =   9.0000E-08                          PKVTH0    =   0.0000E+00                        
+  LLODVTH   =   1.0000E+00                          WLODVTH   =   1.0000E+00                        
+  STK2      =   0.0000E+00                          LODK2     =   1.0000E+00                        
+  LODETA0   =   1.0000E+00                          KU0       =  -1.5200E-08                        
+  LKU0      =  -6.2900E-09                          WKU0      =  -1.0000E-08                        
+  PKU0      =   1.2800E-15                          LLODKU0   =   1.0500E+00                        
+  WLODKU0   =   1.0000E+00                          KVSAT     =   9.9000E-01                        
+  STETA0    =  -2.8000E-11                          TKU0      =   0.0000E+00                        

"""


PMOS_TEMPLATE_MODEL['UMC180'] = """
 .model REPLACEME_MODELNAME PMOS
 *****Model Selectors/Controllers*********************************
 +  LEVEL     =   4.9000E+01                          MOBMOD    =   3.0000E+00
 +  VERSION   =   3.2000E+00                          CAPMOD    =   2.0000E+00
 +  BINUNIT   =   1.0000E+00                          NQSMOD    =   0.0000E+00


 *****Process Parameters******************************************
 +  TOX       =   4.2000E-09                          TOXM      =   4.2000E-09
 +  XJ        =   1.0000E-07                          NCH       =   6.1310E+17
 +  NGATE     =   1.0000E+23

 *****Basic Model Parameters**************************************
 +  VTH0      =  REPLACEME_VTH0                       K1        =   5.7040E-01
 +  K2        =   6.9730E-03                          K3        =  -2.8330E+00
 +  K3B       =   1.3260E+00                          W0        =  -1.9430E-07
 +  NLX       =   2.5300E-07                          DVT0      =   4.8850E-01
 +  DVT1      =   7.5780E-02                          DVT2      =   1.2870E-01
 +  DVT0W     =  -1.2610E-01                          DVT1W     =   2.4790E+04
 +  DVT2W     =   6.9150E-01                          LINT      =  -1.0410E-08
 +  WINT      =  -1.5250E-07                          DWG       =  -1.1510E-07
 +  DWB       =  -1.0390E-07                          U0        =   1.1450E+02
 +  UA        =   1.5400E-09                          UB        =   2.6460E-19
 +  UC        =  -9.5870E-02                          VSAT      =   5.3400E+04
 +  A0        =   1.3500E+00                          AGS       =   3.8180E-01
 +  B0        =  -3.0880E-07                          B1        =   0.0000E+00
 +  KETA      =   1.0440E-02                          A1        =   0.0000E+00
 +  A2        =   1.0000E+00                          VOFF      =  -1.0730E-01
 +  NFACTOR   =   1.5350E-00                          CIT       =  -1.0670E-03
 +  CDSC      =   7.5780E-04                          CDSCD     =  -2.8830E-05
 +  CDSCB     =   1.0000E-04                          ETA0      =   1.0710E+00
 +  ETAB      =  -9.2910E-01                          DSUB      =   1.9191E+00
 +  PCLM      =   2.6530E+00                          PDIBLC1   =   0.0000E+00
 +  PDIBLC2   =   5.0000E-06                          PDIBLCB   =   0.0000E+00
 +  DROUT     =   1.4570E+00                          PSCBE1    =   4.8660E+08
 +  PSCBE2    =   2.8000E-07                          PVAG      =   1.1620E+00


 *****Parameters for Asymmetric and Bias-Dependent Rds Model******
 +  RDSW      =   7.9210E+02                          PRWG      =   0.0000E+00
 +  PRWB      =   0.0000E+00


 *****Impact Ionization Current Model Parameters******************
 +  ALPHA0    =   0.0000E+00                          ALPHA1    =   0.0000E+00
 +  BETA0     =   3.0000E+01


 *****Charge and Capacitance Model Parameters*********************
 +  CGDO      =   2.0540E-10                          CGBO      =   0.0000E+00
 +  CGSO      =   2.0540E-10                          XPART     =   1.0000E+00
 +  CF        =   1.5330E-10                          DLC       =   5.6000E-08
 +  CGSL      =   0.0000E+00                          CGDL      =   0.0000E+00
 +  CKAPPA    =   6.0000E-01
 +  CLC       =   1.0000E-07                          CLE       =   6.0000E-01
 +  DWC       =   0.0000E+00
 +  VFBCV     =  -1.0000E+00                          NOFF      =   1.0000E+00
 +  VOFFCV    =   0.0000E+00                          ACDE      =   1.0000E+00
 +  MOIN      =   1.5000E+01


 *****Layout-Dependent Parasitics Model Parameters****************
 +  LMIN      =   1.8000E-07                          LMAX      =   5.0000E-05
 +  WMIN      =   2.4000E-07                          WMAX      =   1.0000E-04
 +  XL        =  -2.0000E-09                          XW        =  0.0000E+00

 *****Asymmetric Source/Drain Junction Diode Model Parameters*****
 +  JS        =   3.0000E-06                          JSW       =   4.1200E-11
 +  CJ        =   1.1400E-03                          MJ        =   3.9500E-01
 +  PB        =   7.6200E-01                          CJSW      =  1.7400E-10
 +  MJSW      =   3.2400E-01


 *****Temperature Dependence Parameters***************************
 +  TNOM      =   2.5000E+01                          UTE       =  -4.4840E-01
 +  KT1       =  -2.1940E-01                          KT1L      =  -8.2040E-09
 +  KT2       =  -9.4870E-03                          UA1       =   4.5710E-09
 +  UB1       =  -6.0260E-18                          UC1       =  -9.8500E-02
 +  AT        =   1.2030E+04                          PRT       =   0.0000E+00
 +  XTI       =   3.0000E+00


 *****dW and dL Parameters****************************************
 +  WW        =   1.2360E-14                          LW        =  -2.8730E-16
 +  LL        =   6.6350E-15
 +  WL        =   0.0000E+00                          WLN       =   1.0000E+00
 +  WWN       =   1.0000E+00
 +  WWL       =   0.0000E+00
 +  LLN       =   1.0000E+00
 +  LWN       =   1.0000E+00                          LWL       =   0.0000E+00
 +  LLC       =  -7.4500E-15                          LWC       =   0.0000E+00
 +  LWLC      =   0.0000E+00                          WLC       =   0.0000E+00
 +  WWC       =   0.0000E+00                          WWLC      =   0.0000E+00


 *****Other Parameters********************************************
 +  LVTH0     =   4.4000E-03                          WVTH0     = -1.4800E-02
 +  PVTH0     =   3.2000E-03                          LNLX      =  -1.5840E-08
 +  WRDSW     =   1.0070E+01                          WETA0     =   0.0000E+00
 +  WETAB     =   0.0000E+00                          WPCLM     =   0.0000E+00
 +  WUA       =   2.6300E-09                          LUA       =  -8.1530E-11
 +  PUA       =   5.8550E-11                          WUB       =   0.0000E+00
 +  LUB       =   0.0000E+00                          PUB       =   0.0000E+00
 +  WUC       =   0.0000E+00                          LUC       =   0.0000E+00
 +  PUC       =   0.0000E+00                          WVOFF     =  -9.8160E-03
 +  LVOFF     =  -9.8710E-04                          PVOFF     =  -9.8330E-05
 +  WA0       =  -4.8070E-02                          LA0       =  -2.8100E-01
 +  PA0       =   8.6610E-02                          WAGS      =  -4.1770E-02
 +  LAGS      =   4.4540E-02                          PAGS      =  -4.0760E-02
 +  WKETA     =   0.0000E+00                          LKETA     =  -1.2000E-02
 +  PKETA     =   0.0000E+00                          WUTE      =  -2.6820E-01
 +  LUTE      =   0.0000E+00                          PUTE      =   0.0000E+00
 +  WVSAT     =  -1.4200E+04                          LVSAT     =   0.0000E+00
 +  PVSAT     =  -4.3400E+02                          LPDIBLC2  =   3.0120E-03
 +  CJGATE    =   4.200E-10
 +  WAT       =  -6.4050E+03                          WPRT      =   2.1660E+02
 +  N         =   1.0000E+00                          PHP       =   6.6500E-01
 +  CTA       =   1.0000E-03                          CTP       =   7.5300E-04
 +  PTA       =   1.5500E-03                          PTP       =   1.2400E-03
 +  ACM       =   3.0000E+00                          LDIF      =   8.0000E-08
 +  RSH       =   8.0000E+00                          RD        =   0.0000E+00
 +  RSC       =   0.0000E+00                          RDC       =   0.0000E+00
 +  HDIF      =   2.6000E-07                          RS        =   0.0000E+00
"""

PMOS_TEMPLATE_MODEL['UMC90'] = """
.model REPLACEME_MODELNAME PMOS
*****Model Selectors/Controllers*********************************
+  LEVEL     =   5.4000E+01                          VERSION   =   4.3000E+00                        
+  BINUNIT   =   1.0000E+00                          PARAMCHK  =   1.0000E+00                        
+  MOBMOD    =   0.0000E+00                          CAPMOD    =   2.0000E+00                        
+  IGCMOD    =   1.0000E+00                          IGBMOD    =   1.0000E+00                        
+  GEOMOD    =   0.0000E+00                          DIOMOD    =   2.0000E+00                        
+  RDSMOD    =   0.0000E+00                          RBODYMOD  =   0.0000E+00                        
+  PERMOD    =   1.0000E+00                          ACNQSMOD  =   0.0000E+00                        
+  RGEOMOD   =   1.0000E+00                          FNOIMOD   =   1.0000E+00                        
+  TNOIMOD   =   0.0000E+00                       


*****Process Parameters******************************************
+  TOXE      =   2.4500E-09                          TOXP      =   1.9110E-09
+  TOXM      =   2.4500E-09                          EPSROX    =   3.9000E+00                        
+  XJ        =   1.2000E-07                          NGATE     =   1.0000E+20                        
+  NDEP      =   3.6000E+16                          NSD       =   1.0000E+20                        
+  RSH       =   8.0000E+00                        


*****Basic Model Parameters**************************************
+  WINT      =   8.0090E-09                          LINT      =  -2.1220E-08                        
+  VTH0      = REPLACEME_VTH0                        K1        =   2.2500E-01                        
+  K2        =  -2.4750E-02                          K3        =  -8.8950E+00
+  K3B       =   3.9000E+00                          W0        =   2.1220E-06                        
+  DVT0      =   4.6860E+00                          DVT1      =   8.7290E-01                        
+  DVT2      =   1.2770E-02                          DVT0W     =   3.0000E-01                        
+  DVT1W     =   3.9660E+06                          DVT2W     =   2.4940E-01                        
+  DSUB      =   1.0160E+00                          MINV      =   2.8230E-01                        
+  VOFFL     =  -2.5000E-09                          DVTP0     =   6.0620E-06                        
+  DVTP1     =   4.4890E-01                          LPE0      =  -1.2670E-07
+  LPEB      =   6.2500E-08                          PHIN      =   0.0000E+00                        
+  CDSC      =   0.0000E+00                          CDSCB     =  -8.0000E-03                        
+  CDSCD     =   0.0000E+00                          CIT       =   2.7750E-04                        
+  VOFF      =  -1.2000E-01                          NFACTOR   =   2.0000E+00                        
+  ETA0      =   3.0000E-02                          ETAB      =  -5.0310E-01                        
+  VFB       =  -1.0000E+00                          U0        =   9.2600E-03
+  UA        =   4.2790E-10                          UB        =   1.1290E-18                        
+  UC        =   8.5910E-11                          EU        =   1.0000E+00                        
+  VSAT      =   1.3670E+05                          A0        =   1.8600E+00                        
+  AGS       =   1.4670E+00                          A1        =   0.0000E+00                        
+  A2        =   1.0000E+00                          B0        =   7.0000E-07                        
+  B1        =   6.0000E-07                          KETA      =  -5.1120E-02                        
+  DWG       =  -1.7240E-08                          DWB       =   0.0000E+00                        
+  PCLM      =   2.9400E-01                          PDIBLC1   =   5.1850E-08                        
+  PDIBLC2   =   4.0800E-03                          PDIBLCB   =  -5.0000E-01                        
+  DROUT     =   4.6980E-04                          PVAG      =   1.2960E+00                        
+  DELTA     =   2.3890E-03                          PSCBE1    =   6.3370E+09                        
+  PSCBE2    =   3.0000E-03                          FPROUT    =   3.0000E+02                        
+  PDITS     =   2.9810E-01                          PDITSD    =   7.1760E-01                        
+  PDITSL    =   5.0000E+05                        


*****Parameters for Asymmetric and Bias-Dependent Rds Model******
+  RDSW      =   2.2500E+02                          RDSWMIN   =   8.0000E+01                        
+  PRWG      =   0.0000E+00                          PRWB      =   2.0000E-01                        
+  WR        =   1.0000E+00                        


*****Impact Ionization Current Model Parameters******************
+  ALPHA0    =   2.1400E-08                          ALPHA1    =   7.0000E-02                        
+  BETA0     =   1.2000E+01                        


*****Gate-Induced Drain Leakage Model Parameters*****************
+  AGIDL     =   4.4320E-09                          BGIDL     =   4.8080E+09                        
+  CGIDL     =   9.1730E-03                          EGIDL     =  -2.1800E+00                        


*****Gate Dielectric Tunneling Current Model Parameters**********
+  TOXREF    =   2.4500E-09                          DLCIG     =   3.2000E-08                        
+  AIGBACC   =   1.1030E-02                          BIGBACC   =   6.7610E-03                        
+  CIGBACC   =   5.7700E-01                          NIGBACC   =   4.3960E+00                        
+  AIGBINV   =   9.4660E-03                          BIGBINV   =   2.3400E-03                        
+  CIGBINV   =   1.8320E-03                          EIGBINV   =   1.6330E+00                        
+  NIGBINV   =   3.1240E+00                          AIGC      =   6.7900E-03                        
+  BIGC      =   8.8750E-04                          CIGC      =   6.3430E-04                        
+  AIGSD     =   5.6520E-03                          BIGSD     =   7.8050E-05                        
+  CIGSD     =   1.8030E-02                          NIGC      =   7.9250E-01                        
+  POXEDGE   =   1.0000E+00                          PIGCD     =   2.0000E+00                        
+  NTOX      =   1.0000E+00                        


*****Charge and Capacitance Model Parameters*********************
+  DLC       =   3.4200E-08                          DWC       =  -3.0000E-08                        
+  XPART     =   1.0000E+00                          CGSO      =   4.2000E-11
+  CGDO      =   4.2000E-11                          CGBO      =   0.0000E+00                        
+  CGDL      =   2.0000E-10                          CGSL      =   2.0000E-10
+  CLC       =   1.0000E-07                          CLE       =   6.0000E-01                        
+  CF        =   9.0800E-11                          CKAPPAS   =   7.3000E-01                        
+  CKAPPAD   =   7.3000E-01                          ACDE      =   3.5090E-01                        
+  MOIN      =   6.7000E+00                          NOFF      =   2.9360E+00                        
+  VOFFCV    =  -5.2570E-02                        


*****High-Speed/RF Model Parameters******************************


*****Flicker and Thermal Noise Model Parameters******************


+EF=1.103336
+NOIA=1.0635922E+41
+NOIB=6.9613951E+26
+NOIC=5.2897264E+09
+EM=4.1000000E+07
+NTNOI=1.0                     

*****Layout-Dependent Parasitics Model Parameters****************
+  XL        =  -1.0000E-08                          XW        =   0.0000E+00
+  DMCG      =   1.6000E-07                          DMCI      =   1.0000E-07                        
+  DWJ       =   0.0000E+00                        


*****Asymmetric Source/Drain Junction Diode Model Parameters*****
+  JSS       =   1.9950E-07                          JSWS      =   1.0920E-13                        
+  JSWGS     =   1.0000E-13                          IJTHSFWD  =   3.5000E-03                        
+  IJTHSREV  =   2.1750E-03                          BVS       =   8.9640E+00                        
+  XJBVS     =   1.0000E+00                          PBS       =   7.3000E-01                        
+  CJS       =   1.2600E-03                          MJS       =   3.1000E-01                        
+  PBSWS     =   9.9000E-01                          CJSWS     =   1.2900E-10
+  MJSWS     =   1.0000E-01                          PBSWGS    =   6.0000E-01                        
+  CJSWGS    =   2.4500E-10                          MJSWGS    =   9.8900E-01                        


*****Temperature Dependence Parameters***************************
+  TNOM      =   2.5000E+01                          KT1       =  -3.4000E-01                        
+  KT1L      =  -9.5660E-09                          KT2       =  -1.0000E-02                        
+  UTE       =  -1.9620E+00                          UA1       =  -8.3500E-10                        
+  UB1       =  -1.3400E-18                          UC1       =   0.0000E+00                        
+  PRT       =  -1.6750E+02                          AT        =   1.0340E+05                        
+  NJS       =   1.0540E+00                          TPB       =   1.4000E-03                        
+  TCJ       =   8.0000E-04                          TPBSW     =   1.0000E-04                        
+  TCJSW     =   4.0000E-04                          TPBSWG    =   1.5050E-03                        
+  TCJSWG    =   7.6180E-03                          XTIS      =   3.0000E+00                        


*****dW and dL Parameters****************************************
+  LL        =   5.5440E-16                          WL        =   7.1650E-16                        
+  LLN       =   1.0500E+00                          WLN       =   9.7350E-01                        
+  LW        =  -2.1170E-15                          WW        =  -4.3920E-15                        
+  LWN       =   1.0000E+00                          WWN       =   9.9400E-01                        
+  LWL       =   2.3760E-23                          WWL       =  -1.4950E-22
+  LLC       =  -6.7780E-16                          WLC       =   0.0000E+00                        
+  LWC       =   0.0000E+00                          WWC       =   1.0000E-15                        
+  LWLC      =   0.0000E+00                          WWLC      =   0.0000E+00                        


*****Range Parameters for Model Application**********************
+  LMIN      =   8.0000E-08                          LMAX      =   5.0000E-05                        
+  WMIN      =   1.2000E-07                          WMAX      =   1.0000E-04                        


*****Other Parameters********************************************
+  PVTH0     =   0.0000E+00                          LK3       =   1.0000E+00                        
+  PK3       =  -1.4700E-01                          LK3B      =  -7.6900E-01                        
+  WK3B      =   2.1290E+00                          WDSUB     =   1.1010E-02                        
+  PDVTP1    =  -2.0000E-02                          LLPE0     =   2.9370E-08
+  LLPEB     =   1.3590E-08                          LNFACTOR  =   2.6600E-01                        
+  LETAB     =   3.9610E-02                          LAGS      =   1.8360E+00                        
+  PAGS      =  -8.4000E-02                          LB0       =  -5.5000E-08                        
+  LB1       =  -5.8900E-08                          LKETA     =  -1.9200E-03                        
+  LDELTA    =   2.4950E-03                          WRDSW     =   1.0000E+01                        
+  LVOFFCV   =   4.1250E-04                          WKT1      =   5.0000E-03                        
+  WUA1      =   3.9440E-11                          SAREF     =   1.7600E-06                        
+  SBREF     =   1.7600E-06                          WLOD      =   0.0000E+00                        
+  KVTH0     =  -8.0000E-10                          LKVTH0    =  -1.5000E-06                        
+  WKVTH0    =   6.0000E-07                          PKVTH0    =   0.0000E+00                        
+  LLODVTH   =   8.0000E-01                          WLODVTH   =   1.0000E+00                        
+  STK2      =   0.0000E+00                          LODK2     =   1.0000E+00                        
+  LODETA0   =   1.0000E+00                          KU0       =   5.3000E-07                        
+  LKU0      =   5.8000E-04                          WKU0      =  -1.1000E-09                        
+  PKU0      =  -2.5000E-10                          LLODKU0   =   6.8000E-01                        
+  WLODKU0   =   8.5000E-01                          KVSAT     =   1.0000E+00                        
+  STETA0    =   3.8000E-10                          TKU0      =   0.0000E+00                        

"""
