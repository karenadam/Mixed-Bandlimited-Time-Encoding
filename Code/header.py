import numpy as np
import os
import sys
import time
import pickle

sys.path.insert(
    0,
    os.path.split(os.path.realpath(__file__))[0]
    + "/../Multi-Channel-Time-Encoding/Source",
)
from Time_Encoder import *
from Signal import *

To_Svg = False


Figure_Path = os.path.split(os.path.realpath(__file__))[0] + "/../Figures/"
Data_Path = os.path.split(os.path.realpath(__file__))[0] + "/../Data/"

graphical_import = True

if graphical_import:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rc



    if To_Svg:
        plt.rc('text', usetex=False)
        plt.rc('text.latex', unicode = False)
        plt.rc('svg',fonttype = 'none')
    else:
        matplotlib.rc("text", usetex=True)
        matplotlib.rc("font", family="serif")
        matplotlib.rc("font", size=7)
        matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{amssymb}")
    from matplotlib.colors import LogNorm   
