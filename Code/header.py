import numpy as np
import os
import sys
import time
import pickle

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../Multi-Channel-Time-Encoding/Source")
from Time_Encoder import *
from Signal import *


Figure_Path = os.path.split(os.path.realpath(__file__))[0] + "/../Figures/"
Data_Path = os.path.split(os.path.realpath(__file__))[0] + "/../Data/"

graphical_import = True

if graphical_import:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rc
