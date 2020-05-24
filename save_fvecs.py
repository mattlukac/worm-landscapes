import numpy as np 
import pandas as pd

worm_dir = '/projects/haldane/shared/worm_landscapes/2kb/'
worm_file = worm_dir + 'sim_100010039091_Ne_5000_Self_0_Mut_1_FrD_0_FrB_0_SDA_0_SDC_0_SBA_0_SBC_0.12stats.txt'
fvec = pd.read_csv(worm_file, sep='\t')
print(fvec.values.shape)
