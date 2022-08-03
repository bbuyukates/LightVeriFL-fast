import pickle
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

U_array = [71]
plot_LightVeriFL = np.zeros((len(U_array),3))
plot_VeriFL = np.zeros((len(U_array),3))

for i, U in enumerate(U_array):
    t_LightVeriFL  = pickle.load(open('./results/LightVeriFL_N100'+'_U' + str(U)+'_d100000', 'rb'))
    t_VeriFL  = pickle.load(open('./results/VeriFL_N100'+'_U' + str(U)+'_d100000', 'rb'))

    plot_LightVeriFL[i,:] = np.array([t_LightVeriFL[0]['t_Total'], t_LightVeriFL[0]['t_AggTotal'], t_LightVeriFL[0]['t_VeriTotal']])
    plot_VeriFL[i,:] = np.array([t_VeriFL[0]['t_Total'], t_VeriFL[0]['t_AggTotal'], t_VeriFL[0]['t_VeriTotal']])

    print(U)
    print(plot_LightVeriFL[i,:])
    print(plot_VeriFL[i,:])
    print()