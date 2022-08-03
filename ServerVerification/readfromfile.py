import pickle
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

U_array = [70]
plot_LightVeriFL = np.zeros((len(U_array),5))
plot_VeriFL = np.zeros((len(U_array),5))

#for i, U in enumerate(U_array):
#t_LightVeriFL  = pickle.load(open('./results/LightVeriFL_imp_N100'+'_U70'+'_d1206590', 'rb'))
#t_VeriFL  = pickle.load(open('./results/VeriFL_imp_N100'+'_U70'+'_d1206590', 'rb'))

obj1 = pd.read_pickle(r'./results/LightVeriFL_imp_N100'+'_U70'+'_d1206590')

#plot_LightVeriFL[i,:] = np.array([t_LightVeriFL[0]['t_Total'], t_LightVeriFL[0]['t_AggTotal'], t_LightVeriFL[0]['t_AggArray'], t_LightVeriFL[0]['t_VeriTotal'], t_LightVeriFL[0]['t_VeriArray']])
#plot_VeriFL[i,:] = np.array([t_VeriFL[0]['t_Total'], t_VeriFL[0]['t_AggTotal'], t_VeriFL[0]['t_VeriTotal']])

#print(plot_LightVeriFL[i,:])
print(obj1)
#print(plot_VeriFL[i,:])


#'N': N
#            , 'T': T
#            , 'd': d
#            , 't_Total': t_Total_avg
#            , 't_AggTotal': t_AggTotal_avg
#            , 't_AggArray': t_AggArray_hist
#            , 't_VeriTotal': t_VeriTotal_avg
#            , 't_VeriArray': t_VeriArray_hist