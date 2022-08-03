import pickle
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

U_array = [51,71,91]
plot_LightVeriFL = np.zeros((len(U_array),3))
plot_VeriFL = np.zeros((len(U_array),3))

for i, U in enumerate(U_array):
    t_LightVeriFL  = pickle.load(open('./results/LightVeriFL_N100'+'_U' + str(U)+'_d10000', 'rb'))
    t_VeriFL  = pickle.load(open('./results/VeriFL_N100'+'_U' + str(U)+'_d10000', 'rb'))

    plot_LightVeriFL[i,:] = np.array([t_LightVeriFL[0]['t_Total'], t_LightVeriFL[0]['t_AggrTotal'], t_LightVeriFL[0]['t_VeriTotal']])
    plot_VeriFL[i,:] = np.array([t_VeriFL[0]['t_Total'], t_VeriFL[0]['t_AggTotal'], t_VeriFL[0]['t_VeriTotal']])

    print(N)
    print(plot_LightVeriFL[i,:])
    print(plot_VeriFL[i,:])
    print()

plt.figure()
plt.plot(U_array, plot_LightVeriFL[:,0], label='LightVeriFL'    , markevery=1, marker="o",markerfacecolor='None')
plt.plot(U_array, plot_VeriFL[:,0],      label='VeriFL'         , markevery=1, marker="v",markerfacecolor='None')
plt.xlabel('N')
plt.ylabel('Running time (sec)')
plt.grid()
plt.legend()
plt.savefig('./plots/total_runtime_d10000_drop.png',dpi=300, bbox_inches = "tight")
plt.show()

plt.figure()
plt.plot(U_array, plot_LightVeriFL[:,1], label='LightVeriFL (aggregation phase)'    , markevery=1, marker="o",markerfacecolor='None')
plt.plot(U_array, plot_VeriFL[:,1],      label='VeriFL (aggregation phase)'         , markevery=1, marker="v",markerfacecolor='None')
plt.xlabel('N')
plt.ylabel('Running time (sec)')
plt.grid()
plt.legend()
plt.savefig('./plots/aggr_runtime_d10000_drop.png',dpi=300, bbox_inches = "tight")
plt.show()

plt.figure()
plt.plot(U_array, plot_LightVeriFL[:,2], label='LightVeriFL (verification phase)'    , markevery=1, marker="o",markerfacecolor='None')
plt.plot(U_array, plot_VeriFL[:,2],      label='VeriFL (verification phase)'         , markevery=1, marker="v",markerfacecolor='None')
plt.xlabel('N')
plt.ylabel('Running time (sec)')
plt.grid()
plt.legend()
plt.savefig('./plots/verification_runtime_d10000_drop.png',dpi=300, bbox_inches = "tight")
plt.show()