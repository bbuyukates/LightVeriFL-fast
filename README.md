# LightVeriFL-fast

This repo is to implement the LigthVeriFL scheme that enables lightweight verification of the aggregation result provided by the server at each iteration in federated learning.
For fast elliptic curve cryptography, we use the open-source fastecdsa Python library (https://github.com/AntonKueltz/fastecdsa). Please follow the instructions on https://github.com/AntonKueltz/fastecdsa
for the installation of fastecdsa.

In order to run the experiments, in the ServerVerification folder, execute

mpirun -n {N+1} python3 LightVeriFL_EC_fastecdsa_amortized.py {N} {N-D} {d} {L}.

Here, N is the number of users, D is the number of dropped users, d is the model dimension, and L is the batch size for the amortization.

