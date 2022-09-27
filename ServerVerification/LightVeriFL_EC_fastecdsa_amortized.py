import logging
import pickle as pickle
import sys
import time
import random
import math

from fastecdsa.curve import P256
from fastecdsa.point import Point

import numpy as np
from mpi4py import MPI

from utils.function import my_pk_gen, my_key_agreement, BGW_encoding, BGW_decoding, SS_decoding, SS_encoding
from utils.EC import LightVeriFL_enc_EC, LightVeriFL_dec_EC, generate_hash, \
    generate_point_EC, PI_addEC, generate_Pedersen_commitment, curve_g, curve_n

from utils.EC import p_model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logging.basicConfig(level=logging.DEBUG,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

if len(sys.argv) == 1:
    if rank == 0:
        logging.info("ERROR: please check the input arguments")
    exit()
elif len(sys.argv) == 2:
    N = int(sys.argv[1])
    T = int(np.floor(N / 2))
    d = 10 * 1000
elif len(sys.argv) == 3:
    N = int(sys.argv[1])
    U = int(sys.argv[2])
    d = 10 * 1000
elif len(sys.argv) == 4:
    N = int(sys.argv[1])
    U = int(sys.argv[2])
    d = int(sys.argv[3])
    batch = 5
elif len(sys.argv) == 5:
    N = int(sys.argv[1])
    U = int(sys.argv[2])
    d = int(sys.argv[3])
    batch = int(sys.argv[4])
else:
    if rank == 0:
        logging.info("ERROR: please check the input arguments")
    exit()

N_repeat = batch # batch size for amortization.
server_agg_gradient_epochs = [None] * N_repeat
agg_client_hashes_epochs = [None] * N_repeat

num_pk_per_user = 2
# h_array = [0] * N

if __name__ == "__main__":
    U1 = np.arange(
        N)  # surviving users who send the masked model to the server. We assume that all users are surviving in the aggregation phase.
    drop_info = np.zeros((N,), dtype=int)

    T = int(np.floor(N / 2)) #U = int(0.9*N)
    alpha_s = np.array(range(T+1))  # np.arange(0, U)  # T+1 evaluation points for encoding inputs
    beta_s = np.arange(U, U + N)  # N evaluation points for decoding outputs

    h_array = np.arange(1, N + 1)

    if rank == 0:

        logging.info(f"N,U,T={N},{U},{T}, starts!! ")
        surviving_users_indexes = np.random.choice(np.arange(N), U, replace=False).astype(int)
        surviving_users_indexes_actual = random.sample(surviving_users_indexes.tolist(), T+1)
        #print(surviving_users_indexes_actual)
        all_user_set = list(range(N))
        dropped_users_indexes_list = [x for x in all_user_set if
                                      x not in surviving_users_indexes.tolist()]

        for tx_rank in range(1, N + 1):
            comm.Send(surviving_users_indexes, dest=tx_rank)

        for tx_rank in range(1, N + 1):
            comm.Send(np.array(surviving_users_indexes_actual), dest=tx_rank)

        # sample distinct alpha values, alpha_i cannot be 0
        alpha = np.array(random.sample(range(1, d + 1), k=d))  # is fine as long as d < curve.n which is the case for us

        # sample coefficients for the Pedersen commitment scheme. These will be used to generate g and l of the Pedersen commitment scheme.
        Pedersen_coeff = np.array(random.sample(range(1, 100), 2))  # we can use curve.n instead of 100 here.

        for tx_rank in range(1, N + 1):
            comm.Send(alpha, dest=tx_rank)

        for tx_rank in range(1, N + 1):
            comm.Send(Pedersen_coeff, dest=tx_rank)

    elif rank <= N:
        surviving_users_indexes = np.zeros((U,), dtype=int)
        comm.Recv(surviving_users_indexes, source=0)

        surviving_users_indexes_actual = np.zeros((T+1,), dtype=int)
        comm.Recv(surviving_users_indexes_actual, source=0)

        alpha = np.zeros((d,), dtype=int)
        comm.Recv(alpha, source=0)

        Pedersen_coeff = np.zeros((2,), dtype=int)
        comm.Recv(Pedersen_coeff, source=0)
        # logging.info(f'[ rank= {rank} ] surviving_users = {surviving_users}' )

    if rank == 0:

        t_Total_hist = np.zeros((N_repeat,))
        t_AggTotal_hist = np.zeros((N_repeat,))
        t_VeriTotal_hist = np.zeros((N_repeat,))
        t_AggArray_hist = np.zeros((N_repeat, 4))
        t_VeriArray_hist = np.zeros((N_repeat, 2))

        for i_trial in range(N_repeat):

            # comm.Barrier()
            ##########################################
            ##          Server starts HERE          ##
            ##########################################
            '''
            Aggregation Round 0. AdvertiseKeys:
            '''
            print()
            print('----------------------------------')
            print('   Aggregation phase starts!!  ')

            comm.Barrier()
            # 0.0. Receive the public keys
            t0 = time.time()

            public_key_list = np.empty(shape=(num_pk_per_user, N), dtype='int64')
            for i in range(N):
                # drop_info = np.empty(N, dtype='int')
                data = np.empty(num_pk_per_user, dtype='int64')
                comm.Recv(data, source=i + 1)
                public_key_list[:, i] = data

            # 0.1.Broadcast the public keys
            for i in range(N):
                data = np.reshape(public_key_list, num_pk_per_user * N)
                comm.Send(data, dest=i + 1)

            comm.Barrier()
            t_AggRound0 = time.time() - t0

            '''
            Aggregation Round 1. ShareMetadata
            '''
            # Round 1. Share Key

            # 1.0. Receive the SS from users
            comm.Barrier()
            t1 = time.time()

            b_u_SS_list = np.empty((N, N), dtype='int64')
            s_sk_SS_list = np.empty((N, N), dtype='int64')

            for i in range(N):
                data = np.empty(N, dtype='int64')
                comm.Recv(data, source=i + 1)
                b_u_SS_list[i, :] = data

                data = np.empty(N, dtype='int64')
                comm.Recv(data, source=i + 1)
                s_sk_SS_list[i, :] = data

            # 1.1. Send the SS to the users

            for i in range(N):
                data = b_u_SS_list[:, i].astype('int64')
                comm.Send(data, dest=i + 1)

                data = s_sk_SS_list[:, i].astype('int64')
                comm.Send(data, dest=i + 1)

            comm.Barrier()

            # before exchange h_SS
            # comm.Barrier()
            # after exchange h_SS
            # comm.Barrier()

            t_AggRound1 = time.time() - t1

            '''
            Aggregation Round 2. MaskedInputCollection
                - Receives the local model along with masked hash and noise from all users
            '''

            # 2.0 Receive the local model from the clients
            comm.Barrier()
            t2 = time.time()
            rows, cols = (len(U1), d)

            masked_grad_array = [[[] for j in range(cols)] for i in range(rows)]  # [[0] * cols] * rows
            array_idx2 = 0
            for i in U1:
                rx_rank = i + 1
                masked_grad_array[array_idx2] = comm.recv(source=rx_rank)
                array_idx2 += 1

            agg_grad = [sum(x) % p_model for x in zip(*masked_grad_array)]
            #print('Before unmasking, agg masked grad is', agg_grad)

            # comm.Barrier()

            # 2.1 Receive the masked hash and noise from the users
            hz_array = [0] * N
            array_idx = 0
            for i in range(N):
                rx_rank = i + 1
                hz_array[array_idx] = comm.recv(source=rx_rank)
                array_idx += 1

            # comm.Barrier()
            noise_array = [0] * N
            array_idx2 = 0
            for i in range(N):
                rx_rank = i + 1
                noise_array[array_idx2] = comm.recv(source=rx_rank)
                array_idx2 += 1

            # We have two kinds of hz_mul: surviving and dropped
            hz_mul_surviving = PI_addEC([hz_array[x] for x in surviving_users_indexes.tolist()])
            hz_mul_dropped = PI_addEC([hz_array[x] for x in dropped_users_indexes_list])

            noise_mul_surviving = PI_addEC([noise_array[x] for x in surviving_users_indexes.tolist()])
            noise_mul_dropped = PI_addEC([noise_array[x] for x in dropped_users_indexes_list])

            comm.Barrier()
            t_AggRound2 = time.time() - t2

            '''
            Aggregation Round 3. Unmasking
                - 3.0. Reconstruct b_u or s_uv & unmask the aggregate model
                - 3.1. Send the aggregate model back to the users
            '''
            comm.Barrier()
            t3 = time.time()

            # 3.0.0 Receive SS (b_u_SS for surviving users, s_sk_SS for dropped users)
            SS_rx = np.empty((N, N), dtype='int64')
            for i in range(N):
                data = np.empty(N, dtype='int64')
                comm.Recv(data, source=i + 1)
                SS_rx[:, i] = data

            # 3.0.1 Generate PRG based on the seed

            for i in range(N):
                if drop_info[i] == 0:
                    SS_input = np.reshape(SS_rx[i, 0:T + 1], (T + 1,)).tolist()
                    b_u = SS_decoding(SS_input, range(T + 1), p_model)
                    np.random.seed(b_u)

                    temp = np.random.randint(0, p_model, size=d).astype(int)
                    agg_grad = np.mod(agg_grad - temp, p_model)

                else:
                    mask = np.zeros(d, dtype='int')
                    SS_input = np.reshape(SS_rx[i, 0:T + 1], (T + 1, 1))
                    s_sk_dec = BGW_decoding(SS_input, range(T + 1), p_model)

                    for j in range(N):
                        s_pk_list_ = public_key_list[1, :]
                        s_uv_dec = np.mod(s_sk_dec[0][0] * s_pk_list_[j], p_model)

                        if j == i:
                            temp = np.zeros(d, dtype='int')
                        elif j < i:
                            np.random.seed(s_uv_dec)
                            temp = np.random.randint(0, p_model, size=d).astype(int)
                        else:
                            np.random.seed(s_uv_dec)
                            temp = -np.random.randint(0, p_model, size=d).astype(int)
                        mask = np.mod(mask + temp, p_model)
                    agg_grad = np.mod(agg_grad + mask, p_model)

            #print('After  unmasking, agg masked grad is', agg_grad)

            # 3.1 Send the aggregate gradient to the surviving users
            # comm.Barrier()
            for tx_rank in surviving_users_indexes + 1: #range(1, N + 1): #
                comm.Send(np.array(agg_grad), dest=tx_rank)

            comm.Barrier()
            t_AggRound3 = time.time() - t3
            t_AggTotal = time.time() - t0

            # We only have two rounds of verification whereas VeriFL has 3 rounds of verification
            '''
            Verification Round 0. AggregateDecommitting
                - 0.0. Receive the z_tilde_mul from surviving users. 
                Here two kinds of z_tilde_mul is received. One is for the dropped users and the other for the surviving users
                - 0.1. Decodes z_mul
                - 0.2. Broadcasts the reconstructed aggregate hash and noise to the users
            '''
            comm.Barrier()
            t0_verification = time.time()
            print()
            print('----------------------------------')
            print('   Verification phase starts!!  ')
            print('surviving indexes are', surviving_users_indexes)

            # 0.0. Receive z_tilde_mul from surviving users: one for surviving and another for dropped

            # 0.0.0. z_tilde_mul for surviving
            # comm.Barrier()
            z_tilde_mul_array_surviving = [0] * len(surviving_users_indexes_actual)  #len(surviving_users_indexes)
            array_idx = 0
            for i in surviving_users_indexes_actual: #surviving_users_indexes:
                rx_rank = i + 1
                z_tilde_mul_array_surviving[array_idx] = comm.recv(source=rx_rank)
                array_idx += 1

            # 0.0.1. z_tilde_mul for dropped
            # comm.Barrier()
            z_tilde_mul_array_dropped = [0] * len(surviving_users_indexes_actual) #len(surviving_users_indexes)
            array_idx = 0
            for i in surviving_users_indexes_actual: #surviving_users_indexes:
                rx_rank = i + 1
                z_tilde_mul_array_dropped[array_idx] = comm.recv(source=rx_rank)
                array_idx += 1

            # 0.1. Decoding - z_mul
            # Decode the aggregate dropped users' hashes and aggregate surviving users' hashes separately

            # 0.1.0. Decode the aggregate surviving users' hash and mask
            dec_z = LightVeriFL_dec_EC(z_tilde_mul_array_surviving, alpha_s, beta_s[surviving_users_indexes_actual], P256)
            dec_z_minus = (-1 * dec_z)
            res_surviving = (hz_mul_surviving + dec_z_minus)
            noise_surviving = (noise_mul_surviving + dec_z_minus)
            # print(f"multiplication of surviving users' hashes = {res_surviving}\n")

            # 0.1.1. Decode the aggregate dropped users' hash and mask
            dec_z = LightVeriFL_dec_EC(z_tilde_mul_array_dropped, alpha_s, beta_s[surviving_users_indexes_actual], P256)
            dec_z_minus = (-1 * dec_z)
            res_dropped = (hz_mul_dropped + dec_z_minus)
            noise_dropped = (noise_mul_dropped + dec_z_minus)

            # print(f"multiplication of dropped users' hashes = {res_dropped}\n")

            # 0.1.2. Combine the two to reconstruct the aggregate user hash and mask
            res = (res_dropped + res_surviving)
            res_noise = (noise_dropped + noise_surviving)

            # 0.2. Broadcast the reconstructed aggregate hash and noise to the users
            for tx_rank in surviving_users_indexes + 1:
                comm.send(res, dest=tx_rank)

            for tx_rank in surviving_users_indexes + 1:
                comm.send(res_noise, dest=tx_rank)

            comm.Barrier()
            t_VeriRound0 = time.time() - t0_verification

            '''
            Verification Round 1. BatchChecking
                - NoOP in the server
            '''
            comm.Barrier()
            t1_verification = time.time()

            comm.Barrier()
            t_VeriRound1 = time.time() - t1_verification
            t_VeriTotal = time.time() - t0_verification
            t_Total = time.time() - t0

            t_AggArray = np.array([t_AggRound0, t_AggRound1, t_AggRound2, t_AggRound3])
            t_VeriArray = np.array([t_VeriRound0, t_VeriRound1])

            t_Total_hist[i_trial] = t_Total
            t_AggTotal_hist[i_trial] = t_AggTotal
            t_VeriTotal_hist[i_trial] = t_VeriTotal
            t_AggArray_hist[i_trial, :] = t_AggArray
            t_VeriArray_hist[i_trial, :] = t_VeriArray

            print()
            print(f'   ({i_trial} / {N_repeat}) - th Trial ends')
            print(f"t_Total = {t_Total}, t_AggTotal={t_AggTotal}, t_VeriTotal={t_VeriTotal}\n\n")

        t_Total_avg = np.sum(t_Total_hist) / float(N_repeat)
        t_AggTotal_avg = np.sum(t_AggTotal_hist) / float(N_repeat)
        t_VeriTotal_avg = np.sum(t_VeriTotal_hist) / float(N_repeat)
        t_AggArray_avg = np.sum(t_AggArray_hist, axis=0) / float(N_repeat)
        t_VeriArray_avg = np.sum(t_VeriArray_hist, axis=0) / float(N_repeat)

        print("\n\n")
        print(f"N={N}, U={U}, T={T}, d={d}")
        print(f"Average runtime over {N_repeat} repetition.")
        print(f"t_Total = {t_Total_avg}")
        print(f"Aggregation  phase: Total={t_AggTotal_avg}, (round 0, 1, 2, 3)={t_AggArray_avg}")
        print(f"Verification phase: Total={t_VeriTotal_avg}, (round 0, 1, 2)={t_VeriArray_avg}")
        print()

        time_out = []
        result_set = {'N': N
            , 'T': T
            , 'd': d
            , 't_Total': t_Total_avg
            , 't_AggTotal': t_AggTotal_avg
            , 't_AggArray': t_AggTotal_hist
            , 't_VeriTotal': t_VeriTotal_avg
            , 't_VeriArray': t_VeriArray_hist
                      #   'drop_rate': drop_rate
                      }

        time_out.append(result_set)

        pickle.dump(time_out, open('./results/LightVeriFL_fast_amortized_N' + str(N) +'_U' + str(U) + '_d' + str(d) + '_L' + str(batch), 'wb'), -1)

    elif rank <= N:
        for i_trial in range(N_repeat):
            # comm.Barrier()
            ##########################################
            ##           Users start HERE           ##
            ##########################################

            '''
            Aggregation Round 0. AdvertiseKeys:
                - this phase does not depend on the local model
            '''
            comm.Barrier()
            t0 = time.time()

            # 0.0. Send my public keys
            my_sk = np.random.randint(0, p_model, size=(2)).astype('int64')
            my_pk = my_pk_gen(my_sk, p_model, 0)

            my_key = np.concatenate((my_pk, my_sk))
            # print(f"my_key = {my_key}")

            comm.Send(my_key[0:2], dest=0)  # send public key to the server

            # print '[ rank= ',rank,']', my_key[0:2]

            public_key_list = np.empty(num_pk_per_user * N).astype('int64')
            comm.Recv(public_key_list, source=0)

            public_key_list = np.reshape(public_key_list, (num_pk_per_user, N))

            # 0.1 generate z and n_array

            # t0_offline = time.time()

            # t_offline_comm = time.time() - t0_offline_comm
            # t_offline = time.time() - t0_offline
            comm.Barrier()

            '''
            Aggregation Round 1. ShareMetadata:
                - generate z (masking to hash)
                - generate n_array (additional mask used to encoding)
                - LightVeriFL encoding - z_tilde's
                - Exchange z_tilde's
                - 1.0. Share Key
                - 1.1. Local model training 
                - Generate hash (h_i)
                - Generate & exchange commitment (c_i) 
            '''
            comm.Barrier()

            # 1.0 Generate z and n_array
            z = generate_point_EC()  # z is a randomly selected point on EC
            n_array = []  # n_array comes from the EC
            for x in range(T):
                n_array.append(generate_point_EC())

            # 1.1 LightVeriFL encoding to generate z_tilde's
            z_tilde_array = LightVeriFL_enc_EC(z, n_array, alpha_s, beta_s, P256)

            # t_offline_enc = time.time() - t0_offline

            # 1.2 Exchange z_tilde with all other users

            # t0_offline_comm = time.time()
            z_tilde_buffer = [0] * N
            z_tilde_buffer[rank - 1] = z_tilde_array[rank - 1]

            tx_dest = np.delete(range(N), rank - 1)
            for j in range(len(tx_dest)):
                bf_addr = tx_dest[j]
                tx_rank = tx_dest[j] + 1

                tx_data = z_tilde_array[bf_addr]
                comm.send(tx_data, dest=tx_rank)

            rx_source = np.delete(range(N), rank - 1)
            for i in range(len(rx_source)):
                bf_addr = rx_source[i]
                rx_rank = rx_source[i] + 1
                z_tilde_buffer[bf_addr] = comm.recv(source=rx_rank)

            # 1.3 Share key

            # 1.3.1 generate b_u, s_uv

            s_pk_list = public_key_list[1, :]
            my_s_sk = my_key[3]
            my_c_sk = my_key[2]

            b_u = my_c_sk
            s_uv = np.mod(s_pk_list * my_s_sk, p_model)

            mask = np.zeros(d, dtype='int64')
            for i in range(1, N + 1):
                if rank == i:
                    np.random.seed(b_u)
                    # print(f"rank={rank}, b_u={b_u}")
                    temp = np.random.randint(0, p_model, size=d, dtype='int64')
                    # temp = np.zeros(d,dtype='int')
                    mask = np.mod(mask + temp, p_model)
                elif rank > i:
                    np.random.seed(s_uv[i - 1])
                    temp = np.random.randint(0, p_model, size=d, dtype='int64')
                    mask = np.mod(mask + temp, p_model)
                else:
                    np.random.seed(s_uv[i - 1])
                    temp = -np.random.randint(0, p_model, size=d, dtype='int64')
                    mask = np.mod(mask + temp, p_model)
            mask = mask.tolist()

            # 1.3.2. generate SS of b_u, s_sk
            b_u_SS = SS_encoding(my_c_sk, N, T, p_model)
            s_sk_SS = SS_encoding(my_s_sk, N, T, p_model)

            # 1.3.3. Send the SS to the server
            comm.Send(np.array(b_u_SS).astype('int64'), dest=0)
            comm.Send(np.array(s_sk_SS).astype('int64'), dest=0)

            # 1.3.4. Receive the other users' SS from the server

            b_u_SS_others = np.empty(N, dtype='int64')
            s_sk_SS_others = np.empty(N, dtype='int64')

            comm.Recv(b_u_SS_others, source=0)
            comm.Recv(s_sk_SS_others, source=0)

            # comm.Barrier()

            # 1.4 Local model training

            # generate gradient randomly for now. These will come from the training
            x_i = [rank] * d

            # 1.5 Generate hash (h_i) and and mask hash with another point on the EC
            # t0_hash_gen = time.time()

            # generate client hashes based on their local gradients
            h = generate_hash(x_i, alpha, d)
            # print(f"hash of {rank}: {h}\n")
            hz = (h + z)

            # 1.6. Generate & exchange commitment (c_i)

            # 1.6.0 Generate commitment

            Pedersen_noise = generate_point_EC()
            Pedersen_noise_z = (Pedersen_noise + z)  # mask the Pedersen noise
            # Pedersen commitment coefficients
            Pedersen_g = (Pedersen_coeff[0] * curve_g)
            Pedersen_l = (Pedersen_coeff[1] * curve_g)

            commitment = generate_Pedersen_commitment(h, Pedersen_g, Pedersen_l, Pedersen_noise)

            # 1.6.1 Exchange commitment with all the other clients

            for tx_rank in range(1, N + 1):
                if rank is not tx_rank:
                    comm.send(commitment, dest=tx_rank)  # masked user hash

            comm_array = [0] * N
            array_idx3 = 0
            for i in range(N):
                rx_rank = i + 1
                if rank == rx_rank:
                    comm_array[array_idx3] = commitment
                else:
                    comm_array[array_idx3] = comm.recv(source=rx_rank)
                array_idx3 += 1

            comm.Barrier()

            # t_hash_gen = time.time() - t0_hash_gen

            '''
            Aggregation Round 2. MaskedInputCollection
                - Send the local model to the server
                - Send masked hash and noise to the server.
            '''
            comm.Barrier()

            all_user_set = list(range(N))
            dropped_users_indexes_list = [x for x in all_user_set if
                                          x not in surviving_users_indexes.tolist()]

            # 2.0 Send the local model to the server
            
            # comm.Barrier()
            # if rank in U1 + 1:  # same as above
            #     comm.send(x_i, dest=0)

            y_i = [0] * d
            for i in range(d):
                y_i[i] = int(x_i[i] + mask[i] % p_model)
            # y_i = list(y_i)

            # print(f"x ={x_i}, mask = {mask}, y={y_i}")
            if rank in U1 + 1:
                comm.send(y_i, dest=0)
            # comm.Send(np.array(x_i), dest=0)

            # 2.1 Send the masked hash and noise to the server
            # comm.Barrier()
            comm.send(hz, dest=0)  # masked user hash

            # comm.Barrier()
            comm.send(Pedersen_noise_z, dest=0)  # masked user noise

            comm.Barrier()

            '''
            Aggregation Round 3. Unmasking
                - 3.0. Send SS to the server (b_u_SS or s_sk_SS)
                - 3.1. Receive the aggregate model from the server
            '''
            comm.Barrier()

            # 3.0. Send SS
            SS_info = np.empty(N, dtype='int64')
            for i in range(N):
                if drop_info[i] == 0:
                    SS_info[i] = b_u_SS_others[i]
                else:
                    SS_info[i] = s_sk_SS_others[i]
            comm.Send(SS_info, dest=0)

            # 3.1 Receive the aggregate model from the server
            # Now all users receive the aggregate model from the server
            # comm.Barrier()
            if rank in surviving_users_indexes + 1: #range(1, N + 1): #
                # Surviving users receive the aggregate gradient from the server and generate the h_agg
                agg_grad = np.zeros((d,), dtype='int64')
                comm.Recv(agg_grad, source=0)
                # This step occurs at the verification round 1 to make it the same as VeriFL
                # 3.2 Compute the aggregate model's hash
                # h_agg = generate_hash(agg_grad, alpha, d)

            comm.Barrier()

            '''
            Verification Round 0. AggregateDecommitting
            - 0.0. Surviving users send the z_tilde_mul to the server  
            - 0.1. Receive the aggregate hash and noise from the server
            - 0.2. Verify that these reconstructions are correct using the aggregate commitment
            - 0.3. Generate the aggregate hash using the aggregate model received from the server
            - 0.4. Verify the integrity of the aggregation using the aggregate hash
            '''
            comm.Barrier()

            # 0.0.0. Send z_tilde_mul for surviving users
            # comm.Barrier()
            if rank in surviving_users_indexes_actual + 1: #surviving_users_indexes + 1:
                surviving_users_indexes_list = surviving_users_indexes.tolist()
                z_tilde_buffer_surviving = [z_tilde_buffer[i] for i in surviving_users_indexes_list]
                z_tilde_mul_surviving = PI_addEC(z_tilde_buffer_surviving)
                comm.send(z_tilde_mul_surviving, dest=0)

            # 0.0.1. Send z_tilde_mul for dropped users
            if rank in surviving_users_indexes_actual + 1: #surviving_users_indexes + 1:
                z_tilde_buffer_dropped = [z_tilde_buffer[i] for i in dropped_users_indexes_list]
                z_tilde_mul_dropped = PI_addEC(z_tilde_buffer_dropped)  # for the dropped users
                comm.send(z_tilde_mul_dropped, dest=0)

            # 0.1. Receive the aggregate hash and noise from the server

            if rank in surviving_users_indexes + 1:
                t0_verification = time.time()

                # 0.1.0. Receive the aggregate hash from the server
                result = [0]
                result = comm.recv(source=0)
                agg_client_hashes_epochs[i_trial] = result
                #print('result', result)

                # 0.1.1. Receive the aggregate noise from the server
                result_noise = [0]
                result_noise = comm.recv(source=0)

                # 0.2. Verify the received aggregate hash from the server using the commitments

                # first compute the aggregate's commitment (aggregate comes from the server)
                # then compare it against the sum of the individual commitments
                # if equal, it means that the server correctly reconstructed the aggregate hashes
                # then use these to verify the aggregate model

                # 0.2.0. Compute the aggregate commitment using the reconstructed aggregate hash and noise coming from the server
                commitments_agg = generate_Pedersen_commitment(result, Pedersen_g, Pedersen_l, result_noise)
                # 0.2.1. Compute the aggregate commitment using the individual commitments received in Aggregation Round 1
                commitments_summed = Point(0, 0, curve=None)
                for i in range(N):
                    tempp = Point(comm_array[i].x, comm_array[i].y, curve=P256)
                    commitments_summed = ( tempp + commitments_summed)
                    commitments_summed = Point(commitments_summed.x, commitments_summed.y, curve=P256)

                commitments_summed = (commitments_summed + (-(N - 1) * (Pedersen_g + Pedersen_l)))

                # 0.2.2. Verify the commitments

                if commitments_agg == commitments_summed:
                    print(f"Verification @ rank {rank}: Server reconstruction is correct. Proceed.")
                    pass
                else:
                    print(f"Verification @ rank {rank}: Forged reconstruction at the server. Process terminated.")

            comm.Barrier()

            '''
            Verification Round 1. BatchChecking
            - Compute the aggregate hash using the aggregate model received from the server
            - Run verification step over the received aggregate hash (result) and computed aggregate hash based on the received aggregate model (h_agg).
            '''

            comm.Barrier()

            # This part happens only in the end of N_repeat now. When i_trial = N_repeat-1
            # 1.0 Compute the aggregate model's hash
            if rank in surviving_users_indexes + 1:
                server_agg_gradient_epochs[i_trial] = agg_grad
                if i_trial == N_repeat-1:
                    # print('h_agg', h_agg)
                    h_agg = generate_hash([sum(x) for x in zip(*server_agg_gradient_epochs)], alpha, d)
                    #print('h_agg', h_agg)

                    # 1.1 Perform the verification
                    result_amortized = Point(0, 0, curve=None)
                    for ind in range(N_repeat):
                        tempp = Point(agg_client_hashes_epochs[ind].x, agg_client_hashes_epochs[ind].y, curve=P256)
                        result_amortized = (result_amortized + tempp)
                        result_amortized = Point(result_amortized.x, result_amortized.y, curve=P256)

                    if rank in surviving_users_indexes + 1:
                        if h_agg == result_amortized:
                            print(f"Verification @ rank {rank}: Correct aggregation.")
                            pass
                        # otherwise forged aggregation
                        else:
                            print(f"Verification @ rank {rank}: Forged aggregation. Process terminated.")

                t_verification = time.time() - t0_verification

            comm.Barrier()
