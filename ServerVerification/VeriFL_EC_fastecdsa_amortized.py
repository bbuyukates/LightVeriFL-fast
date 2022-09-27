import logging
import pickle as pickle
import sys
import time
import random

from fastecdsa.curve import P256
from fastecdsa.point import Point

import numpy as np
from mpi4py import MPI

from utils.function import my_pk_gen, my_key_agreement, BGW_encoding, BGW_decoding, SS_decoding, SS_encoding
from utils.EC import BGW_encoding_EC, BGW_decoding_EC, generate_hash, generate_point_EC, \
    PI_addEC, generate_Pedersen_commitment, curve_g, curve_n
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

N_repeat = batch # batch size for amortization
server_agg_gradient_epochs = [None] * N_repeat
agg_client_hashes_epochs = [None] * N_repeat

num_pk_per_user = 2

if __name__ == "__main__":

    U1 = np.arange(
        N)  # surviving users who send the masked model to the server. We assume that all users are surviving in the aggregation phase.
    drop_info = np.zeros((N,), dtype=int)

    T = int(np.floor(N / 2)) #U = int(0.9*N)

    alpha_s = np.arange(0, T+1)  # U evaluation points for encoding inputs
    beta_s = np.arange(U, U + N)  # N evaluation points for encoding outputs

    h_array = np.arange(1, N + 1)

    if rank == 0:

        logging.info(f"N,U,T={N},{U},{T}, starts!! ")
        surviving_users_indexes = np.random.choice(np.arange(N), U, replace=False).astype(int)
        surviving_users_indexes = np.sort(surviving_users_indexes)
        surviving_users_indexes_actual = random.sample(surviving_users_indexes.tolist(), T+1)
        surviving_users_indexes_actual = np.sort(surviving_users_indexes_actual)
        print(surviving_users_indexes_actual)

        #print('surviving indexes are', surviving_users_indexes)
        for tx_rank in range(1, N + 1):
            comm.Send(surviving_users_indexes, dest=tx_rank)

        for tx_rank in range(1, N + 1):
            comm.Send(np.array(surviving_users_indexes_actual), dest=tx_rank)

        # sample distinct alpha values, alpha_i cannot be 0
        alpha = np.array(random.sample(range(1, d + 1), k=d))  # is fine as long as d < curve.n which is the case for us
        for tx_rank in range(1, N + 1):
            comm.Send(alpha, dest=tx_rank)

        # sample coefficients for the Pedersen commitment scheme. These will be used to generate g and l of the Pedersen commitment scheme.
        Pedersen_coeff = np.array(random.sample(range(1, 100), 2))  # we can use curve.n instead of 100 here.

        for tx_rank in range(1, N + 1):
            comm.Send(Pedersen_coeff, dest=tx_rank)

    elif rank <= N:
        surviving_users_indexes = np.zeros((U,), dtype=int)
        comm.Recv(surviving_users_indexes, source=0)

        surviving_users_indexes_actual = np.zeros((T+1,), dtype=int)
        comm.Recv(surviving_users_indexes_actual, source=0)

        alpha = np.zeros((d,), dtype=int)
        comm.Recv(alpha, source=0)
        # logging.info(f'[ rank= {rank} ] surviving_users = {surviving_users}' )

        Pedersen_coeff = np.zeros((2,), dtype=int)
        comm.Recv(Pedersen_coeff, source=0)

    if rank == 0:

        t_Total_hist = np.zeros((N_repeat,))
        t_AggTotal_hist = np.zeros((N_repeat,))
        t_VeriTotal_hist = np.zeros((N_repeat,))
        t_AggArray_hist = np.zeros((N_repeat, 4))
        t_VeriArray_hist = np.zeros((N_repeat, 3))

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
            print(f'   ({i_trial} / {N_repeat}) - th Trial!!!!')
            print('   Aggregation phase starts!!  ')

            comm.Barrier()
            t0 = time.time()

            # 0.0. Receive the public keys

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

            # print np.shape(b_u_SS_list)

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
            t_AggRound1 = time.time() - t1

            '''
            Aggregation Round 2. MaskedInputCollection
            '''
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
            #print('Before unmasking, agg masked grad is', agg_grad[:5])

            comm.Barrier()
            t_AggRound2 = time.time() - t2

            '''
            Aggregation Round 3. Unmasking
            '''
            comm.Barrier()
            t3 = time.time()

            # 3.1. Receive SS (b_u_SS for surviving users, s_sk_SS for dropped users)
            SS_rx = np.empty((N, N), dtype='int64')
            for i in range(N):
                data = np.empty(N, dtype='int64')
                comm.Recv(data, source=i + 1)
                SS_rx[:, i] = data

            # 3.2. Generate PRG based on the seed

            for i in range(N):
                if drop_info[i] == 0:

                    SS_input = np.reshape(SS_rx[i, 0:T + 1], (T + 1,)).tolist()
                    b_u = SS_decoding(SS_input, range(T + 1), p_model)
                    np.random.seed(b_u)

                    # print(f"rank = {i+1}, reconstruct b_u = {b_u}")

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

            #print('After  unmasking, agg masked grad is', agg_grad[:5])

            # 3.3. Send the aggregate gradient to the surviving users
            for tx_rank in surviving_users_indexes + 1:
                comm.Send(np.array(agg_grad), dest=tx_rank)

            comm.Barrier()
            t_AggRound3 = time.time() - t3
            t_AggTotal = time.time() - t0

            '''
            Verification Round 0 (Decommitting)
                - Receive hashes from the surviving users.
                - Broadcasts the received hashes to the surviving users
            '''
            
            comm.Barrier()
            t0_verification = time.time()

            print()
            print('----------------------------------')
            print('   Verification phase starts!!  ')
            print('surviving indexes are', surviving_users_indexes)

            comm.Barrier()
            t_VeriRound0 = time.time() - t0_verification

            '''
            Verification Round 1 (DroppedDecommitting)
                - Receive secret shares of hashes from the surviving users.
                - Reconstruct the hashes of the dropped users
            '''
            comm.Barrier()
            t1_verification = time.time()

            rows, cols = (N - len(surviving_users_indexes), len(surviving_users_indexes_actual))
            h_SS_buffer = [[[] for j in range(cols)] for i in range(rows)]
            # np.zeros((len(surviving_users_indexes), N-len(surviving_users_indexes)), dtype = int)
            buffer_idx = 0
            for i in surviving_users_indexes_actual:
                # comm.Recv(h_SS_buffer[buffer_idx], source = i + 1)
                for j in range(rows):
                    h_SS_buffer[j][buffer_idx] = comm.recv(source=i + 1)
                buffer_idx += 1

            # print(f"h_SS_buffer={h_SS_buffer}")
            dropped_users_indexes = list(set(range(N)) - set(surviving_users_indexes))
            h_dec_array = []
            for i in range(len(dropped_users_indexes)):
                h_dec = BGW_decoding_EC(h_SS_buffer[i], surviving_users_indexes_actual, P256)
                h_dec_array.append(h_dec)

            # do a similar reconstruction for noise ss of the dropped users
            noise_SS_buffer = [[[] for j in range(cols)] for i in range(rows)]
            buffer_idx2 = 0
            for i in surviving_users_indexes_actual:
                for j in range(rows):
                    noise_SS_buffer[j][buffer_idx2] = comm.recv(source=i + 1)
                buffer_idx2 += 1

            noise_dec_array = []
            for i in range(len(dropped_users_indexes)):
                noise_dec = BGW_decoding_EC(noise_SS_buffer[i], surviving_users_indexes_actual, P256)
                noise_dec_array.append(noise_dec)

            # server sends the reconstructed hashes and noises of the dropped users back to the surviving users

            for tx_rank in surviving_users_indexes + 1:
                for i in range(len(dropped_users_indexes)):
                    comm.send(h_dec_array[i], dest=tx_rank)
                # comm.Send(np.array(h_dec_array), dest=tx_rank)

            for tx_rank in surviving_users_indexes + 1:
                for i in range(len(dropped_users_indexes)):
                    comm.send(noise_dec_array[i], dest=tx_rank)
                # comm.Send(np.array(noise_dec_array), dest=tx_rank)

            comm.Barrier()
            t_VeriRound1 = time.time() - t1_verification
            '''
            Verification Round 2 (BatchChecking)
                - NoOP in the server
            '''
            comm.Barrier()
            t2_verification = time.time()

            comm.Barrier()
            t_VeriRound2 = time.time() - t2_verification
            t_VeriTotal = time.time() - t0_verification
            t_Total = time.time() - t0

            t_AggArray = np.array([t_AggRound0, t_AggRound1, t_AggRound2, t_AggRound3])
            t_VeriArray = np.array([t_VeriRound0, t_VeriRound1, t_VeriRound2])

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

        pickle.dump(time_out, open('./results/VeriFL_fast_amortized_N' + str(N) +'_U' + str(U) + '_d' + str(d) + '_L' + str(batch), 'wb'), -1)


    elif rank <= N:
        for i_trial in range(N_repeat):
            # comm.Barrier()
            ##########################################
            ##           Users start HERE           ##
            ##########################################

            '''
            Aggregation Round 0. AdvertiseKeys:
            '''
            comm.Barrier()
            t0 = time.time()

            # 0.0. Send my public keys
            my_sk = np.random.randint(0, p_model, size=(2)).astype('int64')
            my_pk = my_pk_gen(my_sk, p_model, 0)

            my_key = np.concatenate((my_pk, my_sk))

            comm.Send(my_key[0:2], dest=0)  # send public key to the server

            # 0.1. Rx public key list from the server
            public_key_list = np.empty(num_pk_per_user * N).astype('int64')
            comm.Recv(public_key_list, source=0)

            public_key_list = np.reshape(public_key_list, (num_pk_per_user, N))

            comm.Barrier()

            '''
            Aggregation Round 1. ShareMetadata:
                - Share Key
                - Local model training 
                - Generate hash (h_i)
                - Generate & exchange Secret Shares
                - Generate & exchange commitment (c_i) 
            '''

            # Round 1.0 Share Key
            # comm.Barrier()
            # 1.0.1 generate b_u, s_uv
            comm.Barrier()
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

            # 1.0.2. generate SS of b_u, s_sk
            b_u_SS = SS_encoding(my_c_sk, N, T, p_model)
            s_sk_SS = SS_encoding(my_s_sk, N, T, p_model)

            # print(f"b_u_SS={b_u_SS[0]}, {b_u_SS_new[0]}")

            # 1.0.3. Send the SS to the server
            comm.Send(np.array(b_u_SS).astype('int64'), dest=0)
            comm.Send(np.array(s_sk_SS).astype('int64'), dest=0)

            # 1.0.4. Receive the other users' SS from the server

            b_u_SS_others = np.empty(N, dtype='int64')
            s_sk_SS_others = np.empty(N, dtype='int64')

            comm.Recv(b_u_SS_others, source=0)
            comm.Recv(s_sk_SS_others, source=0)

            # comm.Barrier()

            ## 1.1. Local training
            x_i = [rank] * d

            ## 1.2. Hash generation
            t0_hash_gen = time.time()

            # generate client hashes based on their local gradients
            h = generate_hash(x_i, alpha, d)
            # print(f"hash of user {rank-1}: {h}\n")

            t_hash_gen = time.time() - t0_hash_gen

            # noise generation, r_i in the VeriFL paper
            Pedersen_noise = generate_point_EC()
            # print(f"noise of user {rank-1}: {Pedersen_noise}\n")

            ## 1.3. Gen secret sharing of hash and noise
            t0_SS = time.time()
            h_SS = BGW_encoding_EC(h, N, T, P256)
            noise_SS = BGW_encoding_EC(Pedersen_noise, N, T, P256)
            t_SS_gen = time.time() - t0_SS

            # comm.Barrier()
            t0_SS_comm = time.time()

            h_SS_buffer = [0] * N

            for i in range(N):
                rx_rank = i + 1

                if rx_rank == rank:
                    h_SS_buffer[i] = h_SS[i]
                    for j in range(N):
                        if j is not rank - 1:
                            # print(f"{rank} sends SS to {j+1}")
                            comm.send(h_SS[j], dest=j + 1)

                else:
                    # print(f"{rank} receives SS from {i+1}")
                    h_SS_buffer[i] = comm.recv(source=i + 1)

            noise_SS_buffer = [0] * N

            for i in range(N):
                rx_rank1 = i + 1

                if rx_rank1 == rank:
                    noise_SS_buffer[i] = noise_SS[i]
                    for j in range(N):
                        if j is not rank - 1:
                            # print(f"{rank} sends SS to {j+1}")
                            comm.send(noise_SS[j], dest=j + 1)

                else:
                    # print(f"{rank} receives SS from {i+1}")
                    noise_SS_buffer[i] = comm.recv(source=i + 1)

            # print(f"rank={rank}, h_SS={h_SS}, h_SS_buffer={h_SS_buffer}")

            ## 1.3. Generate commitment and send commitment to every other client

            # generate Pedersen commitment
            # Pedersen commitment coefficients
            Pedersen_g = (Pedersen_coeff[0] * curve_g)
            Pedersen_l = (Pedersen_coeff[1] * curve_g)

            commitment = generate_Pedersen_commitment(h, Pedersen_g, Pedersen_l, Pedersen_noise)

            # each client sends its commitment to all the other clients
            for tx_rank in range(1, N + 1):
                if rank is not tx_rank:
                    comm.send(commitment, dest=tx_rank)

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

            t_SS_comm = time.time() - t0_SS_comm

            '''
            Aggregation Round 2. MaskedInputCollection
            '''
            comm.Barrier()
            # if rank in U1 + 1:  # same as above
            #     comm.send(x_i, dest=0)

            y_i = [0] * d
            for i in range(d):
                y_i[i] = int(x_i[i] + mask[i] % p_model)

            # print(f"x ={x_i}, mask = {mask}, y={y_i}")
            if rank in U1 + 1:
                comm.send(y_i, dest=0)

            comm.Barrier()

            '''
            Aggregation Round 3. Unmasking
                - Send secret shares to the server
                - Receive the aggregate of models from the server
            '''
            comm.Barrier()

            # 3.1. Send SS
            SS_info = np.empty(N, dtype='int64')
            for i in range(N):
                if drop_info[i] == 0:
                    SS_info[i] = b_u_SS_others[i]
                else:
                    SS_info[i] = s_sk_SS_others[i]
            comm.Send(SS_info, dest=0)

            # 3.3. Receive global model from the server
            agg_grad = np.zeros((d,), dtype=int)
            if rank in surviving_users_indexes + 1:
                comm.Recv(agg_grad, source=0)

            comm.Barrier()

            '''
            Verification Round 0 (Decommitting)
                - Receive hashes from the surviving users.
                - Broadcasts the received hashes to the surviving users
            '''
            comm.Barrier()
            # Surviving users send h_i and r_i to each other.

            if rank in surviving_users_indexes + 1:
                for tx_rank in surviving_users_indexes + 1:
                    if rank is not tx_rank:
                        comm.send(h, dest=tx_rank)  # send plain hash to other surviving users

            if rank in surviving_users_indexes + 1:
                hash_array = [0] * len(surviving_users_indexes)
                arr_idx = 0
                for i in surviving_users_indexes:
                    rx_rank = i + 1
                    if rank == rx_rank:
                        hash_array[arr_idx] = h
                    else:
                        hash_array[arr_idx] = comm.recv(source=rx_rank)
                    arr_idx += 1

            if rank in surviving_users_indexes + 1:
                for tx_rank in surviving_users_indexes + 1:
                    if rank is not tx_rank:
                        comm.send(Pedersen_noise, dest=tx_rank)  # send plain noise to other surviving users

            if rank in surviving_users_indexes + 1:
                noise_array = [0] * len(surviving_users_indexes)
                arr_idx2 = 0
                for i in surviving_users_indexes:
                    rx_rank = i + 1
                    if rank == rx_rank:
                        noise_array[arr_idx2] = Pedersen_noise
                    else:
                        noise_array[arr_idx2] = comm.recv(source=rx_rank)
                    arr_idx2 += 1
            comm.Barrier()

            '''
            Verification Round 1 (DroppedDecommitting)
                - Surviving users send ss of the dropped users to the server
                - Surviving users receive the reconstructed hash and noise of the dropped users
            '''

            comm.Barrier()

            # send h_SS of dropped users
            dropped_users_indexes = list(set(range(N)) - set(surviving_users_indexes))
            if rank in surviving_users_indexes_actual + 1: #surviving_users_indexes + 1:
                for i in dropped_users_indexes:
                    comm.send(h_SS_buffer[i], dest=0)
                # comm.Send(h_SS_buffer[dropped_users_indexes], dest = 0)

            # send noise_SS of dropped users
            if rank in surviving_users_indexes_actual + 1: #surviving_users_indexes + 1:
                for i in dropped_users_indexes:
                    comm.send(noise_SS_buffer[i], dest=0)
                # comm.Send(h_SS_buffer[dropped_users_indexes], dest = 0)

            # surviving users receive the dropped users' hashes and noises
            if rank in surviving_users_indexes + 1:
                h_dropped_buffer = []
                for i in range(N - len(surviving_users_indexes)):
                    tmp = comm.recv(source=0)
                    h_dropped_buffer.append(tmp)
                # print(h_dropped_buffer)

            if rank in surviving_users_indexes + 1:
                noise_dropped_buffer = []
                for i in range(N - len(surviving_users_indexes)):
                    tmp = comm.recv(source=0)
                    noise_dropped_buffer.append(tmp)

            comm.Barrier()

            '''
            Verification Round 2 (BatchChecking)
                - Run verification step over.
                - Implemented amortized-verification 
            '''

            comm.Barrier()

            # Clients first verify the recovered hashes by checking the commitments

            if rank in surviving_users_indexes + 1:
                for i in range(N - len(surviving_users_indexes)):
                    # reconstruct the commitment of ith dropped user
                    tmp_hash = h_dropped_buffer[i]
                    tmp_noise = noise_dropped_buffer[i]
                    comm_reconstructed = generate_Pedersen_commitment(tmp_hash, Pedersen_g, Pedersen_l, tmp_noise)
                    if comm_reconstructed.x == comm_array[dropped_users_indexes[i]].x and comm_reconstructed.y == comm_array[dropped_users_indexes[i]].y:
                        print(f"Verification @ rank {rank}: Server reconstruction is correct. Proceed.")
                        pass
                    else:
                        print(f"Verification @ rank {rank}: Forged reconstruction at the server. Process terminated.")

                # If commitments do not fail, clients verify the aggregation

                server_agg_gradient_epochs[i_trial] = agg_grad
                # Each client computes the aggregate hash only when i_trial = N_repeat - 1
                hashes_summed = Point(0, 0, curve=None)

                # add surviving users' hashes
                for i in range(len(surviving_users_indexes)):
                    temppp = Point(hash_array[i].x, hash_array[i].y, curve=P256)
                    hashes_summed = (temppp + hashes_summed)
                    hashes_summed = Point(hashes_summed.x, hashes_summed.y, curve=P256)

                # add dropped users' hashes
                for i in range(len(dropped_users_indexes)):
                    temppp = Point(h_dropped_buffer[i].x, h_dropped_buffer[i].y, curve=P256)
                    hashes_summed = (temppp + hashes_summed)
                    hashes_summed = Point(hashes_summed.x, hashes_summed.y, curve=P256)

                agg_client_hashes_epochs[i_trial] = hashes_summed

                if i_trial == N_repeat - 1:
                    result_amortized = Point(0, 0, curve=None)
                    for ind in range(N_repeat):
                        temppp = Point(agg_client_hashes_epochs[ind].x, agg_client_hashes_epochs[ind].y, curve=P256)
                        result_amortized = (result_amortized + temppp)
                        result_amortized = Point(result_amortized.x, result_amortized.y, curve=P256)

                    agg_hash = generate_hash([sum(x) for x in zip(*server_agg_gradient_epochs)], alpha, d)
                    if agg_hash == result_amortized:
                        print(f"Verification @ rank {rank}: Correct aggregation.")
                        pass
                    # otherwise forged aggregation
                    else:
                        print(f"Verification @ rank {rank}: Forged aggregation. Process terminated.")

            comm.Barrier()
