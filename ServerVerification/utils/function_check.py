import numpy as np
import time


# np.random.seed(42)  # set the seed of the random number generator for consistency

def modular_inv(a, p):
    x, y, m = 1, 0, p
    while a > 1:
        q = a // m;
        t = m;

        m = np.mod(a, m)
        a = t
        t = y

        y, x = x - np.int64(q) * np.int64(y), t

        if x < 0:
            x = np.mod(x, p)
    return np.mod(x, p)


def divmod(_num, _den, _p):
    # compute num / den modulo prime p
    _num = np.mod(_num, _p)
    _den = np.mod(_den, _p)
    _inv = modular_inv(_den, _p)
    # print(_num,_den,_inv)
    return np.mod(np.int64(_num) * np.int64(_inv), _p)


def PI(vals, p):  # upper-case PI -- product of inputs
    accum = 1
    # print vals
    for v in vals:
        # print accum, v, np.mod(v,p)
        tmp = np.mod(v, p)
        accum = np.mod(accum * tmp, p)
    return accum


def pow(x, a, p):
    out = 1
    for i in range(abs(a)):
        # print(f"out={out}, x={x}")
        out = np.mod(out * x, p)

    if a < 0:
        return modular_inv(out, p)
    else:
        return out


def eff_pow(x, a, p):
    if a < 0:
        return modular_inv(eff_pow(x, -a, p), p)
    elif a == 1:
        return x
    else:
        half = eff_pow(x, a // 2, p)

        if a % 2 == 0:
            return np.mod(half * half, p)
        else:
            return np.mod(np.mod(half * half, p) * x, p)


def my_q(X, q_bit, p):
    X_int = np.round(X * (2 ** q_bit))
    is_negative = (abs(np.sign(X_int)) - np.sign(X_int)) / 2
    out = X_int + p * is_negative
    return out.astype('int64')


def my_q_inv(X_q, q_bit, p):
    flag = X_q - (p - 1) / 2
    is_negative = (abs(np.sign(flag)) + np.sign(flag)) / 2
    X_q = X_q - p * is_negative
    return X_q.astype(float) / (2 ** q_bit)


def gen_Lagrange_coeffs(evalpoints_in, evalpoints_out, p, is_K1=0):
    '''
    input
        - evalpoints_in  : array of evaluation points corresponding to encoding inputs (K)
        - evalpoints_out : array of evaluation points corresponding to encoding outputs (N)
        - p : finite field size
        - is_K1 : True when K=1

    output
        - U : matrix of lagrange coefficients (K x N)
    '''
    U = np.zeros((len(evalpoints_out), len(evalpoints_in)), dtype='int64')

    w = np.zeros((len(evalpoints_in)), dtype='int64')
    for j in range(len(evalpoints_in)):
        cur_beta = evalpoints_in[j]
        den = PI([cur_beta - o for o in evalpoints_in if cur_beta != o], p)
        w[j] = den

    l = np.zeros((len(evalpoints_out)), dtype='int64')
    for i in range(len(evalpoints_out)):
        l[i] = PI([evalpoints_out[i] - o for o in evalpoints_in], p)

    for j in range(len(evalpoints_in)):
        for i in range(len(evalpoints_out)):
            den = np.mod(np.mod(evalpoints_out[i] - evalpoints_in[j], p) * w[j], p)
            U[i][j] = divmod(l[i], den, p)
            print('U ij is', U[i][j])
    return U.astype('int64')


def LightVeriFL_enc(z, n_array, evalpoints_in, evalpoints_out, p):
    '''
    Note
        - all inputs and outputs are beloing to the finite field.
    input
        - z : scalor
        - n : size T array (T: privacy parameter)
        - evalpoints_in : evaluation points for (T+1) original points.
                          evalpoints_in[0] corresponds to evaluation point for z
                          evalpoints_in[1:] corresponds to evaluation points for n_array
        - evalpoints_out : evaluation points for (N) encoded output points.
        - p : finite field size
    output
        - output : size N array which correponds to encoded hash
    '''

    assert len(evalpoints_in) == len(
        n_array) + 1, "@LightVeriFL_enc, length of evalpoints_in should be T(=length of n_array) + 1 (size of z, i.e., scalor) !!"

    W = gen_Lagrange_coeffs(evalpoints_in, evalpoints_out, p)
    print('W is', W)
    flag = W - (p - 1) / 2
    print('flag is', flag)
    is_negative = (abs(np.sign(flag)) + np.sign(flag)) / 2
    print('is negative is', is_negative)
    W = (W - p * is_negative).astype(int)
    print('W becomes', W)

    output = np.zeros((len(evalpoints_out),), dtype='int64')

    for i in range(len(output)):

        output[i] = eff_pow(z, W[i, 0], p)

        for j in range(len(n_array)):
            output[i] = np.mod(output[i] * eff_pow(n_array[j], W[i, j + 1], p), p)

    return output


def LightVeriFL_dec(z_tilde, evalpoints_in, evalpoints_out, p):
    '''
    Note
        - all inputs and outputs are beloing to the finite field.
    input
        - z_tilde : array of size (T+1)
        - evalpoints_in : evaluation points for (T+1) original points.
                          evalpoints_in[0] corresponds to evaluation point for z
                          evalpoints_in[1:] corresponds to evaluation points for n_array
        - evalpoints_out : (T+1) evaluation points corresponding to z_tilde.
        - p : finite field size
    output
        - output : decoded hash result (=scalor)
    '''

    assert len(z_tilde) == len(
        evalpoints_out), "@LightVeriFL_out, length of evalpoints_out and z_tilde should be the same (=T+1)!!"

    W = gen_Lagrange_coeffs(evalpoints_out, [evalpoints_in[0]], p)
    flag = W - (p - 1) / 2
    is_negative = (abs(np.sign(flag)) + np.sign(flag)) / 2
    W = (W - p * is_negative).astype(int)

    _idx_z = 0
    output = eff_pow(z_tilde[_idx_z], W[0, _idx_z], p)

    for j in range(len(evalpoints_out) - 1):
        output = np.mod(output * eff_pow(z_tilde[j + 1], W[0, j + 1], p), p)

    return output

    # we do not need to reconstruct addtional masks, n_i
    # print(f"W={W}")
    # output = np.zeros((len(evalpoints_in),), dtype = 'int64')
    # for i in range(len(output)):
    #     output[i] = np.mod(pow(z_tilde[0], W[i,0], p) * pow(z_tilde[1], W[i,1], p), p)
    # return output[0]


def BGW_encoding(X, N, T, p):
    m = len(X)
    d = len(X[0])

    alpha_s = range(1, N + 1)
    alpha_s = np.int64(np.mod(alpha_s, p))
    X_BGW = np.zeros((N, m, d), dtype='int64')
    R = np.random.randint(p, size=(T + 1, m, d))
    R[0, :, :] = np.mod(X, p)

    for i in range(N):
        for t in range(T + 1):
            X_BGW[i, :, :] = np.mod(X_BGW[i, :, :] + R[t, :, :] * (alpha_s[i] ** t), p)
    return X_BGW


def gen_BGW_lambda_s(alpha_s, p):
    lambda_s = np.zeros((1, len(alpha_s)), dtype='int64')

    for i in range(len(alpha_s)):
        cur_alpha = alpha_s[i];

        den = PI([cur_alpha - o for o in alpha_s if cur_alpha != o], p)
        num = PI([0 - o for o in alpha_s if cur_alpha != o], p)
        lambda_s[0][i] = divmod(num, den, p)
    return lambda_s.astype('int64')


def BGW_decoding(f_eval, worker_idx, p):  # decode the output from T+1 evaluation points
    # f_eval     : [RT X d ]
    # worker_idx : [ 1 X RT]
    # output     : [ 1 X d ]

    # t0 = time.time()
    max = np.max(worker_idx) + 2
    alpha_s = range(1, max)
    alpha_s = np.int64(np.mod(alpha_s, p))
    alpha_s_eval = [alpha_s[i] for i in worker_idx]
    # t1 = time.time()
    # print(alpha_s_eval)
    lambda_s = gen_BGW_lambda_s(alpha_s_eval, p).astype('int64')
    # t2 = time.time()
    # print(lambda_s.shape)
    f_recon = np.mod(np.dot(lambda_s, f_eval), p)
    # t3 = time.time()
    # print 'time info for BGW_dec', t1-t0, t2-t1, t3-t2
    return f_recon


if __name__ == "__main__":

    # check the runtime of pow and eff_pow
    p = 2 ** 15 - 19

    x = np.random.randint(0, p)
    a = np.random.randint((p - 1) // 2, p)

    t0 = time.time()
    out0 = pow(x, a, p)
    t_pow = time.time() - t0
    print(f"runtime of pow={t_pow}, out0 = {out0}")

    t0 = time.time()
    out1 = eff_pow(x, a, p)
    t_eff_pow = time.time() - t0
    print(f"runtime of pow={t_eff_pow}, out0 = {out1}")

    assert False, "stop!!"

    ##########################################################
    # example for LightSecAgg
    # N = 3 (# total users), U = 2 (# surviving users)
    ##########################################################

    if False:
        N = 3
        U = 2
        alpha_s = np.arange(0, U)
        beta_s = np.arange(U, U + N)
        p = 2 ** 10 - 3

        z_array = np.mod(np.arange(1, N + 1) * 100, p)
        n_array = np.random.randint(0, p, size=N)

        ### 1. encoding
        W_enc = gen_Lagrange_coeffs(alpha_s, beta_s, p)
        print(f"W_enc={W_enc}\n")

        # z_tilde[i][j] : encoded data from user i to user j
        z_tilde = np.zeros((N, N), dtype='int')
        for i in range(N):
            for j in range(N):
                z_tilde[i][j] = np.mod(np.sum(np.array([z_array[i], n_array[i]]) * W_enc[j, :]), p)

        ### 2. Select surviving users
        surviving_users = np.random.choice(np.arange(N), U, replace=False)
        print(surviving_users)

        z_tilde_sum = np.zeros((U,), dtype='int64')

        for i in range(len(surviving_users)):
            user_idx = surviving_users[i]

            z_tilde_sum[i] = np.mod(np.sum(z_tilde[surviving_users, user_idx]), p)
            # print(z_tilde[:,user_idx])
            # print(z_tilde[surviving_users,user_idx])
            # print(z_tilde_sum[i])
            # print()

        ### 3. Decoding
        W_dec = gen_Lagrange_coeffs(beta_s[surviving_users], alpha_s, p)
        print(f"W_dec = {W_dec}\n")

        tmp = np.mod(np.matmul(W_dec, W_enc[surviving_users, :]), p)
        print(f"W_dec * W_enc (it should be identity matrix) = \n {tmp} \n")

        dec = np.mod(np.sum(np.matmul(W_dec[0, :], z_tilde_sum[:])), p)
        print(f"original z_sum = {np.mod(np.sum(z_array[surviving_users]),p)},\nreconstructed z_sum = {dec}")

    ##########################################################
    # example for LightVeriFL
    # N = 4 (# total users), T = 2, U = 3 (# surviving users)
    ##########################################################

    N = 4
    T = 2
    U = T + 1
    alpha_s = np.arange(0, U)
    beta_s = np.arange(U, U + N)
    p = 2 ** 10 - 3

    h_array = np.mod(np.arange(1, N + 1) * 2, p)

    z_array = np.random.randint(0, p, size=(N,))
    n_array = np.random.randint(0, p, size=(N, T))

    buffer_at_server = np.mod(h_array * z_array, p)  # server receives h_i * z_i

    print(f"Test of encoding and decoding function of LightVeriFL.")
    print(f"N={N}, T={T}, U=T+1={U}")
    print(f"hash of N (={N}) users = {h_array}")

    ### 1. encoding @ users

    # z_tilde[i][j] : encoded data from user i to user j
    z_tilde = np.zeros((N, N), dtype='int')
    for i in range(N):
        z_tilde[i, :] = LightVeriFL_enc(z_array[i], n_array[i, :], alpha_s, beta_s, p)

    print(f"\n\n encoding output \n={z_tilde}\n")

    ### 2. Select surviving users
    surviving_users = np.random.choice(np.arange(N), U, replace=False)

    z_tilde_mul = np.zeros((U,), dtype='int64')

    for i in range(len(surviving_users)):
        user_idx = surviving_users[i]

        z_tilde_mul[i] = PI(z_tilde[surviving_users, user_idx], p)

        # print(z_tilde[:,user_idx])
        # print(z_tilde[surviving_users,user_idx])
        # print(z_tilde_sum[i])
        # print()

    # dec = LightVeriFL_dec(z_tilde[0,surviving_users], alpha_s, beta_s[surviving_users], p)
    # print(f"dec= {dec}")

    ### 3. decoding @ server

    hz_mul = PI(buffer_at_server[surviving_users], p)
    dec_z = LightVeriFL_dec(z_tilde_mul, alpha_s, beta_s[surviving_users], p)

    res = divmod(hz_mul, dec_z, p)
    print(f"surviving user indexes: {surviving_users}\n")
    print(f"hash of surviving users= { h_array[surviving_users]}, dec = multication of hashes = {res}")

    ## misc (to check the largest coefficient)
    '''
    N = 500
    U = 250
    alpha_s = np.arange(0, U)
    beta_s  = np.arange(U, U+N)
    p = 2 ** 31 - 1

    z_array = np.mod(np.arange(1,N+1)*100, p)
    n_array = np.random.randint(0,p, size=N)

    ### 1. encoding
    t0 = time.time()
    W_enc = gen_Lagrange_coeffs(alpha_s, beta_s, p)

    flag = W_enc - (p-1)/2
    is_negative = (abs(np.sign(flag)) + np.sign(flag))/2
    W_enc = (W_enc - p * is_negative).astype(int)
    print(time.time() - t0)


    print(f"max = {np.max(W_enc)}")

    t0 = time.time()
    a = np.mod(3 ** np.max(W_enc), p)
    print(a, time.time() - t0)
    '''