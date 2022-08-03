import numpy as np
import time


# np.random.seed(42)  # set the seed of the random number generator for consistency

def modular_inv(k, p):
    """Returns the inverse of k modulo p.
    This function returns the only integer x such that (x * k) % p == 1.
    k must be non-zero and p must be a prime.
    """
    if k == 0:
        raise ZeroDivisionError('division by zero')

    if k < 0:
        # k ** -1 = p - (-k) ** -1  (mod p)
        return p - modular_inv(-k, p)

    # Extended Euclidean algorithm.
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = p, k

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    gcd, x, y = old_r, old_s, old_t

    assert gcd == 1
    assert (k * x) % p == 1

    return x % p


def divmod(_num, _den, _p):
    # compute num / den modulo prime p
    _num = np.mod(_num, _p)
    _den = np.mod(_den, _p)
    _inv = modular_inv(_den, _p)
    # print(_num,_den,_inv)
    return _num * _inv % _p  # np.mod(np.int64(_num) * np.int64(_inv), _p)


def PI(vals, p):  # upper-case PI -- product of inputs
    accum = 1
    # print vals
    for v in vals:
        # print accum, v, np.mod(v,p)
        tmp = v % p  # np.mod(v, p)
        accum = (accum * tmp) % p  # np.mod(accum * tmp, p)
    return accum


def pow(x, a, p):
    # print('a is', a)
    # print('x is', x)
    # x = x % p
    # print('range is', abs(int(a)))
    out = 1
    for i in range(abs(int(a))):
        out = (out * x) % p
    if a < 0:
        return modular_inv(out, p)
    else:
        return out


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
    rows, cols = (len(evalpoints_out), len(evalpoints_in))
    U = [[[] for i in range(cols)] for i in range(rows)]  # [[0] * cols] * rows
    # U = np.zeros((len(evalpoints_out), len(evalpoints_in)), dtype='int64')

    w = []  # np.zeros((len(evalpoints_in)), dtype='int64')
    for j in range(len(evalpoints_in)):
        cur_beta = evalpoints_in[j]
        den = PI([cur_beta - o for o in evalpoints_in if cur_beta != o], p)
        w.append(den)  # w[j] = den

    l = []  # np.zeros((len(evalpoints_out)),dtype='int64')
    for i in range(len(evalpoints_out)):
        l.append(PI([evalpoints_out[i] - o for o in evalpoints_in], p))
        # l[i] = PI([evalpoints_out[i] - o for o in evalpoints_in], p)

    for j in range(len(evalpoints_in)):
        for i in range(len(evalpoints_out)):
            den = np.mod(np.mod(evalpoints_out[i] - evalpoints_in[j], p) * w[j], p)
            U[i][j] = divmod(l[i], den, p)
    return U  # U.astype('int64')


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
    # flag = W - (p-1)/2
    flag = []
    for i in range(len(W)):
        innerArray = []
        for j in range(len(W[i])):
            innerArray.append(W[i][j] - (p - 1) / 2)
        flag.append(innerArray)
    is_negative = ((abs(np.sign(flag)) + np.sign(flag)) / 2).tolist()
    Wtemp = []
    for i in range(len(W)):
        innerArray = []
        for j in range(len(W[i])):
            innerArray.append(W[i][j] - p * int(is_negative[i][j]))
        Wtemp.append(innerArray)

    W = Wtemp
    # W = (W - p * is_negative).astype(int)
    output = [0] * len(evalpoints_out)  # np.zeros((len(evalpoints_out),), dtype = 'int64')

    for i in range(len(output)):
        output[i] = (pow(z, W[i][0], p))  # pow(z, W[i,0], p)
        for j in range(len(n_array)):
            output[i] = (output[i] * pow(n_array[j], W[i][j + 1],
                                         p)) % p  # np.mod( output[i] * pow(n_array[j], W[i,j+1], p), p)
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
    flag = []
    for i in range(len(W)):
        innerArray = []
        for j in range(len(W[i])):
            innerArray.append(W[i][j] - (p - 1) / 2)
        flag.append(innerArray)
    # flag = W - (p-1)/2
    is_negative = ((abs(np.sign(flag)) + np.sign(flag)) / 2).tolist()
    Wtemp = []
    for i in range(len(W)):
        innerArray = []
        for j in range(len(W[i])):
            innerArray.append(W[i][j] - p * int(is_negative[i][j]))
        Wtemp.append(innerArray)

    W = Wtemp
    # W = (W - p * is_negative).astype(int)

    _idx_z = 0
    output = pow(z_tilde[_idx_z], W[0][_idx_z], p)  # pow(z_tilde[_idx_z], W[0,_idx_z], p)

    for j in range(len(evalpoints_out) - 1):
        output = (output * pow(z_tilde[j + 1], W[0][j + 1],
                               p)) % p  # np.mod( output * pow(z_tilde[j+1], W[0,j+1], p), p)

    return output

    # we do not need to reconstruct addtional masks, n_i
    # print(f"W={W}")
    # output = np.zeros((len(evalpoints_in),), dtype = 'int64')
    # for i in range(len(output)):
    #     output[i] = np.mod(pow(z_tilde[0], W[i,0], p) * pow(z_tilde[1], W[i,1], p), p)
    # return output[0]


def matmul_mod(X, Y, p):
    assert len(X[0]) == len(Y), 'length of X[0] and Y should be the same!'
    k = len(X)
    m = len(X[0])
    n = len(Y[0])

    out = [[0] * n] * k

    for i in range(k):
        for j in range(n):
            tmp = 0
            for l in range(m):
                tmp = (tmp + X[i][l] * Y[l][j]) % p
            out[i][j] = tmp

    return out


def my_pk_gen(my_sk, p, g):
    # print 'my_pk_gen option: g=',g
    if g == 0:
        return my_sk
    else:
        return np.mod(g ** my_sk, p)


def my_key_agreement(my_sk, u_pk, p, g):
    if g == 0:
        return np.mod(my_sk * u_pk, p)
    else:
        return np.mod(u_pk ** my_sk, p)


def BGW_encoding(X, N, T, p):
    m = len(X)
    d = len(X[0])

    alpha_s = range(1, N + 1)
    alpha_s = np.int64(np.mod(alpha_s, p))
    X_BGW = np.zeros((N, m, d), dtype='int64')
    R = np.random.randint(p, size=(T + 1, m, d)).astype('int64')
    R[0, :, :] = np.mod(X, p)

    for i in range(N):
        for t in range(T + 1):
            X_BGW[i, :, :] = np.mod(X_BGW[i, :, :] + R[t, :, :] * (alpha_s[i] ** t), p)
    return X_BGW


def gen_BGW_lambda_s(alpha_s, p):
    lambda_s = np.zeros((1, len(alpha_s)), dtype='int64')

    for i in range(len(alpha_s)):
        cur_alpha = alpha_s[i]

        den = PI([cur_alpha - o for o in alpha_s if cur_alpha != o], p)
        num = PI([0 - o for o in alpha_s if cur_alpha != o], p)
        lambda_s[0][i] = divmod(num, den, p)
    return np.mod(lambda_s.astype('int64'), p)


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
    f_recon = np.mod(np.dot(lambda_s, f_eval.astype('int64')), p)
    # t3 = time.time()
    # print 'time info for BGW_dec', t1-t0, t2-t1, t3-t2
    return f_recon


def gen_BGW_lambda_s_wo_np(alpha_s, p):
    lambda_s = [0] * len(alpha_s)

    for i in range(len(alpha_s)):
        cur_alpha = alpha_s[i]

        den = PI([cur_alpha - o for o in alpha_s if cur_alpha != o], p)
        num = PI([0 - o for o in alpha_s if cur_alpha != o], p)
        lambda_s[i] = divmod(num, den, p)
    return lambda_s


def SS_encoding(X, N, T, p):
    alpha_s = range(1, N + 1)

    X_SS = [0] * N
    R = np.random.randint(p, size=(T + 1,)).tolist()
    R[0] = X

    for i in range(N):
        for t in range(T + 1):
            X_SS[i] = (X_SS[i] + R[t] * ((alpha_s[i] ** t) % p)) % p
    return X_SS


def SS_decoding(f_eval, worker_idx, p):
    assert len(f_eval) == len(worker_idx)

    max = np.max(worker_idx) + 2
    alpha_s = range(1, max)
    alpha_s_eval = [alpha_s[i] for i in worker_idx]

    lambda_s = gen_BGW_lambda_s_wo_np(alpha_s_eval, p)

    secret = 0

    for i in range(len(lambda_s)):
        secret = (secret + lambda_s[i] * f_eval[i]) % p

    return secret


if __name__ == "__main__":

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