# Implements a discrete logarithm based hash using Elliptic Curves (EC) on NIST P-256/
# EC crytopgrahy implementation from https://asecuritysite.com/encryption/ecdh3

import collections
import hashlib
import random
import binascii
import time
import numpy as np
import math

from fastecdsa.curve import P256
from fastecdsa.point import Point

from .function import gen_Lagrange_coeffs, PI, divmod
from .function import matmul_mod


# Elliptic curve initilization. We use NIST P-256 curve. Can use different curves as well.
#EllipticCurve = collections.namedtuple('EllipticCurve', 'name p a b g n h')

# our initial way of defining the curve, now we do not need this.
#mycurve = EllipticCurve(
#    'p256',
#    a=-3,
#    b=0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b,
#    p=115792089210356248762697446949407573530086143415290314195533631308867097853951,
#    g=(0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296,
#       0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5),
#    n=115792089210356248762697446949407573529996955224135760342422259061068512044369,
#    h=1,
#)

p_model = 2 ** 31 - 1  # field size for model

#xs = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296 #0xde2444bebc8d36e682edd27e0f271508617519b3221a8fa0b77cab3989da97c9
#ys = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5 #0xc093ae7ff36e5380fc01a5aad1e66659702de80f53cec576b6350b243042a256

curve_g = Point(P256.gx, P256.gy, curve=P256) #S
curve_n = P256.q  # 115792089210356248762697446949407573529996955224135760342422259061068512044369
# p_model = 2 ** 16 - 15 # field size for model
# p_model = 2 ** 10 - 3 # field size for model

'''

# Modular arithmetic ##########################################################

def inverse_mod(k, p):
    """Returns the inverse of k modulo p.
    This function returns the only integer x such that (x * k) % p == 1.
    k must be non-zero and p must be a prime.
    """
    if k == 0:
        raise ZeroDivisionError('division by zero')

    if k < 0:
        # k ** -1 = p - (-k) ** -1  (mod p)
        return p - inverse_mod(-k, p)

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

'''

'''
# Functions that work on curve points #########################################

def is_on_curve(point):
    """Returns True if the given point lies on the elliptic curve."""
    if point is None:
        # None represents the point at infinity.
        return True

    x, y = point

    return (y * y - x * x * x - curve.a * x - curve.b) % curve.p == 0
'''

'''

def point_add(point1, point2):
    """Returns the result of point1 + point2 according to the group law."""
    assert is_on_curve(point1), f"point1={point1}"
    assert is_on_curve(point2), f"point1={point1}"

    if point1 is None:
        # 0 + point2 = point2
        return point2
    if point2 is None:
        # point1 + 0 = point1
        return point1

    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2 and y1 != y2:
        # point1 + (-point1) = 0
        return None

    if x1 == x2:
        # This is the case point1 == point2.
        m = (3 * x1 * x1 + curve.a) * inverse_mod(2 * y1, curve.p)
        pass
    else:
        # This is the case point1 != point2.
        m = (y2 - y1) * inverse_mod(x2 - x1, curve.p)  # m is the slope connecting points 1 and 2
        pass

    # https://www.certicom.com/content/certicom/en/22-elliptic-curve-addition-an-algebraic-approach.html
    x3 = m * m - x1 - x2
    y3 = y1 + m * (x3 - x1)
    result = (x3 % curve.p, -y3 % curve.p)

    assert is_on_curve(result)

    return result
'''

'''

# https://andrea.corbellini.name/2015/05/23/elliptic-curve-cryptography-finite-fields-and-discrete-logarithms/

def point_neg(point):
    x_point, y_point = point
    point_negated = (x_point, -y_point % curve.p)
    assert is_on_curve(point_negated)
    return point_negated


# p.x, -p.y % self.q

def scalar_mult(k, point):
    """Returns k * point computed using the double and point_add algorithm."""
    assert is_on_curve(point)

    if k % curve.n == 0 or point is None:
        return None

    if k < 0:
        # k * point = -k * (-point)
        # print('here is the point', point)
        # print('here is the negated point', point_neg(point))
        return scalar_mult(-k, point_neg(point))

    result = None
    addend = point

    while k:
        if k & 1:
            # Add.
            result = point_add(result, addend)

        # Double.
        addend = point_add(addend, addend)

        k >>= 1

    assert is_on_curve(result)

    return result
'''

# Base point curve.g
# We find d distinct points using curve.g as generator
# Alpha_i needs to be distinct so that each g_i = g ** alpha_i is distinct
# We set alpha_i below
def distinct_point_compute(alpha):
    d = len(alpha)
    distinct_bases = tuple(alpha[i] * curve_g for i in range(d))
    return distinct_bases


# Hash generation #
# Takes alpha as input so that distinct g_i points are the same for all clients
def generate_hash(gradient, alpha, d):
    distinct_bases = distinct_point_compute(alpha)
    # compute g_i ** gradient[i] for a client
    temp_hash = tuple(gradient[i] * distinct_bases[i] for i in range(d))
    # compute the product for [d]. This product translates into addition on the elliptic curve
    hash_client = temp_hash[0]
    for i in range(1, d):
        hash_client = hash_client + temp_hash[i]
    return hash_client


# used to verify whether the aggregate has is equal to the "sum" of client hashes
def evaluate_hash(aggregate_hash, client_hashes):
    # aggregate_hash is the hash of the aggregate model at the server
    # client_hashes[i] is the hash of the ith client
    evaluated_hash = client_hashes[0]
    for i in range(1, N):
        evaluated_hash = evaluated_hash + client_hashes[i]
    # if both hashes are equal return correct aggregation
    if aggregate_hash[0] - evaluated_hash[0] == 0:
        print("Correct aggregation.")
        return True
        pass
    # otherwise return forged aggregation
    else:
        print("Forged aggregation. Process terminated.")
        return False


def LightVeriFL_enc_EC(hash, n_array, evalpoints_in, evalpoints_out, curve):
    '''
    input
        - hash : tuple of (x,y)
        - n : size T array (T: privacy parameter) where each element is tuple of (x, y)
        - evalpoints_in : evaluation points for (T+1) original points.
                          evalpoints_in[0] corresponds to evaluation point for z
                          evalpoints_in[1:] corresponds to evaluation points for n_array
        - evalpoints_out : evaluation points for (N) encoded output points.
        - curve : contains elliptic curve info
    output
        - output : size N array which correponds to encoded hash
    '''

    W_enc = gen_Lagrange_coeffs(evalpoints_in, evalpoints_out, curve.q)

    output = [(0, 0)] * len(evalpoints_out)

    for i in range(len(output)):
        output[i] = W_enc[i][0] * hash
        for j in range(len(n_array)):
            output[i] = output[i] + (W_enc[i][j + 1] * n_array[j])
    return output


def LightVeriFL_dec_EC(z_tilde, evalpoints_in, evalpoints_out, curve):
    '''
    input
        - z_tilde : array of size (T+1) where each element is tuple of (x, y)
        - evalpoints_in : evaluation points for (T+1) original points.
                          evalpoints_in[0] corresponds to evaluation point for z
                          evalpoints_in[1:] corresponds to evaluation points for n_array
        - evalpoints_out : (T+1) evaluation points corresponding to z_tilde.
        - curve : contains elliptic curve info
    output
        - output : decoded hash result (=(x,y))
    '''

    assert len(z_tilde) == len(
        evalpoints_out), "@LightVeriFL_out, length of evalpoints_out and z_tilde should be the same (=T+1)!!"

    W_dec = gen_Lagrange_coeffs(evalpoints_out, [evalpoints_in[0]], curve.q)

    _idx_z = 0
    output = W_dec[0][_idx_z] * z_tilde[_idx_z]
    output = Point(output.x, output.y, curve=P256)
    for j in range(len(evalpoints_out) - 1):
        tmp_element = (W_dec[0][j + 1] * z_tilde[j + 1])
        tmp_element = Point(tmp_element.x, tmp_element.y, curve=P256)
        output = output + tmp_element
        output = Point(output.x, output.y, curve=P256)
    return output


# can be made more efficient, look at fastecdsa.
def generate_point_EC():
    """Generates a random point on EC."""
    private_key = random.randrange(1, curve_n)
    public_key_orig = (private_key * curve_g)
    return public_key_orig


def PI_addEC(vals):  # upper-case PI -- addition of inputs on EC
    #print(len(vals))
    #print('vals 0 is', vals[0])
    accum = Point(0, 0, curve=None) #None
    for v in vals: #i in range(1, len(vals)):
    #    #print(P256.is_point_on_curve((v.x, v.y)))
        temp = Point(v.x, v.y, curve=P256)
        accum = accum + temp
        accum = Point(accum.x, accum.y, curve=P256)
    #    print(P256.is_point_on_curve((accum.x, accum.y)))
    #    #print('after addition is', accum)
    return accum


def gen_BGW_lambda_s_EC(alpha_s, curve):
    lambda_s = [0] * len(alpha_s)

    for i in range(len(alpha_s)):
        cur_alpha = alpha_s[i]

        den = PI([cur_alpha - o for o in alpha_s if cur_alpha != o], curve.q)
        num = PI([0 - o for o in alpha_s if cur_alpha != o], curve.q)
        lambda_s[i] = divmod(num, den, curve.q)
    return lambda_s


def BGW_encoding_EC(hash, N, T, curve):  # hash is (x,y)

    alpha_s = range(1, N + 1)
    alpha_s = np.mod(alpha_s, curve.q)
    hash_SS = [(0, 0)] * N
    R = [hash]  # n_array also comes from the EC
    for _ in range(T):
        R.append(generate_point_EC())

    for i in range(N):
        hash_SS[i] = R[0]
        for t in range(1, T + 1):
            # print(f"i={i}, t={t}, hash_SS[i]={hash_SS[i]}, R[t]={R[t]}, alpha={(alpha_s[i] ** t) % curve.n}")
            hash_SS[i] = hash_SS[i] + (((alpha_s[i] ** t) % curve.q) * R[t])
    return hash_SS


def BGW_decoding_EC(hash_SS, worker_idx, curve):  # decode the output from T+1 evaluation points
    # f_eval     : [RT X d ]
    # worker_idx : [ 1 X RT]
    # output     : [ 1 X d ]

    # t0 = time.time()
    max = np.max(worker_idx) + 2
    alpha_s = range(1, max)
    alpha_s = np.mod(alpha_s, curve.q)
    alpha_s_eval = [alpha_s[i] for i in worker_idx]
    # t1 = time.time()
    # print(alpha_s_eval)
    lambda_s = gen_BGW_lambda_s_EC(alpha_s_eval, curve)

    # t2 = time.time()
    # print(lambda_s.shape)

    hash_recon = (lambda_s[0] * hash_SS[0])
    hash_recon = Point(hash_recon.x, hash_recon.y, curve=P256)
    for i in range(1, len(hash_SS)):
        tmp_element = (lambda_s[i] * hash_SS[i])
        tmp_element = Point(tmp_element.x, tmp_element.y, curve=P256)
        hash_recon = tmp_element + hash_recon
        hash_recon = Point(hash_recon.x, hash_recon.y, curve=P256)

    # t3 = time.time()
    # print 'time info for BGW_dec', t1-t0, t2-t1, t3-t2
    return hash_recon


# here noise is the random noise element of each user
def generate_Pedersen_commitment(hash, Pedersen_g, Pedersen_l, noise):
    # Pedersen_g = scalar_mult(Pedersen_coeff[0], curve.g)
    # Pedersen_l = scalar_mult(Pedersen_coeff[1], curve.g)
    Pedersen_g = Point(Pedersen_g.x, Pedersen_g.y, curve=P256)
    Pedersen_l = Point(Pedersen_l.x, Pedersen_l.y, curve=P256)

    noise = Point(noise.x, noise.y, curve=P256)
    hash = Point(hash.x, hash.y, curve=P256)

    tmp1 = (noise + Pedersen_g)
    tmp2 = (hash + Pedersen_l)
    commitment = (tmp1 + tmp2)
    return commitment


# partitions digits of a large integer into certain number of parts. Takes a 2D number
def partition_digits(input_number, parts):
    line_x = str(input_number[0])
    line_y = str(input_number[1])
    nnn = math.ceil(len(line_x) / parts)
    output_x = [line_x[i:i + nnn] for i in range(0, len(line_x), nnn)]
    output_y = [line_y[i:i + nnn] for i in range(0, len(line_y), nnn)]
    return output_x, output_y


# concatenates a given list elements into a single number
def concatenate_digits(input_number):
    my_lst_str = ''.join(map(str, input_number))
    return int(my_lst_str)


if __name__ == "__main__":
    print("==========================")
    print("  Test of MDS codes on EC \n ")

    # Let's check how it works.
    d = 10  # dimension of the gradient
    N = 10  # number of clients

    # R = 2**24 and B = 2**34 (i.e., the maximum number of clients is B/R = 210 = 1024) in VeriFL paper
    # In VeriFL the maximum number of clients is B/R = 2**10 = 1024
    R = 2 ** 24  # gradient elements come from this domain
    B = 2 ** 34  # domain of the aggregate

    gradients = np.zeros((N, d), dtype='int64')
    # sample distinct alpha values, alpha_i cannot be 0
    alpha = random.sample(range(1, d + 1), k=d)  # is fine as long as d < curve.n which is the case for us

    # generate gradients randomly for the time-being. These come from the learning portion.
    for i in range(N):
        gradients[i, :] = random.sample(range(R), k=d)

    H = generate_hash(gradients[0, :], alpha)
    R = [generate_hash(gradients[1, :], alpha),
         generate_hash(gradients[2, :], alpha)]  # this should be changed to random hash selection.

    # print(len(R))

    alpha_s = [0, 1, 2]
    beta_s = [3, 4, 5]

    W_enc = gen_Lagrange_coeffs(alpha_s, beta_s, curve_n)
    V_dec = gen_Lagrange_coeffs(beta_s, [alpha_s[0]], curve_n)

    print(f"W_enc = {W_enc}")
    print(f"V_dec = {V_dec}\n")

    tmp = matmul_mod(V_dec, W_enc, curve_n)

    print(f"V_dec * W_enc = {tmp}")

    # H_tilde_0 = point_add( scalar_mult(W_enc[0][0], H) , scalar_mult(W_enc[0][1], R) )
    # H_tilde_1 = point_add( scalar_mult(W_enc[1][0], H) , scalar_mult(W_enc[1][1], R) )

    H_tilde_s = LightVeriFL_enc_EC(H, R, alpha_s, beta_s, P256)
    H_dec = LightVeriFL_dec_EC(H_tilde_s, [alpha_s[0]], beta_s, P256)

    print(f"original H = {H}")
    print(f"decoded  H = {H_dec}")

    print("\n  Test of MDS codes on EC ends. ")
    print("==========================")