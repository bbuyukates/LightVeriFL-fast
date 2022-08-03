# Implements Elliptic Curve - Diffie Hellman on NIST P-256/
# Code taken from https://asecuritysite.com/encryption/ecdh3

import collections
import hashlib
import random
import binascii
import time
import numpy as np

type = 4

EllipticCurve = collections.namedtuple('EllipticCurve', 'name p a b g n h')

if (type == 4):
    curve = EllipticCurve(
        'p256',
        a=-3,
        b=0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b,
        p=115792089210356248762697446949407573530086143415290314195533631308867097853951,
        g=(0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296,
           0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5),
        n=115792089210356248762697446949407573529996955224135760342422259061068512044369,
        h=1,
    )


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


# Functions that work on curve points #########################################

def is_on_curve(point):
    """Returns True if the given point lies on the elliptic curve."""
    if point is None:
        # None represents the point at infinity.
        return True

    x, y = point

    return (y * y - x * x * x - curve.a * x - curve.b) % curve.p == 0


def point_add(point1, point2):
    """Returns the result of point1 + point2 according to the group law."""
    assert is_on_curve(point1)
    assert is_on_curve(point2)

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

#https://andrea.corbellini.name/2015/05/23/elliptic-curve-cryptography-finite-fields-and-discrete-logarithms/

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
        print('here is the point', point)
        print('here is the negated point', point_neg(point))
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

# Base point curve.g
# Take dimension d, gradient and alpha

def distinct_point_compute (alpha):
    distinct_bases = tuple(scalar_mult(alpha[i], curve.g) for i in range(d))
    return distinct_bases

# Hash generation ################################################
def generate_hash(gradient, alpha):  # NEEDS TO TAKE ALPHA AS INPUT SO THAT IT IS THE SAME FOR ALL CLIENTS
    #exponent = sum(i[0] * i[1] for i in zip(alpha, gradient))
    ##print(exponent)
    ## IS THERE AN ISSUE WHERE EXPONENT GROWS BEYOND CURVE.P????
    #hash_point = scalar_mult(exponent, curve.g)
    ## hash value is the x-point of the hash point
    #hash = hash_point[0]
    #return hash
    distinct_bases = distinct_point_compute(alpha)
    temp_hash = tuple(scalar_mult(gradient[i], distinct_bases[i]) for i in range(d)) #np.zeros(d)
    #for i in range(d):
    #    temp_hash[i] = scalar_mult(gradient[i], distinct_bases[i])
    #    #temp_hash[i] = xxxx[0]
    hash_client = temp_hash[0]
    for i in range(1, d):
        hash_client = point_add(hash_client, temp_hash[i])
        #hash_client = ((hash_client + temp_hash[i]) % curve.p)
    return hash_client


def evaluate_hash(aggregate_hash, client_hashes):
    # aggregate_hash is the hash of the aggregate model at the server
    # client_hashes[i] is the hash of the ith client
    #evaluated_hash = 1
    #for i in range(N):
    #    evaluated_hash = ((evaluated_hash * client_hashes[i]) % curve.p)
    evaluated_hash = client_hashes[0]
    for i in range(1, N):
        evaluated_hash = point_add(evaluated_hash, client_hashes[i])
    if aggregate_hash[0]-evaluated_hash[0] == 0:
        print("Correct aggregation.")
        return True
        pass
    else:
        print("Forged aggregation. Process terminated.")
        return False


start_time = time.time()
# sample distinct alpha values
d = 10

# R = 2**24 and B = 2**34 (i.e., the maximum number of clients is B/R = 210 = 1024) in VeriFL paper
# In VeriFL the maximum number of clients is B/R = 2**10 = 1024
N=4
R = 2 ** 24 # gradient elements come from this domain
B = 2 ** 34

gradients = np.zeros((N, d), dtype='int32')
#hash = np.zeros(N)
alpha = random.sample(range(2, d+2), k=d) # is fine as long as d < curve.n which is the case for us   # [1]*d #
for i in range(N):
    gradients[i, :] = random.sample(range(R), k=d)
    #hash[i] = generate_hash(gradients[i, :], alpha)
#CHECK ALPHA GENERATION alpha_i cannot be 0.
hash = tuple(generate_hash(gradients[i, :], alpha) for i in range(N))




#sanity_aggregate = 1
#for i in range(N):
#    sanity_aggregate = ((sanity_aggregate * hash[i]) % curve.p)

agg_gradient = sum(gradients)
#for i in range(d):
#    agg_gradient[i] = agg_gradient[i]

aggregate_hash = generate_hash(agg_gradient, alpha)

print("==========================")
print('aggregate hash', aggregate_hash)
#print('sanity aggregate hash is', sanity_aggregate)
#print('difference', aggregate_hash-sanity_aggregate)

#print("Name:\t", curve.name)
#print("a:\t", curve.a)
#print("b:\t", curve.b)
#print("G:\t", curve.g)
#print("P:\t", curve.p)


print("==========================")

a = evaluate_hash(aggregate_hash, hash)
#c = evaluate_hash(sanity_aggregate, hash)

print("--- %s seconds ---" % (time.time() - start_time))



# sunu da implement et
# each client does d exponentiations modularly
# and then you multiply the x coordinates of those points
# aynisi mi cikar acaba
# ve de suresini olc
