# This is to compare our method with fastecdsa.
import time
import random
import numpy as np
from utils.EC import point_add, scalar_mult


t0 = time.time()
g = (0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296,
       0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5)

my_d = 100000
my_alpha = np.array(random.sample(range(1, my_d + 1), k=my_d))  # is fine as long as d < curve.n which is the case for us
my_gradient = [1] * my_d

def distinct_point_compute(alpha):
    d = len(alpha)
    distinct_bases = tuple(scalar_mult(alpha[i], g) for i in range(d))
    return distinct_bases

def generate_hash(gradient, alpha, d):
    distinct_bases = distinct_point_compute(alpha)
    # compute g_i ** gradient[i] for a client
    temp_hash = tuple(scalar_mult(gradient[i], distinct_bases[i]) for i in range(d))
    # compute the product for [d]. This product translates into addition on the elliptic curve
    hash_client = temp_hash[0]
    for i in range(1, d):
        hash_client = point_add(hash_client, temp_hash[i])
    return hash_client

computed_hash = generate_hash(my_gradient, my_alpha, my_d)
#print(computed_hash)
t_hash = time.time() - t0

print(t_hash)