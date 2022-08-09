# This is to compare our method with fastecdsa.
# example taken from the document below (section 4.3.2):
# https://koclab.cs.ucsb.edu/teaching/cren/docs/w02/nist-routines.pdf

from fastecdsa.curve import P256
from fastecdsa.point import Point
import numpy as np
import random
import time

#g=(0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296,
#       0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5)

xs = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296 #0xde2444bebc8d36e682edd27e0f271508617519b3221a8fa0b77cab3989da97c9
ys = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5 #0xc093ae7ff36e5380fc01a5aad1e66659702de80f53cec576b6350b243042a256
S = Point(xs, ys, curve=P256) #S

xt = 0x55a8b00f8da1d44e62f6b3b25316212e39540dc861c89575bb8cf92e35e0986b
yt = 0x5421c3209c2d6c704835d82ac4c3dd90f61a8a52598b9e7ab656e9d8c8b24316
T = Point(xt, yt, curve=P256)



t0 = time.time()
my_d = 100000
my_alpha = random.sample(range(1, my_d + 1), k=my_d) # is fine as long as d < curve.n which is the case for us
my_gradient = [1] * my_d

def distinct_point_compute(alpha):
    d = len(alpha)
    distinct_bases = tuple(alpha[i] * S for i in range(d))
    return distinct_bases

def generate_hash(gradient, alpha, d):
    distinct_bases = distinct_point_compute(alpha)
    temp_hash = tuple(gradient[i] * distinct_bases[i] for i in range(d))
    hash_client = temp_hash[0]
    for i in range(1, d):
        hash_client = hash_client + temp_hash[i]
    return hash_client

computed_hash = generate_hash(my_gradient, my_alpha, my_d)

t_hash = time.time() - t0

print(t_hash)



#gives x and y points
#print(computed_hash.__dict__)
#print(computed_hash.__repr__)
#print(computed_hash.x)
