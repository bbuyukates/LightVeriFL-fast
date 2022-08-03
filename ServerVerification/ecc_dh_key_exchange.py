# Implements Elliptic Curve - Diffie Hellman on NIST P-256/
# Code taken from https://asecuritysite.com/encryption/ecdh3

import collections
import hashlib
import random
import binascii

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



# Keypair generation and ECDSA ################################################

def make_keypair():
    """Generates a random private-public key pair."""
    private_key = random.randrange(1, curve.n)
    private_key = -private_key
    public_key_orig = scalar_mult(private_key, curve.g)
    return private_key, public_key_orig #public_key_mod


#print("Name:\t", curve.name)
#print("a:\t", curve.a)
#print("b:\t", curve.b)
#print("G:\t", curve.g)
#print("P:\t", curve.p)


# THESE THINGS ARE BASED ON N.
# SO MDS CODE NEEDS TO BE BASED ON N
# THE SUBGROUP SIZE. T
print("==========================")

P = curve.g
Q = scalar_mult(2, P)
print(f"P={P}")
print(f"Q={Q}")
C = scalar_mult(curve.n-1, P)
print(f"Q*C = {point_add(Q,C)}")







#print(scalar_mult(curve.p + 1, curve.g))
#print(scalar_mult(curve.n + 1, curve.g))
#print(curve.g)

print("==========================")

aliceSecretKey, alicePublicKey = make_keypair()
bobSecretKey, bobPublicKey = make_keypair()

print("Alice\'s secret key:\t", aliceSecretKey)
print("Alice\'s public key:\t", alicePublicKey)

print("Bob\'s secret key:\t", bobSecretKey)
print("Bob\'s public key:\t", bobPublicKey)


# test distributive property of scalar multiplication over addition in EC
# alicePublicKey and bobPublicKey are just points on the EC
H = alicePublicKey
R = bobPublicKey
w11 = 2
w12 = 3
v1 = 5
H1_tilde = point_add(scalar_mult(w11, H), scalar_mult(w12, R))
H_hat = scalar_mult(v1, H1_tilde)
H_hat_check = point_add(scalar_mult(w11*v1, H), scalar_mult(v1*w12, R))
print("==========================")

print(f"H={H}")
print(f"R={R}")
print(f"H1_tilde={H1_tilde}")
print(f"H_hat={H_hat}")
print(f"H_hat_check={H_hat_check}")
print("==========================")


sharedSecret1 = scalar_mult(bobSecretKey, alicePublicKey)
sharedSecret2 = scalar_mult(aliceSecretKey, bobPublicKey)

print("==========================")
print("Alice\'s shared key:\t", sharedSecret1)
print("Bob\'s shared key:\t", sharedSecret2)

print("==========================")
print("The shared value is the x-value:\t", (sharedSecret1[0]))
