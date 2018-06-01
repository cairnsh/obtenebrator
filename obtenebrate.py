import numpy as np, scipy as sp
import random

def productmixture(m, w, p):
    """productmixture(number of samples, r weights, n x r array p[j, i])
    return: n x m samples"""
    r = len(w)
    n = p.shape[0]
    cum = np.cumsum(w)
    np.testing.assert_almost_equal(cum[r-1], 1)
    result = np.zeros((n, m), dtype=bool)
    for start in range(0, m, 1000000):
        end = min(start + 1000000, m)
        choice = np.random.random(end - start)
        unifs = np.random.random((n, end - start))
        result[:, start:end] = np.select(
            [choice < cum[i] for i in range(r)],
            [unifs < p[:, i, np.newaxis] for i in range(r)]
        )
        print("Copied", end, "/", m)
    return result

TEST1 = [0.4960544 , 0.7446443 , 0.1599165 , 0.98831394, 0.01779817,
       0.50439134, 0.64049817, 0.97741259, 0.9152564 , 0.39585037,
       0.73770265, 0.25856659, 0.32972433, 0.13958005, 0.90940388,
       0.26029715, 0.78614366, 0.57336946, 0.49739241, 0.84571439,
       0.66924763, 0.6422609 , 0.04031003, 0.94331702, 0.46493605,
       0.52348284, 0.1060138 , 0.68331466, 0.33120784, 0.65654052,
       0.56638749, 0.12312042, 0.60921863, 0.69454188, 0.14014539,
       0.0567017 , 0.43992813, 0.93484103, 0.15571828, 0.70149569]
TEST2 = [0.32214499, 0.6065709 , 0.76132901, 0.45640184, 0.31847043,
       0.97554779, 0.98200491, 0.64164317, 0.25733759, 0.95981456,
       0.1976516 , 0.18890083, 0.89433485, 0.85395564, 0.22526078,
       0.36294283, 0.17351558, 0.14135642, 0.80551319, 0.92612935,
       0.06757949, 0.35550929, 0.96055928, 0.76562772, 0.09557816,
       0.30417213, 0.24409638, 0.8058958 , 0.03933394, 0.01885413,
       0.71153941, 0.20642357, 0.99278932, 0.48574307, 0.46331452,
       0.61052617, 0.52011088, 0.14503338, 0.70915326, 0.54032259]
TEST3 = [0.62765842, 0.58690705, 0.59761867, 0.75701152, 0.13757999,
       0.24334473, 0.57454775, 0.95784784, 0.65913478, 0.47358131,
       0.69336789, 0.09260875, 0.71486795, 0.68505406, 0.2034375 ,
       0.51632667, 0.66716951, 0.19086042, 0.46890473, 0.31186268,
       0.49215791, 0.60288261, 0.74063561, 0.7921759 , 0.21134583,
       0.38830594, 0.97190682, 0.69043098, 0.30593963, 0.98690276,
       0.40545496, 0.04392348, 0.85462559, 0.93809666, 0.40816224,
       0.08609527, 0.82609488, 0.34133802, 0.34780932, 0.33054138]
TEST4 = [0.46274129, 0.16400384, 0.73840863, 0.63979331, 0.14809107,
       0.21679685, 0.83683985, 0.43233682, 0.84338851, 0.89126778,
       0.51541581, 0.42948991, 0.58163658, 0.22788931, 0.98128305,
       0.55629221, 0.10264535, 0.39130359, 0.75101191, 0.04235404,
       0.34677643, 0.34682469, 0.6020249 , 0.02263641, 0.20462118,
       0.5654853 , 0.4275153 , 0.14324525, 0.16074094, 0.03263522,
       0.64543845, 0.57201279, 0.85197721, 0.60628   , 0.12771382,
       0.03135377, 0.03177461, 0.16075248, 0.07806533, 0.56721768]
TEST5 = [0.4835, 0.334 , 0.3305, 0.3302, 0.837 , 0.0801, 0.2067, 0.324 ,
       0.8799, 0.807 , 0.3286, 0.1977, 0.4565, 0.1808, 0.2211, 0.2565,
       0.2576, 0.1691, 0.1522, 0.2171, 0.4018, 0.1239, 0.5871, 0.0013,
       0.2495, 0.4595, 0.912 , 0.0118, 0.5004, 0.121 , 0.8376, 0.6186,
       0.07  , 0.1058, 0.2285, 0.4451, 0.2695, 0.518 , 0.3238, 0.9523]
TEST6 = [0.6674, 0.9929, 0.2521, 0.9862, 0.7483, 0.2869, 0.7869, 0.7197,
       0.5897, 0.9687, 0.6042, 0.6534, 0.7529, 0.9739, 0.8914, 0.8278,
       0.5415, 0.5661, 0.7884, 0.4801, 0.0142, 0.8353, 0.7383, 0.3883,
       0.8184, 0.9145, 0.8506, 0.5471, 0.6517, 0.0795, 0.2581, 0.9137,
       0.2429, 0.5315, 0.5813, 0.304 , 0.9903, 0.6833, 0.806 , 0.5294]
TEST7 = [0.2549, 0.2739, 0.054 , 0.9487, 0.0117, 0.5567, 0.4024, 0.0986,
       0.7738, 0.1401, 0.7713, 0.6089, 0.1789, 0.6062, 0.2764, 0.532 ,
       0.8684, 0.1028, 0.2507, 0.2736, 0.6871, 0.6104, 0.6779, 0.7317,
       0.3408, 0.1472, 0.5956, 0.5407, 0.3661, 0.5879, 0.9251, 0.5797,
       0.9296, 0.728 , 0.238 , 0.2427, 0.7642, 0.8024, 0.2536, 0.7238]
TEST8 = [0.2274, 0.3004, 0.3388, 0.8252, 0.5525, 0.0285, 0.2637, 0.2422,
       0.7921, 0.3034, 0.398 , 0.778 , 0.0825, 0.9454, 0.0107, 0.5748,
       0.2786, 0.465 , 0.5499, 0.4624, 0.6995, 0.7698, 0.0022, 0.3292,
       0.6178, 0.8027, 0.0229, 0.0183, 0.2104, 0.4713, 0.1038, 0.769 ,
       0.5546, 0.9011, 0.3828, 0.1462, 0.5732, 0.2245, 0.649 , 0.1828]
TESTTUPLE = (TEST1, TEST2, TEST3, TEST4, TEST5, TEST6, TEST7, TEST8)

def test_product_mixture_generator(m):
    #p = [.5, .5]
    #prod = np.stack((TEST1, TEST2), axis=1)
    #p = [.25, .25, .25, .25]
    #prod = np.stack((TEST1, TEST2, TEST3, TEST4), axis=1)
    p = [.125, .125, .125, .125, .125, .125, .125, .125]
    w = np.stack(TESTTUPLE,axis=1)
    return productmixture(m, p, w)

class ObtenebrationStorage:
    def __init__(self, n, rho, d, j):
        self.n = n
        self.rho = rho
        self.d = d
        self.perm, self.reverse = self.choose_permutation(n, j)
        self.storage = np.zeros((d, 2**rho))
        #self.dartboarder = identity_dartboarder(d, n - rho)
    def choose_permutation(self, n, j):
        perm = list(range(j)) + list(range(j+1, n))
        perm = [perm[j] for j in np.random.permutation(n-1)]
        perm = np.array([j] + perm)
        rev = np.zeros(n, dtype=perm.dtype)
        rev[perm] = np.arange(n)
        return perm,  rev
    def add(self, x):
        rho, n = self.rho, self.n
        coords = x[self.perm[0:rho], :]
        vector = x[self.perm[rho:n], :]
        coord = np.zeros(x.shape[1], dtype=np.int64)
        for j in range(rho):
            coord *= 2
            coord += coords[j, :]
        pass
        for j in range(n - rho):
            self.storage[j, :] += np.bincount(coord, weights=vector[j, :], minlength=2**rho)
    def svd(self, x):
        d = np.linalg.svd(x, compute_uv=False)
        # wheee
        return d
    def mix(self, j, q):
        rho = self.rho
        bit_to_flip = 1 << (rho - j - 1)
        zero, one = [], []
        for i in range(2**rho):
            if i & bit_to_flip:
                one.append(i)
            else:
                zero.append(i)
        O = self.storage
        obtenebration = O[:, zero] * q - O[:, one] * (1-q)
        return self.svd(obtenebration)
    def scan(self, j, rank):
        guesses = []
        print("estimating coordinate ", self.perm[j])
        vals = [self.mix(j, i/10000) for i in range(10000)]
        def f(i):
            return vals[i][rank - 1]
        for i in range(1, len(vals)-1):
            if f(i) < f(i-1) and f(i) < f(i+1):
                print("arg min", i/10000, vals[i][rank-2:rank])
                #print("singular values here are", vals[i])
                guesses.append(i/10000)
        print("actually:",
            sorted([x[self.perm[j]] for x in TESTTUPLE]))
        return self.perm[j], guesses

def mixturescan(x, n, rho, rank):
    accumulated_guesses = {i:[] for i in range(n)}
    print("Obtenebrator running...")
    for i in range(n):
        #stor = ObtenebrationStorage(n, rho, rank + 4)
        for l in range(8): # do it many times
            stor = ObtenebrationStorage(n, rho, n - rho, i)
            stor.add(x)
            for j in range(rho):
                coordinate, guesses = stor.scan(j, rank)
                print(coordinate, guesses)
                accumulated_guesses[coordinate].append(guesses)
    for i in range(n):
        print("guesses for coordinate", i)
        print(accumulated_guesses[i])
        print()
    return accumulated_guesses
