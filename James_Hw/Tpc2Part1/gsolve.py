import numpy as np

def w_mid(z):
    return 1 - np.abs(z - 128.0) / 130.0

def w_unif(z):
    return 1/256.0


def gsolve(points, ln_te, lmd, weight_fcn=w_unif):
## Given a set of pixel values observed for several pixels in several images with different exposure times, this function returns the imaging systems response function g as well as the log film irradiance values for the observed pixels.

## Assumes:
#  Zmin = 0
#  Zmax = 255

# Arguments:
#  points[i,j] is the pixel value of pixel number i (among N) in image j (among M)
#  ln_te(j)   is the log delta t, or log shutter speed, for image j
#  lmd      is lamdba, the constant that determines the amount of smoothness
 
# Returns:
#  g(z)   is the log exposure corresponding to pixel value z
#  lnE(i)  is the log film irradiance at pixel location i

    N, M = points.shape
    nlevels = 256  # [0, 255]
    A = np.zeros((N * M + nlevels - 1, nlevels + N))
    b = np.zeros(A.shape[0])
    print('system size:{}'.format(A.shape))

    k = 0
    for i in range(N):
        for j in range(M):
            wij = weight_fcn(points[i, j])
            A[k, int(points[i, j])] = wij
            A[k, nlevels + i] = -wij
            b[k] = wij * ln_te[j]
            k += 1

    A[k, 128] = 1   # Fix the curve by setting its middle value to 0
    k += 1

    for i in range(nlevels-2):
        wi = weight_fcn(i)
        A[k, i] = lmd * wi
        A[k, i+1] = -2 * lmd * wi
        A[k, i+2] = lmd * wi
        k += 1

    x, _, _, _ = np.linalg.lstsq(A, b)
    g = x[:nlevels]
    lnE = x[nlevels:]

    return g, lnE