import pylab as plt
import hickle as hkl
import tables as tb
from leda_cal2.utils import poly_fit, timestamp_to_lst, closest
import numpy as np
import glob
import hickle

def rebin(x, n):
    xx = x.reshape(x.shape[0] / n, n).mean(axis=1)
    return xx

if __name__ == "__main__":
    flist = sorted(glob.glob('cal_out/resid*.h5'))

    d = hkl.load(flist[0])
    f = d['f']
    d252 = d['252A'] - d['252A_m']
    d254 = d['254A'] - d['254A_m']
    d255 = d['255A'] - d['255A_m']


    plt.figure("RESIDS", figsize=(6, 8))
    for filename in flist[1:]:
        d = hkl.load(filename)
        d252 += d['252A'] - d['252A_m']
        d254 += d['254A'] - d['254A_m']
        d255 += d['255A'] - d['255A_m']

        plt.subplot(3, 1, 1)
        plt.plot(f, d['252A'] - d['252A_m'])
        plt.subplot(3, 1, 2)
        plt.plot(f, d['254A'] - d['254A_m'])
        plt.subplot(3, 1, 3)
        plt.plot(f, d['255A'] - d['255A_m'])

    N = len(flist)

    plt.subplot(3, 1, 1)
    plt.plot(f, d252/N, c='#333333')
    plt.xlabel("252A [MHz]")
    plt.subplot(3, 1, 2)
    plt.plot(f, d254/N, c='#333333')
    plt.xlabel("254A [MHz]")
    plt.subplot(3, 1, 3)
    plt.plot(f, d255/N, c='#333333')
    plt.xlabel("255A [MHz]")

    for ii in (1,2,3):
        plt.subplot(3,1,ii)
        plt.ylabel("Temperature [K]")
        plt.ylim(-10, 10)
        plt.xlim(f[0], f[-1])

    plt.tight_layout()
    plt.savefig('img/05_resids.png')
    plt.show()

    plt.plot(rebin(f[4:], 8), rebin(d254[4:]/N, 8))
    plt.show()


