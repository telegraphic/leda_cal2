import pylab as plt
import hickle as hkl
import tables as tb
from leda_cal2.utils import poly_fit, timestamp_to_lst, closest
import numpy as np

plt.rcParams['font.size'] = 12


def calibrate(data, caldata):
    T_H = caldata['T_H']
    T_C = caldata['T_C']
    G_S = caldata['G_S']
    T_NW = caldata['T_NW']
    # S   = caldata['scale']
    # O   = caldata['offset']

    # D = S * ((T_H - T_C) * data + T_C) / G_S + O
    D = ((T_H - T_C) * data + T_C) / G_S
    return D

def rebin(x, n):
    xx = x.reshape(x.shape[0] / n, n).mean(axis=1)
    return xx

def trim(data, f, f0):
    i0 = closest(f, f0)
    try:
        return f[i0:], data[:, i0:]
    except:
        return f[i0:], data[i0:]

def extract_lsts(data, lsts, lst0, lst1):
    i0 = closest(lsts, lst0)
    i1 = closest(lsts, lst1)
    # print i0, i1, data.shape
    lsts = lsts[i0:i1]
    D = data[i0:i1]
    return lsts, D


def load_data_252x(filename, cal_params, f0=49.8, lst0=10, lst1=12):
    # print "opening %s" % filename
    d = tb.open_file(filename)
    a252x = d.root.data.cols.ant252_x[:]
    ts = d.root.data.cols.timestamp[:]
    f = cal_params['f_mhz']
    D = calibrate(a252x, cal_params)
    lsts = timestamp_to_lst(ts)
    ff, D = trim(D, f, f0)
    lsts, D = extract_lsts(D, lsts, lst0, lst1)
    return ff, lsts, D


def load_data_254x(filename, cal_params, f0=49.8, lst0=10, lst1=12):
    # print "opening %s" % filename
    d = tb.open_file(filename)
    a254x = d.root.data.cols.ant254_x[:]
    ts = d.root.data.cols.timestamp[:]
    f = cal_params['f_mhz']
    D = calibrate(a254x, cal_params)
    lsts = timestamp_to_lst(ts)
    ff, D = trim(D, f, f0)
    lsts, D = extract_lsts(D, lsts, lst0, lst1)
    return ff, lsts, D


def load_data_255x(filename, cal_params, f0=49.8, lst0=10, lst1=12):
    # print "opening %s" % filename
    d = tb.open_file(filename)
    a255x = d.root.data.cols.ant255_x[:]
    ts = d.root.data.cols.timestamp[:]
    f = cal_params['f_mhz']
    D = calibrate(a255x, cal_params)
    lsts = timestamp_to_lst(ts)
    ff, D = trim(D, f, f0)
    lsts, D = extract_lsts(D, lsts, lst0, lst1)
    return ff, lsts, D

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Plot antenna spectra and residuals')
    p.add_argument('filename', help='name of file to open')
    p.add_argument('-f0', '--f0', help='start frequency in MHz, default 40 MHz', type=float, default=40)
    p.add_argument('-f1', '--f1', help='end frequency in MHz, default 87.6 MHz', type=float,  default=87.6)
    p.add_argument('-n',  '--n_poly', help='number of terms in log-poly to fit for residuals. Default 5', type=int,  default=5)
    args = p.parse_args()

    f0 = args.f0
    f1 = args.f1
    n_poly = args.n_poly
    antennas = ['252A', '254A', '255A']

    h_252x = hkl.load('cal_data_252A_fit.hkl')
    h_254x = hkl.load('cal_data_254A_fit.hkl')
    h_255x = hkl.load('cal_data_255A_fit.hkl')

    ff, lsts, C0 = load_data_252x('data/outriggers_2018-06-01_10H17M12S.h5', h_252x, f0, lst0=11, lst1=12)
    ff, lsts, D0 = load_data_254x('data/outriggers_2018-06-01_10H17M12S.h5', h_254x, f0, lst0=11, lst1=12)
    ff, lsts, E0 = load_data_255x('data/outriggers_2018-06-01_10H17M12S.h5', h_255x, f0, lst0=11, lst1=12)

    aC = np.median(C0, axis=0)
    rC = aC - poly_fit(ff, aC, n_poly)
    aD = np.median(D0, axis=0)
    rD = aD - poly_fit(ff, aD, n_poly)
    aE = np.median(E0, axis=0)
    rE = aE - poly_fit(ff, aE, n_poly)

    plt.figure("ANT SPECTRUM")
    plt.plot(ff,  aC, label='252A')
    plt.plot(ff,  aD, label='254A')
    plt.plot(ff,  aE, label='255A')

    plt.xlim(f0, f1)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Temperature [K]")
    plt.minorticks_on()
    plt.legend()
    plt.savefig("img/02_ant_spec.png")

    plt.figure("DATA - %i_TERM_LOGPOLY" % n_poly)
    plt.plot(ff,  rC, label='252A')
    plt.plot(ff,  rD, label='254A')
    plt.plot(ff,  rE, label='255A')
    plt.xlim(f0, f1)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Temperature [K]")
    plt.minorticks_on()
    plt.legend()
    plt.savefig("img/02_resids_polyfit.png")
    plt.show()