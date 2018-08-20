import pylab as plt
import hickle as hkl
import tables as tb
from leda_cal2.utils import poly_fit, timestamp_to_lst, closest
import numpy as np
import glob

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
    p.add_argument('files', help='glob of files to open, e.g. data/*2018*.h5', nargs='+')
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

    dfiles = sorted(args.files)
    print("Matching files: %s" % str(dfiles))

    lst0 = 10
    lst1 = 12

    plt.figure("CALIBRATED DATA", figsize=(6, 8))
    Clist, Dlist, Elist = [], [], []
    for filename in dfiles:
        date_str = filename.split('_')[1]
        ff, lsts, D = load_data_252x(filename, h_252x, f0, lst0, lst1)
        plt.subplot(3, 1, 1)
        if D.shape[0] != 0:
            Davg = np.median(D, axis=0)
            plt.plot(ff, Davg, label=date_str)
        plt.legend()
        plt.xlabel("%s [MHz]" % antennas[0])
        plt.ylabel("Temperature [K]")
        Clist.append(Davg)

        ff, lsts, D = load_data_254x(filename, h_254x, f0, lst0, lst1)
        plt.subplot(3, 1, 2)
        if D.shape[0] != 0:
            Davg = np.median(D, axis=0)
            plt.plot(ff, Davg, label=date_str)
        plt.xlabel("%s [MHz]" % antennas[1])
        plt.ylabel("Temperature [K]")
        plt.legend()
        Dlist.append(Davg)

        ff, lsts, D = load_data_255x(filename, h_255x, f0, lst0, lst1)
        plt.subplot(3, 1, 3)
        if D.shape[0] != 0:
            Davg = np.median(D, axis=0)
            plt.plot(ff, Davg, label=date_str)
        plt.xlabel("%s [MHz]" % antennas[2])
        plt.ylabel("Temperature [K]")
        plt.legend()
        Elist.append(Davg)
    plt.tight_layout()
    plt.savefig("img/03_caldata_daily.png")

    # Compute averages
    Carr = np.array(Clist)
    Cavg = np.median(Carr, axis=0)
    Darr = np.array(Dlist)
    Davg = np.median(Darr, axis=0)
    Earr = np.array(Elist)
    Eavg = np.median(Earr, axis=0)

    plt.figure("DAILY DIFF FROM AVG", figsize=(6, 8))
    for ii in range(len(Clist)):
        ll = dfiles[ii].split('_')[1]
        plt.subplot(3, 1, 1)
        plt.plot(ff, Clist[ii] - Cavg, label=ll)
        plt.subplot(3, 1, 2)
        plt.plot(ff, Dlist[ii] - Davg, label=ll)
        plt.subplot(3, 1, 3)
        plt.plot(ff, Elist[ii] - Eavg, label=ll)

    for ii in (1,2,3):
        plt.subplot(3,1,ii)
        plt.ylim(-50, 50)
        plt.xlabel("%s [MHz]" % antennas[ii-1])
        plt.ylabel("Temperature [K]")
        plt.legend()
    plt.tight_layout()
    plt.savefig("img/03_daily_diffs.png")
    plt.show()