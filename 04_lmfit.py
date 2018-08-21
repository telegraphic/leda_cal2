import pylab as plt
import hickle as hkl
import tables as tb
from leda_cal2.utils import poly_fit, timestamp_to_lst, closest
import numpy as np
import glob
from scipy.interpolate import interp1d as interp

from lmfit import minimize, Parameters, fit_report

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

def trim2(data, f, f0, f1):
    i0 = closest(f, f0)
    i1 = closest(f, f1)
    return f[i0:i1], data[i0:i1]

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


def residual(params, x, model, data):
    mm = model(x, params)
    return (data - mm)


def model_sin(x, params):
    PHI = params['PHI'].value
    PHI0 = params['PHI0'].value
    A_c = params['A_c'].value
    mm = A_c * np.sin(PHI * x + PHI0)
    return mm


def fit_model_sin(x, data):
    params = Parameters()
    params.add('PHI', value=0.3398, vary=True)
    params.add('A_c', value=146., vary=True)
    params.add('PHI0', value=-1.44)
    out = minimize(residual, params, args=(x, model_sin, data))
    outvals = out.params
    for param, val in out.params.items():
        print "%08s: %2.4f" % (param, val)
    return outvals


def fit_model_sin_off(x, data):
    params = Parameters()
    params.add('PHI', value=0.3398, vary=True)
    params.add('A_c', value=146., vary=True)
    params.add('PHI0', value=-1.44)
    params.add('B', value=226)
    params.add('M', value=0.2)
    out = minimize(residual, params, args=(x, model_sin_off, data))
    outvals = out.params
    for param, val in out.params.items():
        print "%08s: %2.4f" % (param, val)
    return outvals


def model_sin_off(x, params):
    PHI = params['PHI'].value
    PHI0 = params['PHI0'].value
    A_c = params['A_c'].value
    B = params['B'].value
    M = params['M'].value

    mm = A_c * np.sin(PHI * x + PHI0) + B + M * x
    return mm

def flag_data(f, d, bp, thr=5):
    """ Flag data. Returns compressed arrays

    flags data where abs(data - bandpass) > threshold

    f: frequency
    d: data (1D)
    bp: bandpass estimate
    thr: Threhold from bandpass above which to flag
    """
    r = d - bp
    d = np.ma.array(d)
    fm = np.ma.array(f)
    d.mask = np.abs(r) > thr
    fm.mask = d.mask
    d.mask[0] = False
    d.mask[-1] = False
    fm.mask[0] = False
    fm.mask[-1] = False
    ff, dd = fm.compressed(), d.compressed()

    return interp(ff, dd)(f)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Plot antenna spectra and residuals')
    p.add_argument('filename', help='Name of measured data file')
    p.add_argument('-f0', '--f0', help='start frequency in MHz, default 40 MHz', type=float, default=50)
    p.add_argument('-f1', '--f1', help='end frequency in MHz, default 87.6 MHz', type=float,  default=87.6)
    p.add_argument('-n',  '--n_poly', help='number of terms in log-poly to fit for residuals. Default 5', type=int,  default=3)
    args = p.parse_args()

    f0 = args.f0
    f1 = args.f1
    n_poly = args.n_poly
    antennas = ['252A', '254A', '255A']

    h_252x = hkl.load('cal_data_252A_fit.hkl')
    h_254x = hkl.load('cal_data_254A_fit.hkl')
    h_255x = hkl.load('cal_data_255A_fit.hkl')

    # Load and calibrate data
    ff, lsts, C0 = load_data_252x(args.filename, h_252x, f0, lst0=11, lst1=12)
    ff, lsts, D0 = load_data_254x(args.filename, h_254x, f0, lst0=11, lst1=12)
    ff, lsts, E0 = load_data_255x(args.filename, h_255x, f0, lst0=11, lst1=12)

    # Compute 1D spectrum and poly fit
    aC = np.median(C0, axis=0)
    rC = aC - poly_fit(ff, aC, n_poly)
    aD = np.median(D0, axis=0)
    rD = aD - poly_fit(ff, aD, n_poly)
    aE = np.median(E0, axis=0)
    rE = aE - poly_fit(ff, aE, n_poly)

    # Trim down to region of interest
    f2, rC = trim2(rC, ff, 58, 80)
    f2, rD = trim2(rD, ff, 58, 80)
    f2, rE = trim2(rE, ff, 58, 80)

    # Fit a sine wave
    rC_model_params = fit_model_sin_off(f2, rC)
    rC_sin_model    = model_sin_off(f2, rC_model_params)
    rD_model_params = fit_model_sin_off(f2, rD)
    rD_sin_model    = model_sin_off(f2, rD_model_params)
    rE_model_params = fit_model_sin_off(f2, rE)
    rE_sin_model    = model_sin_off(f2, rE_model_params)

    rC = flag_data(f2, rC, rC_sin_model, thr=8)
    rD = flag_data(f2, rD, rD_sin_model, thr=8)
    rE = flag_data(f2, rE, rE_sin_model, thr=8)

    plt.figure("FLAGGED")
    plt.plot(f2, rC)
    plt.plot(f2, rD)
    plt.plot(f2, rE)
    plt.savefig("img/04_flagged.png")

    rC_model_params = fit_model_sin_off(f2, rC)
    rC_sin_model    = model_sin_off(f2, rC_model_params)
    rD_model_params = fit_model_sin_off(f2, rD)
    rD_sin_model    = model_sin_off(f2, rD_model_params)
    rE_model_params = fit_model_sin_off(f2, rE)
    rE_sin_model    = model_sin_off(f2, rE_model_params)

    # 252A
    plt.figure("252A")
    plt.subplot(2,1,1)
    plt.plot(f2, rC, c='#cc0000')
    plt.plot(f2, rC_sin_model, c='#333333')
    plt.subplot(2,1,2)
    plt.plot(f2, rC - rC_sin_model, c='#cc0000')
    plt.savefig("img/04_r252A.png")

    # 254A
    plt.figure("254A")
    plt.subplot(2,1,1)
    plt.plot(f2, rD, c='#cc0000')
    plt.plot(f2, rD_sin_model, c='#333333')

    plt.subplot(2,1,2)
    plt.plot(f2, rD - rD_sin_model, c='#cc0000')
    plt.savefig("img/04_r254A.png")

    # 255A
    plt.figure("255A")
    plt.subplot(2,1,1)
    plt.plot(f2, rE, c='#cc0000')
    plt.plot(f2, rE_sin_model, c='#333333')

    plt.subplot(2,1,2)
    plt.plot(f2, rE - rE_sin_model, c='#cc0000')
    plt.savefig("img/04_r255A.png")


    plt.figure("ALL_AVG")
    rAll = ((rC - rC_sin_model) + (rD - rD_sin_model) + (rE - rE_sin_model)) / 3
    plt.plot(f2, rAll)
    plt.show()

