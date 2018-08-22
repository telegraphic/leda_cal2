"""
skymodel.py
-----------

Global Sky Model (GSM) class used for calibration of ledaspec data.


"""


import hickle as hkl
import numpy as np
import pylab as plt

from scipy import interpolate
from scipy.optimize import curve_fit

class SkyModel(object):
    """ GSM sky model class"""
    def __init__(self, pol='x', npol=7):
        self.pol = pol
        if pol == 'x':
            self.data = hkl.load('gsm/gsm_ew.hkl')[:, 1:].T
            self.lsts = hkl.load('gsm/gsm_ew.hkl')[:, 0].T

        if pol == 'y':
            self.data = hkl.load('gsm/gsm_ns.hkl')[:, 1:].T
            self.lsts = hkl.load('gsm/gsm_ns.hkl')[:, 0].T

        self.freqs = np.arange(30, 90.01, 5) * 1e6
        self.npol  = npol
        
        self.compute_gsm_splines()
    
    def curve_fit(self, f, d):
        """ Wrapper for curve_fit """
        f_l = np.log10(f / 70e6)
        d_l = np.log10(d)
        fit = np.polyfit(f_l, d_l, self.npol)
        
        return fit
    
    def compute_gsm_splines(self):
        """ Load GSM data and form interpolation splines """

        print "computing GSM for Pol %s" % self.pol
        drift_data, drift_lsts, drift_freqs = self.data, self.lsts, self.freqs
        
        # Extend to full 24 hours then form interpolation spline
        nd = np.zeros((13, 145))
        nd[:, :144] = drift_data
        nd[:, 144]  = drift_data[:, 0]
        drift_lsts  = np.append(drift_lsts, drift_lsts[0]+24)
        drift_lsts[0] = 0.0
        drift_data  = nd
        
        fits = [ [] for ii in range(self.npol + 1)]
        #print fits
        for ii in range(len(drift_lsts)):

            fit = self.curve_fit(self.freqs, drift_data[:, ii])
            #if not ii%10:
                #print fit
            
            for jj in range(len(fit)):
                fits[jj].append(fit[jj])
        
        self.gsm_pols = np.array(fits)
        
        self.gsm_spline = []
        for kk in range(self.gsm_pols.shape[0]):
            self.gsm_spline.append(interpolate.interp1d(drift_lsts, self.gsm_pols[kk, :], kind='cubic'))

    def _generate_tsky(self, lst, freqs):
        try:
            poly = []
            for spl in self.gsm_spline:
                poly.append(spl(lst))

            p = np.poly1d(poly)
            return 10**(p(np.log10(freqs / 70e6)))
        except:
            return np.zeros_like(freqs)

    def generate_tsky(self, lst, freqs):
        """ Compute GSM Tsky for given lst and frequency range """

        if type(lst) is type(np.array([1,2])):
            tsky = np.zeros((len(lst), len(freqs)))
            ii = 0
            for l in lst:
                tsky[ii] = self._generate_tsky(l, freqs)
                ii += 1
            return tsky
        else:
            return self._generate_tsky(lst, freqs)

    def generate_tsky_vec(self, lst, freqs):
        """ TODO: Make this work. """
        tsky = np.zeros((len(lst), len(freqs)))
        np.apply_along_axis(self._generate_tsky, 0, tsky, lst, freqs)
        return tsky

if __name__ == '__main__':        
    sx = SkyModel('x')
    sy = SkyModel('y')
    f = np.linspace(30, 90, 100) * 1e6

    xx = []
    yy = []

    for ii in range(0, 24):
        xx.append(sx.generate_tsky(ii, f))
        yy.append(sy.generate_tsky(ii, f))

    xx, yy = np.array(xx), np.array(yy)

    plt.plot(xx[:, 0])
    plt.plot(yy[:, 0])
    plt.show()