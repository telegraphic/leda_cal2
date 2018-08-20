from leda_cal2.utils import *
import numpy as np

T_0 = 290.0

def enr_to_k(enr, Tamb=290):
    """ Convert ENR to noise temp
    ENR = 10 Log10( (T - Tamb)/ T_0 )
    """
    T = T_0 * lin10(enr) + Tamb
    return T

def k_to_enr(T):
    """ Convert ENR to noise temp
    ENR = 10 Log10( (T - Tamb)/ T_0 )
    """
    ENR = db10((T - T_0) / T_0)
    return ENR

def y_factor(T_hot, T_cold, P_hot, P_cold):
    """ Compute Y-factor via standard method

    Args:
        T_hot: Hot load temperature in K
        T_cold: Cold load temperature in K
        P_hot: Power measured from HOT, linear units
        P_cold: Power measured from COLD, linear units
    """
    y = P_hot / P_cold
    T_dut = ((T_hot) - y * T_cold) / (y - 1)
    return T_dut

def NT_to_NF(T):
    """ Convert noise temperature to noise figure """
    return 1 + T / T_0

def NF_to_NT(F):
    """ Convert noise figure to noise temperature """
    return T_0 * (F - 1)

def s21_to_L(s, db=False):
    """ Convert S21 to L (attenuation) """
    if db:
        return 1.0 - np.abs(lin10(s))
    else:
        return 1.0 - np.abs(s)

def hp346c_enr(freqs, T_amb=290, L_atten=0, L_cable=0.0):
    """ Compute the ENR for a HP346C noise diode + attenuation

    This computes the ENR, taking into account cable losses and
    any extra attenuation added between the diode and the DUT.

    Args:
        freqs: np.array of frequency values, in MHz
        T_amb: Ambient temperature in K
        atten: np.array of attenuation values, in dB,
                note: atten.shape == freq.shape
        cable_loss: np.array of cable losses, in dB
            note: cable_loss.shape == freq.shape

    Returns np.array of ENR in dB or equivalent temperature in K
    """
    f_cal   = np.array([10, 100])
    enr_cal = np.array([14.91, 14.89])
    enr_fit = np.polyfit(f_cal, enr_cal, 1)
    enr     = np.poly1d(enr_fit)

    T_hp346 = enr_to_k(enr(freqs), T_amb)
    T_atten = T_amb * (L_atten + L_cable)

    T_equiv = (1 - L_atten) * T_hp346 + T_atten

    T_equiv_fit = np.poly1d(np.polyfit(freqs, T_equiv, 2))(freqs)
    return T_equiv_fit

def generate_T_amb_hot(length):
    """ T_hot and T_ambient are not loaded from files. """
    T_amb = np.full(length, 32+273.15)	# This is an array, so then T_hot will be an array
    Q = 14.9-6.95
    T_hot = T_amb*(10**(Q/10)+1)

    return T_amb, T_hot

def test_enr_to_k():
    assert int(enr_to_k(5)) == 1207
    print int(enr_to_k(15))
    assert int(enr_to_k(15)) == 9460
    assert np.isclose(enr_to_k(k_to_enr(1000)), 1000)

if __name__ == "__main__":
    test_enr_to_k()
