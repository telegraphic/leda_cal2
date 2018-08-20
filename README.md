## LEDA CAL2


#### 00_compute_noise_params.py

Compute all calibration quantities from calibration measurements.

Usage: `python 00_compute_noise_params.py`

#### 01_fit_cal_data.py

Runs a polynomial fit on calibration data to produce hkl files for calibration.

Usage: `python 01_compute_noise_params.py`

#### 02_compare_antennas.py

Comparison across antennas of calibrated data.


```
usage: 02_compare_antennas.py [-h] [-f0 F0] [-f1 F1] [-n N_POLY] filename

Plot antenna spectra and residuals

positional arguments:
  filename              name of file to open

optional arguments:
  -h, --help            show this help message and exit
  -f0 F0, --f0 F0       start frequency in MHz, default 40 MHz
  -f1 F1, --f1 F1       end frequency in MHz, default 87.6 MHz
  -n N_POLY, --n_poly N_POLY
                        number of terms in log-poly to fit for residuals.
                        Default 5
```


#### 03_compare_days.py

Compare across a number of days.


```
usage: 03_compare_days.py [-h] [-f0 F0] [-f1 F1] [-n N_POLY] files [files ...]

Plot antenna spectra and residuals

positional arguments:
  files                 glob of files to open, e.g. data/*2018*.h5

optional arguments:
  -h, --help            show this help message and exit
  -f0 F0, --f0 F0       start frequency in MHz, default 40 MHz
  -f1 F1, --f1 F1       end frequency in MHz, default 87.6 MHz
  -n N_POLY, --n_poly N_POLY
                        number of terms in log-poly to fit for residuals.
                        Default 5                      
```

#### 04_lmfit.py

Fit a sinusoisal model to remove from data bandpass.

```
usage: 04_lmfit.py [-h] [-f0 F0] [-f1 F1] [-n N_POLY] filename

Plot antenna spectra and residuals

positional arguments:
  filename              glob of files to open, e.g. data/*2018*.h5

optional arguments:
  -h, --help            show this help message and exit
  -f0 F0, --f0 F0       start frequency in MHz, default 40 MHz
  -f1 F1, --f1 F1       end frequency in MHz, default 87.6 MHz
  -n N_POLY, --n_poly N_POLY
                        number of terms in log-poly to fit for residuals.
                        Default 5
```

