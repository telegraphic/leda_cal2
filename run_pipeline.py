import os

os.system('python 00_compute_noise_params.py')
os.system('python 01_fit_cal_data.py')

filelist = ['outriggers_2018-05-03_14H26M18S.h5',
            'outriggers_2018-05-04_14H26M30S.h5',
            'outriggers_2018-05-05_14H26M32S.h5',
            'outriggers_2018-06-01_10H17M12S.h5',
            'outriggers_2018-06-02_10H17M17S.h5',
            'outriggers_2018-06-03_10H17M21S.h5',
            'outriggers_2018-06-04_10H17M30S.h5']

for filename in filelist:
    os.system('python 04_lmfit.py data/%s' % filename)

os.system('python 05_lmfit2.py')