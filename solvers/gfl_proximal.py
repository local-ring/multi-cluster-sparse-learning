import matlab.engine
import os
from utils.communication import save_data, read_result

def call_matlab(datafile, resultfile, rho1, rho2):
    eng = matlab.engine.start_matlab()
    try:
        eng.cd(os.path.abspath('./src/code_fgfl_aaai14/'))
        eng.addpath(os.path.abspath('./src/code_fgfl_aaai14/GFL/'))
        eng.addpath(eng.genpath(os.path.abspath('./code_fgfl_aaai14/')))
        eng.gfl_proximal(datafile, resultfile, rho1, rho2, nargout=0)
    finally:
        eng.quit()


def gfl_proximal(X, y, A, i, rho1=0.5, rho2=0.5, datafile=None, resultfile=None):
    datafile_name = os.path.join(datafile, f'data_{i}.mat')
    resultfile_name = os.path.join(resultfile, f'result_{i}.mat')
    save_data(X=X, y=y, A=A, filename=datafile_name)
    call_matlab(datafile_name, resultfile_name, rho1, rho2)
    u, _ = read_result(resultfile_name)
    return u.flatten() # the original return a vector with shape (d,1), will not work with recovery_accuracy

