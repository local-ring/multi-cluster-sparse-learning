import matlab.engine
import os
from utils.communication import save_data, read_result

def call_matlab(datafile, resultfile, rho, mu, k=None):
    eng = matlab.engine.start_matlab()
    try:
        eng.cd(os.path.abspath('./src/PQN/'))
        eng.addpath(os.path.abspath('./src/PQN/'))
        eng.addpath(eng.genpath(os.path.abspath('./src/PQN/')))
        eng.addpath(eng.genpath(os.path.abspath('./src/PQN/minConF/')))
        eng.gfl_pqn(datafile, resultfile, rho, mu, float(k), nargout=0)
    finally:
        eng.quit()

def gfl_pqn(X, y, L, i, k=None, rho=None, mu=0.01, datafile=None, resultfile=None):
    datafile_name = os.path.join(datafile, f'data_{i}.mat')
    resultfile_name = os.path.join(resultfile, f'result_{i}.mat')
    save_data(X=X, y=y, L=L, filename=datafile_name)
    call_matlab(datafile_name, resultfile_name, rho, mu, k)
    u, _ = read_result(resultfile_name)
    return u.flatten()