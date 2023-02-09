import numpy as np
import scipy.fft 
import scipy.io
import math
import cmath
import struct
import matplotlib.pyplot as plt
import numpy.matlib
from os.path import exists
from pathlib import Path
from bayes_opt import BayesianOptimization
from mpl_toolkits.mplot3d import Axes3D

def shift(register, feedback, output):
    """GPS Shift Register
    :param list feedback: which positions to use as feedback (1 indexed)
    :param list output: which positions are output (1 indexed)
    :returns output of shift register:
    """

    # calculate output
    out = [register[i - 1] for i in output]
    if len(out) > 1:
        out = sum(out) % 2
    else:
        out = out[0]

    # modulo 2 add feedback
    fb = sum([register[i - 1] for i in feedback]) % 2

    # shift to the right
    for i in reversed(range(len(register[1:]))):
        register[i + 1] = register[i]

    # put feedback in position 1
    register[0] = fb

    return out



def cacode(sv):
    """Build the CA code (PRN) for a given satellite ID
    :param int sv: satellite code (1-32)
    :returns list: ca code for chosen satellite
    """

    SV = {
        1: [2, 6],
        2: [3, 7],
        3: [4, 8],
        4: [5, 9],
        5: [1, 9],
        6: [2, 10],
        7: [1, 8],
        8: [2, 9],
        9: [3, 10],
        10: [2, 3],
        11: [3, 4],
        12: [5, 6],
        13: [6, 7],
        14: [7, 8],
        15: [8, 9],
        16: [9, 10],
        17: [1, 4],
        18: [2, 5],
        19: [3, 6],
        20: [4, 7],
        21: [5, 8],
        22: [6, 9],
        23: [1, 3],
        24: [4, 6],
        25: [5, 7],
        26: [6, 8],
        27: [7, 9],
        28: [8, 10],
        29: [1, 6],
        30: [2, 7],
        31: [3, 8],
        32: [4, 9],
    }

    # init registers
    G1 = [1 for i in range(10)]
    G2 = [1 for i in range(10)]

    ca = []  # stuff output in here

    # create sequence
    for i in range(1023):
      g1 = shift(G1, [3, 10], [10])
      g2 = shift(G2, [2, 3, 6, 8, 9, 10], SV[sv])  # <- sat chosen here from table

        # modulo 2 add and append to the code
      ca.append((g1 + g2) % 2)

    # return C/A code!
    return -np.sign(np.array(ca) - 0.5)

def reference_signal(PRN,offset,f_seq,fs,num_periods):

 seq = cacode(PRN)

 seq = np.matlib.repmat(seq,1,num_periods)[0]

 signal = sample_2(seq,offset,f_seq,fs)



 return signal

def sample_2(seq,offset,F_seq,Fs):


  T_seq=len(seq)/F_seq

  N_samples= math.floor(T_seq*Fs)

  seq_2 = [*seq, *seq, *seq, *seq, *seq]

  if offset < 0:
      
      offset_samp= math.ceil(abs(offset)*Fs)
      
      offset = offset_samp/Fs-abs(offset)
      index_samp=np.arange(2*N_samples+1-offset_samp,3*N_samples-offset_samp+1,1)
      
  else:
      index_samp= np.arange(2*N_samples+1,3*N_samples+1,1)
       
  select = []

  for i in range(len(index_samp)):
      select.append(math.floor((index_samp[i]*1/Fs+offset)*F_seq))
  y = []
  for i in select:
      y.append(seq_2[i])
  

  return y


def func(x):
    return (4*x**2+20*x+7)


#Buscar Scipy.Optimize

def ca_acquisition(t,signal,fs,f_seq,num_periods):
  
  # Aquisição FFT
  doppler_bin_vec = np.arange(-6e3,6e3+100,100)
  threshold = 0.34e5
  sats_found = []
  ACQ_DATA = {'cost_function':[],'data':[]}

  for PRN in range(1,33):

  #generate reference signal
    cost_function=[]
    reference = reference_signal(PRN,0,f_seq,fs,num_periods)
    reference = np.array(reference)
    

    def cost_function_ca_acquisition(doppler_bin_vec):


          
  
      doppler_bin_signal = np.exp(1j*2*np.pi*doppler_bin_vec*t)
      COST=(abs(scipy.fft.ifft(np.conjugate(scipy.fft.fft(signal))*scipy.fft.fft(doppler_bin_signal*reference))))**2 #adaptar
          
          
      return -COST.max()

    

    
    optimizer = scipy.optimize.minimize(
      cost_function_ca_acquisition,
      -6e3,
      method='Powell',
      options = {"disp":False,'xtol': 1e-5,'ftol':1e-5,'maxiter':3},
      bounds = [[-6.e3,6e3]]
    )
    
    MAX_Valor=abs(optimizer.fun)
  
    
    #ACQ_DATA['cost_function'].append(cost_function)
    #temp_array = np.array(cost_function)
    #max_val = temp_array.max()
    

    #if PRN == 1:

    #  COST_ARRAY = np.array(cost_function)

    #else:

    #  COST_ARRAY = np.c_[COST_ARRAY,temp_array]

    if MAX_Valor>threshold:
          
          #(max_index_row,max_index_colum) =np.nonzero(temp_array == max_val)[0][0]+1,np.nonzero(temp_array == max_val)[1][0]+1#adaptar

          
          sats_found = [*sats_found, PRN]#adpatar
          
    #else:
          
          #max_index_row = float('nan')
          #max_index_colum = float('nan')

    ACQ_DATA['data'].append( [ MAX_Valor])
     #ACQ_DATA(PRN).max_index=[max_index_row, max_index_colum] #adaptar
     #ACQ_DATA(PRN).max_val=max_val#adaptar
    COST = []

  return (ACQ_DATA,doppler_bin_vec,threshold,sats_found)

fs = 4e6
T_d= 1e-3
f_seq = 1.023e6
Ts=1/fs
Tc=1/f_seq
f_c=1575.42e6
num_periods=1
Delta=0.5*Tc
K=500
N=num_periods*fs*T_d
N_acq=fs*T_d
index_sample_in=100*N
t=np.arange(0,(N-0.5)*Ts,Ts)
x = scipy.io.loadmat('X.mat')
x = x['x']
x = x[0]
(ACQ_DATA,doppler_bin_vec,threshold,sats_found)=ca_acquisition(t[0:int(N_acq)],x[0:int(N_acq)],fs,f_seq,1)
print(sats_found)
