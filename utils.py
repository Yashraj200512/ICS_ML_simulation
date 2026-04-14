import tensorflow as tf
import scipy.io as sio

# --- Basic Settings ---
Nt = 64  # Total number of antennas
P = 1    # Transmit power limit 

# --- Helper Functions ---

def trans_Vrf(temp):
    # Takes the raw angles (phases) predicted by the neural network
    # and turns them into complex numbers (real + imaginary parts).
    # This creates the physical radio frequency (RF) beamformer.
    v_real = tf.cos(temp)
    v_imag = tf.sin(temp)
    vrf = tf.cast(tf.complex(v_real, v_imag), tf.complex64)
    return vrf

def Rate_func_fixed(temp):
    # Calculates how efficiently data can be transmitted (the "rate").
    # We make the final answer negative because Keras tries to make the loss 
    # as low as possible. Minimizing a negative number forces the actual rate to go up!
    h, v, SNR_input = temp
    
    # Step 1: Multiply the channel (h) by the beamformer (v)
    hv = tf.reduce_sum(tf.cast(h, tf.complex64) * v, axis=1, keepdims=True)
    
    # Step 2: Use Shannon's formula to calculate capacity: log2(1 + SNR * |hv|^2)
    rate = tf.math.log(tf.cast(1 + SNR_input / Nt * tf.pow(tf.abs(hv), 2), tf.float32)) / tf.math.log(2.0)
    
    return -rate

def mat_load(path):
    # Simply grabs the data files you originally created in MATLAB.
    # 'pcsi' is the perfect channel knowledge.
    # 'ecsi' is the estimated channel knowledge with errors.
    print('Loading data...')
    h = sio.loadmat(path + '/pcsi.mat')['pcsi']
    h_est = sio.loadmat(path + '/ecsi.mat')['ecsi']
    
    print('Loading complete.')
    print('Shape of the estimated data is: ', h_est.shape)
    
    return h, h_est