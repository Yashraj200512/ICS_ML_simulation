import tensorflow as tf
import scipy.io as sio

# ---------------------
# System Parameters
# ---------------------
Nt = 64  # Total number of transmit antennas
P = 1    # Normalized transmit power constraint

# ---------------------
# Core Functions
# ---------------------

def trans_Vrf(temp):
    """
    Converts the phase angles predicted by the network into a complex-valued analog beamformer.
    
    Args:
        temp: Phase angles (radians).
        
    Returns:
        Complex tensor representing the RF beamforming vector.
    """
    v_real = tf.cos(temp)
    v_imag = tf.sin(temp)
    vrf = tf.cast(tf.complex(v_real, v_imag), tf.complex64)
    return vrf

def Rate_func_fixed(temp):
    """
    Calculates the achievable spectral efficiency (rate) for the current beamformer.
    Returns the negative rate so the model optimizer can minimize the loss function.
    
    Args:
        temp: A tuple containing (perfect CSI, RF beamformer, linear SNR).
        
    Returns:
        Negative spectral efficiency (float32 tensor).
    """
    h, v, SNR_input = temp
    
    # Compute the effective channel gain
    hv = tf.reduce_sum(tf.cast(h, tf.complex64) * v, axis=1, keepdims=True)
    
    # Calculate spectral efficiency using Shannon's capacity formula
    rate = tf.math.log(tf.cast(1 + SNR_input / Nt * tf.pow(tf.abs(hv), 2), tf.float32)) / tf.math.log(2.0)
    
    return -rate

def mat_load(path):
    """
    Loads the perfect and estimated Channel State Information (CSI) matrices from MATLAB outputs.
    
    Args:
        path: Directory path containing the 'pcsi.mat' and 'ecsi.mat' files.
        
    Returns:
        h: Perfect CSI array.
        h_est: Estimated CSI array.
    """
    print('Loading CSI dataset...')
    h = sio.loadmat(path + '/pcsi.mat')['pcsi']
    h_est = sio.loadmat(path + '/ecsi.mat')['ecsi']
    print('Loading complete.')
    print('The shape of the estimated CSI is: ', h_est.shape)
    
    return h, h_est