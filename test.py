import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from utils import mat_load, trans_Vrf, Rate_func_fixed, Nt


# Load data 
H, H_est = mat_load('train_set/example/test')


H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
H = np.squeeze(H)

imperfect_CSI = Input(name='imperfect_CSI', shape=(H_input.shape[1:4]), dtype=tf.float32)
# FIX: Reverted to 1D shape since H is 2-dimensional (Batch_size, 64)
perfect_CSI = Input(name='perfect_CSI', shape=(H.shape[1],), dtype=tf.complex64)
SNR_input = Input(name='SNR_input', shape=(1,), dtype=tf.float32)

temp = BatchNormalization()(imperfect_CSI)
temp = Flatten()(temp)
temp = BatchNormalization()(temp)
temp = Dense(256, activation='relu')(temp)
temp = BatchNormalization()(temp)
temp = Dense(128, activation='relu')(temp)
phase = Dense(Nt)(temp)

V_RF = Lambda(trans_Vrf, output_shape=(Nt,))(phase)
rate = Lambda(Rate_func_fixed, output_shape=(1,))([perfect_CSI, V_RF, SNR_input])

model = Model(inputs=[imperfect_CSI, perfect_CSI, SNR_input], outputs=rate)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

# Load the official pre-trained weights
model.load_weights('./20db.h5')

rate_bfnn = []
snr_range = range(-20, 25, 5)
for snr in snr_range:
    SNR = np.power(10, np.ones([H.shape[0], 1]) * snr / 10)
    y = model.evaluate(x=[H_input, H, SNR], y=H, batch_size=10, verbose=0)
    rate_bfnn.append(-y)

# The correct baseline numbers calculated in MATLAB
rate_omp = [0.6981, 1.5695, 2.8526, 4.3701, 5.9825, 7.6278, 9.2838, 10.9431, 12.6036]

plt.figure(figsize=(8, 6))

plt.plot(snr_range, rate_bfnn, label="Learning-Based Hybrid Beamforming (BFNN)", 
         color='red', marker='o', linewidth=2.5, markersize=8)

plt.plot(snr_range, rate_omp, label="Model-Based Hybrid Beamforming (OMP)", 
         color='blue', marker='s', linestyle='--', linewidth=2.5, markersize=8)

plt.title("Spectral Efficiency: Learning-Based vs. Model-Based Beamforming", fontsize=14, fontweight='bold')
plt.xlabel("Signal-to-Noise Ratio (SNR) in dB", fontsize=12)
plt.ylabel("Achievable Spectral Efficiency (bps/Hz)", fontsize=12)
plt.legend(loc="upper left", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)

plt.savefig("Final_Simulation_Result4.png", dpi=300, bbox_inches='tight')
print("\nSuccess! Graph saved as Final_Simulation_Result.png")
plt.show()