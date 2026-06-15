import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1

# ==========================================
# RF & Antenna Classes
# ==========================================
class Antenna:
    def __init__(self, diameter=1.0, frequency=18.0, antennaClass=4):
        self.diameter = diameter
        self.frequency = frequency
        self.antennaClass = antennaClass
        
    def __str__(self):
        return f"Antenna {self.diameter}ft | {self.frequency}GHz | Class {self.antennaClass}"

    def gain(self):
        lambda1 = 300e6 / (self.frequency * 1e9)
        area = np.pi * (self.diameter * 0.305 / 2)**2
        return 10 * np.log10(4 * np.pi * 0.55 * area / (lambda1**2))
    
    def beamWidth(self):
        lambda1 = 300e6 / (self.frequency * 1e9)
        return 70 * lambda1 / (self.diameter * 0.305)
    
    def antennaAngularAttenuation(self, teta):
        Lambda1 = 300E6 / (self.frequency * 1E9)
        k = 2 * np.pi / Lambda1
        radius = self.diameter / 2
        
        if isinstance(teta, float):
            teta = np.array([[teta]])
            
        rpe1 = np.zeros(teta.shape)
        
        # ETSI Mask Definitions (Simplified mapping for brevity, focusing on Class 3 & 4)
        if 14 <= self.frequency <= 20 and self.antennaClass == 3:
            angleMask = np.array([5, 10, 25, 60, 95, 180]) 
            rpeMask = np.array([18, 9, 2, -4, -27, -27])
        else:
            # Default fallback mask to keep code concise
            angleMask = np.array([5, 10, 20, 40, 80, 100, 180]) 
            rpeMask = np.array([18, 9, -4, -13, -25, -30, -30])

        for row in range(teta.shape[0]):
            for column in range(teta.shape[1]):
                tetaTmp = np.abs(teta[row][column])
                if tetaTmp == 0: tetaTmp = 0.0001
                if tetaTmp > 180: tetaTmp = 360 - tetaTmp 
                        
                if tetaTmp < angleMask[0]:
                    # Optimized Bessel calculation using SciPy
                    x = k * tetaTmp * radius * np.pi / 180
                    rpe1[row][column] = 10 * math.log10((2 * j1(x) / x)**2)
                else:
                    index1 = np.where(angleMask <= tetaTmp)[0][-1]
                    index2 = np.where(angleMask >= tetaTmp)[0][0]
                    if index1 == index2:
                        rpe1[row][column] = rpeMask[index1] - self.gain()
                    else:
                        interp = rpeMask[index1] + (rpeMask[index2] - rpeMask[index1]) * \
                                 (tetaTmp - angleMask[index1]) / (angleMask[index2] - angleMask[index1])
                        rpe1[row][column] = interp - self.gain()
                            
        return rpe1

def freeSpaceLoss(distance, freq):
    return 92.5 + 20 * np.log10(distance * freq)

# ==========================================
# Max-Min Optimization Engine
# ==========================================
def optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, antenna1, chBW, NF, Pmax, Pmin, is_uplink=True):
    N = tail_angles.size
    Gain = antenna1.gain()
    thermalNoise = -174 + 10 * np.log10(chBW * 1e6) + NF
    thermalNoise_linear = 10**(thermalNoise / 10)

    # Angular Attenuation Matrix
    mat1 = np.ones((N, 1)) * tail_angles
    teta = mat1 - mat1.T
    attenMatrixAntenna = antenna1.antennaAngularAttenuation(teta)

    # Path Loss Setup
    tx_power = np.ones(N) * Pmax
    fsl_matrix = np.zeros((N, N))
    
    for row in range(N):
        for col in range(N):
            dist = tail_distances[row] if is_uplink else tail_distances[col]
            fade = rain_fade[row] if is_uplink else rain_fade[col]
            fsl_matrix[row][col] = Gain - fade - freeSpaceLoss(dist, antenna1.frequency) + Gain + attenMatrixAntenna[row][col]

    c2i_history = []
    
    minMax = 1000
    counter = 100

    while minMax > 1 and counter > 0:
        # Vectorized Rx Matrix calculation
        tx_matrix = np.tile(tx_power, (N, 1)).T
        rxMatrix = tx_matrix + fsl_matrix
        rxMatrix_linear = 10**(rxMatrix / 10)

        c2i = np.zeros(N)
        for i in range(N):
            signal = rxMatrix_linear[i, i]
            interference_plus_noise = np.sum(rxMatrix_linear[:, i]) - signal + thermalNoise_linear
            c2i[i] = 10 * np.log10(signal / interference_plus_noise)

        spare = c2i - target_C2I
        minMax = np.max(spare) - np.min(spare)
        
        # Max-Min Power Adjustments
        if np.min(spare) > 1.0:
            tx_power = tx_power - 0.8 * np.min(spare)
            
        best_idx = np.argmax(spare)
        worst_idx = np.argmin(spare)
        
        # Adaptive step size (smaller for Downlink to prevent ping-pong)
        step = 0.5 if is_uplink else 0.05
        
        tx_power[best_idx] -= step * minMax
        tx_power[worst_idx] += step * minMax
        
        # Strict Hardware Enforcement
        tx_power = np.clip(tx_power, Pmin, Pmax)
        
        if minMax < 2:
            counter -= 1

    return c2i, tx_power

# ==========================================
# Streamlit UI
# ==========================================
def main():
    st.set_page_config(page_title="PtMP Interference Optimizer", layout="wide")
    st.title("📡 Max-Min PtMP Link Optimizer")
    
    # Sidebar Controls
    st.sidebar.header("Network Parameters")
    num_links = st.sidebar.slider("Number of Links", min_value=2, max_value=20, value=6)
    target_C2I = st.sidebar.slider("Target C/I (dB)", min_value=10.0, max_value=60.0, value=34.0, step=1.0)
    
    st.sidebar.header("RF Hardware Configuration")
    freq = st.sidebar.number_input("Frequency (GHz)", value=18.0)
    Pmax = st.sidebar.number_input("Max Tx Power (dBm)", value=20.0)
    Pmin = st.sidebar.number_input("Min Tx Power (dBm)", value=-5.0)
    chBW = st.sidebar.number_input("Channel BW (MHz)", value=56.0)
    NF = st.sidebar.number_input("Noise Figure (dB)", value=5.0)
    antenna_size = st.sidebar.selectbox("Antenna Size (ft)", [1, 2, 3, 4], index=2)

    # Generate Topology Data
    np.random.seed(42) # Fixed seed for stable UI testing
    tail_angles = np.sort(np.random.uniform(-180, 180, num_links))
    tail_distances = np.random.uniform(1.0, 12.0, num_links)
    rain_fade = np.zeros(num_links) # Assumed clear sky for baseline

    # Build Antenna
    ant = Antenna(diameter=antenna_size, frequency=freq, antennaClass=3)
    
    st.markdown(f"**Current Hardware:** {ant} | **Channel:** {chBW} MHz | **Limits:** {Pmin} dBm to {Pmax} dBm")

    # Run Calculations
    if st.button("Run Optimization Engine", type="primary"):
        with st.spinner("Calculating matrices and balancing powers..."):
            
            # Baseline (Everyone transmits at Pmax)
            c2i_base_up, _ = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=True)
            # Optimize Uplink
            c2i_opt_up, tx_opt_up = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=True)
            
            # Optimize Downlink
            c2i_opt_down, tx_opt_down = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=False)

        # UI Results Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uplink (Tail to Hub)")
            fig_up, ax_up = plt.subplots(figsize=(6, 4))
            indices = np.arange(num_links)
            width = 0.35
            ax_up.bar(indices - width/2, c2i_base_up, width, label='Before (Pmax)', color='lightgray')
            ax_up.bar(indices + width/2, c2i_opt_up, width, label='After (Optimized)', color='tab:blue')
            ax_up.axhline(y=target_C2I, color='r', linestyle='--', label='Target C/I')
            ax_up.set_ylabel("C/I (dB)")
            ax_up.set_xlabel("Tail Link Index")
            ax_up.legend()
            st.pyplot(fig_up)
            
            st.markdown("**Final Tx Powers (dBm):**")
            st.code(np.round(tx_opt_up, 1))

        with col2:
            st.subheader("Downlink (Hub to Tail)")
            fig_down, ax_down = plt.subplots(figsize=(6, 4))
            # Simulating before state for downlink
            c2i_base_down, _ = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=False)
            
            ax_down.bar(indices - width/2, c2i_base_down, width, label='Before (Pmax)', color='lightgray')
            ax_down.bar(indices + width/2, c2i_opt_down, width, label='After (Optimized)', color='tab:green')
            ax_down.axhline(y=target_C2I, color='r', linestyle='--', label='Target C/I')
            ax_down.set_ylabel("C/I (dB)")
            ax_down.set_xlabel("Tail Link Index")
            ax_down.legend()
            st.pyplot(fig_down)
            
            st.markdown("**Final Tx Powers (dBm):**")
            st.code(np.round(tx_opt_down, 1))

if __name__ == "__main__":
    main()
