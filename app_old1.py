# -*- coding: utf-8 -*-

import numpy as np
import math
import mpmath
import pandas as pd
import streamlit as st

# =================
# Helper Functions & Classes
# =================

class Antenna:
    def __init__(self, diameter=1, frequency=18, antennaClass=4):
        self.diameter = diameter
        self.frequency = frequency
        self.antennaClass = antennaClass
        
    def __str__(self):
        return f"Antenna with diameter {self.diameter} feet at {self.frequency} GHz class {self.antennaClass} "

    def gain(self):
        lambda1 = 300e6 / (self.frequency * 1e9)
        area = np.pi * (self.diameter * 0.305 / 2)**2
        return 10 * np.log10(4 * np.pi * 0.55 * area / (lambda1**2))
    
    def beamWidth(self):
        lambda1 = 300e6 / (self.frequency * 1e9)
        return 70 * lambda1 / (self.diameter * 0.305)
    
    def antennaAngularAttenuation(self, teta):
        # teta is a matrix in degrees
        # Function returns the attenuation of the antenna at teta degrees from center
        Lambda1 = 300E6 / (self.frequency * 1E9)
        k = 2 * np.pi / Lambda1
        radius = self.diameter / 2
    
        if type(teta) == float:
            teta = np.array([[teta]])
        
        rpe1 = np.zeros(teta.shape)
    
        # 3-14GHz Class 4
        if self.frequency >= 3 and self.frequency <= 14 and self.antennaClass == 4:
            angleMask = np.array([5, 10, 20, 50, 70, 85, 105, 180]) 
            rpeMask = np.array([16, 5, -7,-18, -20, -24, -30, -30])

        # 3-14GHz Class 3
        elif self.frequency >= 3 and self.frequency <= 14 and self.antennaClass == 3:
            angleMask = np.array([5, 20, 70, 100, 180]) 
            rpeMask = np.array([20, 8, -5,-25, -25])
    
        # 14-20GHz Class 4
        elif self.frequency >= 14 and self.frequency <= 20 and self.antennaClass == 4:
            angleMask = np.array([5, 10, 20, 40, 80, 100, 180]) 
            rpeMask = np.array([18, 9, -4,-13, -25, -30, -30])

        # 14-20GHz Class 3
        elif self.frequency >= 14 and self.frequency <= 20 and self.antennaClass == 3:
            angleMask = np.array([5, 10, 25, 60, 95, 180]) 
            rpeMask = np.array([18, 9, 2,-4, -27, -27])
            
        # 20-24GHz Class 4
        elif self.frequency >= 20 and self.frequency <= 24 and self.antennaClass == 4:
            angleMask = np.array([5, 10, 20, 40, 80, 100, 180]) 
            rpeMask = np.array([18, 9, -4,-13, -25, -30, -30])

        # 20-24GHz Class 3
        elif self.frequency >= 20 and self.frequency <= 24 and self.antennaClass == 3:
            angleMask = np.array([5, 10, 20, 40, 50, 100, 180]) 
            rpeMask = np.array([20, 12, 7,3, 0, -23, -23])
            
        # 24-30GHz Class 4
        elif self.frequency >= 24 and self.frequency <= 30 and self.antennaClass == 4:
            angleMask = np.array([5, 10, 20, 40, 80, 100, 180]) 
            rpeMask = np.array([18, 9, -4,-13, -25, -30, -30])

        # 24-30GHz Class 3
        elif self.frequency >= 24 and self.frequency <= 30 and self.antennaClass == 3:
            angleMask = np.array([5, 20, 55, 100, 180]) 
            rpeMask = np.array([20, 5, 0,-23, -25])
            
        # 30-47GHz Class 4
        elif self.frequency >= 30 and self.frequency <= 47 and self.antennaClass == 4:
            angleMask = np.array([5, 10, 20, 40, 90, 180]) 
            rpeMask = np.array([12, 5, -4,-13, -24, -24])

        # 30-47GHz Class 3
        elif self.frequency >= 30 and self.frequency <= 47 and self.antennaClass == 3:
            angleMask = np.array([5, 10, 15, 20, 50, 50, 65, 75, 90, 180]) 
            rpeMask = np.array([16, 9, 5,0, -7, -8, -10, -10, -17, -17])
        
        # 71-80GHz Class 3 
        elif self.frequency >= 70 and self.frequency <= 90 and self.antennaClass == 3:
            angleMask = np.array([5, 10, 20, 50, 70, 90, 180]) 
            rpeMask = np.array([16, 9, 1, -1, -4, -17, -17])
        
        # Default fallback
        else:
            angleMask = np.array([5, 10, 20, 50, 70, 90, 180]) 
            rpeMask = np.array([16, 9, 1, -1, -4, -17, -17])

        for row in range(teta.shape[0]):
            for column in range(teta.shape[1]):
                tetaTmp = np.abs(teta[row][column])
                if tetaTmp == 0:
                    tetaTmp = 0.0001
                    
                if tetaTmp > 180:
                    tetaTmp = 360 - tetaTmp 
                        
                if tetaTmp < angleMask[0]:
                    rpe1[row][column] = 10 * math.log10((2 * mpmath.besselj(1, k * tetaTmp * radius * np.pi / 180) / (k * tetaTmp * radius * np.pi / 180))**2)
                else:
                    index1 = np.where(angleMask <= tetaTmp)[0][-1]
                    index2 = np.where(angleMask >= tetaTmp)[0][0]
                    if index1 == index2:
                        rpe1[row][column] = rpeMask[index1] - self.gain()
                    else:
                        rpe1[row][column] = rpeMask[index1] + (rpeMask[index2] - rpeMask[index1]) * (tetaTmp - angleMask[index1]) / (angleMask[index2] - angleMask[index1]) - self.gain()
                            
        return rpe1


def freeSpaceLoss(distance, freq):
    # Freq is in GHz, distance is in Km
    return 92.5 + 20 * np.log10(distance * freq)
    

# =================
# Optimization Logic
# =================

def reuse_app(tail_angles, tail_distances, rain_fade, target_C2I, antenna1, chBW=2000, NF=5, Pmax=15, Pmin=-2):
    freq = antenna1.frequency
    Gain = antenna1.gain() 
    BW = antenna1.beamWidth() 
    thermalNoise = -174 + 10 * np.log10(chBW * 1e6) + NF

    N = tail_angles.size
    mat1 = np.ones((N, 1)) * tail_angles
    mat2 = mat1.transpose()
    teta = mat1 - mat2 

    attenMatrixAntenna = antenna1.antennaAngularAttenuation(teta)

    # ------------------
    # From Tail to Hub
    # ------------------
    tail_tx_power = np.ones(tail_angles.shape) * Pmax 

    rxMatrix = np.zeros(attenMatrixAntenna.shape)
    for row in range(tail_angles.shape[0]):
        for column in range(tail_angles.shape[0]):
            fsl = freeSpaceLoss(tail_distances[row], freq)
            rxMatrix[row][column] = tail_tx_power[row] + Gain - rain_fade[row] - fsl + Gain + attenMatrixAntenna[row][column]
            
    c2i = np.zeros(rxMatrix.shape[1])
    for column in range(rxMatrix.shape[1]):
        noise = 0
        for row in range(rxMatrix.shape[0]): 
            if column == row:
                s = rxMatrix[row][column]
            else:
                noise = noise + 10 ** (rxMatrix[row][column] / 10)
        
        noise = noise + 10**(thermalNoise / 10)
        noiseDbm = 10 * np.log10(noise)
        c2i[column] = s - noiseDbm
    
    tail_tx_power_before_optimization = tail_tx_power.copy()
    tail_c2i_before_optimization = c2i.copy()

    minMax = 1000
    counter1 = 100

    while minMax > 1 or counter1 > 2: 
        rxMatrix = np.zeros(attenMatrixAntenna.shape)
        for row in range(tail_angles.shape[0]):
            for column in range(tail_angles.shape[0]):
                fsl = freeSpaceLoss(tail_distances[row], freq)
                rxMatrix[row][column] = tail_tx_power[row] + Gain - rain_fade[row] - fsl + Gain + attenMatrixAntenna[row][column]
            
        c2i = np.zeros(rxMatrix.shape[1])
        for column in range(rxMatrix.shape[1]):
            noise = 0
            for row in range(rxMatrix.shape[0]): 
                if column == row:
                    s = rxMatrix[row][column]
                else:
                    noise = noise + 10 ** (rxMatrix[row][column] / 10)
        
            noise = noise + 10**(thermalNoise / 10)
            noiseDbm = 10 * np.log10(noise)
            c2i[column] = s - noiseDbm
    
        spare = c2i - target_C2I    
        minMax = np.max(spare) - np.min(spare)
        
        if np.min(spare) > 1.0:
            tail_tx_power = tail_tx_power - 0.8 * np.min(spare)
        tail_tx_power[np.argmax(spare)] = tail_tx_power[np.argmax(spare)] - 0.5 * minMax
        tail_tx_power[np.argmin(spare)] = tail_tx_power[np.argmin(spare)] + 0.5 * minMax
        if tail_tx_power[np.argmin(spare)] > Pmax:
            tail_tx_power[np.argmin(spare)] = Pmax
        
        if minMax < 2:
            counter1 = counter1 - 1

    tail_tx_power_after_optimization = tail_tx_power
    tail_c2i_after_optimization = c2i    


    # ------------------
    # From Hub to Tail
    # ------------------
    hub_tx_power = np.ones(tail_angles.shape) * Pmax 

    rxMatrix = np.zeros(attenMatrixAntenna.shape)
    for row in range(tail_angles.shape[0]):
        for column in range(tail_angles.shape[0]):
            fsl = freeSpaceLoss(tail_distances[column], freq)
            rxMatrix[row][column] = hub_tx_power[row] + Gain - rain_fade[column] - fsl + Gain + attenMatrixAntenna[row][column]
            
    c2i = np.zeros(rxMatrix.shape[1])
    for column in range(rxMatrix.shape[1]):
        noise = 0
        for row in range(rxMatrix.shape[0]): 
            if column == row:
                s = rxMatrix[row][column]
            else:
                noise = noise + 10 ** (rxMatrix[row][column] / 10)
        
        noise = noise + 10**(thermalNoise / 10)
        noiseDbm = 10 * np.log10(noise)
        c2i[column] = s - noiseDbm
    
    hub_tx_power_before_optimization = hub_tx_power.copy()
    hub_c2i_before_optimization = c2i.copy()

    minMax = 1000
    counter1 = 100

    while minMax > 1 or counter1 > 2: 
        rxMatrix = np.zeros(attenMatrixAntenna.shape)
        for row in range(tail_angles.shape[0]):
            for column in range(tail_angles.shape[0]):
                fsl = freeSpaceLoss(tail_distances[column], freq)
                rxMatrix[row][column] = hub_tx_power[row] + Gain - rain_fade[column] - fsl + Gain + attenMatrixAntenna[row][column]
            
        c2i = np.zeros(rxMatrix.shape[1])
        for column in range(rxMatrix.shape[1]):
            noise = 0
            for row in range(rxMatrix.shape[0]): 
                if column == row:
                    s = rxMatrix[row][column]
                else:
                    noise = noise + 10 ** (rxMatrix[row][column] / 10)
        
            noise = noise + 10**(thermalNoise / 10)
            noiseDbm = 10 * np.log10(noise)
            c2i[column] = s - noiseDbm
    
        spare = c2i - target_C2I    
        minMax = np.max(spare) - np.min(spare)
        
        if np.min(spare) > 1.0:
            hub_tx_power = hub_tx_power - 0.9 * np.min(spare)
            
        hub_tx_power[np.argmax(spare)] = hub_tx_power[np.argmax(spare)] - 0.05 * minMax
        hub_tx_power[np.argmin(spare)] = hub_tx_power[np.argmin(spare)] + 0.05 * minMax
        
        if hub_tx_power[np.argmin(spare)] > Pmax:
            hub_tx_power[np.argmin(spare)] = Pmax
            
        if minMax < 2:
            counter1 = counter1 - 1

    hub_tx_power_after_optimization = hub_tx_power
    hub_c2i_after_optimization = c2i
    
    return (tail_c2i_before_optimization, tail_tx_power_before_optimization, 
            tail_c2i_after_optimization, tail_tx_power_after_optimization, 
            hub_c2i_before_optimization, hub_tx_power_before_optimization, 
            hub_c2i_after_optimization, hub_tx_power_after_optimization)

# =================
# Streamlit GUI App
# =================

st.set_page_config(page_title="PtMP Optimization Tool", layout="wide")
st.title("Point-to-Multipoint Network Power & C/I Optimization")
st.markdown("Set network parameters and dynamically allocate randomly placed CPEs to evaluate convergence of Carrier-to-Interference ratios across the sector.")

# Sidebar parameters
st.sidebar.header("System Parameters")
freq = st.sidebar.number_input("Frequency (GHz)", value=18.0, step=0.5)
antenna_size = st.sidebar.number_input("Antenna Size (ft)", value=3.0, step=0.5)
chBW = st.sidebar.number_input("Channel BW (MHz)", value=56.0, step=1.0)
NF = st.sidebar.number_input("Noise Figure (dB)", value=5.0, step=0.5)

st.sidebar.header("Deployment Limits")
Pmax = st.sidebar.number_input("Max Tx Power (dBm)", value=20.0, step=1.0)
Pmin = st.sidebar.number_input("Min Tx Power (dBm)", value=-5.0, step=1.0)
target_c2i_val = st.sidebar.number_input("Target C/I (dB)", value=35.0, step=1.0)

st.sidebar.header("Simulation Config")
num_cpe = st.sidebar.number_input("Number of CPEs", min_value=1, max_value=50, value=6, step=1)

if st.sidebar.button("Run Single Simulation", type="primary"):
    
    # Generate random parameters for the selected number of CPEs
    # Angles uniformly distributed in a -180 to 180 sector (modify if specific sector is needed)
    tail_angles = np.round(np.random.uniform(-180, 180, num_cpe), 1)
    
    # Distances between 1km and 15km
    tail_distances = np.round(np.random.uniform(1.0, 15.0, num_cpe), 2)
    
    # Setting rain fade to 0 for a clean single-run view, or can easily swap to random.
    rain_fade = np.zeros(num_cpe)
    
    targetC2I = np.ones(num_cpe) * target_c2i_val

    # Setup Antenna
    antenna1 = Antenna(diameter=antenna_size, frequency=freq, antennaClass=3)
    
    st.write(f"### Simulation Scenario")
    st.write(f"**{str(antenna1)}**")

    # Run algorithm
    results = reuse_app(tail_angles, tail_distances, rain_fade, targetC2I, antenna1, chBW, NF, Pmax, Pmin)
    (tail_c2i_before, tail_power_before, tail_c2i_after, tail_power_after, 
     hub_c2i_before, hub_power_before, hub_c2i_after, hub_power_after) = results

    # Data formatting for presentation
    df_tail = pd.DataFrame({
        "CPE Node": [f"Tail {i+1}" for i in range(num_cpe)],
        "Distance (km)": tail_distances,
        "Angle (deg)": tail_angles,
        "Before C/I (dB)": np.round(tail_c2i_before, 2),
        "After C/I (dB)": np.round(tail_c2i_after, 2),
        "Before Power (dBm)": np.round(tail_power_before, 2),
        "After Power (dBm)": np.round(tail_power_after, 2)
    })
    
    df_hub = pd.DataFrame({
        "CPE Node": [f"Tail {i+1}" for i in range(num_cpe)],
        "Distance (km)": tail_distances,
        "Angle (deg)": tail_angles,
        "Before C/I (dB)": np.round(hub_c2i_before, 2),
        "After C/I (dB)": np.round(hub_c2i_after, 2),
        "Before Power (dBm)": np.round(hub_power_before, 2),
        "After Power (dBm)": np.round(hub_power_after, 2)
    })

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uplink: Tail to Hub")
        st.dataframe(df_tail, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Downlink: Hub to Tail")
        st.dataframe(df_hub, hide_index=True, use_container_width=True)
else:
    st.info("Adjust parameters in the sidebar and click **Run Single Simulation** to begin.")
