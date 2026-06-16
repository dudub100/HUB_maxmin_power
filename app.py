import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rand
import mpmath

# ==========================================
# Core Classes and Functions (From original)
# ==========================================

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
        else:
            # Fallback
            angleMask = np.array([5, 180])
            rpeMask = np.array([10, -30])
    
        for row in range(teta.shape[0]):
            for column in range(teta.shape[1]):
                tetaTmp = np.abs(teta[row][column])
                if tetaTmp == 0:
                    tetaTmp = 0.0001
                if tetaTmp > 180:
                    tetaTmp = 360 - tetaTmp 
                        
                if tetaTmp < angleMask[0]:
                    val = float(mpmath.besselj(1, k * tetaTmp * radius * np.pi / 180))
                    denom = k * tetaTmp * radius * np.pi / 180
                    rpe1[row][column] = 10 * math.log10((2 * val / denom)**2)
                else:
                    index1 = np.where(angleMask <= tetaTmp)[0][-1]
                    index2 = np.where(angleMask >= tetaTmp)[0][0]
                    if index1 == index2:
                        rpe1[row][column] = rpeMask[index1] - self.gain()
                    else:
                        rpe1[row][column] = rpeMask[index1] + (rpeMask[index2] - rpeMask[index1]) * (tetaTmp - angleMask[index1]) / (angleMask[index2] - angleMask[index1]) - self.gain()
                            
        return rpe1

def freeSpaceLoss(distance, freq):
    return 92.5 + 20 * np.log10(distance * freq)

def reuse_app(tail_angles, tail_distances, rain_fade, target_C2I, antenna1, chBW=2000, NF=5, Pmax=15, Pmin=-2):
    Gain = antenna1.gain()
    thermalNoise = -174 + 10 * np.log10(chBW * 1e6) + NF
    N = tail_angles.size

    mat1 = np.ones((N, 1)) * tail_angles
    mat2 = mat1.transpose()
    teta = mat1 - mat2
    attenMatrixAntenna = antenna1.antennaAngularAttenuation(teta)

    # --- Tail to Hub Optimization ---
    tail_tx_power = np.ones(tail_angles.shape) * Pmax
    
    rxMatrix = np.zeros(attenMatrixAntenna.shape)
    for row in range(N):
        for column in range(N):
            fsl = freeSpaceLoss(tail_distances[row], antenna1.frequency)
            rxMatrix[row][column] = tail_tx_power[row] + Gain - rain_fade[row] - fsl + Gain + attenMatrixAntenna[row][column]

    c2i = np.zeros(N)
    for column in range(N):
        noise = sum(10**(rxMatrix[row][column]/10) for row in range(N) if row != column)
        noise += 10**(thermalNoise/10)
        c2i[column] = rxMatrix[column][column] - 10 * np.log10(noise)
    
    tail_c2i_before = c2i.copy()
    tail_power_before = tail_tx_power.copy()

    minMax, counter1 = 1000, 100
    while minMax > 1 or counter1 > 2:
        for row in range(N):
            for column in range(N):
                fsl = freeSpaceLoss(tail_distances[row], antenna1.frequency)
                rxMatrix[row][column] = tail_tx_power[row] + Gain - rain_fade[row] - fsl + Gain + attenMatrixAntenna[row][column]
        
        for column in range(N):
            noise = sum(10**(rxMatrix[row][column]/10) for row in range(N) if row != column)
            noise += 10**(thermalNoise/10)
            c2i[column] = rxMatrix[column][column] - 10 * np.log10(noise)
    
        spare = c2i - target_C2I    
        minMax = np.max(spare) - np.min(spare)
        
        if np.min(spare) > 1.0:
            tail_tx_power -= 0.8 * np.min(spare)
            
        tail_tx_power[np.argmax(spare)] -= 0.5 * minMax
        tail_tx_power[np.argmin(spare)] += 0.5 * minMax
        tail_tx_power[np.argmin(spare)] = min(tail_tx_power[np.argmin(spare)], Pmax)
        
        if minMax < 2: counter1 -= 1

    tail_c2i_after = c2i
    tail_power_after = tail_tx_power

    # --- Hub to Tail Optimization ---
    hub_tx_power = np.ones(tail_angles.shape) * Pmax
    for row in range(N):
        for column in range(N):
            fsl = freeSpaceLoss(tail_distances[column], antenna1.frequency)
            rxMatrix[row][column] = hub_tx_power[row] + Gain - rain_fade[column] - fsl + Gain + attenMatrixAntenna[row][column]

    for column in range(N):
        noise = sum(10**(rxMatrix[row][column]/10) for row in range(N) if row != column)
        noise += 10**(thermalNoise/10)
        c2i[column] = rxMatrix[column][column] - 10 * np.log10(noise)
    
    hub_c2i_before = c2i.copy()
    hub_power_before = hub_tx_power.copy()

    minMax, counter1 = 1000, 100
    while minMax > 1 or counter1 > 2:
        for row in range(N):
            for column in range(N):
                fsl = freeSpaceLoss(tail_distances[column], antenna1.frequency)
                rxMatrix[row][column] = hub_tx_power[row] + Gain - rain_fade[column] - fsl + Gain + attenMatrixAntenna[row][column]
        
        for column in range(N):
            noise = sum(10**(rxMatrix[row][column]/10) for row in range(N) if row != column)
            noise += 10**(thermalNoise/10)
            c2i[column] = rxMatrix[column][column] - 10 * np.log10(noise)
    
        spare = c2i - target_C2I    
        minMax = np.max(spare) - np.min(spare)
        
        if np.min(spare) > 1.0:
            hub_tx_power -= 0.9 * np.min(spare)
        hub_tx_power[np.argmax(spare)] -= 0.05 * minMax
        hub_tx_power[np.argmin(spare)] += 0.05 * minMax
        hub_tx_power[np.argmin(spare)] = min(hub_tx_power[np.argmin(spare)], Pmax)
        
        if minMax < 2: counter1 -= 1
        
    hub_c2i_after = c2i
    hub_power_after = hub_tx_power

    return (tail_c2i_before, tail_power_before, tail_c2i_after, tail_power_after, 
            hub_c2i_before, hub_power_before, hub_c2i_after, hub_power_after)

# ==========================================
# Streamlit App UI
# ==========================================

st.set_page_config(layout="wide", page_title="PtMP Planning Tool")
st.title("PtMP Monte Carlo Planning Tool")

# Initialize Session State for Coordinates
if "tail_angles" not in st.session_state:
    st.session_state.tail_angles = np.array([])
if "tail_distances" not in st.session_state:
    st.session_state.tail_distances = np.array([])

# Sidebar Configuration
st.sidebar.header("System Parameters")
freq = st.sidebar.number_input("Frequency (GHz)", value=18.0, step=1.0)
antenna_size = st.sidebar.number_input("Antenna Size (ft)", value=3.0, step=0.5)
antenna_class = st.sidebar.selectbox("Antenna Class", [3, 4], index=0)

col1, col2 = st.sidebar.columns(2)
Pmax = col1.number_input("Pmax (dBm)", value=20.0)
Pmin = col2.number_input("Pmin (dBm)", value=-5.0)

chBW = st.sidebar.number_input("Channel BW (MHz)", value=56.0)
NF = st.sidebar.number_input("Noise Figure (dB)", value=5.0)
target_C2I_val = st.sidebar.number_input("Target C/I (dB)", value=35.0)

st.sidebar.markdown("---")
st.sidebar.header("Simulation Parameters")
num_iteration = st.sidebar.number_input("Monte Carlo Iterations", value=1000, step=100)
view_angle = st.sidebar.number_input("Hub View Angle (Degrees)", value=120.0)
max_distance = st.sidebar.number_input("Max Distance (km)", value=15.0)

# Main UI Tabs
tab_random, tab_manual = st.tabs(["Random CPE Placement", "Manual CPE Placement"])

with tab_random:
    num_cpe = st.number_input("Number of CPEs", value=6, min_value=1, step=1)
    if st.button("Generate Random Locations"):
        # Generate angles symmetrically around 0 (e.g., -60 to +60 for a 120 deg sector)
        st.session_state.tail_angles = np.random.uniform(-view_angle/2, view_angle/2, num_cpe)
        # Distances from 0.5km to max_distance
        st.session_state.tail_distances = np.random.uniform(0.5, max_distance, num_cpe)

with tab_manual:
    st.markdown("Enter comma-separated values for your CPEs.")
    man_angles_str = st.text_input("Tail Angles (e.g., -117, -68, -25)", "-117, -68, -25")
    man_dist_str = st.text_input("Tail Distances (km) (e.g., 8.2, 5.1, 2.4)", "8.2, 5.1, 2.4")
    
    if st.button("Set Manual Locations"):
        try:
            st.session_state.tail_angles = np.array([float(x.strip()) for x in man_angles_str.split(",")])
            st.session_state.tail_distances = np.array([float(x.strip()) for x in man_dist_str.split(",")])
            
            if len(st.session_state.tail_angles) != len(st.session_state.tail_distances):
                st.error("Error: The number of angles must match the number of distances.")
        except ValueError:
            st.error("Please enter valid numbers.")

# Display Current Setup & Graph
if len(st.session_state.tail_angles) > 0 and len(st.session_state.tail_distances) > 0:
    st.markdown("### Current Network Layout")
    
    # Plotting Sector and CPEs
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    
    # Setup polar properties based on view angle (pointing "Up" at 90 degrees in Matplotlib)
    ax.set_theta_zero_location("N")  
    ax.set_thetamin(-view_angle/2)
    ax.set_thetamax(view_angle/2)
    ax.set_ylim(0, max_distance * 1.1)
    
    angles_rad = np.deg2rad(st.session_state.tail_angles)
    
    # Plot Hub
    ax.scatter(0, 0, color='red', marker='^', s=150, label="Hub")
    # Plot Tails
    scatter = ax.scatter(angles_rad, st.session_state.tail_distances, c=st.session_state.tail_distances, cmap='viridis', label="CPEs")
    plt.colorbar(scatter, ax=ax, pad=0.1, label="Distance (km)")
    
    # Annotate points
    for i in range(len(st.session_state.tail_angles)):
        ax.text(angles_rad[i], st.session_state.tail_distances[i]*1.05, f"CPE {i+1}", fontsize=8)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)

    # --- Monte Carlo Execution ---
    st.markdown("### Run Simulation")
    if st.button("Run Monte Carlo Optimization"):
        
        antenna1 = Antenna(diameter=antenna_size, frequency=freq, antennaClass=antenna_class)
        targetC2I_array = np.ones(len(st.session_state.tail_angles)) * target_C2I_val
        
        tail_c2i_worst = np.zeros(num_iteration)
        hub_c2i_worst = np.zeros(num_iteration)
        
        shape, scale = 1.1, 6.0
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for index1 in range(num_iteration):
            # Rain fade generation
            rain_fade = np.random.gamma(shape, scale, len(st.session_state.tail_angles))
            rain_fade = np.clip(rain_fade, 0, 25)
            
            res = reuse_app(
                st.session_state.tail_angles, 
                st.session_state.tail_distances, 
                rain_fade, 
                targetC2I_array, 
                antenna1, 
                chBW, NF, Pmax, Pmin
            )
            
            tail_c2i_after = res[2]
            hub_c2i_after = res[6]
            
            tail_c2i_worst[index1] = np.min(tail_c2i_after)
            hub_c2i_worst[index1] = np.min(hub_c2i_after)
            
            if index1 % 10 == 0:
                progress_bar.progress(index1 / num_iteration)
                status_text.text(f"Running iteration {index1}/{num_iteration}...")

        progress_bar.progress(1.0)
        status_text.text("Simulation Complete!")

        # Results Plotting
        st.markdown("### Worst Case Minimum C/I Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.hist(tail_c2i_worst, bins=50, density=True, alpha=0.7, color='blue')
            ax1.set_title("Tail to Hub (Uplink) C/I")
            ax1.set_xlabel("C/I [dB]")
            ax1.grid(True)
            st.pyplot(fig1)
            
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.hist(hub_c2i_worst, bins=50, density=True, alpha=0.7, color='green')
            ax2.set_title("Hub to Tail (Downlink) C/I")
            ax2.set_xlabel("C/I [dB]")
            ax2.grid(True)
            st.pyplot(fig2)
else:
    st.info("Please generate random locations or input manual coordinates to view the network layout and run simulations.")
