# -*- coding: utf-8 -*-

import numpy as np
import math
import mpmath
import pandas as pd
import streamlit as st
import plotly.express as px

# =================
# Helper Functions & Classes
# =================

class Antenna:
    def __init__(self, diameter=1, frequency=18, antennaClass=4):
        self.diameter = diameter
        self.frequency = frequency
        self.antennaClass = antennaClass
        
    def __str__(self):
        return f"Antenna: {self.diameter}ft @ {self.frequency}GHz Class {self.antennaClass}"

    def gain(self):
        lambda1 = 300e6 / (self.frequency * 1e9)
        area = np.pi * (self.diameter * 0.305 / 2)**2
        return 10 * np.log10(4 * np.pi * 0.55 * area / (lambda1**2))
    
    def beamWidth(self):
        lambda1 = 300e6 / (self.frequency * 1e9)
        return 70 * lambda1 / (self.diameter * 0.305)
    
    def antennaAngularAttenuation(self, teta):
        # teta is a matrix in degrees
        Lambda1 = 300E6 / (self.frequency * 1E9)
        k = 2 * np.pi / Lambda1
        radius = self.diameter / 2
    
        if type(teta) == float:
            teta = np.array([[teta]])
        
        rpe1 = np.zeros(teta.shape)
    
        # ETSI RPE masks approximations based on frequency and class
        if self.frequency >= 3 and self.frequency <= 14:
            if self.antennaClass == 4:
                angleMask = np.array([5, 10, 20, 50, 70, 85, 105, 180]) 
                rpeMask = np.array([16, 5, -7,-18, -20, -24, -30, -30])
            else: # Class 3
                angleMask = np.array([5, 20, 70, 100, 180]) 
                rpeMask = np.array([20, 8, -5,-25, -25])
        elif self.frequency > 14 and self.frequency <= 30:
             if self.antennaClass == 4:
                angleMask = np.array([5, 10, 20, 40, 80, 100, 180]) 
                rpeMask = np.array([18, 9, -4,-13, -25, -30, -30])
             else: # Class 3
                angleMask = np.array([5, 10, 25, 60, 95, 180]) 
                rpeMask = np.array([18, 9, 2,-4, -27, -27])
        else: # High frequency fallback (e.g., 70-80 GHz)
            angleMask = np.array([5, 10, 20, 50, 70, 90, 180]) 
            rpeMask = np.array([16, 9, 1, -1, -4, -17, -17])

        for row in range(teta.shape[0]):
            for column in range(teta.shape[1]):
                tetaTmp = np.abs(teta[row][column])
                if tetaTmp == 0: tetaTmp = 0.0001
                if tetaTmp > 180: tetaTmp = 360 - tetaTmp 
                        
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
    return 92.5 + 20 * np.log10(distance * freq)
    
# =================
# Optimization Logic
# =================

def reuse_app(tail_angles, tail_distances, rain_fade, target_C2I, antenna1, chBW, NF, Pmax, Pmin):
    freq = antenna1.frequency
    Gain = antenna1.gain() 
    BW = antenna1.beamWidth() 
    thermalNoise = -174 + 10 * np.log10(chBW * 1e6) + NF

    N = tail_angles.size
    mat1 = np.ones((N, 1)) * tail_angles
    mat2 = mat1.transpose()
    teta = mat1 - mat2 
    attenMatrixAntenna = antenna1.antennaAngularAttenuation(teta)

    # Helper to calculate C/(I+N) for a given set of powers and direction
    def calculate_c2i(current_powers, is_uplink):
        rxMatrix = np.zeros(attenMatrixAntenna.shape)
        for row in range(num_cpe):
            for column in range(num_cpe):
                # Calculate Path Loss (Free Space Loss)
                # Uplink (Tail->Hub): dist depends on tx node (row)
                # Downlink (Hub->Tail): dist depends on rx node (column)
                fsl_dist = tail_distances[row] if is_uplink else tail_distances[column]
                fsl = freeSpaceLoss(fsl_dist, freq)
                
                # Apply Fade: depends on node position
                fade_val = rain_fade[row] if is_uplink else rain_fade[column]
                
                rxMatrix[row][column] = current_powers[row] + Gain - fade_val - fsl + Gain + attenMatrixAntenna[row][column]
        
        c2i_result = np.zeros(num_cpe)
        for column in range(num_cpe):
            interference_linear = 0
            for row in range(num_cpe): 
                if column == row: s = rxMatrix[row][column]
                else: interference_linear += 10 ** (rxMatrix[row][column] / 10)
            
            total_noise_linear = interference_linear + 10**(thermalNoise / 10)
            c2i_result[column] = s - 10 * np.log10(total_noise_linear)
        return c2i_result

    # ------------------
    # Uplink: Tail to Hub
    # ------------------
    tail_powers = np.ones(num_cpe) * Pmax # Start at max power
    c2i_tail_bef = calculate_c2i(tail_powers, is_uplink=True)
    powers_tail_bef = tail_powers.copy()

    # Optimization loop
    minMax, counter1 = 1000, 100
    while minMax > 1 or counter1 > 2: 
        c2i = calculate_c2i(tail_powers, is_uplink=True)
        spare = c2i - target_C2I    
        minMax = np.max(spare) - np.min(spare)
        
        if np.min(spare) > 1.0: tail_powers -= 0.8 * np.min(spare)
        
        # Min-Max balancing
        tail_powers[np.argmax(spare)] -= 0.5 * minMax
        tail_powers[np.argmin(spare)] += 0.5 * minMax
        tail_powers = np.clip(tail_powers, Pmin, Pmax) # Constrain between limits
        
        if minMax < 2: counter1 -= 1

    c2i_tail_aft = calculate_c2i(tail_powers, is_uplink=True)
    powers_tail_aft = tail_powers


    # ------------------
    # Downlink: Hub to Tail
    # ------------------
    hub_powers = np.ones(num_cpe) * Pmax # Start at max power
    c2i_hub_bef = calculate_c2i(hub_powers, is_uplink=False)
    powers_hub_bef = hub_powers.copy()

    # Optimization loop
    minMax, counter1 = 1000, 100
    while minMax > 1 or counter1 > 2: 
        c2i = calculate_c2i(hub_powers, is_uplink=False)
        spare = c2i - target_C2I    
        minMax = np.max(spare) - np.min(spare)
        
        if np.min(spare) > 1.0: hub_powers -= 0.9 * np.min(spare)
            
        hub_powers[np.argmax(spare)] -= 0.05 * minMax
        hub_powers[np.argmin(spare)] += 0.05 * minMax
        hub_powers = np.clip(hub_powers, Pmin, Pmax) # Constrain between limits
            
        if minMax < 2: counter1 -= 1

    c2i_hub_aft = calculate_c2i(hub_powers, is_uplink=False)
    powers_hub_aft = hub_powers
    
    return (c2i_tail_bef, powers_tail_bef, c2i_tail_aft, powers_tail_aft, 
            c2i_hub_bef, powers_hub_bef, c2i_hub_aft, powers_hub_aft)

# =================
# Streamlit GUI App
# =================

st.set_page_config(page_title="PtMP Convergence Vis", layout="wide")
st.title("PtMP Wireless Sector C/I Convergence Tool")
st.markdown("Set network parameters and dynamically allocate randomly placed CPEs to visualize how power control loop manages interference to ensure link-by-link C/I convergence across the sector.")

# Sidebar parameters
st.sidebar.header("System Parameters")
freq = st.sidebar.number_input("Frequency (GHz)", value=18.0, step=0.5)
antenna_size = st.sidebar.number_input("Antenna Size (ft)", value=3.0, step=0.5)
chBW = st.sidebar.number_input("Channel BW (MHz)", value=56.0, step=1.0)
NF = st.sidebar.number_input("Noise Figure (dB)", value=5.0, step=0.5)

st.sidebar.header("Power Limits & Targets")
Pmax = st.sidebar.number_input("Max Tx Power (dBm)", value=20.0, step=1.0)
Pmin = st.sidebar.number_input("Min Tx Power (dBm)", value=-5.0, step=1.0)
target_c2i_val = st.sidebar.number_input("Target C/I (dB)", value=35.0, step=1.0)

st.sidebar.header("Run Config")
num_cpe = st.sidebar.number_input("Number of CPEs", min_value=1, max_value=30, value=6, step=1)
apply_fade = st.sidebar.checkbox("Apply Gamma-distributed Rain Fade", value=False)

if st.sidebar.button("Run Single Convergence Simulation"):
    
    # Generate random deployment scenario
    # Angles uniformly distributed in a -180 to 180 sector
    tail_angles = np.round(np.random.uniform(-180, 180, num_cpe), 1)
    # Distances between 1km and 15km
    tail_distances = np.round(np.random.uniform(1.0, 15.0, num_cpe), 2)
    
    # Generate Fade Events if requested (Gamma distribution)
    if apply_fade:
        shape, scale = 1.1, 6.  # mean=1.44, std=2*sqrt(6)
        rain_fade = np.random.gamma(shape, scale, num_cpe)
        rain_fade = np.clip(rain_fade, 0, 25)
        rain_fade = np.round(rain_fade, 1)
    else:
        rain_fade = np.zeros(num_cpe)
    
    targetC2I = np.ones(num_cpe) * target_c2i_val

    # Setup Antenna
    antenna1 = Antenna(diameter=antenna_size, frequency=freq, antennaClass=3)
    
    st.write(f"### Simulation Scenario: {num_cpe} dynamic CPE nodes deploying")
    st.write(f"**{str(antenna1)}** | Target C/I: {target_c2i_val} dB")

    # Run algorithm
    results = reuse_app(tail_angles, tail_distances, rain_fade, targetC2I, antenna1, chBW, NF, Pmax, Pmin)
    (u_bef, _, u_aft, _, d_bef, _, d_aft, _) = results

    # =================
    # Visualization & Results Section
    # =================

    # 1. Metric display for overall Min C/I before/after
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Uplink Min C/I (Before)", f"{u_bef.min():.1f} dB")
    col2.metric("Uplink Min C/I (Converged)", f"{u_aft.min():.1f} dB", f"{u_aft.min() - u_bef.min():.1f} dB")
    col3.metric("Downlink Min C/I (Before)", f"{d_bef.min():.1f} dB")
    col4.metric("Downlink Min C/I (Converged)", f"{d_aft.min():.1f} dB", f"{d_aft.min() - d_bef.min():.1f} dB")

    # Layout for charts
    vis_col1, vis_col2 = st.columns([1, 2])

    # 2. Geometry Plot: Sector map (Angle and Distance)
    with vis_col1:
        st.write("### Sector Geometry")
        # AP (Hub) is at origin (0,0)
        hub_x, hub_y = 0, 0
        # Convert polar (deg, km) to Cartesian (km, km) for visualization
        tail_x = tail_distances * np.cos(np.radians(tail_angles))
        tail_y = tail_distances * np.sin(np.radians(tail_angles))
        
        # Prepare spatial data for plot
        map_df = pd.DataFrame({
            'NodeID': [f'CPE {i+1}' for i in range(num_cpe)],
            'Distance (km)': tail_distances,
            'Angle (deg)': tail_angles,
            'Fade (dB)': rain_fade,
            'x': tail_x,
            'y': tail_y,
            'Type': 'CPE node'
        })
        
        combined_df = pd.concat([
            map_df,
            pd.DataFrame({'NodeID': ['AP Hub'], 'x': [0], 'y': [0], 'Type': ['Access Point'], 'Fade (dB)': [0]})
        ])

        fig_map = px.scatter(combined_df, x='x', y='y', color='Type', text='NodeID', 
                             labels={'x': 'X Distance (km)', 'y': 'Y Distance (km)'},
                             color_discrete_map={'Access Point': '#FF4B4B', 'CPE node': '#0068C9'},
                             title=f"Geometric Layout of sector (AP at center)")
        fig_map.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers+text'))
        fig_map.update_layout(showlegend=False, xaxis=dict(range=[-16, 16]), yaxis=dict(range=[-16, 16]))
        st.plotly_chart(fig_map, use_container_width=True)


    # Helper function to prep long-form data for grouped bars
    def prep_grouped_bar_data(before_ci, after_ci, num_cpe, target_val):
        nodes = [f"Link {i+1}" for i in range(num_cpe)]
        
        df_bef = pd.DataFrame({'Link': nodes, 'C/I (dB)': before_ci, 'State': 'Before (Initial)'})
        df_aft = pd.DataFrame({'Link': nodes, 'C/I (dB)': after_ci, 'State': 'After Convergence'})
        
        df_long = pd.concat([df_bef, df_aft])
        return df_long

    # 3. Grouped Bar charts for Before/After C/I per Link
    with vis_col2:
        st.write("### C/I Convergence Link-by-Link")
        
        # Define shared target line definition for Plotly charts
        target_line_def = [dict(type='line', yref='y', y0=target_c2i_val, y1=target_c2i_val, xref='paper', x0=0, x1=1, line=dict(color="Red", width=2, dash="dash"))]
        
        # Uplink Chart
        df_u = prep_grouped_bar_data(u_bef, u_aft, num_cpe, target_c2i_val)
        fig_u = px.bar(df_u, x='Link', y='C/I (dB)', color='State', barmode='group', 
                       title="Uplink Link-by-Link: Tail to Hub C/I",
                       color_discrete_map={'Before (Initial)': '#C0C0C0', 'After Convergence': '#0068C9'}) # Gray and Blue
        fig_u.update_layout(shapes=target_line_def, yaxis_range=[0, max(u_bef.max()+2, target_c2i_val+5)])
        st.plotly_chart(fig_u, use_container_width=True)

        # Downlink Chart
        df_d = prep_grouped_bar_data(d_bef, d_aft, num_cpe, target_c2i_val)
        fig_d = px.bar(df_d, x='Link', y='C/I (dB)', color='State', barmode='group', 
                       title="Downlink Link-by-Link: Hub to Tail C/I",
                       color_discrete_map={'Before (Initial)': '#C0C0C0', 'After Convergence': '#FF4B4B'}) # Gray and Red
        fig_d.update_layout(shapes=target_line_def, yaxis_range=[0, max(d_bef.max()+2, target_c2i_val+5)])
        st.plotly_chart(fig_d, use_container_width=True)

else:
    st.info("Adjust network parameters in the sidebar and click **Run Single Convergence Simulation** to visualize how the sector converges under Gamma fade stress.")
