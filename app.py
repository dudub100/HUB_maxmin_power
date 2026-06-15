# -*- coding: utf-8 -*-

import numpy as np
import math
import mpmath
import pandas as pd
import altair as alt
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
    
        # Example 14-20GHz Class 3 definition (update for your specific class if needed)
        if self.antennaClass == 3:
            angleMask = np.array([5, 10, 25, 60, 95, 180]) 
            rpeMask = np.array([18, 9, 2,-4, -27, -27])
        else: # Default or example class 4
            angleMask = np.array([5, 10, 20, 40, 80, 100, 180]) 
            rpeMask = np.array([18, 9, -4,-13, -25, -30, -30])

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
    
def generate_random_cpes(n, apply_fade=False):
    # Angles uniformly distributed -180 to 180, distances 1-15 km
    angles = np.round(np.random.uniform(-180, 180, n), 1)
    distances = np.round(np.random.uniform(1.0, 15.0, n), 2)
    
    # Optional Gamma distributed fade (using provided parameters, clipped 0-25dB)
    fade = np.zeros(n)
    if apply_fade:
        shape, scale = 1.1, 6.0  
        fade = np.random.gamma(shape, scale, n)
        fade = np.clip(fade, 0, 25)
        fade = np.round(fade, 1)

    return angles, distances, fade

# =================
# Optimization Algorithm (Original reuse_app)
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

    def optimize_direction(tx_powers, direction_type):
        """tx_powers initialized to Pmax for single run"""
        
        # Helper to calculate C/I (Uplink/Downlink specific freeSpaceLoss and RX power calcs omitted for brevity)
        def get_ci(current_powers, is_uplink):
            rxMatrix = np.zeros(attenMatrixAntenna.shape)
            for row in range(current_powers.shape[0]):
                for column in range(current_powers.shape[0]):
                    if is_uplink: # Tail to Hub: FSL based on each CPE distance
                        fsl = freeSpaceLoss(tail_distances[row], freq)
                        rxMatrix[row][column] = current_powers[row] + Gain - rain_fade[row] - fsl + Gain + attenMatrixAntenna[row][column]
                    else: # Hub to Tail: FSL of interfering CPE distance
                        fsl = freeSpaceLoss(tail_distances[column], freq)
                        rxMatrix[row][column] = current_powers[row] + Gain - rain_fade[column] - fsl + Gain + attenMatrixAntenna[row][column]
            
            c2i = np.zeros(rxMatrix.shape[1])
            for column in range(rxMatrix.shape[1]):
                noise_linear = 0
                for row in range(rxMatrix.shape[0]): 
                    if column == row: s = rxMatrix[row][column]
                    else: noise_linear += 10 ** (rxMatrix[row][column] / 10)
                noise_linear += 10**(thermalNoise / 10)
                noiseDbm = 10 * np.log10(noise_linear)
                c2i[column] = s - noiseDbm
            return c2i

        is_uplink = (direction_type == 'uplink')
        c2i_before = get_ci(tx_powers, is_uplink)
        powers_before = tx_powers.copy()

        # Iterate and adjust (optimized version for single run)
        minMax, counter1 = 1000, 100
        while minMax > 1 or counter1 > 2: 
            current_c2i = get_ci(tx_powers, is_uplink)
            spare = current_c2i - target_C2I    
            minMax = np.max(spare) - np.min(spare)
            
            # General power decrease if all above target
            if np.min(spare) > 1.0: tx_powers = tx_powers - 0.8 * np.min(spare)
            
            # Min-Max balancing
            # Hub-to-Tail needs gentler adjustments (smaller factor)
            factor = 0.5 if is_uplink else 0.05
            tx_powers[np.argmax(spare)] -= factor * minMax
            tx_powers[np.argmin(spare)] += factor * minMax
            
            tx_powers = np.clip(tx_powers, Pmin, Pmax) # Clip between limits
            
            if minMax < 2: counter1 -= 1
        
        return c2i_before, powers_before, current_c2i, tx_powers

    # Run optimization for both directions
    u_ci_bef, u_pow_bef, u_ci_aft, u_pow_aft = optimize_direction(np.ones(num_cpe) * Pmax, 'uplink')
    d_ci_bef, d_pow_bef, d_ci_aft, d_pow_aft = optimize_direction(np.ones(num_cpe) * Pmax, 'downlink')
    
    return (u_ci_bef, u_pow_bef, u_ci_aft, u_pow_aft, d_ci_bef, d_pow_bef, d_ci_aft, d_pow_aft)

# =================
# Streamlit Layout
# =================
st.set_page_config(page_title="PtMP C/I Convergence Tool", layout="wide")
st.title("PtMP Wireless Sector C/I Convergence Simulation")

# Inputs in sidebar
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
apply_fade = st.sidebar.checkbox("Apply Random Rain Fade", value=False)

if st.sidebar.button("Run Single Convergence Simulation"):
    # Generate and optimize
    tail_angles, tail_distances, rain_fade = generate_random_cpes(num_cpe, apply_fade)
    targetC2I = np.ones(num_cpe) * target_c2i_val
    antenna1 = Antenna(diameter=antenna_size, frequency=freq, antennaClass=3)

    results = reuse_app(tail_angles, tail_distances, rain_fade, targetC2I, antenna1, chBW, NF, Pmax, Pmin)
    (u_ci_bef, u_pow_bef, u_ci_aft, u_pow_aft, d_ci_bef, d_pow_bef, d_ci_aft, d_pow_aft) = results

    # =================
    # Visualization
    # =================
    
    # 1. Metrics for Convergence (Min C/I before/after)
    st.subheader(f"Convergence Summary for {num_cpe} Dynamic CPEs")
    st.write(str(antenna1))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Uplink: Min C/I (Before)", value=f"{u_ci_bef.min():.1f} dB", delta=None)
    col2.metric(label="Uplink: Min C/I (Converged)", value=f"{u_ci_aft.min():.1f} dB", delta=f"{u_ci_aft.min() - u_ci_bef.min():.1f} dB")
    col3.metric(label="Downlink: Min C/I (Before)", value=f"{d_ci_bef.min():.1f} dB", delta=None)
    col4.metric(label="Downlink: Min C/I (Converged)", value=f"{d_ci_aft.min():.1f} dB", delta=f"{d_ci_aft.min() - d_ci_bef.min():.1f} dB")

    # Layout: Spatial Plot, then Bar Charts
    v_col1, v_col2 = st.columns([2, 3]) # Give space to charts

    # 2. Spatial Visualization (Altair Scatter Polar with converted coords)
    with v_col1:
        st.subheader("Sector Layout (Angle & Distance)")
        
        # Convert polar to Cartesian for easy plotting (AP at 0,0)
        # Note: Degrees must be converted to radians for cos/sin
        x_coords = tail_distances * np.cos(np.radians(tail_angles))
        y_coords = tail_distances * np.sin(np.radians(tail_angles))
        
        spatial_df = pd.DataFrame({
            'NodeID': [f"CPE {i+1}" for i in range(num_cpe)],
            'Distance (km)': tail_distances,
            'Angle (deg)': tail_angles,
            'X': x_coords,
            'Y': y_coords,
            'Type': 'CPE',
            'Size': 100
        })
        
        # Add central AP (0,0)
        ap_df = pd.DataFrame({'NodeID': ['AP Hub'], 'Distance (km)': [0], 'Angle (deg)': [0], 'X': [0], 'Y': [0], 'Type': ['AP'], 'Size': 300})
        combined_df = pd.concat([spatial_df, ap_df])

        # Base chart for points
        points = alt.Chart(combined_df).mark_point(filled=True, stroke='black').encode(
            x=alt.X('X', title='X (km)', scale=alt.Scale(domain=(-15, 15))),
            y=alt.Y('Y', title='Y (km)', scale=alt.Scale(domain=(-15, 15))),
            size=alt.Size('Size', legend=None),
            color=alt.Color('Type', legend=alt.Legend(orient="bottom-right")),
            tooltip=['NodeID', 'Distance (km)', 'Angle (deg)']
        ).properties(width=400, height=400)
        
        # Combine with text labels
        text = points.mark_text(dy=-15).encode(text='NodeID')
        st.altair_chart(points + text, use_container_width=False)


    # 3. Bar Charts: Before vs After C/I (Uplink and Downlink)
    with v_col2:
        def generate_convergence_chart(ci_bef, ci_aft, direction_name):
            # Format data for grouped bar chart
            data = []
            for i in range(num_cpe):
                data.append({'Node': f"CPE {i+1}", 'C/I (dB)': ci_bef[i], 'Status': 'Before Optimization'})
                data.append({'Node': f"CPE {i+1}", 'C/I (dB)': ci_aft[i], 'Status': 'Converged State'})
            
            chart_df = pd.DataFrame(data)
            
            base = alt.Chart(chart_df).mark_bar().encode(
                # Grouped bar setup
                x=alt.X('Status', axis=None),
                y=alt.Y('C/I (dB)', title='C/I (dB)', scale=alt.Scale(domain=(0, chart_df['C/I (dB)'].max() + 5))),
                color=alt.Color('Status', legend=alt.Legend(title=None)),
                column=alt.Column('Node', title='CPE Node', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
                tooltip=['Node', 'Status', 'C/I (dB)']
            ).properties(width=50, height=300)

            # Red line at Target C/I
            target_line = alt.Chart(pd.DataFrame({'y': [target_c2i_val]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y')

            return base + target_line

        st.subheader(f"Uplink Link-by-Link C/I Convergence")
        st.altair_chart(generate_convergence_chart(u_ci_bef, u_ci_aft, "Uplink"), use_container_width=True)
        
        st.subheader(f"Downlink Link-by-Link C/I Convergence")
        st.altair_chart(generate_convergence_chart(d_ci_bef, d_ci_aft, "Downlink"), use_container_width=True)

    # Optional: View Raw Data Table
    if st.checkbox("View Detailed Data Table"):
        df_table = pd.DataFrame({
            "CPE": spatial_df['NodeID'],
            "Dist(km)": tail_distances,
            "Angle(deg)": tail_angles,
            "Fade(dB)": rain_fade,
            "U_CI_Bef": np.round(u_ci_bef, 1),
            "U_CI_Aft": np.round(u_ci_aft, 1),
            "D_CI_Bef": np.round(d_ci_bef, 1),
            "D_CI_Aft": np.round(d_ci_aft, 1),
        })
        st.dataframe(df_table, use_container_width=True)

else:
    st.info("Adjust parameters in the sidebar and click **Run Single Convergence Simulation** to begin a single-run analysis with visual confirmation of convergence.")
