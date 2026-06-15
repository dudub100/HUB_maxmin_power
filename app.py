import streamlit as st
import numpy as np
import math
import matplotlib
matplotlib.use('Agg') # Crucial: Prevents headless server memory hangs on Streamlit Cloud
import matplotlib.pyplot as plt
from scipy.special import j1

# ==========================================
# RF & Antenna Class Definition
# ==========================================
class Antenna:
    def __init__(self, diameter=1.0, frequency=18.0, antennaClass=3):
        self.diameter = diameter
        self.frequency = frequency
        self.antennaClass = antennaClass
        
    def __str__(self):
        return f"Antenna {self.diameter}ft | {self.frequency}GHz | ETSI Class {self.antennaClass}"

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
        
        # Reference ETSI standard mask thresholds for modern point-to-multipoint nodes
        angleMask = np.array([5, 10, 20, 40, 80, 100, 180]) 
        rpeMask = np.array([18, 9, -4, -13, -25, -30, -30])

        for row in range(teta.shape[0]):
            for column in range(teta.shape[1]):
                tetaTmp = np.abs(teta[row][column])
                if tetaTmp == 0: tetaTmp = 0.0001
                if tetaTmp > 180: tetaTmp = 360 - tetaTmp 
                        
                if tetaTmp < angleMask[0]:
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
# Max-Min Power Optimization Engine
# ==========================================
def optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, antenna1, chBW, NF, Pmax, Pmin, is_uplink=True):
    N = tail_angles.size
    Gain = antenna1.gain()
    thermalNoise = -174 + 10 * np.log10(chBW * 1e6) + NF
    thermalNoise_linear = 10**(thermalNoise / 10)

    # Angular Inter-link Isolation Matrix
    mat1 = np.ones((N, 1)) * tail_angles
    teta = mat1 - mat1.T
    attenMatrixAntenna = antenna1.antennaAngularAttenuation(teta)

    # Link Loss Formulation
    tx_power = np.ones(N) * Pmax
    fsl_matrix = np.zeros((N, N))
    
    for row in range(N):
        for col in range(N):
            dist = tail_distances[row] if is_uplink else tail_distances[col]
            fade = rain_fade[row] if is_uplink else rain_fade[col]
            fsl_matrix[row][col] = Gain - fade - freeSpaceLoss(dist, antenna1.frequency) + Gain + attenMatrixAntenna[row][col]

    # --- 1. CALCULATE BASELINE SNAPSHOT (All nodes at Pmax) ---
    tx_matrix_base = np.tile(tx_power, (N, 1)).T
    rxMatrix_base = tx_matrix_base + fsl_matrix
    rxMatrix_linear_base = 10**(rxMatrix_base / 10)

    c2i_base = np.zeros(N)
    for i in range(N):
        signal = rxMatrix_linear_base[i, i]
        interference_plus_noise = np.sum(rxMatrix_linear_base[:, i]) - signal + thermalNoise_linear
        c2i_base[i] = 10 * np.log10(signal / interference_plus_noise)

    # --- 2. CONVERGENCE LOOP ---
    minMax = 1000
    max_iterations = 500
    iteration = 0

    while minMax > 1.0 and iteration < max_iterations:
        iteration += 1
        
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
        
        # Attentuate collective network footprint if target criteria are safely exceeded
        if np.min(spare) > 1.0:
            tx_power = tx_power - 0.8 * np.min(spare)
            
        best_idx = np.argmax(spare)
        worst_idx = np.argmin(spare)
        
        # DAMPENED LEARNING RATE: Fixed step multiplier of 0.05 overrides 
        # the traditional 0.5 jump, eliminating spatial oscillation tracking loops.
        step = 0.05 
        tx_power[best_idx] -= step * minMax
        tx_power[worst_idx] += step * minMax
        
        # Enforce physical hardware bounds immediately
        tx_power = np.clip(tx_power, Pmin, Pmax)

    return c2i_base, c2i, tx_power, iteration

# ==========================================
# Streamlit Frontend UI
# ==========================================
def main():
    st.set_page_config(page_title="PtMP Max-Min Planner", layout="wide")
    st.title("📡 Point-to-Multipoint Link Interference Planner")
    
    # Global Parameters Configured via Sidebar
    st.sidebar.header("Global RF Engine Configuration")
    freq = st.sidebar.number_input("Frequency (GHz)", value=18.0, step=1.0)
    Pmax = st.sidebar.number_input("Max Tx Power (dBm)", value=20.0, step=1.0)
    Pmin = st.sidebar.number_input("Min Tx Power (dBm)", value=-5.0, step=1.0)
    chBW = st.sidebar.number_input("Channel BW (MHz)", value=56.0, step=10.0)
    NF = st.sidebar.number_input("Noise Figure (dB)", value=5.0, step=0.5)
    antenna_size = st.sidebar.selectbox("Antenna Size (ft)", [1, 2, 3, 4], index=2)
    
    ant = Antenna(diameter=antenna_size, frequency=freq, antennaClass=3)
    st.markdown(f"**Hardware Context:** {ant} | **Channel:** {chBW} MHz | **Limits:** {Pmin} to {Pmax} dBm")

    # Navigation Layout Structure Tabs
    tab1, tab2 = st.tabs(["🎯 Single Topology Profiler", "🎲 Monte Carlo Statistical Simulator"])
    
    with tab1:
        st.header("Single Link Layout Profiler")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            num_links = st.slider("Number of Sectored CPE Links", min_value=2, max_value=12, value=4, key="single_n")
        with col_c2:
            target_C2I = st.slider("Target C/I Metric (dB)", min_value=10.0, max_value=60.0, value=34.0, step=1.0, key="single_t")
            
        if st.button("Execute Profile Model", type="primary"):
            # Enforce clean memory space allocation before populating calculations
            plt.close('all')
            
            # Static geometric seeds for the individual profiling tab
            np.random.seed(42) 
            tail_angles = np.sort(np.random.uniform(-45, 45, num_links))
            tail_distances = np.random.uniform(1.0, 12.0, num_links)
            rain_fade = np.zeros(num_links)
            
            # Process calculation configurations
            c2i_base_up, c2i_opt_up, tx_opt_up, iters_up = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=True)
            c2i_base_down, c2i_opt_down, tx_opt_down, iters_down = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=False)
            
            # --- 1. GEOGRAPHIC POLAR MAP RENDERING ---
            st.subheader("Spatial Site Orientation Layout")
            fig_map, ax_map = plt.subplots(figsize=(7, 4), subplot_kw={'projection': 'polar'})
            
            angles_rad = np.radians(tail_angles)
            ax_map.plot(0, 0, '^', color='red', markersize=14, label='AP Base Station (Hub)')
            
            for i in range(num_links):
                ax_map.plot(angles_rad[i], tail_distances[i], 'o', markersize=9, label=f'CPE {i} ({tail_distances[i]:.1f}km, {tail_angles[i]:.1f}°)')
            
            # Restructuring polar window to emphasize sector directionality
            ax_map.set_theta_zero_location("N")
            ax_map.set_theta_direction(-1)
            ax_map.set_thetamin(-60)
            ax_map.set_thetamax(60)
            ax_map.set_rmax(14)
            ax_map.grid(True, linestyle=':', alpha=0.6)
            ax_map.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
            
            st.pyplot(fig_map)
            plt.close(fig_map) # Flush canvas resources from RAM memory allocation
            
            # --- 2. PERFORMANCE INSIGHT PLOTS ---
            res_col1, res_col2 = st.columns(2)
            width = 0.35
            indices = np.arange(num_links)
            
            with res_col1:
                st.subheader("Uplink Channel Performance (CPE ➔ AP)")
                if iters_up >= 500:
                    st.error(f"⚠️ Network Geometry Bottleneck: Loop hit limit ({iters_up} runs). Target unachievable.")
                else:
                    st.success(f"Uplink balanced in {iters_up} sequential steps.")
                    
                fig_up, ax_up = plt.subplots(figsize=(6, 3.5))
                ax_up.bar(indices - width/2, c2i_base_up, width, label='Unmanaged Link State', color='lightgray')
                ax_up.bar(indices + width/2, c2i_opt_up, width, label='Max-Min Balanced State', color='tab:blue')
                ax_up.axhline(y=target_C2I, color='r', linestyle='--', label='Target Boundary')
                ax_up.set_ylabel("Carrier-to-Interference (C/I) [dB]")
                ax_up.set_xticks(indices)
                ax_up.set_xticklabels([f"CPE {i}" for i in indices])
                ax_up.grid(axis='y', linestyle=':', alpha=0.5)
                ax_up.legend()
                
                st.pyplot(fig_up)
                plt.close(fig_up) # Flush configuration layer
                
                st.metric("Worst Sector Link C/I Bound", f"{np.min(c2i_opt_up):.2f} dB")
                st.text(f"Optimized Transmit Matrices (dBm):\n{np.round(tx_opt_up, 1)}")
                
            with res_col2:
                st.subheader("Downlink Channel Performance (AP ➔ CPE)")
                if iters_down >= 500:
                    st.error(f"⚠️ Network Geometry Bottleneck: Loop hit limit ({iters_down} runs). Target unachievable.")
                else:
                    st.success(f"Downlink balanced in {iters_down} sequential steps.")
                    
                fig_down, ax_down = plt.subplots(figsize=(6, 3.5))
                ax_down.bar(indices - width/2, c2i_base_down, width, label='Unmanaged Link State', color='lightgray')
                ax_down.bar(indices + width/2, c2i_opt_down, width, label='Max-Min Balanced State', color='tab:green')
                ax_down.axhline(y=target_C2I, color='r', linestyle='--', label='Target Boundary')
                ax_down.set_ylabel("Carrier-to-Interference (C/I) [dB]")
                ax_down.set_xticks(indices)
                ax_down.set_xticklabels([f"CPE {i}" for i in indices])
                ax_down.grid(axis='y', linestyle=':', alpha=0.5)
                ax_down.legend()
                
                st.pyplot(fig_down)
                plt.close(fig_down) # Flush configuration layer
                
                st.metric("Worst Sector Link C/I Bound", f"{np.min(c2i_opt_down):.2f} dB")
                st.text(f"Optimized Transmit Matrices (dBm):\n{np.round(tx_opt_down, 1)}")

    with tab2:
        st.header("Statistical Environmental Simulator")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            mc_runs = st.number_input("Total Simulation Maps (Iterations)", min_value=10, max_value=2000, value=200, step=50)
        with col_m2:
            mc_links = st.slider("CPE Deployment Density Per Sector", min_value=2, max_value=15, value=5)
        with col_m3:
            mc_target = st.slider("Required Outage Threshold Target (dB)", min_value=15.0, max_value=50.0, value=35.0, step=1.0)

        if st.button("Execute Monte Carlo Engine", type="primary"):
            plt.close('all')
            
            worst_uplink_vector = []
            worst_downlink_vector = []
            saturated_tx_count = 0
            total_tx_evaluated = 0
            
            progress_bar = st.progress(0)
            
            for run in range(mc_runs):
                # Map coordinates matching a standardized 120° sector spread
                mc_angles = np.sort(np.random.uniform(-60, 60, mc_links))
                # Spatial layout mapping to clear radius bias near center points
                mc_distances = 12.0 * np.sqrt(np.random.uniform(0.1, 1.0, mc_links))
                
                # Dynamic Gamma Rain Fade Attenuation Configuration
                # Localized rainfall scaling parameterized by linear link range constraints
                shape, scale = 1.1, 3.0
                rain_intensity = np.random.gamma(shape, scale, mc_links)
                mc_fade = np.clip(rain_intensity * (mc_distances / 5.0), 0, 25)
                
                # Extract array solutions via optimization engine profiles
                c2i_up, tx_up = optimize_network(mc_angles, mc_distances, mc_fade, mc_target, ant, chBW, NF, Pmax, Pmin, is_uplink=True)[1:3]
                c2i_down, tx_down = optimize_network(mc_angles, mc_distances, mc_fade, mc_target, ant, chBW, NF, Pmax, Pmin, is_uplink=False)[1:3]
                
                worst_uplink_vector.append(np.min(c2i_up))
                worst_downlink_vector.append(np.min(c2i_down))
                
                # Check for instances where hardware amplifier bounds saturate completely
                saturated_tx_count += np.sum(tx_up >= (Pmax - 0.2))
                total_tx_evaluated += mc_links
                
                if run % max(1, mc_runs // 10) == 0:
                    progress_bar.progress(run / mc_runs)
            
            progress_bar.progress(1.0)
            
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                st.subheader("Outage Performance Probability (CDF)")
                fig_cdf, ax_cdf = plt.subplots(figsize=(6, 4))
                
                # Compute empirical Cumulative Distribution Function tracks
                ax_cdf.plot(np.sort(worst_uplink_vector), np.linspace(0, 1, mc_runs), label='Uplink Sector Bottleneck', linewidth=2.5)
                ax_cdf.plot(np.sort(worst_downlink_vector), np.linspace(0, 1, mc_runs), label='Downlink Sector Bottleneck', linewidth=2.5, linestyle='--')
                ax_cdf.axvline(x=mc_target, color='r', linestyle=':', label='Simulation Target Requirement')
                
                ax_cdf.set_xlabel("Worst Settled Link C/I in Sector [dB]")
                ax_cdf.set_ylabel("Probability Floor (CDF)")
                ax_cdf.grid(True, alpha=0.35, linestyle=':')
                ax_cdf.legend()
                
                st.pyplot(fig_cdf)
                plt.close(fig_cdf) # Flush canvas layout data layers from memory
                
            with m_col2:
                st.subheader("Statistical Sector Metrics Dashboard")
                up_array = np.array(worst_uplink_vector)
                down_array = np.array(worst_downlink_vector)
                
                outage_rate_up = np.mean(up_array < mc_target) * 100
                outage_rate_down = np.mean(down_array < mc_target) * 100
                saturation_rate = (saturated_tx_count / total_tx_evaluated) * 100
                
                st.metric("Uplink Outage Probability (Below Target)", f"{outage_rate_up:.1f} %")
                st.metric("Downlink Outage Probability (Below Target)", f"{outage_rate_down:.1f} %")
                st.metric("Amplifier Power Saturation Factor (At Pmax)", f"{saturation_rate:.1f} %")
                
                st.info(f"Analysis logged over {mc_runs} random layout variants under active Gamma rain attenuation fades. "
                        f"Median sector bottleneck performance: Uplink={np.median(up_array):.1f} dB, Downlink={np.median(down_array):.1f} dB.")

if __name__ == "__main__":
    main()
