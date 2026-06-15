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
        
        # Default fallback mask mapping Class 3/4 characteristics securely
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
# Fixed Max-Min Optimization Engine
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

    # Path Loss Matrix Setup
    tx_power = np.ones(N) * Pmax
    fsl_matrix = np.zeros((N, N))
    
    for row in range(N):
        for col in range(N):
            dist = tail_distances[row] if is_uplink else tail_distances[col]
            fade = rain_fade[row] if is_uplink else rain_fade[col]
            fsl_matrix[row][col] = Gain - fade - freeSpaceLoss(dist, antenna1.frequency) + Gain + attenMatrixAntenna[row][col]

    minMax = 1000
    max_iterations = 200
    iteration = 0

    # Strict hard counter condition prevents infinite loop execution
    while minMax > 1.0 and iteration < max_iterations:
        iteration += 1
        
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
        
        if np.min(spare) > 1.0:
            tx_power = tx_power - 0.8 * np.min(spare)
            
        best_idx = np.argmax(spare)
        worst_idx = np.argmin(spare)
        
        step = 0.5 if is_uplink else 0.05
        tx_power[best_idx] -= step * minMax
        tx_power[worst_idx] += step * minMax
        
        # Clip immediately to enforce strict physical ceilings
        tx_power = np.clip(tx_power, Pmin, Pmax)

    return c2i, tx_power

# ==========================================
# Streamlit UI
# ==========================================
def main():
    st.set_page_config(page_title="PtMP Max-Min Simulator", layout="wide")
    st.title("📡 PtMP Max-Min Interferece & Monte Carlo Engine")
    
    # Sidebar Global Controls
    st.sidebar.header("Global RF Parameters")
    freq = st.sidebar.number_input("Frequency (GHz)", value=18.0, step=1.0)
    Pmax = st.sidebar.number_input("Max Tx Power (dBm)", value=20.0, step=1.0)
    Pmin = st.sidebar.number_input("Min Tx Power (dBm)", value=-5.0, step=1.0)
    chBW = st.sidebar.number_input("Channel BW (MHz)", value=56.0, step=10.0)
    NF = st.sidebar.number_input("Noise Figure (dB)", value=5.0, step=0.5)
    antenna_size = st.sidebar.selectbox("Antenna Size (ft)", [1, 2, 3, 4], index=2)
    
    ant = Antenna(diameter=antenna_size, frequency=freq, antennaClass=3)
    st.markdown(f"**Current Hardware Baseline:** {ant} | **Channel:** {chBW} MHz")

    # Creating Tabs for Single Run vs Monte Carlo
    tab1, tab2 = st.tabs(["🎯 Single Run Inspection", "🎲 Monte Carlo Statistical Analysis"])
    
    with tab1:
        st.header("Single Topology Run")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            num_links = st.slider("Number of CPE Links", min_value=2, max_value=20, value=6, key="single_n")
        with col_c2:
            target_C2I = st.slider("Target C/I (dB)", min_value=10.0, max_value=60.0, value=34.0, step=1.0, key="single_t")
            
        if st.button("Execute Single Run", type="primary"):
            # Uniform geographic setup across an angular slice
            np.random.seed(42) 
            tail_angles = np.sort(np.random.uniform(-45, 45, num_links))
            tail_distances = np.random.uniform(1.0, 10.0, num_links)
            rain_fade = np.zeros(num_links)
            
            c2i_base, _ = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=True)
            c2i_opt_up, tx_opt_up = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=True)
            c2i_opt_down, tx_opt_down = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=False)
            
            res_col1, res_col2 = st.columns(2)
            width = 0.35
            indices = np.arange(num_links)
            
            with res_col1:
                st.subheader("Uplink Matrix (CPE to Hub)")
                fig_up, ax_up = plt.subplots(figsize=(6, 3.5))
                ax_up.bar(indices - width/2, c2i_base, width, label='Unmanaged (Pmax)', color='lightgray')
                ax_up.bar(indices + width/2, c2i_opt_up, width, label='Max-Min Balanced', color='tab:blue')
                ax_up.axhline(y=target_C2I, color='r', linestyle='--', label='Target')
                ax_up.set_ylabel("C/I (dB)")
                ax_up.legend()
                st.pyplot(fig_up)
                st.metric("Worst Settled Uplink C/I", f"{np.min(c2i_opt_up):.2f} dB")
                st.text(f"Optimized Tx Powers (dBm): {np.round(tx_opt_up, 1)}")
                
            with res_col2:
                st.subheader("Downlink Matrix (Hub to CPE)")
                fig_down, ax_down = plt.subplots(figsize=(6, 3.5))
                c2i_base_down, _ = optimize_network(tail_angles, tail_distances, rain_fade, target_C2I, ant, chBW, NF, Pmax, Pmin, is_uplink=False)
                ax_down.bar(indices - width/2, c2i_base_down, width, label='Unmanaged (Pmax)', color='lightgray')
                ax_down.bar(indices + width/2, c2i_opt_down, width, label='Max-Min Balanced', color='tab:green')
                ax_down.axhline(y=target_C2I, color='r', linestyle='--', label='Target')
                ax_down.set_ylabel("C/I (dB)")
                ax_down.legend()
                st.pyplot(fig_down)
                st.metric("Worst Settled Downlink C/I", f"{np.min(c2i_opt_down):.2f} dB")
                st.text(f"Optimized Tx Powers (dBm): {np.round(tx_opt_down, 1)}")

    with tab2:
        st.header("Monte Carlo Simulation Runs")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            mc_runs = st.number_input("Number of Monte Carlo Iterations", min_value=10, max_value=2000, value=200, step=50)
        with col_m2:
            mc_links = st.slider("Links per Sector", min_value=2, max_value=15, value=5)
        with col_m3:
            mc_target = st.slider("Target C/I Threshold (dB)", min_value=15.0, max_value=50.0, value=35.0, step=1.0)

        if st.button("Run Monte Carlo Engine", type="primary"):
            worst_uplink_vector = []
            worst_downlink_vector = []
            saturated_tx_count = 0
            total_tx_evaluated = 0
            
            progress_bar = st.progress(0)
            
            # Monte Carlo Execution Loop
            for run in range(mc_runs):
                # Generate fully independent random map coordinates per loop
                mc_angles = np.sort(np.random.uniform(-60, 60, mc_links))
                # Using area-proportional square root distribution to avoid cluster bias
                mc_distances = 12.0 * np.sqrt(np.random.uniform(0.1, 1.0, mc_links))
                mc_fade = np.zeros(mc_links)
                
                # Run optimization for this snapshot map
                c2i_up, tx_up = optimize_network(mc_angles, mc_distances, mc_fade, mc_target, ant, chBW, NF, Pmax, Pmin, is_uplink=True)
                c2i_down, tx_down = optimize_network(mc_angles, mc_distances, mc_fade, mc_target, ant, chBW, NF, Pmax, Pmin, is_uplink=False)
                
                worst_uplink_vector.append(np.min(c2i_up))
                worst_downlink_vector.append(np.min(c2i_down))
                
                # Check for saturation (transmitters forced to stay maxed out)
                saturated_tx_count += np.sum(tx_up >= (Pmax - 0.2))
                total_tx_evaluated += mc_links
                
                if run % max(1, mc_runs // 10) == 0:
                    progress_bar.progress(run / mc_runs)
            
            progress_bar.progress(1.0)
            
            # Post-Processing Results Visualization
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                st.subheader("Performance Distribution (CDF)")
                fig_cdf, ax_cdf = plt.subplots(figsize=(6, 4))
                
                # Sort vectors to build empirical Cumulative Distribution Function curves
                ax_cdf.plot(np.sort(worst_uplink_vector), np.linspace(0, 1, mc_runs), label='Uplink Worst Case', linewidth=2)
                ax_cdf.plot(np.sort(worst_downlink_vector), np.linspace(0, 1, mc_runs), label='Downlink Worst Case', linewidth=2, linestyle='--')
                ax_cdf.axvline(x=mc_target, color='r', linestyle=':', label='Configured Target')
                
                ax_cdf.set_xlabel("Worst Achieved C/I in Sector (dB)")
                ax_cdf.set_ylabel("Probability (CDF)")
                ax_cdf.grid(True, alpha=0.4)
                ax_cdf.legend()
                st.pyplot(fig_cdf)
                
            with m_col2:
                st.subheader("Statistical Dashboard")
                up_array = np.array(worst_uplink_vector)
                down_array = np.array(worst_downlink_vector)
                
                outage_rate_up = np.mean(up_array < mc_target) * 100
                outage_rate_down = np.mean(down_array < mc_target) * 100
                saturation_rate = (saturated_tx_count / total_tx_evaluated) * 100
                
                st.metric("Uplink Target Outage Probability", f"{outage_rate_up:.1f} %")
                st.metric("Downlink Target Outage Probability", f"{outage_rate_down:.1f} %")
                st.metric("Amplifier Saturation Rate (At Pmax)", f"{saturation_rate:.1f} %")
                
                st.info(f"Analysis compiled over {mc_runs} distinct geographic layout variants. "
                        f"Median sector bottleneck performance: Uplink={np.median(up_array):.1f} dB, Downlink={np.median(down_array):.1f} dB.")

if __name__ == "__main__":
    main()
