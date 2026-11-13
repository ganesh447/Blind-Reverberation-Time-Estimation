import numpy as np
from scipy import signal
import pyfar as pf
import pyrato as pr

def find_ir_onset(ir, fs, threshold_db=-20):
    """
    Detect the onset of an impulse response based on energy threshold.
    
    Parameters
    ----------
    ir : ndarray
        Impulse response signal
    fs : int
        Sample rate
    threshold_db : float
        Energy threshold in dB below the peak (default: -20 dB)
    
    Returns
    -------
    onset_idx : int
        Sample index where IR onset occurs
    """
    # Compute envelope using short-time RMS
    window_ms = 2  # 2ms window for smoothing
    window_samples = max(1, int(fs * window_ms / 1000))
    
    # Compute RMS in sliding windows
    rms = np.sqrt(np.convolve(ir**2, np.ones(window_samples)/window_samples, mode='same'))
    
    # Convert to dB relative to peak RMS
    rms_db = 10 * np.log10(np.abs(rms) + 1e-30)
    peak_db = np.max(rms_db)

    
    # Find first sample before maximum sample exceeding threshold
    power_ir = 10 * np.log10(np.abs(ir)**2 + 1e-30)
    max_db = np.max(power_ir)
    max_sample = np.argmax(power_ir)
    threshold_value = max_db + threshold_db
    pre_rir = power_ir[:max_sample]
    pre_rir = pre_rir[::-1]
    onset_idx = len(pre_rir) - np.argmax(pre_rir <= threshold_value) - 1

    
    return onset_idx

def window_rir(ir, fs, onset_idx, fade_time=0.001, plot_time=2):
    """
    Detect the onset of an impulse response based on energy threshold.
    
    Parameters
    ----------
    ir : ndarray
        Impulse response signal
    fs : int
        Sample rate
    onset_idx : int
        Sample index where IR onset occurs
    fade_time : float
        Duration of fade-in and fade-out in seconds (default: 0.001s)
    plot_time : float
        Duration of the truncated IR to return in seconds (default: 2s)

    Returns
    -------
    truncated_ir : ndarray
        The truncated impulse response with fade-in and fade-out applied.
    """
    
    # truncated_ir = ir[onset_idx - int(fs * fade_time):]
    truncated_ir = ir[onset_idx - int(fs * fade_time):onset_idx + int(fs * (plot_time - fade_time))]
    fade_in = np.linspace(0, 1, int(fs * fade_time))
    fade_out = np.linspace(1, 0, int(fs * fade_time))
    truncated_ir[:len(fade_in)] = truncated_ir[:len(fade_in)] * fade_in
    truncated_ir[-len(fade_out):] = truncated_ir[-len(fade_out):] * fade_out

    return truncated_ir

def farina_rir(inv_sweep, recording, fs, onset_threshold_db=-20):
    """
    Calculate the impulse response using Farina's method of deconvolution.
    
    Parameters
    ----------
    inv_sweep : ndarray
        Reverse sweep signal (inverse filter) of the original sweep used for excitation
    recording : ndarray
        The recorded signal containing the response to the sweep
    fs : int
        Sample rate of the signals, must be the same for both sweep and recording
    onset_threshold_db : float
        Energy threshold in dB below the peak for automatic onset detection (default: -20 dB)
    Returns
    -------
    rir : ndarray
        The calculated impulse response
    onset_idx : int
        Sample index where IR onset occurs
    """
    # Perform convolution using FFT-based method for efficiency, full length to capture entire response
    rir = signal.convolve(inv_sweep, recording, mode='full', method='fft')
    rir = rir / np.max(np.abs(rir))  # Normalize to prevent clipping and maintain relative levels
    
    # Automatically detect IR onset and return the starting Index
    onset_idx = find_ir_onset(rir, fs, threshold_db=onset_threshold_db)
    
    return rir, onset_idx

def rir_quad(ir, fs, schroeder=True):
    """
    Calculate the Schroeder integral (cumulative energy decay curve) of an impulse response.
    
    Parameters
    ----------
    ir : ndarray
        Impulse response signal
    fs : int
        Sample rate of the impulse response
    schroeder : bool
        If True, compute the cumulative energy decay curve (Schroeder integral).
        If False, compute the direct energy decay curve without cumulative summation.
    
    Returns
    -------
    time : ndarray
        Time axis corresponding to the decay curve
    decay_db : ndarray
        Decay curve in dB, normalized to 0 dB at the start of the decay
    """
    # Compute energy of the impulse response
    energy = ir**2
    real_start = np.argmax(energy)  # Find the index of the maximum energy as actual onset of the IR
    energy = energy[real_start:]  # Consider only the part of the IR starting from the actual onset for decay calculation
    # Create time axis
    time = np.arange(len(energy)) / fs
    
    if schroeder==True:
        # Compute cumulative energy decay curve (Schroeder integral)
        cumulative_energy = np.cumsum(energy[::-1])[::-1]
        
        # Normalize and convert to dB
        decay_db = 10 * np.log10(cumulative_energy + 1e-30)  # Add small value to avoid log(0)
        decay_db -= np.max(decay_db)  # Normalize to 0 dB at the start

    else:
        decay_db = 10 * np.log10(energy + 1e-30)  # Add small value to avoid log(0)
        decay_db -= np.max(decay_db)  # Normalize to 0 dB at the start
    
    return decay_db, time


def lundeby_intersection(ir, fs):
    """
    Calculate the intersection point of the Lundeby method for decay curve analysis.

    Parameters
    ----------
    ir : ndarray
        impulse response signal 
        (Should not be the whole convolution product of farina_rir, since the noise level estimation is 
        based on the tail of the IR, which can be affected by pre- and post-echoes if the full convolution 
        result is used. )
    fs : int or float
        Sample rate of the impulse response in Hz

    Returns
    -------
    intersection_time : float
        Time at which the Lundeby intersection occurs
    reverberation_time : float
        Calculated maximum reliable reverberation time 
        (dB level of the decay curve at this point can be used to determine if the decay is sufficient 
        for reliable reverberation time measurement)
    noise_level : float
        Estimated noise level 
        (probably not usable for farina rirs, since the noiselevel changes with the length 
        of the ir, but can be used for direct recordings via impulse noise or pink noise excitation)

    """

    # Create a pyrato Signal object from the impulse response
    ir_signal = pf.Signal(ir, fs)

    # Use pyrato's built-in function to calculate the Lundeby intersection time, reverberation time, and noise level
    # failure_policy='warning' makes pyrato return NaNs (with a warning) instead of raising when
    # the Lundeby parameters cannot be estimated, matching the NaN-handling in decay_curve_lundeby below.
    intersection_time, reverberation_time, noise_level = pr.edc.intersection_time_lundeby(
        ir_signal, failure_policy='warning'
    )
    # Extract only the value from the returned tuples
    intersection_time = intersection_time[0]
    reverberation_time = reverberation_time[0]
    noise_level = noise_level[0]

    return intersection_time, reverberation_time, noise_level


def decay_curve_lundeby(ir, fs):
    """
    Calculate the decay curve using Lundeby algorithm for a given impulse response.

    Parameters:
    ----------
    ir : ndarray
        Impulse response signal
        (Should not be the whole convolution product of farina_rir, since the noise level estimation is 
        based on the tail of the IR, which can be affected by pre- and post-echoes if the full convolution 
        result is used. )
    fs : int or float
        Sample rate of the impulse response in Hz
    
    Returns:
    ----------
    decay_curve : ndarray
        Decay curve truncated by lundeby intersection time
    time : ndarray
        Time axis corresponding to the decay curve
        noise_level : float
        Estimated noise level in dB
    """

    intersection_time, reverberation_time, noise_level = lundeby_intersection(ir, fs)

    # If the Lundeby estimation failed it already warned and returned NaNs;
    # return a safe placeholder so this function does not crash on int(NaN).
    if np.isnan(intersection_time) or np.isnan(reverberation_time):
        return np.array([np.nan]), np.array([0.0]), 0

    intersection_sample = int(intersection_time * fs)
    decay_curve, time = rir_quad(ir[:intersection_sample], fs, schroeder=True)
    noise_level = 10 * np.log10(noise_level)

    return decay_curve, time, noise_level



def calculate_decay_time_schroeder(ir, fs, start_db=0, end_db=-5):

    """
    Calculate decay time between two specified dB levels from an impulse response.
    
    Parameters
    ----------
    ir : ndarray
        Impulse response signal
    fs : int or float
        Sample rate of the impulse response in Hz
    start_db : float
        Starting dB level for decay time calculation (default: 0 dB)
    end_db : float
        Ending dB level for decay time calculation (default: -5 dB)
    
    Returns
    -------
    decay_time : float
        Decay time in seconds between the specified dB levels, or np.nan if insufficient decay is present.
    
    Notes
    -----
    - The function computes the Schroeder integral to obtain the energy decay curve and finds the time difference between the specified dB levels.
    - If the decay does not reach the specified end_db level, np.nan is returned to indicate an unreliable measurement.
    """
    # Compute the Schroeder integral (energy decay curve)
    decay_db, time = rir_quad(ir, fs, schroeder=True)
    
    # Find indices corresponding to start and end dB levels
    idx_start = np.argmax(decay_db <= start_db)
    idx_end = np.argmax(decay_db <= end_db)
    
    # Check validity of indices
    if idx_start == 0 or idx_end == 0:
        if np.min(decay_db) > start_db or np.min(decay_db) > end_db:
            return np.nan
    
    if idx_start >= idx_end:
        return np.nan
    
    # Calculate decay time
    coeffs = np.polyfit(time[idx_start:idx_end], decay_db[idx_start:idx_end], 1)
    decay_slope = coeffs[0]
    time_diff = -60/decay_slope  # Extrapolate to 60dB decay for EDT, T20 and T30

    decay_end_db = calculate_decay_end_schroeder(ir, fs)
    if decay_end_db+10 > end_db:
        print(f"WARNING:Decay does not reach the specified end dB level of {end_db} dB. Minimum decay level is {decay_end_db:.2f} dB. Decay time measurement may be unreliable.")
        
    
    return time_diff


def calculate_decay_time_lundeby(ir, fs, start_db=0, end_db=-5):

    """
    Calculate decay time between two specified dB levels from an impulse response.
    
    Parameters
    ----------
    ir : ndarray
        Impulse response signal
    fs : int or float
        Sample rate of the impulse response in Hz
    start_db : float
        Starting dB level for decay time calculation (default: 0 dB)
    end_db : float
        Ending dB level for decay time calculation (default: -5 dB)
    
    Returns
    -------
    decay_time : float
        Decay time in seconds between the specified dB levels, or np.nan if insufficient decay is present.
    
    Notes
    -----
    - The function computes the Schroeder integral to obtain the energy decay curve and finds the time difference between the specified dB levels.
    - If the decay does not reach the specified end_db level, np.nan is returned to indicate an unreliable measurement.
    """
    # Compute the Schroeder integral (energy decay curve)
    decay_db, time, noise_floor = decay_curve_lundeby(ir, fs)
    
    # Find indices corresponding to start and end dB levels
    idx_start = np.argmax(decay_db <= start_db)
    idx_end = np.argmax(decay_db <= end_db)
    
    # Check validity of indices
    if idx_start == 0 or idx_end == 0:
        if np.min(decay_db) > start_db or np.min(decay_db) > end_db:
            return np.nan
    
    if idx_start >= idx_end:
        return np.nan
    
    # Calculate decay time
    coeffs = np.polyfit(time[idx_start:idx_end], decay_db[idx_start:idx_end], 1)
    decay_slope = coeffs[0]
    time_diff = -60/decay_slope  # Extrapolate to 60dB decay for EDT, T20 and T30

    # Check if reverberation time exceeds the maximum reverberation time of the lundeby curve
    if noise_floor + 10 > end_db:
        print(f"WARNING:Specified end of decay level with {end_db} dB is not 10 dB above noise floor with {noise_floor:.2f} dB. Decay time measurement may be unreliable.")
        
    
    return time_diff


def calculate_reverberation_times(ir, fs, metrics=None, method='lundeby'):
    """
    Calculate reverberation time metrics from an impulse response via Schroeder integral method.
    
    Parameters
    ----------
    ir : ndarray
        Impulse response signal
    fs : int or float
        Sample rate of the impulse response in Hz
    metrics : str or list of str, optional
        Metrics to calculate. Options: 'T20', 'T30', 'T60', 'EDT' (Early Decay Time).
        If None (default), all metrics are calculated.
    
    Returns
    -------
    results : dict
        Dictionary containing calculated reverberation times in seconds.
        Keys: 'T20', 'T30', 'T60', 'EDT'
        Values are np.nan if the decay is insufficient to measure reliably.
    
    Notes
    -----
    - EDT: Extrapolated from decay between 0 dB and -10 dB (factor of 6)
    - T20: Extrapolated from decay between -5 dB and -25 dB (factor of 3)
    - T30: Extrapolated from decay between -5 dB and -35 dB (factor of 2)
    - T60: Measured directly from -5 dB to -65 dB (does not require extrapolation, but may be unreliable if 
            decay does not reach -65 dB)
    """
    if metrics is None:
        metrics = ['EDT', 'T20', 'T30', 'T60']
    elif isinstance(metrics, str):
        metrics = [metrics]
    
    results = {}
    
    # Calculate requested metrics
    if 'EDT' in metrics:
        if method == 'schroeder':
            results['EDT'] = calculate_decay_time_schroeder(ir, fs, 0, -10)
        elif method == 'lundeby':
            results['EDT'] = calculate_decay_time_lundeby(ir, fs, 0, -10)

    if 'T20' in metrics:
        if method == 'schroeder':
            results['T20'] = calculate_decay_time_schroeder(ir, fs, -5, -25)
        elif method == 'lundeby':
            results['T20'] = calculate_decay_time_lundeby(ir, fs, -5, -25)
    
    if 'T30' in metrics:
        if method == 'schroeder':
            results['T30'] = calculate_decay_time_schroeder(ir, fs, -5, -35)
        elif method == 'lundeby':
            results['T30'] = calculate_decay_time_lundeby(ir, fs, -5, -35)
    
    if 'T60' in metrics:
        if method == 'schroeder':
            results['T60'] = calculate_decay_time_schroeder(ir, fs, -5, -65)
        elif method == 'lundeby':
            results['T60'] = calculate_decay_time_lundeby(ir, fs, -5, -65)
    
    return results



def calculate_energy_ratios(ir, fs, metrics=None):
    """
    Calculate energy ratio metrics (C80, C50, D50, Ts) from an impulse response.
    
    Parameters
    ----------
    ir : ndarray
        Impulse response signal
    fs : int or float
        Sample rate of the impulse response in Hz
    metrics : str or list of str, optional
        Metrics to calculate. Options: 'C80', 'C50', 'D50', 'Ts'.
        If None (default), all metrics are calculated.
    
    Returns
    -------
    results : dict
        Dictionary containing calculated energy ratio metrics in dB.
        Keys: 'C80', 'C50', 'D50', 'Ts'
        
    Notes
    -----
    - C80: Clarity for speech/music, ratio of energy in first 80 ms to energy after 80 ms.
    - C50: Clarity for speech, ratio of energy in first 50 ms to energy after 50 ms.
    - D50: Deutlichkeit for speech, ratio of energy in first 50 ms to overall energy.
    - Ts: Center time, representing the time at which the energy of the impulse response is centered.
    """
    if metrics is None:
        metrics = ['C80', 'C50', 'D50', 'Ts']
    elif isinstance(metrics, str):
        metrics = [metrics]
    
    # Create time axis
    time = np.arange(len(ir)) / fs
    
    results = {}
    
    if 'C80' in metrics:
        energy_early = np.sum(ir[time <= 0.08]**2)
        energy_late = np.sum(ir[time > 0.08]**2)
        results['C80'] = 10 * np.log10(energy_early / (energy_late + 1e-30))
    
    if 'C50' in metrics:
        energy_early = np.sum(ir[time <= 0.05]**2)
        energy_late = np.sum(ir[time > 0.05]**2)
        results['C50'] = 10 * np.log10(energy_early / (energy_late + 1e-30))

    if 'D50' in metrics:
        energy_early = np.sum(ir[time <= 0.05]**2)
        energy_all = np.sum(ir**2)
        results['D50'] = energy_early / energy_all

    if 'Ts' in metrics:
        energy = ir**2
        results['Ts'] = np.sum(time * energy) / np.sum(energy + 1e-30)

    return results


def calculate_decay_db(ir, fs, time_cut=0.05):
    """
    Calculate decay in dB for a given time cut.
    
    Parameters
    ----------
    ir : ndarray
        Impulse response signal
    fs : int or float
        Sample rate of the impulse response in Hz
    time_cut : float
        Time in seconds to separate early and late energy for clarity calculation. (default: 0.05 seconds)
    
    Returns
    -------
    results : float
        Decay in dB for the specified time cut.
    
    Notes
    -----
    - This function calculates the decay in dB for a given time cut.
    - It separates the early and late energy based on the specified time cut and computes the ratio.
    - Similar to clarity metrics, but can be used for arbitrary time cuts to analyze decay characteristics 
      at different time intervals.
    """
    
    # Create time axis and partial energy by time_cut
    time = np.arange(len(ir)) / fs
    energy_early = np.sum(ir[time <= time_cut]**2)
    energy_late = np.sum(ir[time > time_cut]**2)

    # Calculate decay in dB for the specified time cut as the ratio of early energy to late energy
    result = 10 * np.log10(energy_early / (energy_late + 1e-30))
    
    return result

def calculate_decay_end_schroeder(ir, fs):
    """
    Calculate the decay end time based on the intersection of the decay curve and noise floor.
    
    Parameters
    ----------
    decay_curve : ndarray
        Decay curve in dB, normalized to 0 dB at the start of the decay
    time : ndarray
        Time axis corresponding to the decay curve
    fs : int or float
        Sample rate of the impulse response in Hz
    
    Returns
    -------
    decay_end_time : float
        Estimated decay end time in seconds based on the intersection of the decay curve and noise floor.
    
    Notes
    -----
    - This function estimates the decay end time by finding the intersection point between the decay curve and an estimated noise floor.
    - The noise floor can be approximated by analyzing the tail of the impulse response or using a fixed threshold.
    - The intersection point is calculated using a line intersection method between the decay curve and noise floor lines.
    - The function returns the estimated decay end time in seconds, which can be used to determine the effective duration of the reverberation.
    """

    decay_curve, time = rir_quad(ir, fs, schroeder=True)

    # Helper function for line intersection
    def line_intersection(p1,p2,p3,p4):
        xdiff = (p1[0] - p2[0], p3[0] - p4[0])
        ydiff = (p1[1] - p2[1], p3[1] - p4[1])

        div = xdiff[0] * ydiff[1] - xdiff[1] * ydiff[0]
        if div == 0:
            raise Exception('lines do not intersect')

        d = (p1[0] * p2[1] - p1[1] * p2[0], p3[0] * p4[1] - p3[1] * p4[0])
        intersection_db = (d[0] * ydiff[1] - d[1] * ydiff[0]) / div
        return intersection_db
    
    # Define decay line and noise floor line based on the decay curve and time axis
    # Tail line is defined by points at half the ir length and at half the ir length + 1s
    decay_line_p1 = [0, decay_curve[0]]  # Starting point of the decay curve (0 dB)
    decay_line_p2 = [time[int(fs*0.2)], decay_curve[int(fs*0.2)]]  # Ending point of decay line at 100 ms (or at the corresponding time index)
    tail_line_p1 = [time[-int(len(time)/2)], decay_curve[-int(len(time)/2)]]  # Starting point of the tail line (noise floor)
    tail_line_p2 = [time[-int(len(time)/2+fs)], decay_curve[-int(len(time)/2+fs)]]  # Ending point of the tail line at 100 ms (or at the corresponding time index)
    
    # Calculate intersection point between decay line and noise floor line
    try:
        intersection_db = line_intersection(decay_line_p1, decay_line_p2, tail_line_p1, tail_line_p2)
    except:
        print('Decay end in dB not calculabe.')

    return intersection_db