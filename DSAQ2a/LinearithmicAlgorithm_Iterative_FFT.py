import cmath
import math
import time
import matplotlib.pyplot as plt
import numpy as np

class IterativeFFT:

    # Reverse the bits of a number
    @staticmethod
    def bit_reverse(num, bits):
        result = 0
        for i in range(bits):
            if num & (1 << i):
                result |= 1 << (bits - 1 - i)
        return result
    
    # Create bit-reversed copy of input array
    @staticmethod
    def bit_reverse_copy(x):
        n = len(x)
        bits = int(math.log2(n))
        X = [0] * n
        
        # Reorder elements according to bit-reversed indices
        for i in range(n):
            rev_i = IterativeFFT.bit_reverse(i, bits)
            X[rev_i] = complex(x[i])  # Ensure complex type
        
        return X
    
    # Compute FFT using iterative Cooley-Tukey algorithm
    @staticmethod
    def fft(x): 
        n = len(x)
        
        if n == 0 or (n & (n - 1)) != 0:
            raise ValueError(f"Input size must be a power of 2, got {n}")
        
        # Step 1: Bit-reversal permutation - O(n log n)
        X = IterativeFFT.bit_reverse_copy(x)
        
        # Step 2: Iterative butterfly operations - O(n log n)
        num_stages = int(math.log2(n))
        
        # Process log₂(n) stages
        for s in range(1, num_stages + 1):
            m = 1 << s  # m = 2^s (DFT size at this stage)
            half_m = m >> 1  # m/2
            
            # Compute principal m-th root of unity
            # ω_m = e^(-2πi/m) = cos(-2π/m) + i·sin(-2π/m)
            omega_m = cmath.exp(-2j * cmath.pi / m)
            
            # Process all groups at this stage
            for k in range(0, n, m):
                omega = 1  # ω^0 = 1
                
                # Butterfly operations within each group
                for j in range(half_m):
                    # Indices for butterfly operation
                    even_idx = k + j
                    odd_idx = k + j + half_m
                    
                    # Butterfly computation
                    t = omega * X[odd_idx]      # Twiddle factor × odd term
                    u = X[even_idx]             # Even term
                    
                    X[even_idx] = u + t         # Combine: even
                    X[odd_idx] = u - t          # Combine: odd
                    
                    # Update twiddle factor for next butterfly
                    omega *= omega_m
        
        return X
    
    # Compute magnitude spectrum from FFT output
    @staticmethod
    def magnitude_spectrum(X):
        return [abs(x) for x in X]
    
    # Compute phase spectrum from FFT output
    @staticmethod
    def phase_spectrum(X):
        return [cmath.phase(x) for x in X]


# Simulate 5G OFDM signals for Singapore telecommunications
class Signal5G:
    
    # Generate composite OFDM signal
    @staticmethod
    def generate_ofdm_signal(n, frequencies, amplitudes, sampling_rate=1000):
 
        t = np.arange(n) / sampling_rate  # Time array
        signal = np.zeros(n)
        
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        # Add noise to simulate real-world conditions
        noise = 0.1 * np.random.randn(n)
        signal += noise
        
        return signal
    
    # Simulate realistic 5G base station scenario in Singapore
    @staticmethod
    def singapore_5g_scenario():
        return {
            'location': 'Marina Bay Financial Centre Base Station',
            'carrier_frequencies': [50, 120, 200, 350],  # Hz (scaled for demo)
            'amplitudes': [1.0, 0.8, 0.6, 0.4],
            'description': 'Multi-carrier OFDM signal with 4 active subcarriers',
            'interference': 'Minimal (urban environment with managed spectrum)'
        }

# Test 1: Test FFT implementation correctness against NumPy
def test_fft_correctness():
    print("=" * 70)
    print("TEST 1: CORRECTNESS VERIFICATION")
    print("=" * 70)
    
    test_sizes = [8, 16, 32, 64, 128]
    
    for n in test_sizes:
        # Generate random test signal
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        # Compute FFT using both methods
        X_custom = IterativeFFT.fft(x)
        X_numpy = np.fft.fft(x)
        
        # Compare results
        max_error = max(abs(X_custom[i] - X_numpy[i]) for i in range(n))
        
        print(f"\nn = {n:4d}")
        print(f"  Max error vs NumPy: {max_error:.2e}")
        print(f"  Status: {'✓ PASS' if max_error < 1e-10 else '✗ FAIL'}")


# Test 2: Demonstrate O(n log n) complexity 
def test_fft_complexity():
    print("\n" + "=" * 70)
    print("TEST 2: COMPLEXITY ANALYSIS - O(n log n) VERIFICATION")
    print("=" * 70)
    
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    times = []
    
    print(f"\n{'n':>6} {'Time (ms)':>12} {'n log n':>12} {'Ratio':>12}")
    print("-" * 50)
    
    for n in sizes:
        # Generate test signal
        x = np.random.randn(n)
        
        # Measure execution time
        start = time.perf_counter()
        X = IterativeFFT.fft(x)
        end = time.perf_counter()
        
        exec_time = (end - start) * 1000  # Convert to milliseconds
        times.append(exec_time)
        
        # Calculate theoretical operations
        theoretical = n * math.log2(n)
        
        # Calculate ratio (should be roughly constant for O(n log n))
        ratio = exec_time / theoretical if theoretical > 0 else 0
        
        print(f"{n:6d} {exec_time:12.4f} {theoretical:12.1f} {ratio:12.6f}")
    
    # Plot complexity graph
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Execution time vs n
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Input Size (n)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('FFT Execution Time vs Input Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Time vs n log n 
    plt.subplot(1, 2, 2)
    nlogn_values = [n * math.log2(n) for n in sizes]
    plt.plot(nlogn_values, times, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('n log n', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Time vs n log n (Linear = O(n log n))', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/fft_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Complexity graph saved as 'fft_complexity_analysis.png' inside folder 'images'")
    
    return sizes, times

# Test 3: Demonstrate FFT on Singapore 5G scenario
def test_singapore_5g_application():
    print("\n" + "=" * 70)
    print("TEST 3: SINGAPORE 5G SIGNAL ANALYSIS")
    print("=" * 70)
    
    # Get Singapore scenario parameters
    scenario = Signal5G.singapore_5g_scenario()
    
    print(f"\nLocation: {scenario['location']}")
    print(f"Description: {scenario['description']}")
    print(f"Active Frequencies: {scenario['carrier_frequencies']} Hz")
    print(f"Signal Amplitudes: {scenario['amplitudes']}")
    
    # Generate 5G OFDM signal
    n = 1024
    sampling_rate = 1000  # Hz
    signal = Signal5G.generate_ofdm_signal(
        n,
        scenario['carrier_frequencies'],
        scenario['amplitudes'],
        sampling_rate
    )
    
    # Apply FFT
    print(f"\nProcessing {n} samples...")
    start = time.perf_counter()
    spectrum = IterativeFFT.fft(signal)
    end = time.perf_counter()
    
    print(f"FFT completed in {(end - start) * 1000:.3f} ms")
    
    # Get magnitude spectrum
    magnitudes = IterativeFFT.magnitude_spectrum(spectrum)
    
    # Find dominant frequencies
    freqs = np.fft.fftfreq(n, 1/sampling_rate)
    peak_indices = sorted(range(n), key=lambda i: magnitudes[i], reverse=True)[:10]
    
    print(f"\nTop 10 Detected Frequency Components:")
    print(f"{'Rank':>5} {'Frequency (Hz)':>15} {'Magnitude':>12}")
    print("-" * 35)
    for rank, idx in enumerate(peak_indices, 1):
        if freqs[idx] >= 0:  # Only positive frequencies
            print(f"{rank:5d} {freqs[idx]:15.2f} {magnitudes[idx]:12.4f}")
    
    # Visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Time domain signal
    plt.subplot(3, 1, 1)
    time_axis = np.arange(n) / sampling_rate
    plt.plot(time_axis[:200], signal[:200], 'b-', linewidth=1)
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Amplitude', fontsize=11)
    plt.title('5G OFDM Signal - Time Domain (First 200 samples)', 
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Frequency domain (full spectrum)
    plt.subplot(3, 1, 2)
    plt.plot(freqs[:n//2], magnitudes[:n//2], 'r-', linewidth=1)
    plt.xlabel('Frequency (Hz)', fontsize=11)
    plt.ylabel('Magnitude', fontsize=11)
    plt.title('Frequency Spectrum - Full Range', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Zoomed view of carrier frequencies
    plt.subplot(3, 1, 3)
    zoom_range = 400
    plt.stem(freqs[:zoom_range], magnitudes[:zoom_range], 'g', 
             markerfmt='go', basefmt=' ')
    plt.xlabel('Frequency (Hz)', fontsize=11)
    plt.ylabel('Magnitude', fontsize=11)
    plt.title('Detected Carrier Frequencies (Zoomed)', 
              fontsize=13, fontweight='bold')
    
    # Mark expected frequencies
    for freq in scenario['carrier_frequencies']:
        plt.axvline(x=freq, color='orange', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'{freq} Hz')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/singapore_5g_fft_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ 5G signal analysis saved as 'singapore_5g_fft_analysis.png' inside folder 'images'")


# Test 4: Compare FFT O(n log n) vs naive DFT O(n²)
def compare_fft_vs_dft():
    print("\n" + "=" * 70)
    print("TEST 4: FFT vs NAIVE DFT COMPARISON")
    print("=" * 70)
    
    # Naive O(n²) DFT implementation for comparison
    def naive_dft(x):
        n = len(x)
        X = [0] * n
        for k in range(n):
            for j in range(n):
                X[k] += x[j] * cmath.exp(-2j * cmath.pi * k * j / n)
        return X
    
    sizes = [16, 32, 64, 128, 256, 512]
    fft_times = []
    dft_times = []
    
    print(f"\n{'n':>5} {'FFT (ms)':>12} {'DFT (ms)':>12} {'Speedup':>12}")
    print("-" * 50)
    
    for n in sizes:
        x = np.random.randn(n)
        
        # FFT timing
        start = time.perf_counter()
        X_fft = IterativeFFT.fft(x)
        fft_time = (time.perf_counter() - start) * 1000
        fft_times.append(fft_time)
        
        # DFT timing (skip large sizes for DFT)
        if n <= 256:
            start = time.perf_counter()
            X_dft = naive_dft(x)
            dft_time = (time.perf_counter() - start) * 1000
            dft_times.append(dft_time)
            speedup = dft_time / fft_time if fft_time > 0 else 0
        else:
            dft_times.append(None)
            speedup = float('inf')
        
        speedup_str = f"{speedup:.1f}x" if speedup != float('inf') else "N/A"
        dft_str = f"{dft_times[-1]:.4f}" if dft_times[-1] is not None else "too slow"
        
        print(f"{n:5d} {fft_time:12.4f} {dft_str:>12} {speedup_str:>12}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    valid_sizes_dft = [s for s, t in zip(sizes, dft_times) if t is not None]
    valid_dft_times = [t for t in dft_times if t is not None]
    
    plt.plot(sizes, fft_times, 'bo-', linewidth=2, markersize=8, 
             label='FFT O(n log n)')
    plt.plot(valid_sizes_dft, valid_dft_times, 'rs-', linewidth=2, 
             markersize=8, label='Naive DFT O(n²)')
    
    plt.xlabel('Input Size (n)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('FFT vs Naive DFT Performance Comparison', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to show both curves
    
    plt.tight_layout()
    plt.savefig('images/fft_vs_dft_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Comparison graph saved as 'fft_vs_dft_comparison.png' inside folder 'images'")


# Test 5: Structured signals with known DFTs
def test_structured_signals():
    print("\n" + "=" * 70)
    print("TEST 5: STRUCTURED SIGNALS WITH KNOWN DFT")
    print("=" * 70)

    # Helper to compare against NumPy
    def compare_signal(x, description):
        n = len(x)
        X_custom = IterativeFFT.fft(x)
        X_numpy = np.fft.fft(x)
        max_error = max(abs(X_custom[i] - X_numpy[i]) for i in range(n))
        print(f"\n{description}")
        print(f"n = {n}")
        print(f"Max error vs NumPy: {max_error:.2e}")
        print(f"Status: {'✓ PASS' if max_error < 1e-10 else '✗ FAIL'}")

    # 5.1 Impulse (delta) signal: [1, 0, 0, ..., 0]
    n = 16
    impulse = np.zeros(n, dtype=complex)
    impulse[0] = 1.0
    compare_signal(impulse, "Case 5.1: Impulse signal (delta at index 0)")

    # 5.2 Single cosine at an FFT bin frequency
    # x[j] = cos(2π m j / n) for integer m
    m = 3
    t = np.arange(n)
    cosine = np.cos(2 * np.pi * m * t / n)
    compare_signal(cosine, f"Case 5.2: Cosine at bin frequency m = {m}")

    # 5.3 All zeros
    zeros = np.zeros(n, dtype=complex)
    compare_signal(zeros, "Case 5.3: All-zero signal")


# Test 6: Round-trip correctness using NumPy IFFT
def test_round_trip():
    print("\n" + "=" * 70)
    print("TEST 6: ROUND-TRIP CHECK (FFT + IFFT)")
    print("=" * 70)

    test_sizes = [8, 16, 32, 64, 128]

    for n in test_sizes:
        # Random complex input
        x = np.random.randn(n) + 1j * np.random.randn(n)

        # Forward FFT using custom implementation
        X_custom = IterativeFFT.fft(x)

        # Inverse using NumPy's IFFT
        x_recovered = np.fft.ifft(X_custom)

        # Compare original and recovered
        max_error = max(abs(x[i] - x_recovered[i]) for i in range(n))

        print(f"\nn = {n:4d}")
        print(f"Max error (original vs IFFT(FFT(x))): {max_error:.2e}")
        print(f"Status: {'✓ PASS' if max_error < 1e-10 else '✗ FAIL'}")

# Test 7: Robustness and boundary conditions
def test_robustness():
    print("\n" + "=" * 70)
    print("TEST 7: ROBUSTNESS AND BOUNDARY CONDITIONS")
    print("=" * 70)

    # 7.1 Real-only vs complex input
    n = 32
    real_signal = np.random.randn(n)
    complex_signal = np.random.randn(n) + 1j * np.random.randn(n)

    X_real_custom = IterativeFFT.fft(real_signal)
    X_real_numpy = np.fft.fft(real_signal)
    max_err_real = max(abs(X_real_custom[i] - X_real_numpy[i]) for i in range(n))

    X_complex_custom = IterativeFFT.fft(complex_signal)
    X_complex_numpy = np.fft.fft(complex_signal)
    max_err_complex = max(abs(X_complex_custom[i] - X_complex_numpy[i]) for i in range(n))

    print(f"\nCase 7.1: Real-only input (n = {n})")
    print(f"Max error vs NumPy: {max_err_real:.2e}")
    print(f"Status: {'✓ PASS' if max_err_real < 1e-10 else '✗ FAIL'}")

    print(f"\nCase 7.2: Complex input (n = {n})")
    print(f"Max error vs NumPy: {max_err_complex:.2e}")
    print(f"Status: {'✓ PASS' if max_err_complex < 1e-10 else '✗ FAIL'}")

    # 7.2 Different amplitude scales
    n = 64
    for scale in [1e-6, 1.0, 1e6]:
        x = scale * (np.random.randn(n) + 1j * np.random.randn(n))
        X_custom = IterativeFFT.fft(x)
        X_numpy = np.fft.fft(x)
        max_error = max(abs(X_custom[i] - X_numpy[i]) for i in range(n))
        rel_error = max_error / (np.max(np.abs(X_numpy)) + 1e-15)

        print(f"\nCase 7.3: Random complex input, scale = {scale:g}, n = {n}")
        print(f"Max abs error: {max_error:.2e}")
        print(f"Max relative error: {rel_error:.2e}")

    # 7.3 Boundary sizes and invalid sizes
    print("\nCase 7.4: Boundary sizes and invalid n")

    # Smallest valid sizes
    for n_valid in [2, 4]:
        x = np.random.randn(n_valid)
        X_custom = IterativeFFT.fft(x)
        X_numpy = np.fft.fft(x)
        max_error = max(abs(X_custom[i] - X_numpy[i]) for i in range(n_valid))
        print(f"  n = {n_valid}: max error vs NumPy = {max_error:.2e} "
              f"{'✓ PASS' if max_error < 1e-10 else '✗ FAIL'}")

    # Invalid sizes: n = 0 and non-power-of-2
    for n_invalid in [0, 12]:
        try:
            x = np.random.randn(n_invalid)
            IterativeFFT.fft(x)
            print(f"  n = {n_invalid}: EXPECTED error, but no error raised ✗ FAIL")
        except ValueError as e:
            print(f"  n = {n_invalid}: correctly raised ValueError ('{e}') ✓ PASS")



def main():

    # Run all tests
    test_fft_correctness()
    test_fft_complexity()
    test_singapore_5g_application()
    compare_fft_vs_dft()
    test_structured_signals()
    test_round_trip()
    test_robustness()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  1. fft_complexity_analysis.png")
    print("  2. singapore_5g_fft_analysis.png")
    print("  3. fft_vs_dft_comparison.png")
    print("\n")


if __name__ == "__main__":
    main()
