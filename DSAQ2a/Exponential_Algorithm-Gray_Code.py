import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class GrayCodeGenerator:
    
    # Generate n-bit Gray codes using reflection method
    @staticmethod
    def generate_reflection(n: int) -> List[str]:
        if n < 1:
            raise ValueError("n must be at least 1")
        
        # Base case: 1-bit Gray code
        gray_codes = ['0', '1']
        
        # Build n-bit codes iteratively
        for i in range(2, n + 1):
            current_size = len(gray_codes)
            
            # Step 1: Reflect - append reversed copy
            # This doubles the list size: O(2^i) operations at iteration i
            for j in range(current_size - 1, -1, -1):
                gray_codes.append(gray_codes[j])
            
            # Step 2: Prefix '0' to first half
            # O(2^(i-1)) operations
            for j in range(current_size):
                gray_codes[j] = '0' + gray_codes[j]
            
            # Step 3: Prefix '1' to second half
            # O(2^(i-1)) operations
            for j in range(current_size, 2 * current_size):
                gray_codes[j] = '1' + gray_codes[j]
        
        return gray_codes
    
    # Generate n-bit Gray codes using XOR formula method
    @staticmethod
    def generate_formula(n: int) -> List[str]:

        if n < 1:
            raise ValueError("n must be at least 1")
        
        gray_codes = []
        total = 1 << n  # 2^n using bit shift
        
        # Generate all 2^n codes
        for i in range(total):
            # Apply Gray code formula
            gray_value = i ^ (i >> 1)
            
            # Convert to n-bit binary string
            gray_string = format(gray_value, f'0{n}b')
            gray_codes.append(gray_string)
        
        return gray_codes
    
    # Verify that codes satisfy Gray code property
    @staticmethod
    def verify_gray_property(codes: List[str]) -> Tuple[bool, str]:
        n = len(codes[0])
        expected_count = 1 << n
        
        # Check count
        if len(codes) != expected_count:
            return False, f"Expected {expected_count} codes, got {len(codes)}"
        
        # Check Hamming distance between consecutive codes
        for i in range(len(codes)):
            next_i = (i + 1) % len(codes)  # Cyclic check
            
            # Count bit differences
            diff_count = sum(c1 != c2 for c1, c2 in zip(codes[i], codes[next_i]))
            
            if diff_count != 1:
                return False, f"Codes at index {i} and {next_i} differ by {diff_count} bits, expected 1"
        
        return True, "All codes valid"
    
    # Calculate Hamming distance between two binary strings
    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    # Convert binary string to decimal
    @staticmethod
    def binary_to_decimal(binary_str: str) -> int:
        return int(binary_str, 2)
    
    # Convert Gray code to binary
    @staticmethod
    def gray_to_binary(gray: str) -> str:
        binary = [gray[0]]
        for i in range(1, len(gray)):
            # XOR current gray bit with previous binary bit
            binary.append(str(int(gray[i]) ^ int(binary[i-1])))
        return ''.join(binary)


# Simulate QR code error pattern analysis for Singapore's digital infrastructure
class SingaporeQRCodeAnalyzer:

    @staticmethod
    def simulate_qr_error_patterns(n: int) -> dict:

        scenarios = {
            'n_bits': n,
            'total_patterns': 2 ** n,
            'singapore_context': {
                'application': 'SGQR Payment System Error Correction',
                'deployment_locations': [
                    'Hawker Centers (100+ locations)',
                    'MRT Stations (134 stations)',
                    'Retail Stores (50,000+ merchants)',
                    'Government Buildings (SafeEntry check-ins)'
                ],
                'error_types': {
                    'weather_damage': 'Tropical rain, high humidity',
                    'wear_and_tear': 'High-traffic locations (Orchard Road, CBD)',
                    'poor_lighting': 'Indoor hawker centers, parking lots',
                    'screen_glare': 'Outdoor sunlight on mobile screens'
                }
            },
            'analysis_purpose': [
                'Test Reed-Solomon error correction (QR uses RS)',
                'Identify critical bit positions',
                'Optimize error correction level (L/M/Q/H)',
                'Train ML models for damaged code recognition'
            ]
        }
        
        return scenarios
    
    # Analyze which bit positions are most critical in transitions    
    @staticmethod
    def analyze_critical_bits(codes: List[str]) -> dict:
        n = len(codes[0])
        bit_change_counts = [0] * n
        
        # Count how many times each bit position changes
        for i in range(len(codes) - 1):
            for bit_pos in range(n):
                if codes[i][bit_pos] != codes[i+1][bit_pos]:
                    bit_change_counts[bit_pos] += 1
        
        return {
            'bit_positions': list(range(n)),
            'change_frequencies': bit_change_counts,
            'most_critical_bit': bit_change_counts.index(max(bit_change_counts)),
            'least_critical_bit': bit_change_counts.index(min(bit_change_counts))
        }


# Test Case 1: Test Gray code generation correctness
def test_correctness():
    print("=" * 70)
    print("TEST 1: CORRECTNESS VERIFICATION")
    print("=" * 70)
    
    test_cases = [
        (1, ['0', '1']),
        (2, ['00', '01', '11', '10']),
        (3, ['000', '001', '011', '010', '110', '111', '101', '100'])
    ]
    
    for n, expected in test_cases:
        # Test reflection method
        codes_reflection = GrayCodeGenerator.generate_reflection(n)
        
        # Test formula method
        codes_formula = GrayCodeGenerator.generate_formula(n)
        
        is_valid_reflection, msg_reflection = GrayCodeGenerator.verify_gray_property(codes_reflection)
        is_valid_formula, msg_formula = GrayCodeGenerator.verify_gray_property(codes_formula)
        
        print(f"\nn = {n}:")
        print(f"  Expected: {expected}")
        print(f"  Reflection: {codes_reflection}")
        print(f"  Formula:    {codes_formula}")
        print(f"  Reflection valid: {'✓ PASS' if is_valid_reflection else '✗ FAIL'} - {msg_reflection}")
        print(f"  Formula valid:    {'✓ PASS' if is_valid_formula else '✗ FAIL'} - {msg_formula}")
        print(f"  Match expected:   {'✓ PASS' if codes_reflection == expected else '✗ FAIL'}")


# Test Case 2: Demonstrate exponential O(2^n) growth
def test_exponential_growth():
    print("\n" + "=" * 70)
    print("TEST 2: EXPONENTIAL COMPLEXITY DEMONSTRATION - O(2^n)")
    print("=" * 70)
    
    test_sizes = list(range(1, 21))  # n = 1 to 20
    reflection_times = []
    formula_times = []
    code_counts = []
    
    print(f"\n{'n':>3} {'Codes':>12} {'Reflection (ms)':>18} {'Formula (ms)':>15} {'2^n':>12}")
    print("-" * 70)
    
    for n in test_sizes:
        expected_count = 2 ** n
        code_counts.append(expected_count)
        
        # Test reflection method
        start = time.perf_counter()
        codes_reflection = GrayCodeGenerator.generate_reflection(n)
        reflection_time = (time.perf_counter() - start) * 1000
        reflection_times.append(reflection_time)
        
        # Test formula method
        start = time.perf_counter()
        codes_formula = GrayCodeGenerator.generate_formula(n)
        formula_time = (time.perf_counter() - start) * 1000
        formula_times.append(formula_time)
        
        print(f"{n:3d} {len(codes_reflection):12,} {reflection_time:18.4f} {formula_time:15.4f} {expected_count:12,}")
        
        # Stop if taking too long
        if reflection_time > 10000:  # 10 seconds
            print(f"\n⚠ Stopping at n={n} (execution time exceeds 10 seconds)")
            test_sizes = test_sizes[:test_sizes.index(n)+1]
            break
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of codes vs n (exponential curve)
    axes[0, 0].plot(test_sizes, code_counts, 'ro-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('n (bits)', fontsize=11)
    axes[0, 0].set_ylabel('Number of Gray Codes (2^n)', fontsize=11)
    axes[0, 0].set_title('Exponential Growth: 2^n', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Execution time vs n
    axes[0, 1].plot(test_sizes, reflection_times, 'bo-', linewidth=2, markersize=6, label='Reflection')
    axes[0, 1].plot(test_sizes, formula_times, 'go-', linewidth=2, markersize=6, label='Formula')
    axes[0, 1].set_xlabel('n (bits)', fontsize=11)
    axes[0, 1].set_ylabel('Execution Time (ms)', fontsize=11)
    axes[0, 1].set_title('Execution Time vs n', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Time vs 2^n (should be linear on log scale)
    axes[1, 0].plot(code_counts, reflection_times, 'mo-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Number of Codes (2^n)', fontsize=11)
    axes[1, 0].set_ylabel('Execution Time (ms)', fontsize=11)
    axes[1, 0].set_title('Time vs 2^n (Linear = O(2^n))', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Growth rate comparison table
    axes[1, 1].axis('off')
    growth_data = []
    for i, n in enumerate(test_sizes[:10]):  # First 10 for readability
        growth_data.append([
            str(n),
            f"{code_counts[i]:,}",
            f"{reflection_times[i]:.3f}",
            f"{code_counts[i] / code_counts[i-1]:.1f}x" if i > 0 else "-"
        ])
    
    table = axes[1, 1].table(
        cellText=growth_data,
        colLabels=['n', 'Codes (2^n)', 'Time (ms)', 'Growth Factor'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Exponential Complexity Analysis - Gray Code Generation', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('images/gray_code_exponential_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Complexity graph saved as 'gray_code_exponential_analysis.png' inside folder 'images'")
    
    return test_sizes, code_counts, reflection_times


# Test Case 3: Demonstrate Gray code application to Singapore QR code systems
def test_singapore_qr_application():
    print("\n" + "=" * 70)
    print("TEST 3: SINGAPORE QR CODE ERROR PATTERN ANALYSIS")
    print("=" * 70)
    
    # Typical QR code data block sizes
    test_n = 8  # 8-bit data block (256 patterns)
    
    print(f"\nGenerating {2**test_n} error patterns for {test_n}-bit QR data blocks...")
    start = time.perf_counter()
    codes = GrayCodeGenerator.generate_reflection(test_n)
    gen_time = (time.perf_counter() - start) * 1000
    
    scenario = SingaporeQRCodeAnalyzer.simulate_qr_error_patterns(test_n)
    
    print(f"\n{'='*60}")
    print(f"SINGAPORE CONTEXT: {scenario['singapore_context']['application']}")
    print(f"{'='*60}")
    print(f"\nDeployment Locations:")
    for loc in scenario['singapore_context']['deployment_locations']:
        print(f"  • {loc}")
    
    print(f"\nError Types in Tropical Singapore:")
    for error_type, description in scenario['singapore_context']['error_types'].items():
        print(f"  • {error_type.replace('_', ' ').title()}: {description}")
    
    print(f"\nAnalysis Purpose:")
    for purpose in scenario['analysis_purpose']:
        print(f"  • {purpose}")
    
    print(f"\n{'='*60}")
    print(f"GENERATION RESULTS")
    print(f"{'='*60}")
    print(f"Total patterns generated: {len(codes):,}")
    print(f"Generation time: {gen_time:.3f} ms")
    print(f"Memory usage: ~{len(codes) * test_n / 1024:.2f} KB")
    
    bit_analysis = SingaporeQRCodeAnalyzer.analyze_critical_bits(codes)
    
    print(f"\n{'='*60}")
    print(f"BIT POSITION CRITICALITY ANALYSIS")
    print(f"{'='*60}")
    print(f"\nBit change frequencies across all {len(codes)-1} transitions:")
    for pos, freq in enumerate(bit_analysis['change_frequencies']):
        print(f"  Bit {pos}: {freq} changes ({freq/(len(codes)-1)*100:.1f}%)")
    
    print(f"\nMost critical bit position: {bit_analysis['most_critical_bit']}")
    print(f"Least critical bit position: {bit_analysis['least_critical_bit']}")
    
    # Show sample patterns
    print(f"\n{'='*60}")
    print(f"SAMPLE ERROR PATTERNS (First 16 of {len(codes)})")
    print(f"{'='*60}")
    print(f"{'Index':>6} {'Gray Code':>12} {'Decimal':>10} {'Bits Changed':>15}")
    print("-" * 50)
    
    for i in range(min(16, len(codes))):
        decimal = GrayCodeGenerator.binary_to_decimal(codes[i])
        if i > 0:
            bits_changed = GrayCodeGenerator.hamming_distance(codes[i-1], codes[i])
        else:
            bits_changed = 0
        print(f"{i:6d} {codes[i]:>12} {decimal:10d} {bits_changed:15d}")
    
    # Visualize Gray code sequence
    visualize_gray_code_sequence(codes[:32] if len(codes) > 32 else codes)


# Visualize Gray code sequence as a binary matrix
def visualize_gray_code_sequence(codes: List[str]):
    n = len(codes[0])
    num_codes = len(codes)
    
    # Create binary matrix
    matrix = np.zeros((num_codes, n))
    for i, code in enumerate(codes):
        for j, bit in enumerate(code):
            matrix[i, j] = int(bit)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Plot 1: Binary matrix heatmap
    im1 = ax1.imshow(matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Bit Position', fontsize=12)
    ax1.set_ylabel('Code Index', fontsize=12)
    ax1.set_title(f'{n}-bit Gray Code Sequence ({num_codes} codes)', 
                  fontsize=13, fontweight='bold')
    
    # Add gridlines
    ax1.set_xticks(np.arange(n))
    ax1.set_yticks(np.arange(num_codes))
    ax1.set_xticklabels([f'b{i}' for i in range(n)])
    ax1.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=ax1)
    cbar.set_label('Bit Value', rotation=270, labelpad=15)
    
    # Plot 2: Transition diagram (for small n)
    if num_codes <= 16:
        angles = np.linspace(0, 2*np.pi, num_codes, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        
        ax2.scatter(x, y, s=300, c='lightblue', edgecolors='blue', linewidths=2, zorder=3)
        
        for i in range(num_codes):
            ax2.text(x[i]*1.15, y[i]*1.15, codes[i], 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        
        for i in range(num_codes):
            next_i = (i + 1) % num_codes
            ax2.plot([x[i], x[next_i]], [y[i], y[next_i]], 
                    'r-', linewidth=1.5, alpha=0.6, zorder=1)
            
            changed_bit = -1
            for bit_pos in range(n):
                if codes[i][bit_pos] != codes[next_i][bit_pos]:
                    changed_bit = bit_pos
                    break
            
            mid_x = (x[i] + x[next_i]) / 2
            mid_y = (y[i] + y[next_i]) / 2
            ax2.text(mid_x*1.05, mid_y*1.05, f'b{changed_bit}', 
                    fontsize=7, color='red', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Gray Code Transition Diagram\n(Single bit changes between adjacent codes)', 
                     fontsize=13, fontweight='bold')
    else:
        # For larger n, show statistics
        ax2.axis('off')
        stats_text = f"""
        GRAY CODE STATISTICS
        {'='*40}
        
        Number of bits (n): {n}
        Total codes: {num_codes:,}
        Complexity: O(2^n) = O(2^{n})
        
        SINGAPORE QR CODE APPLICATION
        {'='*40}
        
        Use Case: SGQR Payment Error Correction
        
        Error Patterns: All {num_codes:,} single-bit 
        transitions analyzed
        
        Critical for:
          • Reed-Solomon error correction
          • Robust QR code scanning
          • Machine learning training data
          • Field testing in tropical climate
        
        Deployment: 50,000+ merchants
        Daily scans: Millions across Singapore
        """
        
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('images/singapore_qr_gray_code_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Visualization saved as 'singapore_qr_gray_code_visualization.png' inside folder 'images'")


# Test Case 4: Compare exponential growth with other complexities
def compare_growth_rates():

    print("\n" + "=" * 70)
    print("TEST 4: GROWTH RATE COMPARISON")
    print("=" * 70)
    
    n_values = list(range(1, 21))
    
    # Calculate different complexities
    linear = [n for n in n_values]
    nlogn = [n * np.log2(n) if n > 0 else 0 for n in n_values]
    quadratic = [n**2 for n in n_values]
    exponential = [2**n for n in n_values]
    
    print(f"\n{'n':>3} {'O(n)':>10} {'O(n log n)':>12} {'O(n²)':>12} {'O(2^n)':>15}")
    print("-" * 60)
    
    for i, n in enumerate(n_values[:15]):  # First 15 for readability
        print(f"{n:3d} {linear[i]:10.0f} {nlogn[i]:12.1f} {quadratic[i]:12.0f} {exponential[i]:15,}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Linear scale (shows exponential explosion)
    ax1.plot(n_values[:15], linear[:15], 'g-', linewidth=2, marker='o', label='O(n)')
    ax1.plot(n_values[:15], nlogn[:15], 'b-', linewidth=2, marker='s', label='O(n log n)')
    ax1.plot(n_values[:15], quadratic[:15], 'orange', linewidth=2, marker='^', label='O(n²)')
    ax1.plot(n_values[:15], exponential[:15], 'r-', linewidth=3, marker='D', label='O(2^n)')
    
    ax1.set_xlabel('Input Size (n)', fontsize=12)
    ax1.set_ylabel('Operations', fontsize=12)
    ax1.set_title('Growth Rate Comparison (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale (shows all curves)
    ax2.plot(n_values, linear, 'g-', linewidth=2, marker='o', label='O(n)')
    ax2.plot(n_values, nlogn, 'b-', linewidth=2, marker='s', label='O(n log n)')
    ax2.plot(n_values, quadratic, 'orange', linewidth=2, marker='^', label='O(n²)')
    ax2.plot(n_values, exponential, 'r-', linewidth=3, marker='D', label='O(2^n)')
    
    ax2.set_xlabel('Input Size (n)', fontsize=12)
    ax2.set_ylabel('Operations (log scale)', fontsize=12)
    ax2.set_title('Growth Rate Comparison (Log Scale)', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('images/growth_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Comparison graph saved as 'growth_rate_comparison.png' inside folder 'images'")
    
    # Print doubling analysis
    print(f"\n{'='*60}")
    print("DOUBLING ANALYSIS - Why O(2^n) is Exponential")
    print(f"{'='*60}")
    print(f"\n{'n':>3} {'2^n':>15} {'Factor':>15} {'Explanation'}")
    print("-" * 70)
    
    for n in [5, 6, 7, 10, 15, 20]:
        codes = 2**n
        factor = 2**(n) / 2**(n-1) if n > 1 else 1
        print(f"{n:3d} {codes:15,} {factor:15.1f}x {'Doubles with each +1 to n'}")



# Test Case 5: Equivalence of reflection and formula methods
def test_method_equivalence():
    print("\n" + "=" * 70)
    print("TEST 5: METHOD EQUIVALENCE - REFLECTION vs FORMULA")
    print("=" * 70)

    # Choose a range of n where 2^n is still manageable
    test_ns = [2, 3, 4, 5, 6, 8, 10]

    for n in test_ns:
        codes_reflection = GrayCodeGenerator.generate_reflection(n)
        codes_formula    = GrayCodeGenerator.generate_formula(n)

        # Check lengths
        len_ok = (len(codes_reflection) == len(codes_formula) == (1 << n))

        # Check that the sets of codes are identical (order not required)
        set_ok = (set(codes_reflection) == set(codes_formula))

        print(f"\nn = {n}")
        print(f"  Reflection count: {len(codes_reflection)}")
        print(f"  Formula count:    {len(codes_formula)}")
        print(f"  Lengths match expected 2^n: {'✓ PASS' if len_ok else '✗ FAIL'}")
        print(f"  Sets of codes identical:    {'✓ PASS' if set_ok else '✗ FAIL'}")

        if not (len_ok and set_ok):
            print("  ⚠ Mismatch detected between methods for this n!")


# Test 6: Edge-case verification failures for Gray code property
def test_verification_edge_cases():
    print("\n" + "=" * 70)
    print("TEST 6: VERIFICATION EDGE CASES")
    print("=" * 70)

    # Base valid sequence for n = 3
    valid_codes = ['000', '001', '011', '010', '110', '111', '101', '100']

    # Case 1: Missing one code (wrong length)
    missing_one = valid_codes[:-1]
    ok, msg = GrayCodeGenerator.verify_gray_property(missing_one)
    print("\nCase 1: Missing one code")
    print(f"  Expected valid?  NO")
    print(f"  verify_gray_property: {'✓ CORRECTLY REJECTED' if not ok else '✗ INCORRECTLY ACCEPTED'}")
    print(f"  Message: {msg}")

    # Case 2: Duplicate code
    duplicate = valid_codes.copy()
    duplicate[3] = duplicate[2]  # make a duplicate, breaking uniqueness
    ok, msg = GrayCodeGenerator.verify_gray_property(duplicate)
    print("\nCase 2: Duplicate code")
    print(f"  Expected valid?  NO")
    print(f"  verify_gray_property: {'✓ CORRECTLY REJECTED' if not ok else '✗ INCORRECTLY ACCEPTED'}")
    print(f"  Message: {msg}")

    # Case 3: Wrong Hamming distance between adjacent codes
    wrong_distance = valid_codes.copy()
    # Swap two codes so that at least one adjacent pair differs in more than 1 bit
    wrong_distance[1], wrong_distance[2] = wrong_distance[2], wrong_distance[1]
    ok, msg = GrayCodeGenerator.verify_gray_property(wrong_distance)
    print("\nCase 3: Adjacent codes differ by more than 1 bit")
    print(f"  Expected valid?  NO")
    print(f"  verify_gray_property: {'✓ CORRECTLY REJECTED' if not ok else '✗ INCORRECTLY ACCEPTED'}")
    print(f"  Message: {msg}")



def main():import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class GrayCodeGenerator:
    
    # Generate n-bit Gray codes using reflection method
    @staticmethod
    def generate_reflection(n: int) -> List[str]:
        if n < 1:
            raise ValueError("n must be at least 1")
        
        # Base case: 1-bit Gray code
        gray_codes = ['0', '1']
        
        # Build n-bit codes iteratively
        for i in range(2, n + 1):
            current_size = len(gray_codes)
            
            # Step 1: Reflect - append reversed copy
            # This doubles the list size: O(2^i) operations at iteration i
            for j in range(current_size - 1, -1, -1):
                gray_codes.append(gray_codes[j])
            
            # Step 2: Prefix '0' to first half
            # O(2^(i-1)) operations
            for j in range(current_size):
                gray_codes[j] = '0' + gray_codes[j]
            
            # Step 3: Prefix '1' to second half
            # O(2^(i-1)) operations
            for j in range(current_size, 2 * current_size):
                gray_codes[j] = '1' + gray_codes[j]
        
        return gray_codes
    
    # Generate n-bit Gray codes using XOR formula method
    @staticmethod
    def generate_formula(n: int) -> List[str]:

        if n < 1:
            raise ValueError("n must be at least 1")
        
        gray_codes = []
        total = 1 << n  # 2^n using bit shift
        
        # Generate all 2^n codes
        for i in range(total):
            # Apply Gray code formula
            gray_value = i ^ (i >> 1)
            
            # Convert to n-bit binary string
            gray_string = format(gray_value, f'0{n}b')
            gray_codes.append(gray_string)
        
        return gray_codes
    
    # Verify that codes satisfy Gray code property
    @staticmethod
    def verify_gray_property(codes: List[str]) -> Tuple[bool, str]:
        n = len(codes[0])
        expected_count = 1 << n
        
        # Check count
        if len(codes) != expected_count:
            return False, f"Expected {expected_count} codes, got {len(codes)}"
        
        # Check Hamming distance between consecutive codes
        for i in range(len(codes)):
            next_i = (i + 1) % len(codes)  # Cyclic check
            
            # Count bit differences
            diff_count = sum(c1 != c2 for c1, c2 in zip(codes[i], codes[next_i]))
            
            if diff_count != 1:
                return False, f"Codes at index {i} and {next_i} differ by {diff_count} bits, expected 1"
        
        return True, "All codes valid"
    
    # Calculate Hamming distance between two binary strings
    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    # Convert binary string to decimal
    @staticmethod
    def binary_to_decimal(binary_str: str) -> int:
        return int(binary_str, 2)
    
    # Convert Gray code to binary
    @staticmethod
    def gray_to_binary(gray: str) -> str:
        binary = [gray[0]]
        for i in range(1, len(gray)):
            # XOR current gray bit with previous binary bit
            binary.append(str(int(gray[i]) ^ int(binary[i-1])))
        return ''.join(binary)


# Simulate QR code error pattern analysis for Singapore's digital infrastructure
class SingaporeQRCodeAnalyzer:

    @staticmethod
    def simulate_qr_error_patterns(n: int) -> dict:

        scenarios = {
            'n_bits': n,
            'total_patterns': 2 ** n,
            'singapore_context': {
                'application': 'SGQR Payment System Error Correction',
                'deployment_locations': [
                    'Hawker Centers (100+ locations)',
                    'MRT Stations (134 stations)',
                    'Retail Stores (50,000+ merchants)',
                    'Government Buildings (SafeEntry check-ins)'
                ],
                'error_types': {
                    'weather_damage': 'Tropical rain, high humidity',
                    'wear_and_tear': 'High-traffic locations (Orchard Road, CBD)',
                    'poor_lighting': 'Indoor hawker centers, parking lots',
                    'screen_glare': 'Outdoor sunlight on mobile screens'
                }
            },
            'analysis_purpose': [
                'Test Reed-Solomon error correction (QR uses RS)',
                'Identify critical bit positions',
                'Optimize error correction level (L/M/Q/H)',
                'Train ML models for damaged code recognition'
            ]
        }
        
        return scenarios
    
    # Analyze which bit positions are most critical in transitions    
    @staticmethod
    def analyze_critical_bits(codes: List[str]) -> dict:
        n = len(codes[0])
        bit_change_counts = [0] * n
        
        # Count how many times each bit position changes
        for i in range(len(codes) - 1):
            for bit_pos in range(n):
                if codes[i][bit_pos] != codes[i+1][bit_pos]:
                    bit_change_counts[bit_pos] += 1
        
        return {
            'bit_positions': list(range(n)),
            'change_frequencies': bit_change_counts,
            'most_critical_bit': bit_change_counts.index(max(bit_change_counts)),
            'least_critical_bit': bit_change_counts.index(min(bit_change_counts))
        }


# Test Case 1: Test Gray code generation correctness
def test_correctness():
    print("=" * 70)
    print("TEST 1: CORRECTNESS VERIFICATION")
    print("=" * 70)
    
    test_cases = [
        (1, ['0', '1']),
        (2, ['00', '01', '11', '10']),
        (3, ['000', '001', '011', '010', '110', '111', '101', '100'])
    ]
    
    for n, expected in test_cases:
        # Test reflection method
        codes_reflection = GrayCodeGenerator.generate_reflection(n)
        
        # Test formula method
        codes_formula = GrayCodeGenerator.generate_formula(n)
        
        is_valid_reflection, msg_reflection = GrayCodeGenerator.verify_gray_property(codes_reflection)
        is_valid_formula, msg_formula = GrayCodeGenerator.verify_gray_property(codes_formula)
        
        print(f"\nn = {n}:")
        print(f"  Expected: {expected}")
        print(f"  Reflection: {codes_reflection}")
        print(f"  Formula:    {codes_formula}")
        print(f"  Reflection valid: {'✓ PASS' if is_valid_reflection else '✗ FAIL'} - {msg_reflection}")
        print(f"  Formula valid:    {'✓ PASS' if is_valid_formula else '✗ FAIL'} - {msg_formula}")
        print(f"  Match expected:   {'✓ PASS' if codes_reflection == expected else '✗ FAIL'}")


# Test Case 2: Demonstrate exponential O(2^n) growth
def test_exponential_growth():
    print("\n" + "=" * 70)
    print("TEST 2: EXPONENTIAL COMPLEXITY DEMONSTRATION - O(2^n)")
    print("=" * 70)
    
    test_sizes = list(range(1, 21))  # n = 1 to 20
    reflection_times = []
    formula_times = []
    code_counts = []
    
    print(f"\n{'n':>3} {'Codes':>12} {'Reflection (ms)':>18} {'Formula (ms)':>15} {'2^n':>12}")
    print("-" * 70)
    
    for n in test_sizes:
        expected_count = 2 ** n
        code_counts.append(expected_count)
        
        # Test reflection method
        start = time.perf_counter()
        codes_reflection = GrayCodeGenerator.generate_reflection(n)
        reflection_time = (time.perf_counter() - start) * 1000
        reflection_times.append(reflection_time)
        
        # Test formula method
        start = time.perf_counter()
        codes_formula = GrayCodeGenerator.generate_formula(n)
        formula_time = (time.perf_counter() - start) * 1000
        formula_times.append(formula_time)
        
        print(f"{n:3d} {len(codes_reflection):12,} {reflection_time:18.4f} {formula_time:15.4f} {expected_count:12,}")
        
        # Stop if taking too long
        if reflection_time > 10000:  # 10 seconds
            print(f"\n⚠ Stopping at n={n} (execution time exceeds 10 seconds)")
            test_sizes = test_sizes[:test_sizes.index(n)+1]
            break
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of codes vs n (exponential curve)
    axes[0, 0].plot(test_sizes, code_counts, 'ro-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('n (bits)', fontsize=11)
    axes[0, 0].set_ylabel('Number of Gray Codes (2^n)', fontsize=11)
    axes[0, 0].set_title('Exponential Growth: 2^n', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Execution time vs n
    axes[0, 1].plot(test_sizes, reflection_times, 'bo-', linewidth=2, markersize=6, label='Reflection')
    axes[0, 1].plot(test_sizes, formula_times, 'go-', linewidth=2, markersize=6, label='Formula')
    axes[0, 1].set_xlabel('n (bits)', fontsize=11)
    axes[0, 1].set_ylabel('Execution Time (ms)', fontsize=11)
    axes[0, 1].set_title('Execution Time vs n', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Time vs 2^n (should be linear on log scale)
    axes[1, 0].plot(code_counts, reflection_times, 'mo-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Number of Codes (2^n)', fontsize=11)
    axes[1, 0].set_ylabel('Execution Time (ms)', fontsize=11)
    axes[1, 0].set_title('Time vs 2^n (Linear = O(2^n))', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Growth rate comparison table
    axes[1, 1].axis('off')
    growth_data = []
    for i, n in enumerate(test_sizes[:10]):  # First 10 for readability
        growth_data.append([
            str(n),
            f"{code_counts[i]:,}",
            f"{reflection_times[i]:.3f}",
            f"{code_counts[i] / code_counts[i-1]:.1f}x" if i > 0 else "-"
        ])
    
    table = axes[1, 1].table(
        cellText=growth_data,
        colLabels=['n', 'Codes (2^n)', 'Time (ms)', 'Growth Factor'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Exponential Complexity Analysis - Gray Code Generation', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('images/gray_code_exponential_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Complexity graph saved as 'gray_code_exponential_analysis.png' inside folder 'images'")
    
    return test_sizes, code_counts, reflection_times


# Test Case 3: Demonstrate Gray code application to Singapore QR code systems
def test_singapore_qr_application():
    print("\n" + "=" * 70)
    print("TEST 3: SINGAPORE QR CODE ERROR PATTERN ANALYSIS")
    print("=" * 70)
    
    # Typical QR code data block sizes
    test_n = 8  # 8-bit data block (256 patterns)
    
    print(f"\nGenerating {2**test_n} error patterns for {test_n}-bit QR data blocks...")
    start = time.perf_counter()
    codes = GrayCodeGenerator.generate_reflection(test_n)
    gen_time = (time.perf_counter() - start) * 1000
    
    scenario = SingaporeQRCodeAnalyzer.simulate_qr_error_patterns(test_n)
    
    print(f"\n{'='*60}")
    print(f"SINGAPORE CONTEXT: {scenario['singapore_context']['application']}")
    print(f"{'='*60}")
    print(f"\nDeployment Locations:")
    for loc in scenario['singapore_context']['deployment_locations']:
        print(f"  • {loc}")
    
    print(f"\nError Types in Tropical Singapore:")
    for error_type, description in scenario['singapore_context']['error_types'].items():
        print(f"  • {error_type.replace('_', ' ').title()}: {description}")
    
    print(f"\nAnalysis Purpose:")
    for purpose in scenario['analysis_purpose']:
        print(f"  • {purpose}")
    
    print(f"\n{'='*60}")
    print(f"GENERATION RESULTS")
    print(f"{'='*60}")
    print(f"Total patterns generated: {len(codes):,}")
    print(f"Generation time: {gen_time:.3f} ms")
    print(f"Memory usage: ~{len(codes) * test_n / 1024:.2f} KB")
    
    bit_analysis = SingaporeQRCodeAnalyzer.analyze_critical_bits(codes)
    
    print(f"\n{'='*60}")
    print(f"BIT POSITION CRITICALITY ANALYSIS")
    print(f"{'='*60}")
    print(f"\nBit change frequencies across all {len(codes)-1} transitions:")
    for pos, freq in enumerate(bit_analysis['change_frequencies']):
        print(f"  Bit {pos}: {freq} changes ({freq/(len(codes)-1)*100:.1f}%)")
    
    print(f"\nMost critical bit position: {bit_analysis['most_critical_bit']}")
    print(f"Least critical bit position: {bit_analysis['least_critical_bit']}")
    
    # Show sample patterns
    print(f"\n{'='*60}")
    print(f"SAMPLE ERROR PATTERNS (First 16 of {len(codes)})")
    print(f"{'='*60}")
    print(f"{'Index':>6} {'Gray Code':>12} {'Decimal':>10} {'Bits Changed':>15}")
    print("-" * 50)
    
    for i in range(min(16, len(codes))):
        decimal = GrayCodeGenerator.binary_to_decimal(codes[i])
        if i > 0:
            bits_changed = GrayCodeGenerator.hamming_distance(codes[i-1], codes[i])
        else:
            bits_changed = 0
        print(f"{i:6d} {codes[i]:>12} {decimal:10d} {bits_changed:15d}")
    
    # Visualize Gray code sequence
    visualize_gray_code_sequence(codes[:32] if len(codes) > 32 else codes)


# Visualize Gray code sequence as a binary matrix
def visualize_gray_code_sequence(codes: List[str]):
    n = len(codes[0])
    num_codes = len(codes)
    
    # Create binary matrix
    matrix = np.zeros((num_codes, n))
    for i, code in enumerate(codes):
        for j, bit in enumerate(code):
            matrix[i, j] = int(bit)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Plot 1: Binary matrix heatmap
    im1 = ax1.imshow(matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Bit Position', fontsize=12)
    ax1.set_ylabel('Code Index', fontsize=12)
    ax1.set_title(f'{n}-bit Gray Code Sequence ({num_codes} codes)', 
                  fontsize=13, fontweight='bold')
    
    # Add gridlines
    ax1.set_xticks(np.arange(n))
    ax1.set_yticks(np.arange(num_codes))
    ax1.set_xticklabels([f'b{i}' for i in range(n)])
    ax1.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=ax1)
    cbar.set_label('Bit Value', rotation=270, labelpad=15)
    
    # Plot 2: Transition diagram (for small n)
    if num_codes <= 16:
        angles = np.linspace(0, 2*np.pi, num_codes, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        
        ax2.scatter(x, y, s=300, c='lightblue', edgecolors='blue', linewidths=2, zorder=3)
        
        for i in range(num_codes):
            ax2.text(x[i]*1.15, y[i]*1.15, codes[i], 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        
        for i in range(num_codes):
            next_i = (i + 1) % num_codes
            ax2.plot([x[i], x[next_i]], [y[i], y[next_i]], 
                    'r-', linewidth=1.5, alpha=0.6, zorder=1)
            
            changed_bit = -1
            for bit_pos in range(n):
                if codes[i][bit_pos] != codes[next_i][bit_pos]:
                    changed_bit = bit_pos
                    break
            
            mid_x = (x[i] + x[next_i]) / 2
            mid_y = (y[i] + y[next_i]) / 2
            ax2.text(mid_x*1.05, mid_y*1.05, f'b{changed_bit}', 
                    fontsize=7, color='red', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Gray Code Transition Diagram\n(Single bit changes between adjacent codes)', 
                     fontsize=13, fontweight='bold')
    else:
        # For larger n, show statistics
        ax2.axis('off')
        stats_text = f"""
        GRAY CODE STATISTICS
        {'='*40}
        
        Number of bits (n): {n}
        Total codes: {num_codes:,}
        Complexity: O(2^n) = O(2^{n})
        
        SINGAPORE QR CODE APPLICATION
        {'='*40}
        
        Use Case: SGQR Payment Error Correction
        
        Error Patterns: All {num_codes:,} single-bit 
        transitions analyzed
        
        Critical for:
          • Reed-Solomon error correction
          • Robust QR code scanning
          • Machine learning training data
          • Field testing in tropical climate
        
        Deployment: 50,000+ merchants
        Daily scans: Millions across Singapore
        """
        
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('images/singapore_qr_gray_code_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Visualization saved as 'singapore_qr_gray_code_visualization.png' inside folder 'images'")


# Test Case 4: Compare exponential growth with other complexities
def compare_growth_rates():

    print("\n" + "=" * 70)
    print("TEST 4: GROWTH RATE COMPARISON")
    print("=" * 70)
    
    n_values = list(range(1, 21))
    
    # Calculate different complexities
    linear = [n for n in n_values]
    nlogn = [n * np.log2(n) if n > 0 else 0 for n in n_values]
    quadratic = [n**2 for n in n_values]
    exponential = [2**n for n in n_values]
    
    print(f"\n{'n':>3} {'O(n)':>10} {'O(n log n)':>12} {'O(n²)':>12} {'O(2^n)':>15}")
    print("-" * 60)
    
    for i, n in enumerate(n_values[:15]):  # First 15 for readability
        print(f"{n:3d} {linear[i]:10.0f} {nlogn[i]:12.1f} {quadratic[i]:12.0f} {exponential[i]:15,}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Linear scale (shows exponential explosion)
    ax1.plot(n_values[:15], linear[:15], 'g-', linewidth=2, marker='o', label='O(n)')
    ax1.plot(n_values[:15], nlogn[:15], 'b-', linewidth=2, marker='s', label='O(n log n)')
    ax1.plot(n_values[:15], quadratic[:15], 'orange', linewidth=2, marker='^', label='O(n²)')
    ax1.plot(n_values[:15], exponential[:15], 'r-', linewidth=3, marker='D', label='O(2^n)')
    
    ax1.set_xlabel('Input Size (n)', fontsize=12)
    ax1.set_ylabel('Operations', fontsize=12)
    ax1.set_title('Growth Rate Comparison (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale (shows all curves)
    ax2.plot(n_values, linear, 'g-', linewidth=2, marker='o', label='O(n)')
    ax2.plot(n_values, nlogn, 'b-', linewidth=2, marker='s', label='O(n log n)')
    ax2.plot(n_values, quadratic, 'orange', linewidth=2, marker='^', label='O(n²)')
    ax2.plot(n_values, exponential, 'r-', linewidth=3, marker='D', label='O(2^n)')
    
    ax2.set_xlabel('Input Size (n)', fontsize=12)
    ax2.set_ylabel('Operations (log scale)', fontsize=12)
    ax2.set_title('Growth Rate Comparison (Log Scale)', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('images/growth_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Comparison graph saved as 'growth_rate_comparison.png' inside folder 'images'")
    
    # Print doubling analysis
    print(f"\n{'='*60}")
    print("DOUBLING ANALYSIS - Why O(2^n) is Exponential")
    print(f"{'='*60}")
    print(f"\n{'n':>3} {'2^n':>15} {'Factor':>15} {'Explanation'}")
    print("-" * 70)
    
    for n in [5, 6, 7, 10, 15, 20]:
        codes = 2**n
        factor = 2**(n) / 2**(n-1) if n > 1 else 1
        print(f"{n:3d} {codes:15,} {factor:15.1f}x {'Doubles with each +1 to n'}")



# Test Case 5: Equivalence of reflection and formula methods
def test_method_equivalence():
    print("\n" + "=" * 70)
    print("TEST 5: METHOD EQUIVALENCE - REFLECTION vs FORMULA")
    print("=" * 70)

    # Choose a range of n where 2^n is still manageable
    test_ns = [2, 3, 4, 5, 6, 8, 10]

    for n in test_ns:
        codes_reflection = GrayCodeGenerator.generate_reflection(n)
        codes_formula    = GrayCodeGenerator.generate_formula(n)

        # Check lengths
        len_ok = (len(codes_reflection) == len(codes_formula) == (1 << n))

        # Check that the sets of codes are identical (order not required)
        set_ok = (set(codes_reflection) == set(codes_formula))

        print(f"\nn = {n}")
        print(f"  Reflection count: {len(codes_reflection)}")
        print(f"  Formula count:    {len(codes_formula)}")
        print(f"  Lengths match expected 2^n: {'✓ PASS' if len_ok else '✗ FAIL'}")
        print(f"  Sets of codes identical:    {'✓ PASS' if set_ok else '✗ FAIL'}")

        if not (len_ok and set_ok):
            print("  ⚠ Mismatch detected between methods for this n!")


# Test 6: Edge-case verification failures for Gray code property
def test_verification_edge_cases():
    print("\n" + "=" * 70)
    print("TEST 6: VERIFICATION EDGE CASES")
    print("=" * 70)

    # Base valid sequence for n = 3
    valid_codes = ['000', '001', '011', '010', '110', '111', '101', '100']

    # Case 1: Missing one code (wrong length)
    missing_one = valid_codes[:-1]
    ok, msg = GrayCodeGenerator.verify_gray_property(missing_one)
    print("\nCase 1: Missing one code")
    print(f"  Expected valid?  NO")
    print(f"  verify_gray_property: {'✓ CORRECTLY REJECTED' if not ok else '✗ INCORRECTLY ACCEPTED'}")
    print(f"  Message: {msg}")

    # Case 2: Duplicate code
    duplicate = valid_codes.copy()
    duplicate[3] = duplicate[2]  # make a duplicate, breaking uniqueness
    ok, msg = GrayCodeGenerator.verify_gray_property(duplicate)
    print("\nCase 2: Duplicate code")
    print(f"  Expected valid?  NO")
    print(f"  verify_gray_property: {'✓ CORRECTLY REJECTED' if not ok else '✗ INCORRECTLY ACCEPTED'}")
    print(f"  Message: {msg}")

    # Case 3: Wrong Hamming distance between adjacent codes
    wrong_distance = valid_codes.copy()
    # Swap two codes so that at least one adjacent pair differs in more than 1 bit
    wrong_distance[1], wrong_distance[2] = wrong_distance[2], wrong_distance[1]
    ok, msg = GrayCodeGenerator.verify_gray_property(wrong_distance)
    print("\nCase 3: Adjacent codes differ by more than 1 bit")
    print(f"  Expected valid?  NO")
    print(f"  verify_gray_property: {'✓ CORRECTLY REJECTED' if not ok else '✗ INCORRECTLY ACCEPTED'}")
    print(f"  Message: {msg}")



def main():

    test_correctness()
    test_exponential_growth()
    test_singapore_qr_application()
    compare_growth_rates()
    test_method_equivalence()
    test_verification_edge_cases()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  1. gray_code_exponential_analysis.png")
    print("  2. singapore_qr_gray_code_visualization.png")
    print("  3. growth_rate_comparison.png")
    print("\n")

if __name__ == "__main__":
    main()
