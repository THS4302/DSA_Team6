import cmath 
import math

# Reorder the input into bit-revered index order
def bit_reverse_copy(a):
    n = len(a)
    result = [0] * n
    bits = int(math.log2(n))  # Number of bits needed to represent indices

    for i in range(n):
        # Reverse the binary bits of index i
        rev = int('{:0{width}b}'.format(i, width=bits)[::-1], 2)
        result[rev] = a[i]  # Place value at its bit-reversed position

    return result

# Compute the FFT of the input array using iterative (loop based) method
def iterative_fft(a):
    n = len(a)
    A = bit_reverse_copy(a)  # Reorder inputs using bit reversal
    stages = int(math.log2(n))  # Total number of stages

    # Outer loop over each FFT stage
    for s in range(1, stages + 1):
        m = 2 ** s  # Size of sub-groups to process (e.g. 2, 4, 8, ...)
        wm = cmath.exp(-2j * cmath.pi / m)  # "Twiddle factor" (complex root of unity)

        # Process all groups of size m across the array
        for k in range(0, n, m):
            w = 1  # Initialize twiddle multiplier
            for j in range(m // 2):
                # Perform the butterfly operation on each pair
                t = w * A[k + j + m // 2]  # Twiddled value
                u = A[k + j]               # Top element of the butterfly

                # Update the two elements in place
                A[k + j] = u + t           # Even index result
                A[k + j + m // 2] = u - t  # Odd index result

                w *= wm  # Advance twiddle multiplier

    return A  

# example usage
if __name__ == "__main__":
    input_signal = [1, 1, 1, 1, 0, 0, 0, 0]  # Real-valued time-domain signal
    fft_result = iterative_fft([complex(x) for x in input_signal])  

    print("Input signal:")
    print(input_signal)

    print("\nFFT Output (Frequency domain):")
    for val in fft_result:
        print(f"{val:.4f}")
