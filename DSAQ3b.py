import time
import random

def counting_sort_for_radix(arr, exp):
    """
    Sort arr[] based on the digit represented by exp.
    """
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # Base 10 for decimal numbers

    # Store count of occurrences in count[]
    for i in range(n):
        index = (arr[i] // exp)
        count[index % 10] += 1

    # Change count[i] so that it contains the actual position in output[]
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array (traverse backwards for stability)
    i = n - 1
    while i >= 0:
        index = (arr[i] // exp)
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    # Copy the output array to arr[]
    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    """
    Main Radix Sort function.
    """
    if not arr:
        return
    max1 = max(arr)  # Find max number to know number of digits
    exp = 1
    while max1 // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10

# BENCHMARKING
def run_benchmark():
    # Test Data - Arrays of increasing size up to 1 million
    sizes = [10000, 100000, 500000, 1000000]
    
    print(f"{'Input Size (n)':<15} | {'Radix Sort (s)':<15} | {'O(n lg n) Sort (s)':<20}")
    print("-" * 55)

    for n in sizes:
        # Generate random test data (int between 0 and 10 million)
        data = [random.randint(0, 10000000) for _ in range(n)]
        
        # Create copy for the comparison sort
        data_copy = data.copy()

        # Measure Radix Sort (Linear Time)
        start_time = time.time()
        radix_sort(data)
        radix_time = time.time() - start_time

        # Measure Timsort (built-in O(n lg n) sort)
        start_time = time.time()
        data_copy.sort()
        builtin_time = time.time() - start_time

        print(f"{n:<15} | {radix_time:<15.4f} | {builtin_time:<20.4f}")

if __name__ == "__main__":
    run_benchmark()