#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static void radix_sort_u32(uint32_t *arr, size_t n) {
    if (n <= 1) return;

    uint32_t *tmp = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!tmp) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    size_t count[256];

    for (unsigned pass = 0; pass < 4; pass++) {
        memset(count, 0, sizeof(count));
        unsigned shift = pass * 8;

        for (size_t i = 0; i < n; i++) {
            uint8_t key = (uint8_t)((arr[i] >> shift) & 0xFFu);
            count[key]++;
        }

        size_t sum = 0;
        for (size_t b = 0; b < 256; b++) {
            size_t c = count[b];
            count[b] = sum;
            sum += c;
        }

        for (size_t i = 0; i < n; i++) {
            uint8_t key = (uint8_t)((arr[i] >> shift) & 0xFFu);
            tmp[count[key]++] = arr[i];
        }

        memcpy(arr, tmp, n * sizeof(uint32_t));
    }

    free(tmp);
}

static int is_sorted_u32(const uint32_t *arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (arr[i - 1] > arr[i]) return 0;
    }
    return 1;
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *xs, int k) {
    qsort(xs, (size_t)k, sizeof(double), cmp_double);
    if (k % 2 == 1) return xs[k / 2];
    return 0.5 * (xs[k/2 - 1] + xs[k/2]);
}

static void fill_random_u32(uint32_t *arr, size_t n, uint32_t seed0) {
    uint32_t seed = seed0;
    for (size_t i = 0; i < n; i++) {
        arr[i] = xorshift32(&seed);
    }
}

static void print_test_header(void) {
    printf("\n");
    printf("===================================================================================\n");
    printf("                               COMPREHENSIVE TEST                               \n");
    printf("===================================================================================\n\n");
}

static void print_test_result(const char *test_name, int passed, const char *details) {
    printf("%-40s [%s] %s\n", test_name, passed ? "PASS" : "FAIL", details);
    if (!passed) {
        fprintf(stderr, "ERROR: Test '%s' failed!\n", test_name);
        exit(1);
    }
}

static void test_empty_array(void) {
    uint32_t *arr = NULL;
    radix_sort_u32(arr, 0);
    print_test_result("Empty Array (n=0)", 1, "No operations performed");
}

static void test_single_element(void) {
    uint32_t arr[1] = {42};
    radix_sort_u32(arr, 1);
    int passed = (arr[0] == 42 && is_sorted_u32(arr, 1));
    print_test_result("Single Element (n=1)", passed, "Trivially sorted");
}

static void test_already_sorted(void) {
    uint32_t arr[5] = {1, 2, 3, 4, 5};
    radix_sort_u32(arr, 5);
    int passed = (arr[0] == 1 && arr[1] == 2 && arr[2] == 3 && 
                  arr[3] == 4 && arr[4] == 5 && is_sorted_u32(arr, 5));
    print_test_result("Already Sorted Array", passed, "4 passes still performed");
}

static void test_reverse_sorted(void) {
    uint32_t arr[5] = {5, 4, 3, 2, 1};
    radix_sort_u32(arr, 5);
    int passed = (arr[0] == 1 && arr[1] == 2 && arr[2] == 3 && 
                  arr[3] == 4 && arr[4] == 5 && is_sorted_u32(arr, 5));
    print_test_result("Reverse Sorted Array", passed, "Worst case for some algorithms");
}

static void test_duplicate_values(void) {
    uint32_t arr[5] = {3, 1, 3, 2, 1};
    uint32_t expected[5] = {1, 1, 2, 3, 3};
    radix_sort_u32(arr, 5);
    int passed = 1;
    for (int i = 0; i < 5; i++) {
        if (arr[i] != expected[i]) {
            passed = 0;
            break;
        }
    }
    passed = passed && is_sorted_u32(arr, 5);
    print_test_result("Duplicate Values", passed, "Stable sorting maintained");
}

static void test_maximum_values(void) {
    uint32_t arr[5] = {0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFF0, 0xFFFFFFFF};
    radix_sort_u32(arr, 5);
    int passed = (arr[0] == 0xFFFFFFF0 && arr[1] == 0xFFFFFFFE && 
                  arr[4] == 0xFFFFFFFF && is_sorted_u32(arr, 5));
    print_test_result("Maximum Values (0xFFFFFFFF)", passed, "MSB pass handles 0xFF");
}

static void test_minimum_value(void) {
    uint32_t arr[5] = {0, 5, 0, 3, 0};
    radix_sort_u32(arr, 5);
    int passed = (arr[0] == 0 && arr[1] == 0 && arr[2] == 0 && 
                  arr[3] == 3 && arr[4] == 5 && is_sorted_u32(arr, 5));
    print_test_result("Minimum Value (0x00000000)", passed, "Multiple zeros sorted stably");
}

static void test_mixed_boundary_values(void) {
    uint32_t arr[6] = {0xFFFFFFFF, 0, 0x80000000, 1, 0x7FFFFFFF, 0xFFFFFFFE};
    radix_sort_u32(arr, 6);
    int passed = (arr[0] == 0 && arr[1] == 1 && arr[2] == 0x7FFFFFFF && 
                  arr[3] == 0x80000000 && arr[4] == 0xFFFFFFFE && 
                  arr[5] == 0xFFFFFFFF && is_sorted_u32(arr, 6));
    print_test_result("Mixed Boundary Values", passed, "Full range tested");
}

static void test_small_random(void) {
    const size_t n = 1000;
    uint32_t *arr = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!arr) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    
    fill_random_u32(arr, n, 123456789);
    radix_sort_u32(arr, n);
    int passed = is_sorted_u32(arr, n);
    
    free(arr);
    print_test_result("Small Random (n=1000)", passed, "Random data sorted correctly");
}

static void run_all_tests(void) {
    print_test_header();
    
    test_empty_array();
    test_single_element();
    test_already_sorted();
    test_reverse_sorted();
    test_duplicate_values();
    test_maximum_values();
    test_minimum_value();
    test_mixed_boundary_values();
    test_small_random();
    
    printf("\n");
    printf("===================================================================================\n");
    printf("                      ALL TESTS PASSED SUCCESSFULLY                               \n");
    printf("===================================================================================\n\n");
}

// ============================================================================
// PERFORMANCE BENCHMARKS (ORIGINAL CODE)
// ============================================================================

int main(void) {
    // Run comprehensive test suite first
    run_all_tests();
    
    // Continue with performance benchmarks
    printf("\n");
    printf("===================================================================================\n");
    printf("                         PERFORMANCE BENCHMARKS                                    \n");
    printf("===================================================================================\n\n");
    
    const size_t sizes[] = { 10000, 100000, 500000, 1000000, 2000000 };
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    const int TRIALS = 7;     
    const int WARMUP = 1; 

    printf("%-10s | %-10s | %-10s | %-12s | %-12s | %-14s\n",
           "n", "Radix(s)", "qsort(s)", "Radix/n", "qsort/n", "qsort/(nlog2n)");
    printf("-----------------------------------------------------------------------------------------\n");

    for (size_t si = 0; si < num_sizes; si++) {
        size_t n = sizes[si];

        uint32_t *data = (uint32_t *)malloc(n * sizeof(uint32_t));
        uint32_t *copy = (uint32_t *)malloc(n * sizeof(uint32_t));
        if (!data || !copy) {
            fprintf(stderr, "malloc failed for n=%zu\n", n);
            return 1;
        }

        double radix_times[TRIALS];
        double qsort_times[TRIALS];

        for (int t = 0; t < TRIALS + WARMUP; t++) {
            uint32_t seed = 123456789u + (uint32_t)(t * 1013904223u);

            fill_random_u32(data, n, seed);
            memcpy(copy, data, n * sizeof(uint32_t));

            // Radix
            uint64_t t0 = now_ns();
            radix_sort_u32(data, n);
            uint64_t t1 = now_ns();
            double radix_s = (double)(t1 - t0) / 1e9;

            if (!is_sorted_u32(data, n)) {
                fprintf(stderr, "Radix sort failed for n=%zu\n", n);
                return 1;
            }

            // qsort
            t0 = now_ns();
            qsort(copy, n, sizeof(uint32_t), cmp_uint32);
            t1 = now_ns();
            double qsort_s = (double)(t1 - t0) / 1e9;

            if (!is_sorted_u32(copy, n)) {
                fprintf(stderr, "qsort failed for n=%zu\n", n);
                return 1;
            }

            if (t >= WARMUP) {
                int idx = t - WARMUP;
                radix_times[idx] = radix_s;
                qsort_times[idx] = qsort_s;
            }
        }

        double radix_med = median(radix_times, TRIALS);
        double qsort_med = median(qsort_times, TRIALS);

        double log2n = log((double)n) / log(2.0);
        double q_over_nlogn = qsort_med / ((double)n * log2n);

        printf("%-10zu | %-10.6f | %-10.6f | %-12.3e | %-12.3e | %-14.3e\n",
               n,
               radix_med,
               qsort_med,
               radix_med / (double)n,
               qsort_med / (double)n,
               q_over_nlogn);

        free(data);
        free(copy);
    }

    printf("\n");
    printf("===================================================================================\n");
    printf("                      BENCHMARKING COMPLETED SUCCESSFULLY                          \n");
    printf("===================================================================================\n\n");

    return 0;
}
