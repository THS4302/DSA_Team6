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

int main(void) {
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

        // keep input distribution consistent across trials but change seed each trial.
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

            // Record after warm-up
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

    return 0;
}