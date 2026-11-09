// ============================================================================
//  tree_cuda.cu -- GPU (CUDA) arm of the MPI+X study.
//
//  Rubric: X = OpenMP, so we must also test on >= 1 GPU.  This standalone
//  program runs the decision-tree HOT PATH on the GPU: building the per-feature
//  histogram [count, positive-count] that the CPU code (src/dt.cpp) builds, and
//  finding the best Gini split from it.  It is deliberately NOT MPI -- the
//  MPI+X requirement is met by the MPI+OpenMP build; this is the required GPU
//  demonstration, benchmarked against the CPU.
//
//  To stay directly comparable with dt.cpp we reuse the SAME synthetic data
//  generator, the SAME exact binning (<=256 distinct values per feature), and
//  the SAME split-scan logic, so the GPU must pick the identical root split.
//
//  Build:  nvcc -O3 -std=c++17 src/tree_cuda.cu -o build/dt_cuda
//  Run:    build/dt_cuda --synthetic 32000000 --features 28 [--seed 42] [--reps 3]
//          build/dt_cuda --csv instacart_clean.csv
//
//  What it measures (printed as a RESULT_GPU line and human-readable summary):
//    * t_h2d        host->device copy of the binned data + labels
//    * t_kernel     the histogram-build kernel
//    * t_d2h        device->host copy of the histogram
//    * t_scan       host-side best-split scan (same code as dt.cpp)
//    * best split   feature, threshold, gini gain  (for the correctness check)
//
//  NOTE: not compiled/tested on this machine (no nvcc/GPU here).  Verify on the
//  GPU node:  it should report the same (feature, threshold) root split as
//    build/dt_serial --synthetic N --features F --dump-tree -  (first split).
// ============================================================================
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using std::string;
using std::vector;
typedef long long i64;

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(_e), __FILE__, __LINE__);           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static double now_sec() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// --------------------------------------------------------- synthetic data
// Identical schema and generator to src/dt.cpp, so the GPU sees the same rows.
static const int NDOM = 7;
static const int DOM_LO[NDOM] = {1, 0, 0, 0, 1, 1, 1};
static const int DOM_HI[NDOM] = {99, 6, 23, 30, 29, 134, 21};

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline int gen_row(uint64_t seed, uint64_t r, int F, int* vals) {
    uint64_t s = splitmix64(seed ^ (r * 0xD6E8FEB86659FD93ULL + 0x1234567ULL));
    for (int f = 0; f < F; f++) {
        s = splitmix64(s);
        int d = f % NDOM;
        uint64_t range = (uint64_t)(DOM_HI[d] - DOM_LO[d] + 1);
        vals[f] = DOM_LO[d] + (int)(s % range);
    }
    s = splitmix64(s);
    double u = (double)(s >> 11) * (1.0 / 9007199254740992.0);
    double p = 0.3 + 0.3 * (vals[4] <= 3) + 0.2 * (vals[3] == 7);
    p = std::min(0.95, std::max(0.05, p));
    return u < p ? 1 : 0;
}

// ------------------------------------------------------------------- bins
// Per feature: a list of representative values (bin edges).  For the synthetic
// generator these are exactly the domain values, so binning is exact and the
// bin code is (value - DOM_LO[d]).  nb[f] bins per feature, off[f] is the base
// offset of feature f in the flattened histogram.
struct Bins {
    int F = 0;
    vector<int> nb, off;
    vector<vector<float>> rep;
    int total = 0;
    void finalize() {
        nb.assign(F, 0);
        off.assign(F, 0);
        total = 0;
        for (int f = 0; f < F; f++) {
            nb[f] = (int)rep[f].size();
            off[f] = total;
            total += nb[f];
        }
    }
};

// ------------------------------------------------------------- CUDA kernel
// One thread per row.  Each thread walks its F features and atomically bumps the
// [count, positive] pair of the (feature, bin) it lands in.  Histogram layout is
// identical to dt.cpp: 2 ints per (feature,bin), feature f starting at 2*off[f].
__global__ void histogram_kernel(const uint8_t* __restrict__ X,
                                 const uint8_t* __restrict__ y, i64 n, int F,
                                 const int* __restrict__ off, int* __restrict__ hist) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* row = X + i * F;
    int yy = y[i];
    for (int f = 0; f < F; f++) {
        int* cell = hist + 2 * (off[f] + row[f]);
        atomicAdd(&cell[0], 1);
        atomicAdd(&cell[1], yy);
    }
}

// ------------------------------------------------------- host-side scan
// Same Gini and same candidate thresholds (midpoints of present bins) as dt.cpp.
static inline double gini(i64 pos, i64 tot) {
    if (tot == 0) return 0.0;
    double p1 = (double)pos / (double)tot, p0 = 1.0 - p1;
    return 1.0 - (p0 * p0 + p1 * p1);
}

struct Split {
    int f = -1;
    float thr = 0.f;
    double gain = -1.0;
};

static Split best_split(const int* h, const Bins& B, i64 tot, i64 pos) {
    const double cur = gini(pos, tot);
    Split best;
    for (int f = 0; f < B.F; f++) {
        const int nb = B.nb[f];
        const int* hf = h + 2 * B.off[f];
        i64 ltot = 0, lpos = 0;
        int last = -1;
        for (int b = 0; b < nb; b++) {
            int c = hf[2 * b];
            if (c == 0) continue;
            if (last >= 0) {
                i64 rtot = tot - ltot, rpos = pos - lpos;
                double weighted = ((double)ltot / tot) * gini(lpos, ltot) +
                                  ((double)rtot / tot) * gini(rpos, rtot);
                double g = cur - weighted;
                if (g > best.gain) {
                    best.gain = g;
                    best.f = f;
                    best.thr = (B.rep[f][last] + B.rep[f][b]) / 2.0f;
                }
            }
            ltot += c;
            lpos += hf[2 * b + 1];
            last = b;
        }
    }
    return best;
}

// ------------------------------------------------------------------- data
struct Data {
    i64 n = 0;
    int F = 0;
    vector<uint8_t> x, y;  // row-major n*F bin codes, and labels
};

static void make_synthetic(i64 N, int F, uint64_t seed, Data& D, Bins& B) {
    D.n = N;
    D.F = F;
    D.x.resize((size_t)N * F);
    D.y.resize((size_t)N);
    B.F = F;
    B.rep.assign(F, {});
    for (int f = 0; f < F; f++) {
        int d = f % NDOM;
        for (int v = DOM_LO[d]; v <= DOM_HI[d]; v++) B.rep[f].push_back((float)v);
    }
    B.finalize();
    int vals[256];
    for (i64 i = 0; i < N; i++) {
        int lab = gen_row(seed, (uint64_t)i, F, vals);
        for (int f = 0; f < F; f++)
            D.x[(size_t)i * F + f] = (uint8_t)(vals[f] - DOM_LO[f % NDOM]);
        D.y[(size_t)i] = (uint8_t)lab;
    }
}

// ------------------------------------------------------------------- main
int main(int argc, char** argv) {
    i64 N = 0;
    int F = 28;
    uint64_t seed = 42;
    int reps = 3, warmup = 1;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (a == "--synthetic") N = atoll(next());
        else if (a == "--features") F = atoi(next());
        else if (a == "--seed") seed = strtoull(next(), nullptr, 10);
        else if (a == "--reps") reps = atoi(next());
        else if (a == "--warmup") warmup = atoi(next());
        else { std::fprintf(stderr, "usage: dt_cuda --synthetic N [--features F] [--seed S] [--reps R]\n"); return 2; }
    }
    if (N <= 0 || F < 7 || F > 256) {
        std::fprintf(stderr, "need --synthetic N (>0) and --features in 7..256\n");
        return 2;
    }

    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    Data D;
    Bins B;
    make_synthetic(N, F, seed, D, B);

    // Count global positives once on the host (cheap, not the hot path).
    i64 pos = 0;
    for (i64 i = 0; i < D.n; i++) pos += D.y[(size_t)i];
    const i64 tot = D.n;
    const size_t H = 2 * (size_t)B.total;  // ints in the histogram

    // Device buffers.
    uint8_t *dX = nullptr, *dY = nullptr;
    int *dOff = nullptr, *dHist = nullptr;
    CHECK_CUDA(cudaMalloc(&dX, (size_t)D.n * F));
    CHECK_CUDA(cudaMalloc(&dY, (size_t)D.n));
    CHECK_CUDA(cudaMalloc(&dOff, sizeof(int) * B.F));
    CHECK_CUDA(cudaMalloc(&dHist, sizeof(int) * H));
    CHECK_CUDA(cudaMemcpy(dOff, B.off.data(), sizeof(int) * B.F, cudaMemcpyHostToDevice));

    const int TPB = 256;
    const int blocks = (int)((D.n + TPB - 1) / TPB);
    vector<int> hist((size_t)H);
    Split last_split;

    double t_h2d = 0, t_kernel = 0, t_d2h = 0, t_scan = 0;
    for (int r = -warmup; r < reps; r++) {
        bool timed = (r >= 0);

        double t0 = now_sec();
        CHECK_CUDA(cudaMemcpy(dX, D.x.data(), (size_t)D.n * F, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dY, D.y.data(), (size_t)D.n, cudaMemcpyHostToDevice));
        if (timed) t_h2d += now_sec() - t0;

        CHECK_CUDA(cudaMemset(dHist, 0, sizeof(int) * H));
        CHECK_CUDA(cudaDeviceSynchronize());
        t0 = now_sec();
        histogram_kernel<<<blocks, TPB>>>(dX, dY, D.n, F, dOff, dHist);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        if (timed) t_kernel += now_sec() - t0;

        t0 = now_sec();
        CHECK_CUDA(cudaMemcpy(hist.data(), dHist, sizeof(int) * H, cudaMemcpyDeviceToHost));
        if (timed) t_d2h += now_sec() - t0;

        t0 = now_sec();
        last_split = best_split(hist.data(), B, tot, pos);
        if (timed) t_scan += now_sec() - t0;
    }

    t_h2d /= reps; t_kernel /= reps; t_d2h /= reps; t_scan /= reps;
    double t_gpu_total = t_h2d + t_kernel + t_d2h + t_scan;

    std::fprintf(stderr,
                 "[dt_cuda] GPU=%s N=%lld F=%d bins=%d reps=%d\n"
                 "  root split: feature=%d threshold=%.4g gini_gain=%.6f\n"
                 "  t_h2d=%.6f  t_kernel=%.6f  t_d2h=%.6f  t_scan=%.6f  total=%.6f (s)\n",
                 prop.name, (long long)N, F, B.total, reps,
                 last_split.f, last_split.thr, last_split.gain,
                 t_h2d, t_kernel, t_d2h, t_scan, t_gpu_total);

    // Machine-readable line (compare against dt.cpp's serial/MPI timings).
    std::printf("RESULT_GPU,cuda,%s,%lld,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.4g,%.6f\n",
                prop.name, (long long)N, F, B.total,
                t_gpu_total, t_h2d, t_kernel, t_d2h, t_scan,
                last_split.f, last_split.thr, last_split.gain);

    cudaFree(dX); cudaFree(dY); cudaFree(dOff); cudaFree(dHist);
    return 0;
}
