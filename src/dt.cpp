// ============================================================================
//  dt.cpp -- Histogram-based CART decision tree (Gini) with exact global
//            split search.  ONE source file for three builds:
//
//      serial        g++ -O3 src/dt.cpp                       (no MPI, no OpenMP)
//      MPI           mpicxx -DUSE_MPI ...                     (data parallel)
//      MPI + OpenMP  mpicxx -DUSE_MPI -fopenmp ...            (hybrid)
//
//  Why this replaces the old tree_mpi_omp.cpp split search
//  -------------------------------------------------------
//  Rows are sharded across MPI ranks, so the left/right class counts of a
//  candidate threshold are SUMS over all ranks.  The old code computed the
//  gain from rank-local left counts mixed with global totals and then took
//  MPI_MAXLOC over those local gains, which is not the global Gini gain.
//
//  Here every rank builds a local histogram (total / positive count per
//  feature bin) for the node, one MPI_Allreduce(SUM) turns it into the global
//  histogram, and every rank then scans that same histogram with the same
//  code.  Consequences:
//    * the chosen split is the true global best split,
//    * all ranks reach identical decisions without a broadcast,
//    * counts are integers, so the sum does not depend on the order of the
//      reduction: the tree is bit-for-bit identical for any rank count and any
//      thread count (contrast with floating-point sums, Lecture 8).
//
//  Two growth strategies (identical trees, different communication pattern):
//    --growth node   depth-first, one Allreduce per internal node
//    --growth level  breadth-first, one Allreduce per tree level
//  Two OpenMP strategies for building histograms (identical results):
//    --omp sample    threads split the rows, private histograms, then reduce
//    --omp feature   threads split the features, shared histogram
//
//  Data: --csv FILE (label,feat...)  or  --synthetic N [--features F]
//        (deterministic Instacart-like generator; row r depends only on
//         (seed, r), so the dataset is the same for every rank count).
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using std::string;
using std::vector;
typedef int64_t i64;

static const uint16_t NONE = 0xFFFF;      // sample is not in any active node
static const i64 OMP_MIN_ROWS = 4096;     // below this, threading a pass costs more than it saves

static inline double now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static inline int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
static inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// ----------------------------------------------------------------- options
struct Params {
    string csv;
    i64 synthetic_n = 0;
    int features = 7;
    uint64_t seed = 42;
    int depth = 6;
    int min_split = 20;        // node with tot <= min_split is a leaf (same rule as the old code)
    string growth = "level";   // node | level
    string omp = "sample";     // sample | feature
    int reps = 3;
    int warmup = 1;
    int profile = 0;           // 1: barrier before every Allreduce so wait and transfer are timed separately
    int check = 0;             // 1: extra consistency checks (slower)
    string tag = "run";
    string dump_tree;
    string export_csv;
    bool header = false;
    bool quiet = false;
};

static void usage() {
    std::fprintf(stderr,
        "usage: dt (--csv FILE | --synthetic N [--features F] [--seed S])\n"
        "          [--depth 6] [--min-split 20] [--growth node|level] [--omp sample|feature]\n"
        "          [--reps 3] [--warmup 1] [--profile 0|1] [--check 0|1] [--tag NAME]\n"
        "          [--dump-tree FILE] [--export-csv FILE] [--header] [--quiet]\n");
}

static bool parse_args(int argc, char** argv, Params& P) {
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--csv") P.csv = need("--csv");
        else if (a == "--synthetic") P.synthetic_n = std::atoll(need("--synthetic"));
        else if (a == "--features") P.features = std::atoi(need("--features"));
        else if (a == "--seed") P.seed = std::strtoull(need("--seed"), nullptr, 10);
        else if (a == "--depth") P.depth = std::atoi(need("--depth"));
        else if (a == "--min-split") P.min_split = std::atoi(need("--min-split"));
        else if (a == "--growth") P.growth = need("--growth");
        else if (a == "--omp") P.omp = need("--omp");
        else if (a == "--reps") P.reps = std::atoi(need("--reps"));
        else if (a == "--warmup") P.warmup = std::atoi(need("--warmup"));
        else if (a == "--profile") P.profile = std::atoi(need("--profile"));
        else if (a == "--check") P.check = std::atoi(need("--check"));
        else if (a == "--tag") P.tag = need("--tag");
        else if (a == "--dump-tree") P.dump_tree = need("--dump-tree");
        else if (a == "--export-csv") P.export_csv = need("--export-csv");
        else if (a == "--header") P.header = true;
        else if (a == "--quiet") P.quiet = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return false; }
    }
    if (P.header) return true;
    if (P.csv.empty() == (P.synthetic_n <= 0)) {
        std::fprintf(stderr, "give exactly one of --csv or --synthetic\n"); usage(); return false;
    }
    if (P.growth != "node" && P.growth != "level") { std::fprintf(stderr, "bad --growth\n"); return false; }
    if (P.omp != "sample" && P.omp != "feature") { std::fprintf(stderr, "bad --omp\n"); return false; }
    if (P.depth < 1 || P.depth > 15) { std::fprintf(stderr, "--depth must be in 1..15\n"); return false; }
    if (P.synthetic_n > 0 && (P.features < 7 || P.features > 256)) {
        std::fprintf(stderr, "--features must be in 7..256 for synthetic data\n"); return false;
    }
    if (P.synthetic_n >= 2000000000LL) { std::fprintf(stderr, "N must be < 2^31 (int32 histogram counts)\n"); return false; }
    return true;
}

// -------------------------------------------------------------- MPI wrapper
// Everything that touches MPI lives here, so the algorithm code is identical
// in the serial and parallel builds.
struct Comm {
    int rank = 0, size = 1;
    bool profile = false;
    double t_wait = 0, t_allr = 0;   // seconds in barrier-before-Allreduce / in the Allreduce itself
    i64 n_allr = 0;
    double bytes = 0;

    void init(int* argc, char*** argv) {
#ifdef USE_MPI
        int provided = 0;
        MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
        (void)argc; (void)argv;
#endif
    }
    void finalize() {
#ifdef USE_MPI
        MPI_Finalize();
#endif
    }
    void abort_all(int code) {
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, code);
#endif
        std::exit(code);
    }
    void reset() { t_wait = t_allr = 0; n_allr = 0; bytes = 0; }
    void barrier() {
#ifdef USE_MPI
        if (size > 1) MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    // The only collective on the training critical path.
    void sum_hist(int32_t* buf, i64 n) {
#ifdef USE_MPI
        if (size > 1) {
            if (profile) {
                double a = now_sec();
                MPI_Barrier(MPI_COMM_WORLD);
                t_wait += now_sec() - a;
            }
            double b = now_sec();
            MPI_Allreduce(MPI_IN_PLACE, buf, (int)n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            t_allr += now_sec() - b;
            n_allr++;
            bytes += 4.0 * (double)n;
        }
#else
        (void)buf; (void)n;
#endif
    }
    // Untimed helpers (setup / reporting).
    void sum_i64(i64* buf, int n) {
#ifdef USE_MPI
        if (size > 1) MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
        (void)buf; (void)n;
#endif
    }
    void min_i64(i64* buf, int n) {
#ifdef USE_MPI
        if (size > 1) MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
#else
        (void)buf; (void)n;
#endif
    }
    void max_i64(i64* buf, int n) {
#ifdef USE_MPI
        if (size > 1) MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#else
        (void)buf; (void)n;
#endif
    }
    void sum_f64(double* buf, int n) {
#ifdef USE_MPI
        if (size > 1) MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
        (void)buf; (void)n;
#endif
    }
    void max_f64(double* buf, int n) {
#ifdef USE_MPI
        if (size > 1) MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#else
        (void)buf; (void)n;
#endif
    }
};

// -------------------------------------------------------------------- data
struct Bins {
    int F = 0;
    vector<vector<float>> rep;   // rep[f][b]: representative value of bin b (exact data value when not approximate)
    vector<int> nb;              // bins per feature
    vector<int> off;             // cumulative bin offsets
    int total = 0;               // sum of nb
    bool approx = false;         // true if some feature had > 256 distinct values (then bins are a subsample)
    void finalize() {
        nb.assign(F, 0); off.assign(F, 0); total = 0;
        for (int f = 0; f < F; f++) { nb[f] = (int)rep[f].size(); off[f] = total; total += nb[f]; }
    }
};

struct Data {
    i64 n = 0;                   // local rows
    i64 N = 0;                   // global rows
    int F = 0;
    vector<uint8_t> x;           // local bin codes, row-major n*F
    vector<uint8_t> y;           // local labels
};

static void block_range(i64 N, int size, int rank, i64* start, i64* count) {
    i64 base = N / size, rem = N % size;
    *count = base + (rank < rem ? 1 : 0);
    *start = (i64)rank * base + std::min<i64>(rank, rem);
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Instacart-like schema, same domains as preprocess.py.
static const int NDOM = 7;
static const int DOM_LO[NDOM] = {1, 0, 0, 0, 1, 1, 1};
static const int DOM_HI[NDOM] = {99, 6, 23, 30, 29, 134, 21};
// 0 order_number, 1 order_dow, 2 order_hour_of_day, 3 days_since_prior_order,
// 4 add_to_cart_order, 5 aisle_id, 6 department_id; feature f >= 7 repeats domain f % 7.

// Row r of the synthetic dataset: a pure function of (seed, r).
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

static void make_synthetic(const Params& P, Comm& C, Data& D, Bins& B) {
    D.F = P.features; D.N = P.synthetic_n;
    i64 start, cnt;
    block_range(D.N, C.size, C.rank, &start, &cnt);
    D.n = cnt;
    D.x.resize((size_t)cnt * D.F);
    D.y.resize((size_t)cnt);
    B.F = D.F; B.rep.assign(D.F, {});
    for (int f = 0; f < D.F; f++) {
        int d = f % NDOM;
        for (int v = DOM_LO[d]; v <= DOM_HI[d]; v++) B.rep[f].push_back((float)v);
    }
    B.finalize();
    const int F = D.F;
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < cnt; i++) {
        int vals[256];
        int lab = gen_row(P.seed, (uint64_t)(start + i), F, vals);
        for (int f = 0; f < F; f++) D.x[(size_t)i * F + f] = (uint8_t)(vals[f] - DOM_LO[f % NDOM]);
        D.y[(size_t)i] = (uint8_t)lab;
    }
}

static void export_synthetic_csv(const Params& P, const string& path) {
    if (P.synthetic_n > 5000000) { std::fprintf(stderr, "refusing to export more than 5M rows\n"); return; }
    std::ofstream out(path);
    out << "reordered";
    for (int f = 0; f < P.features; f++) out << ",f" << f;
    out << "\n";
    int vals[256];
    for (i64 r = 0; r < P.synthetic_n; r++) {
        int lab = gen_row(P.seed, (uint64_t)r, P.features, vals);
        out << lab;
        for (int f = 0; f < P.features; f++) out << "," << vals[f];
        out << "\n";
    }
}

static bool load_csv_root(const string& path, vector<float>& X, vector<uint8_t>& y, i64& n, int& F) {
    std::ifstream in(path);
    if (!in) return false;
    string line;
    if (!std::getline(in, line)) return false;
    F = (int)std::count(line.begin(), line.end(), ',');   // columns - 1
    if (F < 1) return false;
    n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const char* p = line.c_str();
        char* end = nullptr;
        double v = std::strtod(p, &end);
        y.push_back(v > 0.5 ? 1 : 0);
        p = end;
        for (int f = 0; f < F; f++) {
            if (*p == ',') p++;
            v = std::strtod(p, &end);
            if (end == p) return false;
            X.push_back((float)v);
            p = end;
        }
        n++;
    }
    return true;
}

static inline uint8_t code_of(const vector<float>& rep, float x) {
    int c = (int)(std::upper_bound(rep.begin(), rep.end(), x) - rep.begin()) - 1;
    return (uint8_t)(c < 0 ? 0 : c);
}

// CSV path: rank 0 reads, scatters raw floats; bin edges come from the global
// set of distinct values (exact whenever every feature has <= 256 of them).
static bool make_from_csv(const Params& P, Comm& C, Data& D, Bins& B) {
    vector<float> Xfull; vector<uint8_t> yfull;
    i64 N = 0; int F = 0; int ok = 1;
    if (C.rank == 0) ok = load_csv_root(P.csv, Xfull, yfull, N, F) ? 1 : 0;
#ifdef USE_MPI
    i64 meta[3] = {N, F, ok};
    if (C.size > 1) MPI_Bcast(meta, 3, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    N = meta[0]; F = (int)meta[1]; ok = (int)meta[2];
#endif
    if (!ok) { if (C.rank == 0) std::fprintf(stderr, "cannot read CSV %s\n", P.csv.c_str()); return false; }
    D.N = N; D.F = F;
    i64 start, cnt;
    block_range(N, C.size, C.rank, &start, &cnt);
    D.n = cnt;
    vector<float> Xraw((size_t)cnt * F);
    D.y.resize((size_t)cnt);
#ifdef USE_MPI
    if (C.size > 1) {
        vector<int> cx(C.size), dx(C.size), cy(C.size), dy(C.size);
        for (int r = 0; r < C.size; r++) {
            i64 s, c; block_range(N, C.size, r, &s, &c);
            cy[r] = (int)c; dy[r] = (int)s; cx[r] = (int)(c * F); dx[r] = (int)(s * F);
        }
        MPI_Scatterv(Xfull.data(), cx.data(), dx.data(), MPI_FLOAT, Xraw.data(), (int)(cnt * F), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(yfull.data(), cy.data(), dy.data(), MPI_BYTE, D.y.data(), (int)cnt, MPI_BYTE, 0, MPI_COMM_WORLD);
    } else
#endif
    {
        Xraw = std::move(Xfull);
        D.y = std::move(yfull);
    }

    B.F = F; B.rep.assign(F, {}); B.approx = false;
    int local_trunc = 0;
    for (int f = 0; f < F; f++) {
        vector<float> col((size_t)cnt);
        for (i64 i = 0; i < cnt; i++) {
            float v = Xraw[(size_t)i * F + f];
            if (v != v) { std::fprintf(stderr, "NaN in feature %d (fill missing values first)\n", f); C.abort_all(3); }
            col[(size_t)i] = v;
        }
        std::sort(col.begin(), col.end());
        col.erase(std::unique(col.begin(), col.end()), col.end());
        const size_t CAP = 4096;
        if (col.size() > CAP) {
            vector<float> sub;
            for (size_t b = 0; b < CAP; b++) sub.push_back(col[(size_t)((double)b * (col.size() - 1) / (CAP - 1))]);
            col.swap(sub);
            local_trunc = 1;
        }
        vector<float> all = col;
#ifdef USE_MPI
        if (C.size > 1) {
            int mycnt = (int)col.size();
            vector<int> counts(C.size), displs(C.size);
            MPI_Allgather(&mycnt, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            int tot = 0;
            for (int r = 0; r < C.size; r++) { displs[r] = tot; tot += counts[r]; }
            all.assign((size_t)tot, 0.f);
            MPI_Allgatherv(col.data(), mycnt, MPI_FLOAT, all.data(), counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
            std::sort(all.begin(), all.end());
            all.erase(std::unique(all.begin(), all.end()), all.end());
        }
#endif
        if (all.size() > 256) {
            vector<float> sub;
            for (int b = 0; b < 256; b++) sub.push_back(all[(size_t)((double)b * (all.size() - 1) / 255.0)]);
            sub.erase(std::unique(sub.begin(), sub.end()), sub.end());
            all.swap(sub);
            local_trunc = 1;
        }
        B.rep[f] = all;
    }
    i64 t = local_trunc;
    C.max_i64(&t, 1);
    B.approx = (t != 0);
    B.finalize();

    D.x.resize((size_t)cnt * F);
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < cnt; i++)
        for (int f = 0; f < F; f++) D.x[(size_t)i * F + f] = code_of(B.rep[f], Xraw[(size_t)i * F + f]);
    return true;
}

// -------------------------------------------------------------------- tree
struct Node {
    int leaf = 0, pred = 0;
    int feat = -1, bin = -1;
    float thr = 0.f;
    int left = -1, right = -1;
    i64 tot = 0, pos = 0;
};

struct Split {
    int f = -1, bin = -1;
    float thr = 0.f;
    double gain = -1.0;
    i64 ltot = 0, lpos = 0;      // global left counts at the chosen boundary
};

static inline double gini(i64 pos, i64 tot) {
    if (tot == 0) return 0.0;
    double p1 = (double)pos / (double)tot;
    double p0 = 1.0 - p1;
    return 1.0 - (p0 * p0 + p1 * p1);
}

// Best boundary of one feature in one node's GLOBAL histogram.  Boundaries are
// only evaluated between bins that are present in the node, with the threshold
// at the midpoint of the two neighbouring present values: same candidates as
// the sort-based serial code and as scikit-learn.  Strict '>' keeps the first
// maximum, so ties are broken the same way on every rank.
static Split scan_feature(const int32_t* h, const Bins& B, int f, i64 tot, i64 pos, double cur) {
    Split best;
    const int nb = B.nb[f];
    const int32_t* hf = h + 2 * (size_t)B.off[f];
    i64 ltot = 0, lpos = 0;
    int last = -1;
    for (int b = 0; b < nb; b++) {
        int32_t c = hf[2 * b];
        if (c == 0) continue;
        if (last >= 0) {
            i64 rtot = tot - ltot, rpos = pos - lpos;
            double weighted = ((double)ltot / (double)tot) * gini(lpos, ltot) +
                              ((double)rtot / (double)tot) * gini(rpos, rtot);
            double g = cur - weighted;
            if (g > best.gain) {
                best.gain = g; best.f = f; best.bin = last;
                best.thr = (B.rep[f][last] + B.rep[f][b]) / 2.0f;
                best.ltot = ltot; best.lpos = lpos;
            }
        }
        ltot += c; lpos += hf[2 * b + 1];
        last = b;
    }
    return best;
}

static Split best_split(const int32_t* h, const Bins& B, i64 tot, i64 pos) {
    const double cur = gini(pos, tot);
    Split best;
    for (int f = 0; f < B.F; f++) {
        Split s = scan_feature(h, B, f, tot, pos, cur);
        if (s.gain > best.gain) best = s;
    }
    return best;
}

struct Timers { double hist = 0, scan = 0, part = 0; };

class Trainer {
public:
    Trainer(const Data& d, const Bins& b, Comm& c, const Params& p) : D(d), B(b), C(c), P(p) {}
    vector<Node> tree;
    Timers T;

    void train() {
        tree.clear();
        T = Timers();
        if (P.growth == "level") train_level(); else train_node();
    }

private:
    const Data& D; const Bins& B; Comm& C; const Params& P;
    vector<int32_t> hist;

    bool terminal(int depth, i64 tot, i64 pos) const {
        return depth >= P.depth || tot <= P.min_split || pos == 0 || pos == tot;
    }
    int new_node(i64 tot, i64 pos) {
        Node nd; nd.tot = tot; nd.pos = pos;
        tree.push_back(nd);
        return (int)tree.size() - 1;
    }
    void make_leaf(int id) {
        Node& nd = tree[id];
        nd.leaf = 1;
        nd.pred = (nd.pos >= nd.tot - nd.pos) ? 1 : 0;
    }
    i64 root_positives() {
        i64 p = 0;
#pragma omp parallel for reduction(+ : p) schedule(static)
        for (i64 i = 0; i < D.n; i++) p += D.y[(size_t)i];
        int32_t buf[1] = {(int32_t)p};
        C.sum_hist(buf, 1);
        return buf[0];
    }

    // Local histogram of the rows selected by rowfn(k, row, slot) -> bool.
    template <class RowFn>
    void accumulate(i64 cnt, RowFn rowfn, int m, int32_t* out) {
        const int F = D.F;
        const size_t H = 2 * (size_t)B.total;
        const size_t len = (size_t)m * H;
        const uint8_t* X = D.x.data();
        const uint8_t* Y = D.y.data();
        const int* off = B.off.data();
        const int Th = (cnt >= OMP_MIN_ROWS) ? max_threads() : 1;

        if (Th > 1 && P.omp == "feature") {
#pragma omp parallel for schedule(static) num_threads(Th)
            for (int f = 0; f < F; f++) {
                for (i64 k = 0; k < cnt; k++) {
                    i64 r; int s;
                    if (!rowfn(k, r, s)) continue;
                    int32_t* p = out + (size_t)s * H + 2 * (size_t)(off[f] + X[(size_t)r * F + f]);
                    p[0]++; p[1] += Y[r];
                }
            }
            return;
        }
        if (Th == 1) {
            for (i64 k = 0; k < cnt; k++) {
                i64 r; int s;
                if (!rowfn(k, r, s)) continue;
                const uint8_t* row = X + (size_t)r * F;
                int32_t* hb = out + (size_t)s * H;
                int yy = Y[r];
                for (int f = 0; f < F; f++) {
                    int32_t* p = hb + 2 * (size_t)(off[f] + row[f]);
                    p[0]++; p[1] += yy;
                }
            }
            return;
        }
        vector<vector<int32_t>> priv((size_t)Th);
#pragma omp parallel num_threads(Th)
        {
            int t = thread_id();
            priv[(size_t)t].assign(len, 0);
            int32_t* h = priv[(size_t)t].data();
#pragma omp for schedule(static)
            for (i64 k = 0; k < cnt; k++) {
                i64 r; int s;
                if (!rowfn(k, r, s)) continue;
                const uint8_t* row = X + (size_t)r * F;
                int32_t* hb = h + (size_t)s * H;
                int yy = Y[r];
                for (int f = 0; f < F; f++) {
                    int32_t* p = hb + 2 * (size_t)(off[f] + row[f]);
                    p[0]++; p[1] += yy;
                }
            }
#pragma omp for schedule(static)
            for (i64 e = 0; e < (i64)len; e++) {
                int32_t sum = 0;
                for (int u = 0; u < Th; u++) sum += priv[(size_t)u][(size_t)e];
                out[e] = sum;
            }
        }
    }

    void check_hist(const int32_t* h, i64 tot, i64 pos, int node_slot) {
        for (int f = 0; f < D.F; f++) {
            i64 t = 0, p = 0;
            const int32_t* hf = h + 2 * (size_t)B.off[f];
            for (int b = 0; b < B.nb[f]; b++) { t += hf[2 * b]; p += hf[2 * b + 1]; }
            if (t != tot || p != pos) {
                std::fprintf(stderr, "[check] histogram mismatch node-slot %d feature %d: hist(%lld,%lld) vs stats(%lld,%lld)\n",
                             node_slot, f, (long long)t, (long long)p, (long long)tot, (long long)pos);
                C.abort_all(4);
            }
        }
    }

    // ---------------------------------------------------- level-wise growth
    void train_level() {
        const int F = D.F;
        const size_t H = 2 * (size_t)B.total;
        vector<uint16_t> slot((size_t)D.n, NONE);

        i64 tot = D.N, pos = 0;
        {
            double t0 = now_sec();
            pos = root_positives();
            T.scan += now_sec() - t0;
        }
        int root = new_node(tot, pos);
        vector<int> cur; vector<i64> cur_tot, cur_pos;
        if (terminal(0, tot, pos)) {
            make_leaf(root);
        } else {
            cur.push_back(root); cur_tot.push_back(tot); cur_pos.push_back(pos);
#pragma omp parallel for schedule(static)
            for (i64 i = 0; i < D.n; i++) slot[(size_t)i] = 0;
        }

        for (int depth = 0; !cur.empty(); depth++) {
            const int m = (int)cur.size();
            hist.assign((size_t)m * H, 0);

            double t0 = now_sec();
            const uint16_t* sl = slot.data();
            accumulate(D.n, [&](i64 k, i64& r, int& s) -> bool {
                uint16_t v = sl[k];
                if (v == NONE) return false;
                r = k; s = v; return true;
            }, m, hist.data());
            T.hist += now_sec() - t0;

            C.sum_hist(hist.data(), (i64)hist.size());
            if (P.check) for (int s = 0; s < m; s++) check_hist(hist.data() + (size_t)s * H, cur_tot[(size_t)s], cur_pos[(size_t)s], s);

            t0 = now_sec();
            vector<Split> sp((size_t)m);
#pragma omp parallel for schedule(dynamic) if (m > 1)
            for (int s = 0; s < m; s++)
                sp[(size_t)s] = best_split(hist.data() + (size_t)s * H, B, cur_tot[(size_t)s], cur_pos[(size_t)s]);

            vector<int> nxt; vector<i64> nxt_tot, nxt_pos;
            vector<int> lslot((size_t)m, -1), rslot((size_t)m, -1), sfeat((size_t)m, 0), sbin((size_t)m, 0);
            vector<uint8_t> isleaf((size_t)m, 0);
            for (int s = 0; s < m; s++) {
                int id = cur[(size_t)s];
                const Split& S = sp[(size_t)s];
                if (S.f < 0 || S.gain <= 0.0) { make_leaf(id); isleaf[(size_t)s] = 1; continue; }
                tree[id].feat = S.f; tree[id].bin = S.bin; tree[id].thr = S.thr;
                i64 lt = S.ltot, lp = S.lpos;
                i64 rt = cur_tot[(size_t)s] - lt, rp = cur_pos[(size_t)s] - lp;
                int L = new_node(lt, lp);
                int R = new_node(rt, rp);
                tree[id].left = L; tree[id].right = R;
                if (terminal(depth + 1, lt, lp)) make_leaf(L);
                else { lslot[(size_t)s] = (int)nxt.size(); nxt.push_back(L); nxt_tot.push_back(lt); nxt_pos.push_back(lp); }
                if (terminal(depth + 1, rt, rp)) make_leaf(R);
                else { rslot[(size_t)s] = (int)nxt.size(); nxt.push_back(R); nxt_tot.push_back(rt); nxt_pos.push_back(rp); }
                sfeat[(size_t)s] = S.f; sbin[(size_t)s] = S.bin;
            }
            T.scan += now_sec() - t0;

            t0 = now_sec();
            const uint8_t* X = D.x.data();
#pragma omp parallel for schedule(static)
            for (i64 i = 0; i < D.n; i++) {
                uint16_t s = slot[(size_t)i];
                if (s == NONE) continue;
                if (isleaf[s]) { slot[(size_t)i] = NONE; continue; }
                int c = X[(size_t)i * F + sfeat[s]];
                int ns = (c <= sbin[s]) ? lslot[s] : rslot[s];
                slot[(size_t)i] = (ns < 0) ? NONE : (uint16_t)ns;
            }
            T.part += now_sec() - t0;

            cur.swap(nxt); cur_tot.swap(nxt_tot); cur_pos.swap(nxt_pos);
        }
    }

    // ----------------------------------------------------- node-wise growth
    vector<int32_t> idx, tmp;

    // Parallel partition of idx[lo,hi): rows with code[feat] <= bin first.
    i64 partition_idx(i64 lo, i64 hi, int feat, int bin) {
        const i64 n = hi - lo;
        const int F = D.F;
        const uint8_t* X = D.x.data();
        const int Th = (n >= OMP_MIN_ROWS) ? max_threads() : 1;
        vector<i64> cL((size_t)Th, 0), oL((size_t)Th, 0), oR((size_t)Th, 0);
        i64 totalL = 0;
#pragma omp parallel num_threads(Th)
        {
            int t = thread_id();
            i64 a = lo + n * t / Th, b = lo + n * (t + 1) / Th;
            i64 c = 0;
            for (i64 k = a; k < b; k++) if (X[(size_t)idx[(size_t)k] * F + feat] <= bin) c++;
            cL[(size_t)t] = c;
#pragma omp barrier
#pragma omp single
            {
                i64 accL = 0, accR = 0;
                for (int u = 0; u < Th; u++) {
                    i64 ua = lo + n * u / Th, ub = lo + n * (u + 1) / Th;
                    oL[(size_t)u] = accL; accL += cL[(size_t)u];
                    oR[(size_t)u] = accR; accR += (ub - ua) - cL[(size_t)u];
                }
                totalL = accL;
            }
            i64 pl = lo + oL[(size_t)t], pr = lo + totalL + oR[(size_t)t];
            for (i64 k = a; k < b; k++) {
                int32_t v = idx[(size_t)k];
                if (X[(size_t)v * F + feat] <= bin) tmp[(size_t)pl++] = v; else tmp[(size_t)pr++] = v;
            }
#pragma omp barrier
#pragma omp for schedule(static)
            for (i64 k = lo; k < hi; k++) idx[(size_t)k] = tmp[(size_t)k];
        }
        return lo + totalL;
    }

    int rec(i64 lo, i64 hi, int depth, i64 tot, i64 pos) {
        int id = new_node(tot, pos);
        if (terminal(depth, tot, pos)) { make_leaf(id); return id; }
        const size_t H = 2 * (size_t)B.total;
        hist.assign(H, 0);

        double t0 = now_sec();
        const int32_t* ix = idx.data();
        accumulate(hi - lo, [&](i64 k, i64& r, int& s) -> bool { r = ix[lo + k]; s = 0; return true; }, 1, hist.data());
        T.hist += now_sec() - t0;

        C.sum_hist(hist.data(), (i64)H);
        if (P.check) check_hist(hist.data(), tot, pos, 0);

        t0 = now_sec();
        Split S = best_split(hist.data(), B, tot, pos);
        T.scan += now_sec() - t0;
        if (S.f < 0 || S.gain <= 0.0) { make_leaf(id); return id; }
        tree[id].feat = S.f; tree[id].bin = S.bin; tree[id].thr = S.thr;

        t0 = now_sec();
        i64 mid = partition_idx(lo, hi, S.f, S.bin);
        T.part += now_sec() - t0;

        int L = rec(lo, mid, depth + 1, S.ltot, S.lpos);
        int R = rec(mid, hi, depth + 1, tot - S.ltot, pos - S.lpos);
        tree[id].left = L; tree[id].right = R;
        return id;
    }

    void train_node() {
        idx.resize((size_t)D.n);
        tmp.resize((size_t)D.n);
#pragma omp parallel for schedule(static)
        for (i64 i = 0; i < D.n; i++) idx[(size_t)i] = (int32_t)i;
        double t0 = now_sec();
        i64 pos = root_positives();
        T.scan += now_sec() - t0;
        rec(0, D.n, 0, D.N, pos);
    }
};

// --------------------------------------------------------- tree utilities
static void canon(const vector<Node>& t, int id, string& s) {
    const Node& nd = t[id];
    if (nd.leaf) { s += "L"; s += std::to_string(nd.pred); s += ";"; return; }
    s += "N"; s += std::to_string(nd.feat); s += "."; s += std::to_string(nd.bin); s += "(";
    canon(t, nd.left, s); canon(t, nd.right, s); s += ")";
}
static uint64_t fnv1a(const string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}
static void dump_rec(const vector<Node>& t, int id, std::ostream& o) {
    const Node& nd = t[id];
    if (nd.leaf) { o << "L " << nd.pred << " " << nd.tot << " " << nd.pos << "\n"; return; }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9g", (double)nd.thr);
    o << "N " << nd.feat << " " << nd.bin << " " << buf << " " << nd.tot << " " << nd.pos << "\n";
    dump_rec(t, nd.left, o); dump_rec(t, nd.right, o);
}

static i64 count_correct(const Data& D, const vector<Node>& t) {
    const int F = D.F;
    const uint8_t* X = D.x.data();
    i64 correct = 0;
#pragma omp parallel for reduction(+ : correct) schedule(static)
    for (i64 i = 0; i < D.n; i++) {
        int id = 0;
        while (!t[id].leaf) id = (X[(size_t)i * F + t[id].feat] <= t[id].bin) ? t[id].left : t[id].right;
        if (t[id].pred == D.y[(size_t)i]) correct++;
    }
    return correct;
}

static const char* CSV_HEADER =
    "RESULT,tag,impl,growth,omp,profile,ranks,threads,cores,N,F,depth,rep,"
    "t_total,t_hist,t_scan,t_part,t_wait,t_allreduce,hist_imbalance,accuracy,"
    "tree_hash,n_nodes,n_allreduce,mbytes_allreduce,approx,t_setup";

// -------------------------------------------------------------------- main
int main(int argc, char** argv) {
#ifdef _OPENMP
    omp_set_dynamic(0);   // the code sizes per-thread buffers from omp_get_max_threads()
#endif
    Comm C;
    C.init(&argc, &argv);
    Params P;
    if (!parse_args(argc, argv, P)) { C.finalize(); return 2; }
    if (P.header) { if (C.rank == 0) std::puts(CSV_HEADER); C.finalize(); return 0; }
    C.profile = (P.profile != 0);

    // ---- data (untimed setup)
    C.barrier();
    double ts0 = now_sec();
    Data D; Bins B;
    if (!P.csv.empty()) {
        if (!make_from_csv(P, C, D, B)) { C.abort_all(1); }
    } else {
        if (C.rank == 0 && !P.export_csv.empty()) export_synthetic_csv(P, P.export_csv);
        make_synthetic(P, C, D, B);
    }
    C.barrier();
    double setup = now_sec() - ts0;
    C.max_f64(&setup, 1);
    if (B.total > 100000) { if (C.rank == 0) std::fprintf(stderr, "too many bins\n"); C.abort_all(5); }

    const int threads = max_threads();
#ifdef USE_MPI
    const char* impl = "mpi";
#else
    const char* impl = "serial";
#endif
    if (C.rank == 0 && !P.quiet) {
        std::fprintf(stderr, "[dt] impl=%s ranks=%d threads=%d N=%lld F=%d bins=%d%s depth=%d growth=%s omp=%s setup=%.2fs\n",
                     impl, C.size, threads, (long long)D.N, D.F, B.total, B.approx ? " (APPROXIMATE bins)" : "",
                     P.depth, P.growth.c_str(), P.omp.c_str(), setup);
    }

    Trainer tr(D, B, C, P);
    for (int w = 0; w < P.warmup; w++) tr.train();

    string canon_str; uint64_t hash0 = 0;
    for (int rep = 0; rep < P.reps; rep++) {
        C.reset();
        C.barrier();
        double t0 = now_sec();
        tr.train();
        C.barrier();
        double total = now_sec() - t0;

        double v[6] = {tr.T.hist, tr.T.scan, tr.T.part, C.t_wait, C.t_allr, total};
        double vmax[2] = {tr.T.hist, total};
        C.sum_f64(v, 6);
        C.max_f64(vmax, 2);
        double mean_hist = v[0] / C.size;
        double imb = mean_hist > 0 ? vmax[0] / mean_hist : 1.0;

        i64 cc = count_correct(D, tr.tree);
        C.sum_i64(&cc, 1);
        double acc = (double)cc / (double)D.N;

        canon_str.clear();
        canon(tr.tree, 0, canon_str);
        uint64_t h = fnv1a(canon_str);
        if (P.check && C.size > 1) {
            i64 hv[2] = {(i64)(h >> 1), (i64)(h >> 1)};
            i64 lo = hv[0], hi = hv[1];
            C.min_i64(&lo, 1); C.max_i64(&hi, 1);
            if (lo != hi) { std::fprintf(stderr, "[check] ranks built different trees\n"); C.abort_all(6); }
        }
        if (rep == 0) hash0 = h;
        else if (h != hash0 && C.rank == 0) std::fprintf(stderr, "[warn] tree hash changed between repetitions!\n");

        if (C.rank == 0) {
            std::printf("RESULT,%s,%s,%s,%s,%d,%d,%d,%d,%lld,%d,%d,%d,"
                        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.6f,"
                        "%016llx,%zu,%lld,%.4f,%d,%.3f\n",
                        P.tag.c_str(), impl, P.growth.c_str(), P.omp.c_str(), P.profile, C.size, threads,
                        C.size * threads, (long long)D.N, D.F, P.depth, rep,
                        vmax[1], v[0] / C.size, v[1] / C.size, v[2] / C.size, v[3] / C.size, v[4] / C.size, imb, acc,
                        (unsigned long long)h, tr.tree.size(), (long long)C.n_allr, C.bytes / 1.0e6,
                        B.approx ? 1 : 0, setup);
            std::fflush(stdout);
        }
    }

    if (C.rank == 0 && !P.dump_tree.empty()) {
        std::ofstream o(P.dump_tree);
        dump_rec(tr.tree, 0, o);
    }
    C.finalize();
    return 0;
}
