/* Trajectory-distance kernels for the Route Similarity Preservation Problem.
 *
 * All curves are passed as flat arrays of interleaved planar coordinates,
 * P = [x0,y0,x1,y1,...], with `m` points (so 2*m doubles).  Coordinates are
 * assumed to be already projected into a metric frame (metres); no geodetic
 * work happens here.
 *
 * Every routine is O(m*n) time and O(min(m,n)) working memory: the dynamic
 * programmes keep two rows only, which matters because these are called
 * once per route per individual per generation.
 *
 * Build:  cc -O3 -shared -fPIC -o libkernels.so kernels.c -lm
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline double pt_dist(const double *P, long i, const double *Q, long j) {
    double dx = P[2 * i] - Q[2 * j];
    double dy = P[2 * i + 1] - Q[2 * j + 1];
    return sqrt(dx * dx + dy * dy);
}

static inline double dmin3(double a, double b, double c) {
    double t = a < b ? a : b;
    return t < c ? t : c;
}

static inline double dmax2(double a, double b) { return a > b ? a : b; }

/* ---------------------------------------------------------------- Frechet */
/* Discrete Frechet distance (Eiter & Mannila 1994), coupling formulation:
 *   CA(i,j) = max( d(i,j), min( CA(i-1,j), CA(i-1,j-1), CA(i,j-1) ) ).
 * Returns CA(m-1,n-1), i.e. the bottleneck of the best monotone coupling. */
double discrete_frechet(const double *P, long m, const double *Q, long n) {
    if (m <= 0 || n <= 0) return -1.0;
    double *prev = (double *)malloc(sizeof(double) * (size_t)n);
    double *cur  = (double *)malloc(sizeof(double) * (size_t)n);
    if (!prev || !cur) { free(prev); free(cur); return -1.0; }

    /* first row: only rightward moves are available */
    prev[0] = pt_dist(P, 0, Q, 0);
    for (long j = 1; j < n; ++j)
        prev[j] = dmax2(prev[j - 1], pt_dist(P, 0, Q, j));

    for (long i = 1; i < m; ++i) {
        cur[0] = dmax2(prev[0], pt_dist(P, i, Q, 0)); /* only downward moves */
        for (long j = 1; j < n; ++j)
            cur[j] = dmax2(pt_dist(P, i, Q, j),
                           dmin3(prev[j], prev[j - 1], cur[j - 1]));
        memcpy(prev, cur, sizeof(double) * (size_t)n);
    }
    double out = prev[n - 1];
    free(prev); free(cur);
    return out;
}

/* -------------------------------------------------------------------- DTW */
/* Dynamic Time Warping with Euclidean ground distance.  Returned value is the
 * accumulated cost divided by (m+n), which is the usual path-length
 * normalisation; without it DTW is not comparable across route lengths. */
double dtw_distance(const double *P, long m, const double *Q, long n) {
    if (m <= 0 || n <= 0) return -1.0;
    double *prev = (double *)malloc(sizeof(double) * (size_t)n);
    double *cur  = (double *)malloc(sizeof(double) * (size_t)n);
    if (!prev || !cur) { free(prev); free(cur); return -1.0; }

    prev[0] = pt_dist(P, 0, Q, 0);
    for (long j = 1; j < n; ++j)
        prev[j] = prev[j - 1] + pt_dist(P, 0, Q, j);

    for (long i = 1; i < m; ++i) {
        cur[0] = prev[0] + pt_dist(P, i, Q, 0);
        for (long j = 1; j < n; ++j)
            cur[j] = pt_dist(P, i, Q, j) +
                     dmin3(prev[j], prev[j - 1], cur[j - 1]);
        memcpy(prev, cur, sizeof(double) * (size_t)n);
    }
    double out = prev[n - 1] / (double)(m + n);
    free(prev); free(cur);
    return out;
}

/* -------------------------------------------------------------- Hausdorff */
/* Symmetric (bidirectional) Hausdorff distance.  Order-blind by construction,
 * which is exactly the property we want to contrast against Frechet. */
double hausdorff_distance(const double *P, long m, const double *Q, long n) {
    if (m <= 0 || n <= 0) return -1.0;
    double sup_pq = 0.0, sup_qp = 0.0;
    for (long i = 0; i < m; ++i) {
        double best = INFINITY;
        for (long j = 0; j < n; ++j) {
            double d = pt_dist(P, i, Q, j);
            if (d < best) best = d;
        }
        if (best > sup_pq) sup_pq = best;
    }
    for (long j = 0; j < n; ++j) {
        double best = INFINITY;
        for (long i = 0; i < m; ++i) {
            double d = pt_dist(P, i, Q, j);
            if (d < best) best = d;
        }
        if (best > sup_qp) sup_qp = best;
    }
    return dmax2(sup_pq, sup_qp);
}

/* ------------------------------------------------------------------- LCSS */
/* Longest Common SubSequence distance with spatial tolerance `eps` (metres)
 * and warping window `delta` (index units).  Normalised to [0,1] and then
 * rescaled by `eps` so the result carries metric units like the others. */
double lcss_distance(const double *P, long m, const double *Q, long n,
                     double eps, long delta) {
    if (m <= 0 || n <= 0) return -1.0;
    long *prev = (long *)calloc((size_t)(n + 1), sizeof(long));
    long *cur  = (long *)calloc((size_t)(n + 1), sizeof(long));
    if (!prev || !cur) { free(prev); free(cur); return -1.0; }

    for (long i = 1; i <= m; ++i) {
        cur[0] = 0;
        for (long j = 1; j <= n; ++j) {
            long lag = i - j; if (lag < 0) lag = -lag;
            if (lag <= delta && pt_dist(P, i - 1, Q, j - 1) < eps)
                cur[j] = prev[j - 1] + 1;
            else
                cur[j] = prev[j] > cur[j - 1] ? prev[j] : cur[j - 1];
        }
        memcpy(prev, cur, sizeof(long) * (size_t)(n + 1));
    }
    long lcs = prev[n];
    long lo = m < n ? m : n;
    free(prev); free(cur);
    return eps * (1.0 - (double)lcs / (double)lo);
}
