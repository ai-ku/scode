#include <stdio.h>
#include <math.h>
#include "svec.h"
#include "rng.h"

void svec_print(svec x) {
  for (int i = 0; i < x->size; i++) {
    if (i > 0) printf("\t");
    printf("%.6f", svec_get(x, i));
  }
}

float svec_sqdist(svec x, svec y) {
  float sqdist = 0;
  for (int i = x->size - 1; i >= 0; i--) {
    float xi = svec_get(x, i);
    float yi = svec_get(y, i);
    float xy = xi - yi;
    sqdist += xy * xy;
  }
  return sqdist;
}

void svec_randomize(svec x) {
  for (int i = x->size - 1; i >= 0; i--) {
    svec_set(x, i, -1 + 2 * gsl_rng_uniform(rng_R));
  }
  svec_normalize(x);
}

float svec_pull(svec x, svec y, float d) {
  float sumsq = 0;
  for (int i = x->size - 1; i >= 0; i--) {
    float xi = svec_get(x, i);
    float yi = svec_get(y, i);
    float move = d * (yi - xi);
    svec_set(x, i, xi + move);
    sumsq += move * move;
  }
  svec_normalize(x);
  return sumsq;
}

float svec_push(svec x, svec y, float d) {
  float sumsq = 0;
  for (int i = x->size - 1; i >= 0; i--) {
    float xi = svec_get(x, i);
    float yi = svec_get(y, i);
    float move = d * (xi - yi);
    svec_set(x, i, xi + move);
    sumsq += move * move;
  }
  svec_normalize(x);
  return sumsq;
}

void svec_normalize(svec x) {
  float sumsq = 0;
  for (int i = x->size - 1; i >= 0; i--) {
    float xi = svec_get(x, i);
    sumsq += xi * xi;
  }
  svec_scale(x, 1.0 / sqrt(sumsq));
}
