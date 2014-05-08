#include <math.h>
#include "dlib.h"

/*** svec_t: vectors on unit sphere representing tokens.  svec_t is a
 pointer to a struct that contains a token string, its count, and its
 vector.  The vectors are unit-length, NDIM dimensional, float
 arrays. */

typedef struct svec_s {
  char *key;
  size_t cnt;
  float *vec;
} *svec_t;

static inline svec_t svec_new(const char *key) {
  svec_t s = _d_malloc(sizeof(struct svec_s));
  s->key = _d_strdup(key);
  s->cnt = 0;
  s->vec = NULL;
  return s;
}

#define svec_key(s) ((s)->key)

/* This defines a hash table for svec_t and sget(), see dlib.h for details. */

D_HASH(s, svec_t, char *, svec_key, d_strmatch, fnv1a, svec_new, d_isnull, d_mknull)

#define svec(h,k) (*sget(h,k,true))

/* d2(x, y, ndim) returns the squared distance between two ndim
   dimensional vectors x and y. */

static inline double d2(float *x, float *y, size_t ndim) {
  double ans = 0;
  for (size_t j = 0; j < ndim; j++) {
    double dj = x[j] - y[j];
    ans += dj * dj;
  }
  return ans;
}

/* normalize(x, ndim) scales the ndim dimensional vector x to unit
   size. */

static inline float *normalize(float *x, size_t ndim) {
  double s = 0;
  for (size_t j = 0; j < ndim; j++) 
    s += x[j] * x[j];
  s = sqrt(s);
  for (size_t j = 0; j < ndim; j++) 
    x[j] /= s;
  return x;
}

/* Things defined in scode-model.c */

typedef struct model_s {
  size_t ntok;			// Number of tokens per tuple
  size_t ndim;			// Number of embedding dimensions
  darr_t *v;			// ntok hash tables of string -> svec
  size_t *n;			// ntok counts (for denominator)
  double *z;			// ntok normalization constants (z[0]=0)
} *model_t;

model_t load_model(char *modelfile, double z);
model_t new_model(size_t ntok, size_t ndim);
void print_model(model_t m);
void free_model(model_t m);
size_t calcN(model_t m, size_t i);
double calcZ(model_t m, size_t i);
