/* scode-logl model < data */

const char *usage = "Usage: scode-logl model < data\n";

/*
Computes the average log likelihood of the tuples in data based on
the embeddings given in model output by scode.  The model file has
the format:

i:token <tab> cnt <tab> vec1 <tab> vec2 ...

Where i is the column of the tuple at which token was observed, cnt
is the number of times it was observed and [vec1, vec2, ...] is the
embedding vector.  

The Globerson07 MM model defines the likelihood of a pair as:

p(x,y) = (1/Z) * phat(x) * phat(y) * exp(-d2(x,y))
Z = Sum_{x,y} phat(x) * phat(y) * exp(-d2(x,y))

where phat are the empirical probabilities and d2 is the squared
distance between the embeddings of the two tokens.

Globerson07 Section 6.2 suggests the following for more than two
variables:

logp(x0,x1,...,xn) = Sum_{i=1..n} wi * (Sum_{x0,xi} phat(x0,xi) * logp(x0,xi))

Assuming all wi = 1, this is just the sum of pairwise average logp
between x0 and xi.

On the issue of Z, we should calculate the real Z instead of
relying on the approximation like the scode training algorithm.

On the issue of missing values: when column i is empty do not
include that tuple in the (x0,xi) calculation.
*/

#include <stdio.h>
#include <math.h>
#include "dlib.h"

typedef struct svec_s {
  char *key;
  size_t cnt;
  float *vec;
} *svec_t;


/* This defines a hash table for svec_t and sget(), see dlib.h for details. */

static svec_t svec_new(const char *key) {
  svec_t s = malloc(sizeof(struct svec_s));
  s->key = strdup(key);
  s->cnt = 0;
  s->vec = NULL;
  return s;
}

#define svec_key(s) ((s)->key)
#define svec(h,k) (*sget(h,k,true))

D_HASH(s, svec_t, char *, svec_key, d_strmatch, fnv1a, svec_new, d_isnull, d_mknull)


/* d2(x, y, ndim) returns the squared distance between two ndim
   dimensional vectors x and y. */

double d2(float *x, float *y, size_t ndim) {
  double ans = 0;
  for (size_t j = 0; j < ndim; j++) {
    double dj = x[j] - y[j];
    ans += dj * dj;
  }
  return ans;
}

typedef struct model_s {
  size_t ntok;
  size_t ndim;
  darr_t *v;
  size_t *n;
  double *z;
} *model_t;

size_t calcN(model_t m, size_t i) {
  size_t n = 0;
  forhash (svec_t, yptr, m->v[i], d_isnull) {
    svec_t y = *yptr;
    n += y->cnt;
  }
  return n;
}

double calcZ(model_t m, size_t i) {
  if (i == 0) return 0;
  msg("calcZ(%d)...", i);
  double z = 0;
  size_t nx = m->n[0];
  size_t ny = m->n[i];
  size_t dot = 0;
  forhash (svec_t, xptr, m->v[0], d_isnull) {
    if (dot++ % 1000 == 0) fputc('.', stderr);
    svec_t x = *xptr;
    double px = ((double) x->cnt) / nx;
    forhash (svec_t, yptr, m->v[i], d_isnull) {
      svec_t y = *yptr;
      double py = ((double) y->cnt) / ny;
      z += px * py * exp(-d2(x->vec, y->vec, m->ndim));
    }
  }
  fprintf(stderr, "\nz=%g\n", z);
  return z;
}

model_t load_model(char *modelfile) {  
  darr_t v = darr(0, darr_t);	// hash tables of embedding vectors: v[ntok]
  size_t ndim = 0;		// dimensionality of the embedding
  char **toks = NULL;		  // array to tokenize each line
  size_t len1 = 0;		  // length of the first line (used as upper bound on number of toks)

  msg("Reading %s...", modelfile);
  forline (line, modelfile) {
    line[strlen(line)-1] = 0;	// chop newline
    if (len1 == 0) {
      len1 = strlen(line);
      toks = malloc(len1 * sizeof(char *));
    }
    size_t n = split(line, "\t", toks, len1);
    if (ndim == 0) {
      ndim = n - 2;
    }
    assert(n == ndim + 2);

    char *ptr = NULL;
    long int index = strtol(toks[0], &ptr, 10);
    assert(ptr > toks[0] && *ptr == ':' && index >= 0);
    if (index >= len(v)) {
      val(v, index, darr_t) = darr(0, svec_t);
    }

    char *token = ptr+1;
    svec_t s = svec(val(v, index, darr_t), token);
    assert(!strcmp(s->key, token));

    long int count = strtol(toks[1], &ptr, 10);
    assert(ptr > toks[1] && *ptr == 0 && count > 0);
    s->cnt = count;

    s->vec = malloc(ndim * sizeof(float));
    for (size_t i = 0; i < ndim; i++) {
      s->vec[i] = strtof(toks[i+2], &ptr);
      assert(ptr > toks[i+2] && *ptr == 0);
    }
  }
  free(toks);
  model_t m = malloc(sizeof(struct model_s));
  m->ntok = len(v);
  m->ndim = ndim;
  m->v = calloc(m->ntok, sizeof(darr_t));
  m->n = calloc(m->ntok, sizeof(size_t));
  m->z = calloc(m->ntok, sizeof(double));
  for (int i = 0; i < m->ntok; i++) {
    m->v[i] = val(v, i, darr_t);
    m->n[i] = calcN(m, i);
    m->z[i] = calcZ(m, i);  // Takes about 450 seconds
 // m->z[i] = (i==0 ? 0 : 0.166);
    msg("[%d]: v=%zu n=%zu z=%g", i, len(m->v[i]), m->n[i], m->z[i]);
  }
  darr_free(v);
  return m;
}


/*** main() */

int main(int argc, char **argv) {
  if (argc != 2) die(usage);
  model_t m = load_model(argv[1]);
  char **toks = malloc(m->ntok * sizeof(char *));
  svec_t *x = malloc(m->ntok * sizeof(svec_t));
  double logL = 0;
  size_t nline = 0;
  forline (line, NULL) {
    nline++;
    line[strlen(line)-1] = 0;	// chop newline
    size_t ntok = split(line, "\t", toks, m->ntok);
    assert(ntok == m->ntok);
    for (size_t i = 0; i < m->ntok; i++) {
      if (*toks[i] == '\0') {
	x[i] = NULL;
      } else {
	x[i] = svec(m->v[i], toks[i]);
	assert(x[i]->vec != NULL && x[i]->cnt > 0);
      }
    }
    assert(x[0] != NULL);
    double logx = log(((double) x[0]->cnt) / m->n[0]);
    for (size_t i = 1; i < m->ntok; i++) {
      double logy = log(((double) x[i]->cnt) / m->n[i]);
      double logz = log(m->z[i]);
      logL += logx + logy - logz - d2(x[0]->vec, x[i]->vec, m->ndim);
    }
  }
  logL /= nline;
  msg("nline=%zu logL=%g", nline, logL);
}

