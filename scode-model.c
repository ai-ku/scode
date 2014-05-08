#include <stdio.h>
#include <math.h>
#include "dlib.h"
#include "scode-model.h"

/* load_model(): Allocates and loads model from file.  If Z <= 0 the
   real Z's are calculated and saved in m->z[i] for i=1..(ntok-1),
   which takes a long time.  For a quick load, the caller can provide
   an approximate Z > 0 in which case all m->z[i] is set to Z.
   m->z[0] is always 0. */

model_t load_model(char *modelfile, double Z) {  
  model_t m = _d_calloc(1, sizeof(struct model_s));
  char **toks = NULL;   	// array to tokenize each line
  size_t len1 = 0;		// length of the first line (used as upper bound on number of toks)

  // msg("Reading %s...", modelfile);
  forline (line, modelfile) {
    line[strlen(line)-1] = 0;	// chop newline
    if (len1 == 0) {
      len1 = strlen(line);
      toks = _d_calloc(len1, sizeof(char *));
    }
    size_t n = split(line, "\t", toks, len1);
    if (m->ndim == 0) {
      m->ndim = n - 2;
    }
    assert(n == m->ndim + 2);

    char *ptr = NULL;
    long int index = strtol(toks[0], &ptr, 10);
    assert(ptr > toks[0] && *ptr == ':' && index >= 0);
    if (index >= m->ntok) {
      size_t oldn = m->ntok;
      m->ntok = index + 1;
      m->v = _d_realloc(m->v, m->ntok * sizeof(darr_t));
      m->n = _d_realloc(m->n, m->ntok * sizeof(size_t));
      m->z = _d_realloc(m->z, m->ntok * sizeof(double));
      for (int i = oldn; i < m->ntok; i++) {
	m->v[i] = darr(0, svec_t);
	m->n[i] = 0;
	m->z[i] = 0;
      }
    }

    char *token = ptr+1;
    svec_t s = svec(m->v[index], token);
    assert(!strcmp(s->key, token));

    long int count = strtol(toks[1], &ptr, 10);
    assert(ptr > toks[1] && *ptr == 0 && count > 0);
    assert(s->cnt == 0);
    s->cnt = count;
    m->n[index] += count;

    assert(s->vec == NULL);
    s->vec = _d_calloc(m->ndim, sizeof(float));
    for (size_t i = 0; i < m->ndim; i++) {
      s->vec[i] = strtof(toks[i+2], &ptr);
      assert(ptr > toks[i+2] && *ptr == 0);
    }
  }
  _d_free(toks);
  for (int i = 1; i < m->ntok; i++) {
    m->z[i] = (Z <= 0 ? calcZ(m, i) : Z);
  }
  return m;
}

model_t new_model(size_t ntok, size_t ndim) {
  model_t m = _d_calloc(1, sizeof(struct model_s));
  m->ntok = ntok; m->ndim = ndim;
  m->v = _d_calloc(ntok, sizeof(darr_t));
  m->n = _d_calloc(ntok, sizeof(size_t));
  m->z = _d_calloc(ntok, sizeof(double));
  for (size_t i = 0; i < ntok; i++) {
    m->v[i] = darr(0, svec_t);
  }
  return m;
}

/* print_model(): Prints model to stdout. */

void print_model(model_t m) {
  for (size_t i = 0; i < m->ntok; i++) {
    forhash (svec_t, sptr, m->v[i], d_isnull) {
      svec_t s = *sptr;
      printf("%zu:%s\t%zu", i, s->key, s->cnt);
      for (size_t j = 0; j < m->ndim; j++) {
	printf("\t%g", s->vec[j]);
      }
      putchar('\n');
    }
  }
}

/* free_model(m): Deallocates memory for m. */

void free_model(model_t m) {
  for (size_t i = 0; i < m->ntok; i++) {
    forhash (svec_t, sptr, m->v[i], d_isnull) {
      svec_t s = *sptr;
      _d_free(s->key);
      _d_free(s->vec);
      _d_free(s);
    }
    darr_free(m->v[i]);
  }
  _d_free(m->v);
  _d_free(m->n);
  _d_free(m->z);
  _d_free(m);
}

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
  // msg("calcZ(%d)...", i);
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
  // fprintf(stderr, "\nz=%g\n", z);
  return z;
}

