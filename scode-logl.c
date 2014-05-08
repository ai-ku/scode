/* scode-logl model < data */

const char *usage = "Usage: scode-logl [-z Z] model < data\n";

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
#include <unistd.h>
#include <math.h>
#include "dlib.h"
#include "scode-model.h"


/*** main() */

int main(int argc, char **argv) {
  // Load the model from the file given as first non-option argument.
  // Normalization constant Z, will be calculated (takes long) if not provided.
  double Z = 0;
  int opt;
  while((opt = getopt(argc, argv, "z:")) != -1) {
    switch(opt) {
    case 'z': Z = atof(optarg); break;
    default: die("%s", usage);
    }
  }
  if (optind == argc) die("%s", usage);
  msg("Loading model from %s.", argv[optind]);
  if (Z > 0) msg("Will use fixed Z of %g.", Z);
  else msg("Will calculate Z, this may take some time...");
  model_t m = load_model(argv[optind], Z);

  // Read data from stdin and calculate logL
  char **toks = _d_calloc(m->ntok, sizeof(char *));
  svec_t *x = _d_calloc(m->ntok, sizeof(svec_t));
  double *logZ = _d_calloc(m->ntok, sizeof(double));
  for (size_t i = 1; i < m->ntok; i++) {
    logZ[i] = log(m->z[i]);
  }
  double logL = 0;
  size_t nline = 0;

  msg("Reading data from stdin (each dot = 1M lines)...");
  forline (line, NULL) {
    if ((++nline & ((1<<20)-1)) == 0) fputc('.', stderr);
    line[strlen(line)-1] = 0;	// chop newline
    size_t ntok = split(line, "\t", toks, m->ntok);
    if (ntok != m->ntok) die("Wrong number of columns.");
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
      if (x[i] == NULL) continue;
      double logy = log(((double) x[i]->cnt) / m->n[i]);
      logL += logx + logy - logZ[i] - d2(x[0]->vec, x[i]->vec, m->ndim);
    }
  }
  fputc('\n', stderr);
  logL /= nline;
  _d_free(toks); _d_free(x); _d_free(logZ);
  free_model(m);
  msg("nlines=%zu avg-logL=%g", nline, logL);
}

