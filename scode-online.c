#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#include "dlib.h"
#define vmsg(...) if(VERBOSE)msg(__VA_ARGS__)

/*** Command line options */

const char *usage = "Usage: scode-online [OPTIONS] < file\n"
  "file should have tab separated columns of arbitrary tokens\n"
  "-d NDIM: number of dimensions (default 25)\n"
  "-z Z: partition function approximation (default 0.166)\n"
  "-p PHI0: learning rate parameter (default 50.0)\n"
  "-u ETA0: learning rate parameter (default 0.2)\n"
  "-s SEED: random seed (default 1)\n"
  "-m MAXHIST: max number of tokens to remember (default 1e6)\n"
  "-v verbose messages (default false)\n";

size_t NDIM = 25;
double Z = 0.166;
double PHI0 = 50.0;
double ETA0 = 0.2;
unsigned SEED = 1;
size_t MAXHIST = 1e6;
bool VERBOSE = false;


/*** svec_t: vectors on unit sphere representing tokens.  svec_t is a
 pointer to a struct that contains a token string, its count, and its
 vector.  The vectors are unit-length, NDIM dimensional, float
 arrays. */

typedef struct svec_s {
  char *key;
  size_t cnt;
  float *vec;
} *svec_t;


/*** Helper functions */

svec_t rand_token(darr_t m, svec_t x);
double d2(float *x, float *y, size_t ndim);
float *normalize(float *x, size_t ndim);
float *rand_unit_vector(size_t ndim);


/*** S-CODE Algorithm */

void scode(svec_t x[], darr_t marginal[], size_t ndim, size_t ntok)
{
  /* x[0]..x[ntok-1] are svecs for the last tuple observed.  We are
     going to update their vectors based on the multivariate extension
     (Glob07, Sec 6.2) of the S-CODE algorithm (Maron10). The first
     step is to sample a random token for each position of the tuple
     from its marginal distribution which gives us a random tuple
     r[i].  Some x[i] (except x[0]) could be NULL, corresponding to
     unobserved features, those are skipped. */

  assert(x[0] != NULL);
  svec_t *r = malloc(ntok * sizeof(svec_t));
  for (size_t i = 0; i < ntok; i++) {
    r[i] = (x[i] == NULL ? NULL : rand_token(marginal[i], x[i]));
  }

  /* The update rule for the binary S-CODE algorithm after observing a
     token pair with vectors (x0,xi) and sampling a random pair
     (r0,ri) from marginals is (Maron10, Eq. 10,11,12):

     x0 += e0 * [(xi - x0) + (1/Z) * exp(-|x0-ri|^2) * (x0 - ri)]
     xi += ei * [(x0 - xi) + (1/Z) * exp(-|xi-r0|^2) * (xi - r0)]
     ei = eta0 * phi0 / (phi0 + cnt(xi))

     The idea is to apply this binary update rule to each pair (x0,x1),
     (x0,x2), etc.  Note that x0 has special status, it has ntok-1
     updates which we accumulate in vector dx0. */     

  float *dx0 = calloc(ndim, sizeof(float));
  for (size_t i = 1; i < ntok; i++) {
    if (x[i] == NULL) continue;
    double ei = ETA0 * PHI0 / (PHI0 + x[i]->cnt);
    double zi = exp(-d2(x[i]->vec, r[0]->vec, ndim)) / Z;
    double z0 = exp(-d2(x[0]->vec, r[i]->vec, ndim)) / Z;
    for (size_t j = 0; j < ndim; j++) {
      x[i]->vec[j] += ei * (x[0]->vec[j] - x[i]->vec[j] + 
			    zi * (x[i]->vec[j] - r[0]->vec[j]));
      dx0[j] += (x[i]->vec[j] - x[0]->vec[j] +
		 z0 * (x[0]->vec[j] - r[i]->vec[j]));
    }
    normalize(x[i]->vec, ndim);
  }
  double e0 = ETA0 * PHI0 / (PHI0 + x[0]->cnt);
  for (size_t j = 0; j < ndim; j++) {
    x[0]->vec[j] += e0 * dx0[j];
  }
  normalize(x[0]->vec, ndim);
  free(dx0);
  free(r);
}

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

/* normalize(x, ndim) scales the ndim dimensional vector x to unit
   size. */

float *normalize(float *x, size_t ndim) {
  double s = 0;
  for (size_t j = 0; j < ndim; j++) 
    s += x[j] * x[j];
  s = sqrt(s);
  for (size_t j = 0; j < ndim; j++) 
    x[j] /= s;
  return x;
}

/* rand_token() first adds the observed token x to the marginal
   distribution array m, then returns a random element from m.  
   In order to do this with limited memory we keep at most MAXHIST
   tokens in m, after that x overwrites a random element. */

svec_t rand_token(darr_t m, svec_t x) {
  size_t n = len(m);
  if (n < MAXHIST) {
    val(m, n, svec_t) = x;
    n++;
  } else {
    assert(n == MAXHIST);
    size_t r = rand() % n;
    val(m, r, svec_t) = x;
  }
  size_t r = rand() % n;
  return val(m, r, svec_t);
}

/* rand_unit_vector(ndim) allocates and returns a random unit vector
   of ndim dimensions.  Picking dimensions uniformly in [-1,1] and
   normalizing does not give us a random vector, corners of the
   hypercube are more likely.  Using the gaussian distribution for
   each dimension fixes the problem.  We generate two standard
   gaussians from two uniform (0,1] using the Box-Muller transform. */

#define TWO_PI 6.2831853071795864769252866
 
float *rand_unit_vector(size_t ndim) {
  float *r = malloc(ndim * sizeof(float));
  double rand1 = 0;
  double rand2 = 0;
  for (size_t i = 0; i < ndim; i++) {
    if (rand1 == 0) {
      rand1 = rand() / ((double) RAND_MAX);
      if(rand1 < 1e-100) rand1 = 1e-100;
      rand1 = sqrt(-2 * log(rand1));
      rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;
      r[i] = rand1 * cos(rand2);
    } else {
      r[i] = rand1 * sin(rand2);
      rand1 = 0;
    }
  }
  return normalize(r, ndim);
}

void get_options(int argc, char **argv) {
  int opt;
  while((opt = getopt(argc, argv, "d:z:p:u:s:m:v")) != -1) {
    switch(opt) {
    case 'd': NDIM = atoi(optarg); break;
    case 'z': Z = atof(optarg); break;
    case 'p': PHI0 = atof(optarg); break;
    case 'u': ETA0 = atof(optarg); break;
    case 's': SEED = atoi(optarg); break;
    case 'm': MAXHIST = atoi(optarg); break;
    case 'v': VERBOSE = true; break;
    default: die("%s",usage);
    }
  }
}


/* This defines a hash table for svec_t and sget(), see dlib.h for details. */

static svec_t svec_new(const char *key) {
  svec_t s = malloc(sizeof(struct svec_s));
  s->key = strdup(key);
  s->cnt = 0;
  s->vec = rand_unit_vector(NDIM);
  return s;
}

#define svec_key(s) ((s)->key)
#define svec(h,k) (*sget(h,k,true))

D_HASH(s, svec_t, char *, svec_key, d_strmatch, fnv1a, svec_new, d_isnull, d_mknull)


/*** main() */

int main(int argc, char **argv) {
  get_options(argc, argv);
  vmsg("scode-online -d %u -z %g -p %g -u %g -s %lu -m %lu %s",
       NDIM, Z, PHI0, ETA0, SEED, MAXHIST, (VERBOSE ? "-v " : ""));
  srand(SEED);
  darr_t *v = NULL;		// hash tables of embedding vectors
  darr_t *m = NULL;		// arrays to sample from marginals
  svec_t *x = NULL;		// last tuple read
  char **toks = NULL;		// tokens on last line
  size_t ntok = 0;		// number of tokens on each line
  size_t len1 = 0;		// strlen of first line
  
  forline (line, NULL) {	// Process input
    line[strlen(line)-1] = 0;	// chop newline
    if (len1 == 0) {
      len1 = strlen(line);
      toks = malloc(len1 * sizeof(char *));
    }
    size_t n = split(line, "\t", toks, len1);
    if (ntok == 0) {		// Alloc if first line
      ntok = n;
      assert(ntok > 1);
      x = malloc(ntok * sizeof(svec_t));
      v = malloc(ntok * sizeof(darr_t));
      m = malloc(ntok * sizeof(darr_t));
      for (size_t i = 0; i < ntok; i++) {
	v[i] = darr(0, svec_t);
	m[i] = darr(0, svec_t);
      }
    }
    assert(n == ntok);
    for (size_t i = 0; i < ntok; i++) {
      if (*toks[i] == '\0') {
	x[i] = NULL;
      } else {
	x[i] = svec(v[i], toks[i]);
	x[i]->cnt++;
      }
    }
    scode(x, m, NDIM, ntok);
  }

  vmsg("Writing...");
  for (size_t i = 0; i < ntok; i++) { // Print and free
    forhash (svec_t, sptr, v[i], d_isnull) {
      svec_t s = *sptr;
      printf("%zu:%s\t%zu", i, s->key, s->cnt);
      for (size_t j = 0; j < NDIM; j++) {
	printf("\t%g", s->vec[j]);
      }
      putchar('\n');
      free(s->key);
      free(s->vec);
      free(s);
    }
    darr_free(v[i]);
    darr_free(m[i]);
  }
  free(v); free(m); free(x); free(toks);
  vmsg("done");
}

