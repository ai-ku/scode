// TODO: print average logl for -v2
// figure out xargs, ncat, repeat etc.

#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include "dlib.h"
#include "scode-model.h"
#define vmsg(...) if(VERBOSE)msg(__VA_ARGS__)

/*** Command line options */

const char *usage = "Usage: scode-online [OPTIONS] < file\n"
  "file should have tab separated columns of arbitrary tokens\n"
  "-d NDIM: number of dimensions (default 25)\n"
  "-z Z: partition function approximation (default 0.166)\n"
  "-p PHI0: learning rate parameter (default 50.0)\n"
  "-u ETA0: learning rate parameter (default 0.2)\n"
  "-s SEED: random seed (default 1)\n"
  "-m MODEL: model file to initialize with (default none)\n"
  "-n MAXHIST: max number of tokens to remember (default 1e6)\n"
  "-v VERBOSE: 0 (default) nothing, 1 dots, 2 logl\n";

size_t NDIM = 25;
double Z = 0.166;
double PHI0 = 50.0;
double ETA0 = 0.2;
unsigned SEED = 1;
char *MODEL = NULL;
size_t MAXHIST = 1e6;
int VERBOSE = 0;
size_t NTOK = 2; // read from input

void get_options(int argc, char **argv) {
  int opt;
  while((opt = getopt(argc, argv, "d:z:p:u:s:m:n:v:")) != -1) {
    switch(opt) {
    case 'd': NDIM = atoi(optarg); break;
    case 'z': Z = atof(optarg); break;
    case 'p': PHI0 = atof(optarg); break;
    case 'u': ETA0 = atof(optarg); break;
    case 's': SEED = atoi(optarg); break;
    case 'm': MODEL = optarg; break;
    case 'n': MAXHIST = atoi(optarg); break;
    case 'v': VERBOSE = atoi(optarg); break;
    default: die("%s",usage);
    }
  }
}

#define vmsg_options() \
  vmsg("scode-online -d %u -z %g -p %g -u %g -s %lu -m %s -n %lu -v %d", \
       NDIM, Z, PHI0, ETA0, SEED, (MODEL ? MODEL : "NULL"), MAXHIST, VERBOSE);

/*** Helper functions */

svec_t rand_token(darr_t m, svec_t x);
float *rand_unit_vector(size_t ndim);


/*** S-CODE Algorithm */

void scode(svec_t x[], darr_t marginal[]) {
  static svec_t *r = NULL;
  static float *dx0 = NULL;
  if (r == NULL) r = malloc(NTOK * sizeof(svec_t));
  if (dx0 == NULL) dx0 = malloc(NDIM * sizeof(float));
  for (size_t i = 0; i < NDIM; i++) dx0[i] = 0;

  /* x[0]..x[ntok-1] are svecs for the last tuple observed.  We are
     going to update their vectors based on the multivariate extension
     (Glob07, Sec 6.2) of the S-CODE algorithm (Maron10). The first
     step is to sample a random token for each position of the tuple
     from its marginal distribution which gives us a random tuple
     r[i].  Some x[i] (except x[0]) could be NULL, corresponding to
     unobserved features, those are skipped. */

  assert(x[0] != NULL);
  for (size_t i = 0; i < NTOK; i++) {
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

  for (size_t i = 1; i < NTOK; i++) {
    if (x[i] == NULL) continue;
    double ei = ETA0 * PHI0 / (PHI0 + x[i]->cnt);
    double zi = exp(-d2(x[i]->vec, r[0]->vec, NDIM)) / Z;
    double z0 = exp(-d2(x[0]->vec, r[i]->vec, NDIM)) / Z;
    for (size_t j = 0; j < NDIM; j++) {
      x[i]->vec[j] += ei * (x[0]->vec[j] - x[i]->vec[j] + 
			    zi * (x[i]->vec[j] - r[0]->vec[j]));
      dx0[j] += (x[i]->vec[j] - x[0]->vec[j] +
		 z0 * (x[0]->vec[j] - r[i]->vec[j]));
    }
    normalize(x[i]->vec, NDIM);
  }
  double e0 = ETA0 * PHI0 / (PHI0 + x[0]->cnt);
  for (size_t j = 0; j < NDIM; j++) {
    x[0]->vec[j] += e0 * dx0[j];
  }
  normalize(x[0]->vec, NDIM);
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
  float *r = _d_malloc(ndim * sizeof(float));
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

static inline void report_progress(svec_t *x, model_t m) {
  static double logL_avg = 0;
  static u64 ncall = 0;
  static double logZ = 0;
  ncall++;
  if (VERBOSE == 1) {
    if (!(ncall % 1000000)) fputc('.', stderr);
  } else {
    if (logZ == 0) logZ = log(Z);
    assert(x[0] != NULL);
    double logx = log(((double) x[0]->cnt) / m->n[0]);
    for (size_t i = m->ntok - 1; i > 0; i--) {
      if (x[i] == NULL) continue;
      double logy = log(((double) x[i]->cnt) / m->n[i]);
      double logp = logx + logy - logZ - d2(x[0]->vec, x[i]->vec, m->ndim);
      logL_avg += (logp - logL_avg) * (ncall < 1e7 ? 1.0/ncall : 1e-7);
      //logL_avg = (1.0/ncall) * logp + ((ncall-1.0)/ncall) * logL_avg;
    }
    if (!(ncall % 1000000)) msg("%dM %g", ncall/1000000, logL_avg);
  }
}

/*** main() */

int main(int argc, char **argv) {
  get_options(argc, argv);
  vmsg_options();
  srand(SEED);

  model_t m = NULL;		// scode model
  darr_t *b = NULL;		// arrays to sample from marginals
  svec_t *x = NULL;		// last tuple read
  char **toks = NULL;		// tokens on last line
  size_t len1 = 0;		// strlen of first line

  if (MODEL != NULL) {
    vmsg("Loading model %s", MODEL);
    // this loads v, sets ntok, ndim, calculates n, z.
    m = load_model(MODEL, Z);
    NTOK = m->ntok; NDIM = m->ndim;
  }
  
  vmsg("Reading stdin (each dot = 1M lines)");
  forline (line, NULL) {
    
    // split line
    line[strlen(line)-1] = 0;
    if (toks == NULL) {
      len1 = strlen(line);
      toks = _d_malloc(len1 * sizeof(char *));
    }
    size_t n = split(line, "\t", toks, len1);

    // alloc if necessary
    if (x == NULL) { 
      assert(n > 1);
      NTOK = n;
      x = _d_calloc(n, sizeof(svec_t));
      b = _d_calloc(n, sizeof(darr_t));
      for (size_t i = 0; i < n; i++)
	b[i] = darr(0, svec_t);
      if (m == NULL)
	m = new_model(n, NDIM);
    }
    assert(n == m->ntok);

    // lookup vectors
    for (size_t i = 0; i < n; i++) {
      if (*toks[i] == '\0') {
	x[i] = NULL;
      } else {
	x[i] = svec(m->v[i], toks[i]);
	x[i]->cnt++;
	m->n[i]++;
	if (x[i]->vec == NULL)
	  x[i]->vec = rand_unit_vector(m->ndim);
      }
    }

    // update vectors
    scode(x, b);
    if (VERBOSE) report_progress(x, m);
  }
  if (VERBOSE) fputc('\n', stderr);
  vmsg("Printing model...");
  print_model(m);
  for (size_t i = 0; i < m->ntok; i++) {
    darr_free(b[i]);
  }
  _d_free(b); _d_free(x); _d_free(toks);
  free_model(m); 
  vmsg("done");
}

