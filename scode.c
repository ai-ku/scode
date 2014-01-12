#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include "dlib.h"
#include "svec.h"
#include "rng.h"

const char *usage = "Usage: scode [OPTIONS] < file\n"
  "file should have columns of arbitrary tokens\n"
  "-r RESTART: number of restarts (default 1)\n"
  "-i NITER: number of iterations over data (default UINT32_MAX)\n"
  "-t THRESHOLD: quit if logL increase for iter <= this (default .001)\n"
  "-d NDIM: number of dimensions (default 25)\n"
  "-z Z: partition function approximation (default 0.166)\n"
  "-p PHI0: learning rate parameter (default 50.0)\n"
  "-u NU0: learning rate parameter (default 0.2)\n"
  "-s SEED: random seed (default 0)\n"
  "-c calculate real Z (default false)\n"
  "-w The first line of the input is weights (default false)\n"
  "-v verbose messages (default false)\n";

int RESTART = 1;
int NITER = UINT32_MAX;
double THRESHOLD = 0.001;
int NDIM = 25;
double Z = 0.166;
double PHI0 = 50.0;
double NU0 = 0.2;
int SEED = 0;
int CALCZ = 0;
int WEIGHT = 0;
int VERBOSE = 0;

int NTOK = -1;
int NTUPLE = 0;

const gsl_rng_type *rng_T;
gsl_rng *rng_R = NULL;
darr_t data;
uint32_t **update_cnt;
double * weight = NULL;
double * uweight = NULL; /*Updated weights*/
uint32_t * cweight = NULL; /*Weights Counts*/
uint32_t **cnt;
#define frq(i,j) ((double)cnt[i][j]*NTOK/len(data))
svec **vec;
svec **best_vec;
svec dummy_vec;
sym_t qmax;
sym_t NULLFEATID;
#define NULLFEATMARKER "/XX/"

int main(int argc, char **argv);
void init_rng();
void free_rng();
int init_data();
int init_weight();
void free_weight();
void randomize_vectors();
void copy_best_vec();
void free_data();
void update_tuple(sym_t *t);
double logL();
double calcZ();

#define vmsg(...) if(VERBOSE)msg(__VA_ARGS__)

int main(int argc, char **argv) {
  int opt;
  while((opt = getopt(argc, argv, "r:i:t:d:z:p:u:s:cwv")) != -1) {
    switch(opt) {
    case 'r': RESTART = atoi(optarg); break;
    case 'i': NITER = atoi(optarg); break;
    case 't': THRESHOLD = atof(optarg); break;
    case 'd': NDIM = atoi(optarg); break;
    case 'z': Z = atof(optarg); break;
    case 'p': PHI0 = atof(optarg); break;
    case 'u': NU0 = atof(optarg); break;
    case 's': SEED = atoi(optarg); break;
    case 'c': CALCZ = 1; break;
    case 'w': WEIGHT = 1; break;
    case 'v': VERBOSE = 1; break;
    default: die("%s",usage);
    }
  }

  vmsg("scode -r %d -i %d -t %g -d %d -z %g -p %g -u %g -s %d %s%s%s",
       RESTART, NITER, THRESHOLD, NDIM, Z, NU0, PHI0, SEED,
       (CALCZ ? "-c " : ""), (WEIGHT ? "-w " : ""), (VERBOSE ? "-v " : ""));

  init_rng();
  if (SEED) gsl_rng_set(rng_R, SEED);
  if (WEIGHT) NTOK = init_weight();
  NTUPLE = init_data();
  vmsg("Read %d tuples %d uniq tokens", NTUPLE, qmax);

  double best_logL = 0;
  for (int start = 0; start < RESTART; start++) {
    randomize_vectors();
    double ll = logL();
    vmsg("Restart %d/%d logL0=%g best=%g", 1+start, RESTART, ll, best_logL);
    if (CALCZ) vmsg("Z=%g (approx %g)", calcZ(), Z);
    for (int iter = 0; iter < NITER; iter++) {
      for (int di = 0; di < NTUPLE; di++) {
	update_tuple(&val(data, di * NTOK, sym_t));
      }
      double ll0 = ll;
      ll = logL();
      vmsg("Iteration %d/%d logL=%g", 1+iter, NITER, ll);
      if (ll - ll0 <= THRESHOLD) break;
    }
    if (start == 0 || ll > best_logL) {
      vmsg("Updating best_vec with logL=%g", ll);
      best_logL = ll;
      copy_best_vec();
    }
    vmsg("Restart %d/%d logL1=%g best=%g", 1+start, RESTART, ll, best_logL);
    if (CALCZ) vmsg("Z=%g (approx %g)", calcZ(), Z);
  }
  for (uint32_t t = 0; t < NTOK; t++) {
    for (uint32_t q = 1; q <= qmax; q++) {
      if (best_vec[t][q] == NULL) continue;
      printf("%d:%s\t%d\t", t, sym2str(q), cnt[t][q]);
      svec_print(best_vec[t][q]);
      putchar('\n');
    }
  }
  fflush(stdout);
  free_data();
  free_rng();
  if (WEIGHT) free_weight();
  symtable_free();
  dfreeall();
  fprintf(stderr, "%f\n", best_logL);
  vmsg("bye");
}

double logL() {
  double l = 0;
  for (int i = 0; i < NTUPLE; i++) {
    sym_t *t = &val(data, i * NTOK, sym_t);
    sym_t x = t[0];
    sym_t y = t[1];
    float px = frq(0, x);
    float py = frq(1, y);
    svec vx = vec[0][x];
    svec vy = vec[1][y];
    float xy = svec_sqdist(vx, vy);
    l += log(px * py) - xy;
  }
  return (l / NTUPLE - log(Z));
}

double calcZ() {
  double z = 0;
  for (uint32_t x = 1; x <= qmax; x++) {
    if (VERBOSE && (x % 1000 == 0)) fputc('.', stderr);
    if (cnt[0][x] == 0) continue;
    float px = frq(0, x);
    svec vx = vec[0][x];
    for (uint32_t y = 1; y <= qmax; y++) {
      if (cnt[1][y] == 0) continue;
      float py = frq(1, y);
      svec vy = vec[1][y];
      float xy = svec_sqdist(vx, vy);
      z += px * py * exp(-xy);
    }
  }
  if (VERBOSE) fputc('\n', stderr);
  return z;
}

void update_tuple(sym_t *t) {
  /*weighted update*/
  static svec *u = NULL;
  static svec *v = NULL;
  static svec dx = NULL;
  if (u == NULL) u = _d_malloc(NTOK * sizeof(svec));
  if (v == NULL) v = _d_malloc(NTOK * sizeof(svec));
  if (dx == NULL) dx = svec_alloc(NDIM);
  for (int i = 0; i < NTOK; i++) u[i] = vec[i][t[i]];
  for (int i = 0; i < NTOK; i++) {
    /* Sampling values from the marginal distributions. */
    /* Can this be done once, or do we have to resample for every x? */
    if(i > 0 && t[i] == NULLFEATID) continue;
    for (int j = 0; j < NTOK; j++) {
      if (j==i) { v[j] = u[i]; continue;}
      uint32_t r = gsl_rng_uniform_int(rng_R, NTUPLE);
      sym_t y = val(data, r * NTOK + j, sym_t);
      v[j] = vec[j][y];
      if(i > 0) break;
    }          
    /* Compute the move for u[i] */
    svec_set_zero(dx);
    double ww;
    for (int j = 0; j < NTOK; j++) {
      if (j == i) continue;
      ww = weight == NULL ? 1 : (i > 0 ? weight[i] : weight[j]);
      double push = 0, pull = 0;
      if (v[j] == NULL) v[j] = dummy_vec;
      else push = exp(-svec_sqdist(u[i], v[j])) / Z;
      if(u[j] == NULL)  u[j] = dummy_vec;
      else pull = 1;
      if(push != 0 || pull != 0){
	for (int d = 0; d < NDIM; d++) {                    
	  float dxd = svec_get(dx, d);
	  float x = svec_get(u[i], d);
	  float y = svec_get(u[j], d);
	  float z = svec_get(v[j], d);
	  svec_set(dx, d, dxd + ww * ( pull * (y - x) + push * (x - z)));
	}
      }
      /*restore the vectors to original forms*/
      if(push == 0) v[j] = NULL;
      if(pull == 0) u[j] = NULL;
      if(i > 0) break;
      }
    /* Apply the move scaled by learning parameter */
    uint32_t cx = update_cnt[i][t[i]]++;
    float nx = NU0 * (PHI0 / (PHI0 + cx));
    svec_scale(dx, nx);
    svec_add(u[i], dx);
    svec_normalize(u[i]);
  }
}

int init_weight(){
  int size = 100, i = 0;
  weight = _d_malloc(size * sizeof(double));
  forline (buf, NULL) {
    fortok (tok, buf) {
      weight[i] = atof(tok);
      assert(weight[i++] >= 0);
      if(i >= 100) {
	size *= 2; 
	weight = _d_realloc(weight, size);
      }
    }
    assert(i > 0);
    break;
  }
  return i;
}

void free_weight() {
  if (weight != NULL) _d_free(weight);
}

int init_data() {
  qmax = 0;
  data = darr(0, sym_t);
  forline (buf, NULL) {
    size_t ntok = 0;
    fortok (tok, buf) {
      sym_t q = str2sym(tok, true);
      if (q > qmax) qmax = q;
      val(data, len(data), sym_t) = q;
      if(strcmp(tok, NULLFEATMARKER) == 0) NULLFEATID = q;
      ntok++;
    }
    if(NTOK == -1) NTOK = ntok;
    assert(ntok == NTOK); //Each line has equal number of tokens
  }
  assert(NTOK > 0);
  update_cnt = _d_malloc(NTOK * sizeof(ptr_t));
  cnt = _d_malloc(NTOK * sizeof(ptr_t));
  vec = _d_malloc(NTOK * sizeof(ptr_t));
  best_vec = _d_malloc(NTOK * sizeof(ptr_t));     
  dummy_vec = svec_alloc(NDIM);
  svec_zero(dummy_vec);
  uweight = _d_calloc(NTOK, sizeof(double));
  cweight = _d_calloc(NTOK, sizeof(uint32_t));
  for (int i = 0; i < NTOK; i++) {
    update_cnt[i] = _d_calloc(qmax+1, sizeof(uint32_t));
    cnt[i] = _d_calloc(qmax+1, sizeof(uint32_t));
    vec[i] = _d_calloc(qmax+1, sizeof(svec));
    best_vec[i] = _d_calloc(qmax+1, sizeof(svec));
  }
  int N = len(data) / NTOK;
  for (int i = 0; i < N; i++) {
    sym_t *p = &val(data, i * NTOK, sym_t);
    for (int j = 0; j < NTOK; j++) {
      int k = p[j];
      assert(k <= qmax);
      cnt[j][k]++;
      if(k == NULLFEATID){
	vec[j][k] = best_vec[j][k] = NULL;
      }
      else if (vec[j][k] == NULL) {
	vec[j][k] = svec_alloc(NDIM);
	best_vec[j][k] = svec_alloc(NDIM);
      }
    }
  }
  return N;
}

void free_data() {
  for (int i = 0; i < NTOK; i++) {
    for (int j = 0; j <= qmax; j++) {
      if (vec[i][j] != NULL) {
	svec_free(vec[i][j]);
	svec_free(best_vec[i][j]);
      }
    }
    _d_free(best_vec[i]);
    _d_free(vec[i]);
    _d_free(cnt[i]);
    _d_free(update_cnt[i]);
  }
  _d_free(cweight);
  _d_free(uweight);
  svec_free(dummy_vec);
  _d_free(best_vec);
  _d_free(vec);
  _d_free(cnt);
  _d_free(update_cnt);
  darr_free(data);
}

void randomize_vectors() {
  for (int j = 0; j < NTOK; j++) {
    for (uint32_t q = 1; q <= qmax; q++) {
      if (vec[j][q] != NULL) {
	svec_randomize(vec[j][q]);
	update_cnt[j][q] = 0;
      }
    }
  }
}

void copy_best_vec() {
  for (int j = 0; j < NTOK; j++) {
    for (uint32_t q = 1; q <= qmax; q++) {
      if (vec[j][q] != NULL) {
	svec_memcpy(best_vec[j][q], vec[j][q]);
      }
    }
  }
}

void init_rng() {
  gsl_rng_env_setup();
  rng_T = gsl_rng_default;
  rng_R = gsl_rng_alloc(rng_T);
}

void free_rng() {
  gsl_rng_free(rng_R);
}

