const char *usage = "Usage: scode [OPTIONS] < file\n"
  "file should have two columns of arbitrary tokens\n"
  "-r RESTART: number of restarts (default 1)\n"
  "-i NITER: number of iterations over data (default 20)\n"
  "-d NDIM: number of dimensions (default 25)\n"
  "-z Z: partition function approximation (default 0.166)\n"
  "-p PHI0: learning rate parameter (default 50.0)\n"
  "-u NU0: learning rate parameter (default 0.2)\n"
  "-s SEED: random seed (default 0)\n"
  "-c: calculate real Z (default false)\n"
  "-m: merge vectors at output (default false)\n"
  "-v: verbose messages (default false)\n";

#define NTOK 2		      /* number of tokens per input line */
int RESTART = 1;
int NITER = 50;
int NDIM = 25;
double Z = 0.166;
double PHI0 = 50.0;
double NU0 = 0.2;
int SEED = 0;
int CALCZ = 0;
int VMERGE = 0;
int VERBOSE = 0;

#include <stdio.h>
#include <unistd.h>
#include <glib.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include "foreach.h"
#include "procinfo.h"
#include "svec.h"
#include "rng.h"

typedef GQuark Tuple[NTOK];
const gsl_rng_type *rng_T;
gsl_rng *rng_R = NULL;
GArray *data;
guint *update_cnt[NTOK];
guint *cnt[NTOK];
#define frq(i,j) ((double)cnt[i][j]/data->len)
svec *vec[NTOK];
svec *best_vec[NTOK];
GQuark qmax;

int main(int argc, char **argv);
void init_rng();
void free_rng();
void init_data();
void randomize_vectors();
void copy_best_vec();
void free_data();
float update_tuple(Tuple t);
float update_svec(svec x, svec y, svec y2, float xy2, float nx);
double logL();
double calcZ();

#define msg if(VERBOSE)g_message

int main(int argc, char **argv) {
  int opt;
  while((opt = getopt(argc, argv, "r:i:d:z:u:p:s:cmv")) != -1) {
    switch(opt) {
    case 'r': RESTART = atoi(optarg); break;
    case 'i': NITER = atoi(optarg); break;
    case 'd': NDIM = atoi(optarg); break;
    case 'z': Z = atof(optarg); break;
    case 'u': NU0 = atof(optarg); break;
    case 'p': PHI0 = atof(optarg); break;
    case 's': SEED = atoi(optarg); break;
    case 'c': CALCZ = 1; break;
    case 'm': VMERGE = 1; break;
    case 'v': VERBOSE = 1; break;
    default: g_error(usage);
    }
  }

  g_message_init();
  msg("scode -r %d -i %d -d %d -z %g -u %g -p %g -s %d %s%s%s",
      RESTART, NITER, NDIM, Z, NU0, PHI0, SEED,
      (CALCZ ? "-c " : ""), (VMERGE ? "-m " : ""), (VERBOSE ? "-v " : ""));

  init_rng();
  if (SEED) gsl_rng_set(rng_R, SEED);

  init_data();
  msg("Read %d tuples %d uniq tokens", data->len, qmax);

  double best_logL = 0;
  for (int start = 0; start < RESTART; start++) {
    randomize_vectors();
    double ll = logL();
    msg("Restart %d/%d logL0=%g best=%g", 1+start, RESTART, ll, best_logL);
    if (CALCZ) msg("Z=%g (approx %g)", calcZ(), Z);
    for (int iter = 0; iter < NITER; iter++) {
      for (int di = 0; di < data->len; di++) {
	update_tuple(g_array_index(data, Tuple, di));
      }
      ll = logL();
      msg("Iteration %d/%d logL=%g", 1+iter, NITER, ll);
    }
    if (start == 0 || ll > best_logL) {
      msg("Updating best_vec with logL=%g", ll);
      best_logL = ll;
      copy_best_vec();
    }
    msg("Restart %d/%d logL1=%g best=%g", 1+start, RESTART, ll, best_logL);
    if (CALCZ) msg("Z=%g (approx %g)", calcZ(), Z);
  }

  if (VMERGE) {			/* output for Maron et.al. 2010 bigram s-code model */
    for (guint q = 1; q <= qmax; q++) {
      printf("%s\t%d", g_quark_to_string(q), cnt[0][q]);
      for (guint t = 0; t < NTOK; t++) {
	g_assert(best_vec[t][q] != NULL);
	putchar('\t');
	svec_print(best_vec[t][q]);
      }
      putchar('\n');
    }
  } else {			/* regular output */
    for (guint t = 0; t < NTOK; t++) {
      for (guint q = 1; q <= qmax; q++) {
	if (best_vec[t][q] == NULL) continue;
	printf("%d:%s\t%d\t", t, g_quark_to_string(q), cnt[t][q]);
	svec_print(best_vec[t][q]);
	putchar('\n');
      }
    }
  }

  fflush(stdout);
  free_data();
  free_rng();
  fprintf(stderr, "%f\n", best_logL);
  msg("bye");
}

double logL() {
  double l = 0;
  for (int i = 0; i < data->len; i++) {
    GQuark *t = g_array_index(data, Tuple, i);
    GQuark x = t[0];
    GQuark y = t[1];
    float px = frq(0, x);
    float py = frq(1, y);
    svec vx = vec[0][x];
    svec vy = vec[1][y];
    float xy = svec_sqdist(vx, vy);
    l += log(px * py) - xy;
  }
  return (l / data->len - log(Z));
}

double calcZ() {
  double z = 0;
  for (guint x = 1; x <= qmax; x++) {
    if (VERBOSE && (x % 1000 == 0)) fprintf(stderr, ".");
    if (cnt[0][x] == 0) continue;
    float px = frq(0, x);
    svec vx = vec[0][x];
    for (guint y = 1; y <= qmax; y++) {
      if (cnt[1][y] == 0) continue;
      float py = frq(1, y);
      svec vy = vec[1][y];
      float xy = svec_sqdist(vx, vy);
      z += px * py * exp(-xy);
    }
  }
  if (VERBOSE) fprintf(stderr, "\n");
  return z;
}

float update_tuple(Tuple t) {
  GQuark x1 = t[0];
  GQuark y1 = t[1];
  guint cx = update_cnt[0][x1]++;
  guint cy = update_cnt[1][y1]++;
  float nx = NU0 * (PHI0 / (PHI0 + cx));
  float ny = NU0 * (PHI0 / (PHI0 + cy));
  svec vx1 = vec[0][x1];
  svec vy1 = vec[1][y1];
  guint ry = gsl_rng_uniform_int(rng_R, data->len);
  GQuark y2 = g_array_index(data, Tuple, ry)[1];
  guint rx = gsl_rng_uniform_int(rng_R, data->len);
  GQuark x2 = g_array_index(data, Tuple, rx)[0];
  svec vx2 = vec[0][x2];
  svec vy2 = vec[1][y2];
  float x1y2 = svec_sqdist(vx1, vy2);
  float y1x2 = svec_sqdist(vx2, vy1);
  float dx = update_svec(vx1, vy1, vy2, x1y2, nx);
  float dy = update_svec(vy1, vx1, vx2, y1x2, ny);
  return (dx > dy ? dx : dy);
}

float update_svec(svec x, svec y, svec y2, float xy2, float nx) {
  float sum_move2 = 0;
  float sum_x2 = 0;
  float exy2z = exp(-xy2) / Z;
  for (int i = x->size - 1; i >= 0; i--) {
    float xi = svec_get(x, i);
    float yi = svec_get(y, i);
    float y2i = svec_get(y2, i);
    float move = nx * (yi - xi + exy2z * (xi - y2i));
    xi += move;
    svec_set(x, i, xi);
    sum_move2 += move * move;
    sum_x2 += xi * xi;
  }
  svec_scale(x, 1 / sqrt(sum_x2));
  return sum_move2;
}
  
void init_data() {
  Tuple t;
  qmax = 0;
  data = g_array_new(FALSE, FALSE, sizeof(Tuple));
  foreach_line(buf, "") {
    int i = 0;
    foreach_token(tok, buf) {
      g_assert(i < NTOK);
      GQuark q = g_quark_from_string(tok);
      if (q > qmax) qmax = q;
      t[i++] = q;
    }
    g_assert(i == NTOK);
    g_array_append_val(data, t);
  }
  for (int i = 0; i < NTOK; i++) {
    update_cnt[i] = g_new0(guint, qmax+1);
    cnt[i] = g_new0(guint, qmax+1);
    vec[i] = g_new0(svec, qmax+1);
    best_vec[i] = g_new0(svec, qmax+1);
  }
  for (int i = 0; i < data->len; i++) {
    GQuark *p = g_array_index(data, Tuple, i);    
    for (int j = 0; j < NTOK; j++) {
      int k = p[j];
      g_assert(k <= qmax);
      cnt[j][k]++;
      if (vec[j][k] == NULL) {
	vec[j][k] = svec_alloc(NDIM);
	best_vec[j][k] = svec_alloc(NDIM);
      }
    }
  }
}

void randomize_vectors() {
  for (int j = 0; j < NTOK; j++) {
    for (guint q = 1; q <= qmax; q++) {
      if (vec[j][q] != NULL) {
	svec_randomize(vec[j][q]);
	update_cnt[j][q] = 0;
      }
    }
  }
}

void copy_best_vec() {
  for (int j = 0; j < NTOK; j++) {
    for (guint q = 1; q <= qmax; q++) {
      if (vec[j][q] != NULL) {
	svec_memcpy(best_vec[j][q], vec[j][q]);
      }
    }
  }
}

void free_data() {
  for (int i = 0; i < NTOK; i++) {
    for (int j = 0; j <= qmax; j++) {
      if (vec[i][j] != NULL) {
	svec_free(vec[i][j]);
	svec_free(best_vec[i][j]);
      }
    }
    g_free(vec[i]);
    g_free(best_vec[i]);
    g_free(update_cnt[i]);
  }
  g_array_free(data, TRUE);
}

void init_rng() {
  gsl_rng_env_setup();
  rng_T = gsl_rng_default;
  rng_R = gsl_rng_alloc(rng_T);
}

void free_rng() {
  gsl_rng_free(rng_R);
}

