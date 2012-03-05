#include <stdio.h>
#include <glib.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include "foreach.h"
#include "procinfo.h"
#include "svec.h"
#include "rng.h"
const gsl_rng_type *rng_T;
gsl_rng *rng_R = NULL;

#define NTOK 2		      /* number of tokens per input line */
#define NITER 5		      /* how many times to go over the data */
#define NDIM 25		      /* dimensionality of the embedding */
#define PHI0 100.0	      /* learning rate parameter */
#define NU0 0.1		      /* learning rate parameter */
#define Z 0.146		      /* partition function approximation */

typedef GQuark Tuple[NTOK];

GArray *data;
guint *cnt[NTOK];
float *frq[NTOK];
svec *vec[NTOK];
GQuark qmax;
float maxmove;

int main(int argc, char **argv);
void init_rng();
void free_rng();
void init_data();
void free_data();
void svec_update(Tuple t);
double logL();
double calcZ();

int main(int argc, char **argv) {
  g_message_init();
  g_message("hello");
  init_rng();
  g_message("Reading data");
  init_data();
  g_message("Read %d tuples %d uniq tokens", data->len, qmax);
  g_message("logL=%g", logL());
  g_message("Z=%g (approx %g)", calcZ(), Z);
  for (int iter = 0; iter < NITER; iter++) {
    g_message("Iteration %d", iter);
    maxmove = 0;
    for (int di = 0; di < data->len; di++) {
      svec_update(g_array_index(data, Tuple, di));
    }
    g_message("maxmove=%g", maxmove);
    g_message("logL=%g", logL());
  }
  g_message("Z=%g (approx %g)", calcZ(), Z);
  free_data();
  free_rng();
  g_message("bye");
}

double logL() {
  double l = 0;
  for (int i = 0; i < data->len; i++) {
    GQuark *t = g_array_index(data, Tuple, i);
    GQuark x = t[0];
    GQuark y = t[1];
    float px = frq[0][x];
    float py = frq[1][y];
    svec vx = vec[0][x];
    svec vy = vec[1][y];
    float xy = svec_sqdist(vx, vy);
    l += log(px * py) - xy;
  }
  return (l / data->len - log(Z));
}

double calcZ() {
  double z = 0;
  for (guint x = 0; x <= qmax; x++) {
    if (x % 1000 == 0) fprintf(stderr, ".");
    if (frq[0][x] == 0) continue;
    float px = frq[0][x];
    svec vx = vec[0][x];
    for (guint y = 0; y <= qmax; y++) {
      if (frq[1][y] == 0) continue;
      float py = frq[1][y];
      svec vy = vec[1][y];
      float xy = svec_sqdist(vx, vy);
      z += px * py * exp(-xy);
    }
  }
  fprintf(stderr, "\n");
  return z;
}

void svec_update(Tuple t) {
  float d;
  GQuark x1 = t[0];
  GQuark y1 = t[1];
  guint cx = cnt[0][x1]++;
  guint cy = cnt[1][y1]++;
  float nx = NU0 * (PHI0 / (PHI0 + cx));
  float ny = NU0 * (PHI0 / (PHI0 + cy));
  svec vx1 = vec[0][x1];
  svec vy1 = vec[1][y1];
  d = svec_pull(vx1, vy1, nx);
  if (d > maxmove) maxmove = d;
  d = svec_pull(vy1, vx1, ny);
  if (d > maxmove) maxmove = d;
  guint rx = gsl_rng_uniform_int(rng_R, data->len);
  GQuark x2 = g_array_index(data, Tuple, rx)[0];
  guint ry = gsl_rng_uniform_int(rng_R, data->len);
  GQuark y2 = g_array_index(data, Tuple, ry)[1];
  svec vx2 = vec[0][x2];
  svec vy2 = vec[1][y2];
  float x1y2 = svec_sqdist(vx1, vy2);
  float x2y1 = svec_sqdist(vx2, vy1);
  d = svec_push(vx1, vy2, nx * exp(-x1y2) / Z);
  if (d > maxmove) maxmove = d;
  d = svec_push(vy1, vx2, ny * exp(-x2y1) / Z);
  if (d > maxmove) maxmove = d;
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
    cnt[i] = g_new0(guint, qmax+1);
    frq[i] = g_new0(float, qmax+1);
    vec[i] = g_new0(svec, qmax+1);
  }
  for (int i = 0; i < data->len; i++) {
    GQuark *p = g_array_index(data, Tuple, i);    
    for (int j = 0; j < NTOK; j++) {
      int k = p[j];
      g_assert(k <= qmax);
      frq[j][k]++;
      if (vec[j][k] == NULL) {
	svec v = svec_alloc(NDIM);
	svec_randomize(v);
	vec[j][k] = v;
      }
    }
  }
  for (int i = 0; i < NTOK; i++) {
    for (int j = 0; j <= qmax; j++) {
      if (frq[i][j] == 0) continue;
      frq[i][j] /= data->len;
    }
  }
}

void free_data() {
  for (int i = 0; i < NTOK; i++) {
    for (int j = 0; j <= qmax; j++) {
      if (vec[i][j] != NULL) svec_free(vec[i][j]);
    }
    g_free(vec[i]);
    g_free(cnt[i]);
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

