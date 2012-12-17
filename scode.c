const char *usage = "Usage: scode [OPTIONS] < file\n"
     "file should have columns of arbitrary tokens\n"
     //     "-n NTOK: number of tokens per input line\n"
     "-r RESTART: number of restarts (default 1)\n"
     "-i NITER: number of iterations over data (default 20)\n"
     "-d NDIM: number of dimensions (default 25)\n"
     "-z Z: partition function approximation (default 0.166)\n"
     "-p PHI0: learning rate parameter (default 50.0)\n"
     "-u NU0: learning rate parameter (default 0.2)\n"
     "-s SEED: random seed (default 0)\n"
     "-c calculate real Z (default false)\n"
     "-m merge vectors at output (default false)\n"
     "-w The first line of the input is weights(default false)\n"
     "-e MUL experimental features (default 0)\n"
     "-a prints all embeddings (X,Y1,Y2...) (default false)\n"
     "-v verbose messages (default false)\n";

int NTOK = -1;
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
int NTUPLE = 0;
int WEIGHT = 0;
int EXPER = 0;
int PRINTALL = 0;

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

const gsl_rng_type *rng_T;
gsl_rng *rng_R = NULL;
GArray *data;
guint **update_cnt;
gdouble * weight = NULL;
gdouble * uweight = NULL; /*Updated weights*/
guint * cweight = NULL; /*Weights Counts*/
guint **cnt;
#define frq(i,j) ((double)cnt[i][j]*NTOK/data->len)
svec **vec;
svec **best_vec;
svec dummy_vec;
GQuark qmax;
GQuark NULLFEATID;
#define NULLFEATMARKER "/XX/"

int main(int argc, char **argv);
void init_rng();
void free_rng();
int init_data();
int init_weight();
void randomize_vectors();
void copy_best_vec();
void free_data();
void update_tuple(GQuark *t);
double logL();
double calcZ();

#define msg if(VERBOSE)g_message

int main(int argc, char **argv) {
     int opt;
     while((opt = getopt(argc, argv, "n:r:i:d:z:u:p:s:e:acmwv")) != -1) {
          switch(opt) {
               //               case 'n': NTOK = atoi(optarg); break;
               case 'n': break;
               case 'r': RESTART = atoi(optarg); break;
               case 'i': NITER = atoi(optarg); break;
               case 'd': NDIM = atoi(optarg); break;
               case 'z': Z = atof(optarg); break;
               case 'u': NU0 = atof(optarg); break;
               case 'p': PHI0 = atof(optarg); break;
               case 's': SEED = atoi(optarg); break;
               case 'e': EXPER = atoi(optarg); break;
               case 'c': CALCZ = 1; break;
               case 'm': VMERGE = 1; break;
               case 'w': WEIGHT = 1; break;
               case 'v': VERBOSE = 1; break;
               case 'a': PRINTALL = 1; break;
               default: g_error("%s",usage);
          }
     }

     g_message_init();
     msg("scode -n %d -r %d -i %d -d %d -z %g -u %g -p %g -s %d %s%s%s",
          NTOK, RESTART, NITER, NDIM, Z, NU0, PHI0, SEED,
          (CALCZ ? "-c " : ""), (VMERGE ? "-m " : ""), (VERBOSE ? "-v " : ""));

     init_rng();
     if (SEED) gsl_rng_set(rng_R, SEED);
     if (WEIGHT) NTOK = init_weight();
     /* for(int i = 0 ; i < NTOK ; i++){ */
     /*      fprintf(stderr,"%d:%f ",i,weight[i]); */
     /* } */
     /* fprintf(stderr,"\n"); */
     NTUPLE = init_data();
     msg("Read %d tuples %d uniq tokens", NTUPLE, qmax);

     double best_logL = 0;
     for (int start = 0; start < RESTART; start++) {
          randomize_vectors();
          double ll = logL();
          msg("Restart %d/%d logL0=%g best=%g", 1+start, RESTART, ll, best_logL);
          if (CALCZ) msg("Z=%g (approx %g)", calcZ(), Z);
          for (int iter = 0; iter < NITER; iter++) {
               for (int di = 0; di < NTUPLE; di++) {
                    update_tuple(&g_array_index(data, GQuark, di * NTOK));
               }
               /* fprintf(stderr,"Effect on Likelihood[NTOK:%d]\n",NTOK); */
               //gdouble sum = 0;
               //               for (int di = 1; di < NTOK; di++) {
               //sum += (uweight[di]/cweight[di]);
               //               }
               /* for (int di = 1; di < NTOK; di++) { */
               /*      //                    uweight[di] = uweight[di] / (cweight[di] * sum); */
               /*      fprintf(stderr,"[%d:%f/%d(%f)] ",di,uweight[di],cweight[di],uweight[di]/cweight[di]); */
               /*      uweight[di] = cweight[di] = 0; */
               /* } */
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
          /* Actually there is no need for this, a script can do this with normal output. */
          for (guint q = 1; q <= qmax; q++) {
               printf("%s\t%d", g_quark_to_string(q), cnt[0][q]);
               for (guint t = 0; t < NTOK; t++) {
                    g_assert(best_vec[t][q] != NULL);
                    putchar('\t');
                    svec_print(best_vec[t][q]);
               }
               putchar('\n');
          }
     } else if(EXPER){
          fprintf(stderr,"<<<Experimental features: %d X + Y>>>\n", EXPER);
          for (int di = 0; di < NTUPLE; di++) {
               GQuark *tar = &g_array_index(data, GQuark, di * NTOK);
               for (guint t = 0; t < NTOK; t++) {
                    int q = tar[t];
                    g_assert(best_vec[t][q] != NULL);
                    putchar('\t');
                    if (t == 0){                         
                         svec_mul_print(best_vec[t][q], EXPER);
                    }
                    else{
                         putchar('\t');
                         svec_print(best_vec[t][q]);
                    }
               }
               putchar('\n');
          }
     }
     else if(PRINTALL){
          for (guint t = 0; t < NTOK; t++) {
               for (guint q = 1; q <= qmax; q++) {
                    if (best_vec[t][q] == NULL) continue;
                    printf("%d:%s\t%d\t", t, g_quark_to_string(q), cnt[t][q]);
                    svec_print(best_vec[t][q]);
                    putchar('\n');
               }
          }
     }
     else {			/* regular output */
          //NTOK is set to 1
          for (guint t = 0; t < 1; t++) {
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
     for (int i = 0; i < NTUPLE; i++) {
          GQuark *t = &g_array_index(data, GQuark, i * NTOK);
          GQuark x = t[0];
          GQuark y = t[1];
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

void update_tuple(GQuark *t) {
     /*weighted update*/
     static svec *u = NULL;
     static svec *v = NULL;
     static svec dx = NULL;
     if (u == NULL) u = malloc(NTOK * sizeof(svec));
     if (v == NULL) v = malloc(NTOK * sizeof(svec));
     if (dx == NULL) dx = svec_alloc(NDIM);
     for (int i = 0; i < NTOK; i++) u[i] = vec[i][t[i]];
     for (int i = 0; i < NTOK; i++) {
          /* Sampling values from the marginal distributions. */
          /* Can this be done once, or do we have to resample for every x? */
          if(i > 0 && t[i] == NULLFEATID) continue;
          for (int j = 0; j < NTOK; j++) {
               if (j==i) { v[j] = u[i]; continue;}
               guint r = gsl_rng_uniform_int(rng_R, NTUPLE);
               GQuark y = g_array_index(data, GQuark, r * NTOK + j);
               v[j] = vec[j][y];
               if(i > 0) break;
          }          
          /* Compute the move for u[i] */
          svec_set_zero(dx);
          gdouble ww;
          for (int j = 0; j < NTOK; j++) {
               if (j == i) continue;
               ww = weight == NULL ? 1 : (i > 0 ? weight[i] : weight[j]);
               gdouble push = 0, pull = 0;
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
               /* /\*calculate the effect on likelihood(gradient of weight)*\/ */
               /* if(pull != 0 || push != 0){ */
               /*      uweight[j] += -log(Z) + log(frq(i,t[i]) * frq(j,t[j])) - push * Z; */
               /*      cweight[j] += 1; */
               /* } */
          }
          /* Apply the move scaled by learning parameter */
          guint cx = update_cnt[i][t[i]]++;
          float nx = NU0 * (PHI0 / (PHI0 + cx));
          svec_scale(dx, nx);
          svec_add(u[i], dx);
          svec_normalize(u[i]);
     }
}

#ifdef OLD
void update_tuple_old(GQuark *t) {
     static svec *u = NULL;
     static svec *v = NULL;
     static svec dx = NULL;
     if (u == NULL) u = malloc(NTOK * sizeof(svec));
     if (v == NULL) v = malloc(NTOK * sizeof(svec));
     if (dx == NULL) dx = svec_alloc(NDIM);
     guint tok = NTOK;
     for (int i = 0; i < tok; i++) u[i] = vec[i][t[i]];

     for (int i = 0; i < tok; i++) {

          /* Sampling values from the marginal distributions. */
          /* Can this be done once, or do we have to resample for every x? */
          for (int j = 0; j < tok; j++) {
               if (j==i) { v[j] = u[i]; continue; }
               guint r = gsl_rng_uniform_int(rng_R, NTUPLE);
               GQuark y = g_array_index(data, GQuark, r * NTOK + j);
               v[j] = vec[j][y];
          }

          /* Compute sum distance squared and the push coefficient */
          gdouble d2 = 0;
          for (int j = 0; j < tok; j++) {
               for (int k = 0; k < j; k++) {
                    d2 += svec_sqdist(v[k], v[j]);
               }
          }
          gdouble push = exp(-d2) / Z;

          /* Compute the move for u[i] */
          svec_set_zero(dx);
          for (int j = 0; j < tok; j++) {
               if (j == i) continue;
               for (int d = 0; d < NDIM; d++) {
                    float dxd = svec_get(dx, d);
                    float x = svec_get(u[i], d);
                    float y = svec_get(u[j], d);
                    float z = svec_get(v[j], d);
                    svec_set(dx, d, dxd + (y - x + push * (x - z)));
               }
          }
    
          /* Apply the move scaled by learning parameter */
          guint cx = update_cnt[i][t[i]]++;
          float nx = NU0 * (PHI0 / (PHI0 + cx));
          svec_scale(dx, nx);
          svec_add(u[i], dx);
          svec_normalize(u[i]);
     }
}


float update_tuple_old(GQuark *t) {
     GQuark x1 = t[0];
     GQuark y1 = t[1];
     guint cx = update_cnt[0][x1]++;
     guint cy = update_cnt[1][y1]++;
     float nx = NU0 * (PHI0 / (PHI0 + cx));
     float ny = NU0 * (PHI0 / (PHI0 + cy));
     svec vx1 = vec[0][x1];
     svec vy1 = vec[1][y1];
     guint rx = gsl_rng_uniform_int(rng_R, NTUPLE);
     GQuark x2 = g_array_index(data, GQuark, rx * NTOK);
     guint ry = gsl_rng_uniform_int(rng_R, NTUPLE);
     GQuark y2 = g_array_index(data, GQuark, ry * NTOK + 1);
     svec vx2 = vec[0][x2];
     svec vy2 = vec[1][y2];
     float x1y2 = svec_sqdist(vx1, vy2);
     float y1x2 = svec_sqdist(vx2, vy1);
     float dx = update_svec(vx1, vy1, vy2, x1y2, nx);
     float dy = update_svec(vy1, vx1, vx2, y1x2, ny);
     return (dx > dy ? dx : dy);
}

float update_svec_old(svec x, svec y, svec y2, float xy2, float nx) {
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
#endif
  
int init_weight(){
     int size = 100, i = 0;
     weight = malloc(size * sizeof(gdouble));
     foreach_line(buf, "") {
          foreach_token(tok, buf) {
               weight[i] = atof(tok);
               g_assert(weight[i++] >= 0);
               if(i >= 100) {
                    size *= 2; 
                    weight = realloc(weight, size);
               }
          }
          g_assert(i > 0);
          break;
     }
     return i;
}

int init_data() {
     qmax = 0;
     data = g_array_new(FALSE, FALSE, sizeof(GQuark));
     foreach_line(buf, "") {
          int i = 0;
          foreach_token(tok, buf) {
               GQuark q = g_quark_from_string(tok);               
               if (q > qmax) qmax = q;
               g_array_append_val(data, q);
               if(strcmp(tok,NULLFEATMARKER) == 0) NULLFEATID = q;
               i++;
          }
          if(NTOK == -1) NTOK = i;
          g_assert(i == NTOK); //Each line has equal number of tokens
     }
     g_assert(NTOK > 0);
     update_cnt = malloc(NTOK * sizeof(gpointer));
     cnt = malloc(NTOK * sizeof(gpointer));
     vec = malloc(NTOK * sizeof(gpointer));
     best_vec = malloc(NTOK * sizeof(gpointer));     
     dummy_vec = svec_alloc(NDIM);
     svec_zero(dummy_vec);
     uweight = calloc(NTOK, sizeof(gdouble));
     cweight = calloc(NTOK, sizeof(guint));
     for (int i = 0; i < NTOK; i++) {
          update_cnt[i] = g_new0(guint, qmax+1);
          cnt[i] = g_new0(guint, qmax+1);
          vec[i] = g_new0(svec, qmax+1);
          best_vec[i] = g_new0(svec, qmax+1);
     }
     int N = data->len / NTOK;
     for (int i = 0; i < N; i++) {
          GQuark *p = &g_array_index(data, GQuark, i * NTOK);    
          for (int j = 0; j < NTOK; j++) {
               int k = p[j];
               g_assert(k <= qmax);
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
     svec_free(dummy_vec);
     g_free(weight);
     g_free(cweight);
     g_free(uweight);
}

void init_rng() {
     gsl_rng_env_setup();
     rng_T = gsl_rng_default;
     rng_R = gsl_rng_alloc(rng_T);
}

void free_rng() {
     gsl_rng_free(rng_R);
}

