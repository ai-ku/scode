#ifndef __SVEC_H__
#define __SVEC_H__

#define HAVE_INLINE
#include <gsl/gsl_vector.h>

typedef gsl_vector_float *svec;

#define svec_alloc gsl_vector_float_alloc
#define svec_free gsl_vector_float_free

void svec_randomize(svec x);
void svec_normalize(svec x);
float svec_pull(svec x, svec y, float d);
float svec_push(svec x, svec y, float d);
float svec_sqdist(svec x, svec y);

#endif
