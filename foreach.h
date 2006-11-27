#ifndef __FOREACH_H__
#define __FOREACH_H__

#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <errno.h>
#include <stdlib.h>

/* FOREACH: C needs some good looping macros: */

#define foreach(type, var, array)\
  for (register type (var), *_p = (type*)(array)->pdata;\
       _p != NULL; _p = NULL)\
  for (register int _i = 0, _l = (array)->len;\
       (_i < _l) && ((var = _p[_i]) || 1); _i++)

#define foreach_ptr(ptr, array)\
  for (register gpointer *ptr = (array)->pdata;\
       ptr != NULL; ptr = NULL)\
  for (register int _i = 0, _l = (array)->len;\
       (_i < _l); _i++, ptr++)

#define foreach_int(var, lo, hi)\
  for (register int var = (lo), _hi = (hi); var <= _hi; var++)

#define foreach_char(var, str)\
  for (register char var, *_p = (str);\
       (var = *_p) != 0; _p++)

#define LINE (1<<16)

extern FILE *popen (__const char *__command, __const char *__modes);
extern int pclose (FILE *__stream);

static FILE *_myopen(const char *fname) {
  if (NULL == fname) return stdin;
  else if (*fname == '|') return popen((fname)+1, "r");
  else return fopen(fname, "r");
}
static FILE *_myclose(const char *fname, FILE *fp) {
  if (NULL == fname) return NULL;
  else if (*fname == '|') pclose(fp);
  else fclose(fp);
  return NULL;
}

#define foreach_line(str, fname)\
  errno = 0; \
  for (FILE *_fp = _myopen(fname); \
       (_fp != NULL) || (errno && (perror(fname), exit(errno), 0)); \
       _fp = _myclose(fname, _fp)) \
  for (char str[LINE];\
       ((str[LINE - 1] = -1) &&\
        fgets(str, LINE, _fp) &&\
        ((str[LINE - 1] != 0) ||\
         (perror("Line too long"), exit(-1), 0))); )

#define foreach_token(tok, str)\
  for (register char *(tok) = strtok((str), " \t\n\r\f\v");\
       (tok) != NULL; (tok) = strtok(NULL," \t\n\r\f\v"))


#endif

