#ifndef __PROCINFO_H__
#define __PROCINFO_H__
#include <stdio.h>
#include <time.h>
#include <sys/sysinfo.h>

#define dot(n) (((n)==0) ? procinfo() : ((++procinfo_cnt)%(n>>5) == 0) ? fputc('.', stderr))

static int procinfo_cnt;

static void procinfo() {
  static clock_t clk;
  static unsigned long mem;
  static struct sysinfo info;

  procinfo_cnt = 0;
  sysinfo(&info);
  if (clk == 0) {
    clk = clock();
    mem = info.freeram;
    fprintf(stderr, "t=%.2f m=%d\n", (float)clk/CLOCKS_PER_SEC, mem);
  } else {
    fprintf(stderr, " t=%.2f m=%d\n", 
	    (float)(clock() - clk) / CLOCKS_PER_SEC,
	    mem - info.freeram);
  }
}

#endif
