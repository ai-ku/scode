#ifndef __PROCINFO_H__
#define __PROCINFO_H__
#include <stdio.h>
#include <time.h>
#include <sys/sysinfo.h>

extern int procinfo_cnt;
void procinfo();
void my_log_func(const gchar *log_domain,
		 GLogLevelFlags log_level,
		 const gchar *message,
		 gpointer user_data);

#define dot(n) (((n)==0) ? procinfo_cnt = (fputc('\n', stderr), 0) : ((++procinfo_cnt)%(n>>5) == 0) ? fputc('.', stderr) : 0)

#endif
