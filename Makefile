CC=gcc
CFLAGS=-O3 -D_XOPEN_SOURCE -Wall -std=c99 -I. `pkg-config --cflags glib-2.0`
LIBS=`pkg-config --libs glib-2.0` -lm -lgsl -lgslcblas

scode: scode.o svec.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

scode.o: scode.c svec.h
	$(CC) -c $(CFLAGS) $< -o $@

svec.o: svec.c svec.h
	$(CC) -c $(CFLAGS) $< -o $@

kmeans.o: kmeans.c kmeans.h svec.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	-rm *.o scode