CC=gcc
CFLAGS=-O3 -D_GNU_SOURCE -Wall -std=c99 -I. `pkg-config --cflags glib-2.0`
LIBS=`pkg-config --libs glib-2.0` -lm -lgsl -lgslcblas

scode: scode.o svec.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

scode.o: scode.c svec.h rng.h foreach.h procinfo.h
	$(CC) -c $(CFLAGS) $< -o $@

svec.o: svec.c svec.h rng.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	-rm *.o scode
