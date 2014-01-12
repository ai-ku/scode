CC=gcc
CFLAGS=-O3 -D_GNU_SOURCE -Wall -std=c99 -pedantic
LIBS=-lm -lgsl -lgslcblas

scode: scode.o svec.o dlib.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

scode.o: scode.c svec.h rng.h dlib.h
	$(CC) -c $(CFLAGS) $< -o $@

svec.o: svec.c svec.h rng.h
	$(CC) -c $(CFLAGS) $< -o $@

dlib.o: dlib.c dlib.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	-rm *.o scode
