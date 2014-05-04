CC=gcc
#CFLAGS=-O3 -D_GNU_SOURCE -Wall -std=c99 -pedantic
CFLAGS=-O3 -DNDEBUG -D_POSIX_C_SOURCE=200809L -D_NO_MUSABLE -Wall -std=c99 -pedantic
#CFLAGS=-g -D_POSIX_C_SOURCE=200809L -Wall -std=c99 -pedantic
LIBS=-lm -lgsl -lgslcblas

all: scode scode-online scode-logl

scode-logl: scode-logl.o dlib.o
	$(CC) $(CFLAGS) $^ -lm -o $@

scode-logl.o: scode-logl.c dlib.h
	$(CC) -c $(CFLAGS) $< -o $@

scode-online: scode-online.o dlib.o
	$(CC) $(CFLAGS) $^ -lm -o $@

scode-online.o: scode-online.c dlib.h
	$(CC) -c $(CFLAGS) $< -o $@

scode: scode.o svec.o dlib.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

scode.o: scode.c svec.h rng.h dlib.h
	$(CC) -c $(CFLAGS) $< -o $@

svec.o: svec.c svec.h rng.h
	$(CC) -c $(CFLAGS) $< -o $@

dlib.o: dlib.c dlib.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	-rm *.o scode scode-online
