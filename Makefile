CXX=aarch64-linux-gnu-g++
DEPS_CFLAGS?=$(shell env PKG_CONFIG_PATH=/usr/local/frc/lib/pkgconfig pkg-config --cflags wpilibc)
DEPS_LIBS?=$(shell env PKG_CONFIG_PATH=/usr/local/frc/lib/pkgconfig pkg-config --libs wpilibc)
EXE=multiCameraServerExample
DESTDIR?=/home/pi/

.PHONY: clean build install

build: ${EXE}

install: build
	cp ${EXE} runCamera ${DESTDIR}

clean:
	rm ${EXE} *.o

OBJS=main.o

${EXE}: ${OBJS}
	${CXX} -pthread -Wno-psabi -g -o $@ $^ ${DEPS_LIBS} -Wl,--unresolved-symbols=ignore-in-shared-libs

.cpp.o:
	${CXX} -pthread -Wno-psabi- -g -Og -c -o $@ -std=c++20 ${CXXFLAGS} ${DEPS_CFLAGS} $<
