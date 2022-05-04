CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs --cflags opencv`


.PHONY: clean

Convolution: Convolution.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

Convolution-cu: Convolution.cu
	nvcc -o $@ $< $(LDLIBS)

clean:
	rm -f Convolution Convolution-cu
