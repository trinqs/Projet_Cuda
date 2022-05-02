CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`


.PHONY: clean

BlurCovalution: BlurCovalution.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

BlurCovalution-cu: BlurCovalution.cu
	nvcc -o $@ $< $(LDLIBS)

clean:
	rm -f BlurCovalution BlurCovalution-cu