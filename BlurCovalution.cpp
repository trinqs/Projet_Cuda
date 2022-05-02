#include "BlurCovalution.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;

using ui32 = unsigned int;


struct complex {
    float r; float i;
    complex(float r, float i) : r(r), i(i) {}
    float magnitude() {return r*r + i*i;}
    complex operator*(const complex& c) {
        return complex(r * c.r - i * c.i, i * c.r + r * c.i);
    }
    complex operator+(const complex& c) {
        return complex(r + c.r, i + c.i);
    }
};


unsigned char julia( int x, int y )
{
    const float scale = 1.5;

    float jx = scale * (float)(dim/2.0f - x)/(dim/2.0f);
    float jy = scale * (float)(dim/2.0f - y)/(dim/2.0f);

    ::complex c(-0.8, 0.156);
    ::complex a(jx, jy);

    for(unsigned int i = 0 ; i < 200 ; ++i) {

        a = a * a + c;

        if(a.magnitude() > 1000) {
            return 0;
        }

    }

    return 255;
}

void pasAlpha( unsigned char* rgb, unsigned char* g, size_t imgCols,size_t imgRow){
    for(int col = 0; col< imgCols;col++){
        for(int row = 0; row< imgRow; row++){
            if(col >0 && col< imgCols && row >0 && row< imgRow){
                //red
                unsigned char ne = rgb[3*((row-1)*imgCols+(col-1))];
                unsigned char n = rgb[3*((row-1)*imgCols+(col))];
                unsigned char no = rgb[3*((row-1)*imgCols+(col+1))];
                unsigned char o = rgb[3*((row)*imgCols+(col+1))];
                unsigned char so = rgb[3*((row+1)*imgCols+(col+1))];
                unsigned char s = rgb[3*((row+1)*imgCols+(col))];
                unsigned char se = rgb[3*((row+1)*imgCols+(col-1))];
                unsigned char e = rgb[3*((row)*imgCols+(col-1))];
                //green
                unsigned char ne = rgb[3*((row-1)*imgCols+(col-1)+1)];
                unsigned char n = rgb[3*((row-1)*imgCols+(col)+1)];
                unsigned char no = rgb[3*((row-1)*imgCols+(col+1)+1)];
                unsigned char o = rgb[3*((row)*imgCols+(col+1)+1)];
                unsigned char so = rgb[3*((row+1)*imgCols+(col+1)+1)];
                unsigned char s = rgb[3*((row+1)*imgCols+(col)+1)];
                unsigned char se = rgb[3*((row+1)*imgCols+(col-1)+1)];
                unsigned char e = rgb[3*((row)*imgCols+(col-1)+1)];
                //blue
                unsigned char ne = rgb[3*((row-1)*imgCols+(col-1)+2)];
                unsigned char n = rgb[3*((row-1)*imgCols+(col)+2)];
                unsigned char no = rgb[3*((row-1)*imgCols+(col+1)+2)];
                unsigned char o = rgb[3*((row)*imgCols+(col+1)+2)];
                unsigned char so = rgb[3*((row+1)*imgCols+(col+1)+2)];
                unsigned char s = rgb[3*((row+1)*imgCols+(col)+2)];
                unsigned char se = rgb[3*((row+1)*imgCols+(col-1)+2)];
                unsigned char e = rgb[3*((row)*imgCols+(col-1)+2)];
            }
        }
    }
}


int main()
{
    Mat m_in = cv::imread("in.jpg", IMREAD_UNCHANGED );
    auto rgb = m_in.data;

    std::vector< unsigned char > g( m_in.rows * m_in.cols );
    cv::Mat m_out( m_in.rows, m_in.cols, CV_8UC1, g.data() );

    if(rgb.size%3==0){
        //pas d'alpha
    }
    if(rgb.size%4==0){
        //de l'alpha
    }




    return 0;
}