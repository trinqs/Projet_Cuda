#include "BlurCovalution.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;

using ui32 = unsigned int;

/*
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
*/

void pasAlpha( unsigned char* rgb, unsigned char* g, size_t imgCols,size_t imgRow){
    for(int col = 0; col< imgCols;col++){
        for(int row = 0; row< imgRow; row++){
            if(col >0 && col< imgCols && row >0 && row< imgRow){
                for( int i=0; i<3; i++){
                    unsigned char ne = rgb[3*((row-1)*imgCols+(col-1))+i];
                    unsigned char n = rgb[3*((row-1)*imgCols+(col))+i];
                    unsigned char no = rgb[3*((row-1)*imgCols+(col+1))+i];
                    unsigned char o = rgb[3*((row)*imgCols+(col+1))+i];
                    unsigned char so = rgb[3*((row+1)*imgCols+(col+1))+i];
                    unsigned char s = rgb[3*((row+1)*imgCols+(col))+i];
                    unsigned char se = rgb[3*((row+1)*imgCols+(col-1))+i];
                    unsigned char e = rgb[3*((row)*imgCols+(col-1))+i];
                    unsigned char milieu = rgb[3*((row)*imgCols+col)+i];

                    g[3*((row)*imgCols+col)+i] = ne* (1/9)
                                                + n* (1/9)
                                                + no* (1/9)
                                                + o* (1/9)
                                                + so* (1/9)
                                                + s* (1/9)
                                                + se* (1/9)
                                                + e* (1/9)
                                                + milieu * (1/9);
                    cout << (int)ne << endl;
                }
            }
            else{
                for(int i= 0; i<3;i++){
                    g[3*((row)*imgCols+col)+i] = 0;
                }
            }
        }
    }
}

vector< unsigned char > pasAlpha2( unsigned char* rgb, size_t imgCols,size_t imgRow){
    std::vector< unsigned char > res( 3*(imgRow * imgCols));


    for(int col = 0; col< imgCols;col++){
        for(int row = 0; row< imgRow; row++){
            if(col >0 && col< imgCols && row >0 && row< imgRow){
                //cout << "coucou on est rentre" << endl;
                for( int i=0; i<3; i++){
                    unsigned char ne = rgb[3*((row-1)*imgCols+(col-1))+i];
                    unsigned char n = rgb[3*((row-1)*imgCols+(col))+i];
                    unsigned char no = rgb[3*((row-1)*imgCols+(col+1))+i];
                    unsigned char o = rgb[3*((row)*imgCols+(col+1))+i];
                    unsigned char so = rgb[3*((row+1)*imgCols+(col+1))+i];
                    unsigned char s = rgb[3*((row+1)*imgCols+(col))+i];
                    unsigned char se = rgb[3*((row+1)*imgCols+(col-1))+i];
                    unsigned char e = rgb[3*((row)*imgCols+(col-1))+i];
                    unsigned char milieu = rgb[3*((row)*imgCols+col)+i];

                    unsigned char sum = ne* (1/9)
                                        + n* (1/9)
                                        + no* (1/9)
                                        + o* (1/9)
                                        + so* (1/9)
                                        + s* (1/9)
                                        + se* (1/9)
                                        + e* (1/9)
                                        + milieu * (1/9);

                    res[3*((row)*imgCols+col)+i] = sum;
                }
            }
            else{
                for(int i= 0; i<3;i++){

                    res[3*((row)*imgCols+col)+i] = 0;
                }
            }
        }
    }
    return res;
}


int main()
{
    Mat m_in = cv::imread("in.jpeg", IMREAD_UNCHANGED );
    uchar* rgb = m_in.data;
    auto cols = m_in.cols;
    auto rows = m_in.rows;
    auto sizeRGB = 3*(rows * cols);
    for (int i =0; i<3*(rows * cols); i++){
        cout<<rgb[i]<<endl;
    }





    auto type = m_in.type();

    unsigned char* rgb_d = rgb;

    uchar* g = new uchar[ 3*(rows * cols)]();

    if(sizeRGB%3==0){
        pasAlpha(rgb,g,cols,rows);
        //g = pasAlpha2(rgb_d,cols,rows);
    }
    if(sizeRGB%4==0){
        //de l'alpha
    }

    cv::Mat m_out( rows, cols, type, g );
    cv::imwrite( "out.jpeg", m_out );

    return 0;
}