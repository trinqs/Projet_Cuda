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
                    for (int decalageRow = -1; decalageRow < 2; decalageRow++){
                        for (int decalageCol = -1; decalageCol < 2; decalageCol++ ){
                            g[3*((row)*imgCols+col)+i] += rgb[3*(( row + decalageRow )*imgCols+( col + decalageCol ))+i] * 1; //1 = coefficient de la matrice de convolution à l'indice associé
                        }
                    }
                    //normalisation en dehors de la boucle pour faire moins d'arrondis
                    g[3*((row)*imgCols+col)+i] = g[3*((row)*imgCols+col)+i]/9; //9 = somme des coefficients de la matrice de convolution


                    unsigned char ne = rgb[3*((row-1)*imgCols+(col-1))+i];
                    unsigned char n = rgb[3*((row-1)*imgCols+(col))+i];
                    unsigned char no = rgb[3*((row-1)*imgCols+(col+1))+i];
                    unsigned char o = rgb[3*((row)*imgCols+(col+1))+i];
                    unsigned char so = rgb[3*((row+1)*imgCols+(col+1))+i];
                    unsigned char s = rgb[3*((row+1)*imgCols+(col))+i];
                    unsigned char se = rgb[3*((row+1)*imgCols+(col-1))+i];
                    unsigned char e = rgb[3*((row)*imgCols+(col-1))+i];
                    unsigned char milieu = rgb[3*((row)*imgCols+col)+i];

                     g[3*((row)*imgCols+col)+i] = (ne
                                                + n
                                                + no
                                                + o
                                                + so
                                                + s
                                                + se
                                                + e
                                                + milieu)
                                                /9;

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


int main(int n, char* params[])
{
    Mat m_in;

    cout << n <<endl;

    if (n==2){
        cout << params[0] << endl;
        m_in = cv::imread(params[1], IMREAD_UNCHANGED );
    }else{
        m_in = cv::imread("in.jpeg", IMREAD_UNCHANGED );
    }

    uchar* rgb = m_in.data;
    auto cols = m_in.cols;
    auto rows = m_in.rows;
    auto sizeRGB = 3*(rows * cols);

    auto type = m_in.type();

    uchar* g = new uchar[ 3*(rows * cols)]();

    if(sizeRGB%3==0){
        pasAlpha(rgb,g,cols,rows);
    }
    if(sizeRGB%4==0){
        //de l'alpha
    }

    cv::Mat m_out( rows, cols, type, g );
    cv::imwrite( "out.jpeg", m_out );

    return 0;
}