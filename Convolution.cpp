#include "Convolution.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
using namespace cv;

using ui32 = unsigned int;


struct matriceConvolution {
    vector<vector<int>> matrice;
    int cols;
    int rows;
    int sommeCoefficients;
    int facteurMax;

    matriceConvolution(vector<vector<int>> _matrice) : matrice(_matrice) {
        this->cols = _matrice[0].size();
        this->rows = _matrice.size();
        this->sommeCoefficients = 0;
        int sommeNegative = 0;
        int sommePositive = 0;
        for (int i=0; i<_matrice.size(); i++){
            for (int j=0; j< _matrice[0].size(); j++){
                this->sommeCoefficients += _matrice[i][j];
                if (_matrice[i][j] < 0){
                    sommeNegative +=_matrice[i][j];
                }else{
                    sommePositive += _matrice[i][j];
                }
            }
        }
        this->facteurMax = max(sommePositive,(sommeNegative*-1));
    }

};

void pasAlpha( unsigned char* bgr, unsigned char* g, size_t imgCols,size_t imgRow, matriceConvolution noyau){
    int limCols = noyau.cols/2;
    int limRows = noyau.rows/2;
    for(int col = 0; col< imgCols;col++){
        for(int row = 0; row< imgRow; row++){
            if(col >= limCols && col< imgCols-limCols && row >= limRows && row < imgRow-limRows){

                for( int i=0; i<3; i++){

                    auto sum=0;

                    //cout << "\n" << endl;

                    for (int decalageCol = -limCols; decalageCol < limCols+1; decalageCol++){
                        for (int decalageRow = -limRows; decalageRow < limRows+1; decalageRow++){
                            sum += bgr[3*(( row + decalageRow )*imgCols+( col + decalageCol ))+i] * noyau.matrice[ decalageRow + limRows ][ decalageCol + limCols ];//coefficient de la matrice de convolution à l'indice associé, on fait la rotation en même temps par le calcul d'indice
                        }
                    }
                    //normalisation en dehors de la boucle pour faire moins d'arrondis
                    if (noyau.sommeCoefficients==noyau.facteurMax){
                        sum/= noyau.facteurMax;
                    }

                    if (sum < 0){
                        sum=0;
                    } else if(sum >255){
                        sum=255;
                    }



                    g[3*(row*imgCols+col)+i] = sum;
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

void blur3Convolution(int n, char* params[], unsigned char* bgr, size_t cols, size_t rows, int sizebgr, auto type){
     uchar* g = new uchar[ 3*(rows * cols)]();
                    matriceConvolution noyau = matriceConvolution(
                            vector<vector<int>>({ {1,1,1} , {1,1,1} , {1,1,1} })
                    );
                    if(sizebgr%3==0){
                        pasAlpha(bgr,g,cols,rows, noyau);

                    }else if(sizebgr%4==0){
                        //de l'alpha
                    }

                    cv::Mat m_out( rows, cols, type, g );
                    if (n==3){
                        string res = "out_blur3_";
                        res.append(params[2]);
                        cv::imwrite( res, m_out );
                    }else if(n==2){
                        string res = "out_blur3_";
                        res.append(params[1]);
                        cv::imwrite( res, m_out );
                    }else{
                        string res = "out_blur3";
                        res.append(".jpeg");
                        cv::imwrite( res, m_out );
                    }
}

void blur5Convolution(int n, char* params[], unsigned char* bgr, size_t cols, size_t rows, int sizebgr, auto type){
    uchar* g = new uchar[ 3*(rows * cols)]();
                        matriceConvolution noyau = matriceConvolution(
                                vector<vector<int>>({ {1,1,1,1,1} , {1,1,1,1,1} , {1,1,1,1,1}, {1,1,1,1,1}, {1,1,1,1,1} })
                        );
                        if(sizebgr%3==0){
                            pasAlpha(bgr,g,cols,rows, noyau);

                        }else if(sizebgr%4==0){
                            //de l'alpha
                        }

                        cv::Mat m_out( rows, cols, type, g );
                        if (n==3){
                            string res = "out_blur5_";
                            res.append(params[2]);
                            cv::imwrite( res, m_out );
                        }else if(n==2){
                            string res = "out_blur5_";
                            res.append(params[1]);
                            cv::imwrite( res, m_out );
                        }else{
                            string res = "out_blur5";
                            res.append(".jpeg");
                            cv::imwrite( res, m_out );
                        }
}

void blur11Convolution(int n, char* params[], unsigned char* bgr, size_t cols, size_t rows, int sizebgr, auto type){
    uchar* g = new uchar[ 3*(rows * cols)]();
                            matriceConvolution noyau = matriceConvolution(
                                    vector<vector<int>>({ {1,1,1,1,1,1,1,1,1,1,1} , {1,1,1,1,1,1,1,1,1,1,1} , {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1} })
                            );
                            if(sizebgr%3==0){
                                pasAlpha(bgr,g,cols,rows, noyau);

                            }else if(sizebgr%4==0){
                                //de l'alpha
                            }

                            cv::Mat m_out( rows, cols, type, g );
                            if (n==3){
                                string res = "out_blur11_";
                                res.append(params[2]);
                                cv::imwrite( res, m_out );
                            }else if(n==2){
                                string res = "out_blur11_";
                                res.append(params[1]);
                                cv::imwrite( res, m_out );
                            }else{
                                string res = "out_blur11";
                                res.append(".jpeg");
                                cv::imwrite( res, m_out );
                            }
}

void gaussianBlur3Convolution(int n, char* params[], unsigned char* bgr, size_t cols, size_t rows, int sizebgr, auto type){
    uchar* g = new uchar[ 3*(rows * cols)]();
                                matriceConvolution noyau = matriceConvolution(
                                    vector<vector<int>>({ {1,2,1} , {2,4,2} , {1,2,1} })
                                );
                                if(sizebgr%3==0){
                                    pasAlpha(bgr,g,cols,rows, noyau);

                                }else if(sizebgr%4==0){
                                    //de l'alpha
                                }

                                cv::Mat m_out( rows, cols, type, g );
                                if (n==3){
                                    string res = "out_gaussianBlur3_";
                                    res.append(params[2]);
                                    cv::imwrite( res, m_out );
                                }else if(n==2){
                                    string res = "out_gaussianBlur3_";
                                    res.append(params[1]);
                                    cv::imwrite( res, m_out );
                                }else{
                                    string res = "out_gaussianBlur3";
                                    res.append(".jpeg");
                                    cv::imwrite( res, m_out );
                                }
}

void sharpness3Convolution(int n, char* params[], unsigned char* bgr, size_t cols, size_t rows, int sizebgr, auto type){
        uchar* g = new uchar[ 3*(rows * cols)]();
                                    matriceConvolution noyau = matriceConvolution(
                                        vector<vector<int>>({ {0,-1,0} , {-1,5,-1} , {0,-1,0} })
                                    );
                                    if(sizebgr%3==0){
                                        pasAlpha(bgr,g,cols,rows, noyau);

                                    }else if(sizebgr%4==0){
                                        //de l'alpha
                                    }

                                    cv::Mat m_out( rows, cols, type, g );
                                    if (n==3){
                                        string res = "out_sharpness3_";
                                        res.append(params[2]);
                                        cv::imwrite( res, m_out );
                                    }else if(n==2){
                                        string res = "out_sharpness3_";
                                        res.append(params[1]);
                                        cv::imwrite( res, m_out );
                                    }else{
                                        string res = "out_sharpness3";
                                        res.append(".jpeg");
                                        cv::imwrite( res, m_out );
                                    }
}

void detectEdges3Convolution(int n, char* params[], unsigned char* bgr, size_t cols, size_t rows, int sizebgr, auto type){
        uchar* g = new uchar[ 3*(rows * cols)]();
                                    matriceConvolution noyau = matriceConvolution(
                                        vector<vector<int>>({ {-1,-1,-1} , {-1,8,-1} , {-1,-1,-1} })
                                    );
                                    if(sizebgr%3==0){
                                        pasAlpha(bgr,g,cols,rows, noyau);

                                    }else if(sizebgr%4==0){
                                        //de l'alpha
                                    }

                                    cv::Mat m_out( rows, cols, type, g );
                                    if (n==3){
                                        string res = "out_detectEdges3_";
                                        res.append(params[2]);
                                        cv::imwrite( res, m_out );
                                    }else if(n==2){
                                        string res = "out_detectEdges3_";
                                        res.append(params[1]);
                                        cv::imwrite( res, m_out );
                                    }else{
                                        string res = "out_detectEdges3";
                                        res.append(".jpeg");
                                        cv::imwrite( res, m_out );
                                    }
}

int main(int n, char* params[])
{
    Mat m_in;
    if (n==2 || n==3){
        m_in = cv::imread(params[1], IMREAD_UNCHANGED );
    }else{
        m_in = cv::imread("in.jpeg", IMREAD_UNCHANGED );
    }

    uchar* bgr = m_in.data;

    auto cols = m_in.cols;
    auto rows = m_in.rows;
    auto sizebgr = 3*(rows * cols);

    auto type = m_in.type();

    uchar* g = new uchar[ 3*(rows * cols)]();

    blur3Convolution(n,params,bgr,cols,rows, sizebgr, type);
    blur5Convolution(n,params,bgr,cols,rows, sizebgr, type);
    blur11Convolution(n,params,bgr,cols,rows, sizebgr, type);
    gaussianBlur3Convolution(n,params,bgr,cols,rows, sizebgr, type);
    detectEdges3Convolution(n,params,bgr,cols,rows, sizebgr, type);
    sharpness3Convolution(n,params,bgr,cols,rows, sizebgr, type);

    return 0;
}
