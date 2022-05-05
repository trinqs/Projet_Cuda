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

void pasAlpha( unsigned char* rgb, unsigned char* g, size_t imgCols,size_t imgRow, matriceConvolution noyau){
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

                            sum += rgb[3*(( row + decalageRow )*imgCols+( col + decalageCol ))+i] * noyau.matrice[ decalageRow + limRows ][ decalageCol + limCols ];//coefficient de la matrice de convolution à l'indice associé, on fait la rotation en même temps par le calcul d'indice
                            if (row == 192 && col == 211 ){
                                cout << "\n coordonnees du pixel dans le calcul(" << (row+decalageRow) << "," << ( col + decalageCol ) << ")\n valeur du pixel : " <<  (int)rgb[3*(( row + decalageRow )*imgCols+( col + decalageCol ))+i] <<endl;
                                cout << "coefficient de la matrice " << noyau.matrice[ decalageRow + limRows ][ decalageCol + limCols ] << endl;
                                cout << "\n" << endl;
                            }
                        }

                    }

                    //normalisation en dehors de la boucle pour faire moins d'arrondis
                    if (row == 192 && col == 211 ){
                        cout << "somme des coefficients : " << noyau.sommeCoefficients << endl;
                        cout << "\n Valeur de la sum AVANT division : " << sum << endl;
                    }
                    /*if (noyau.sommeCoefficients !=0){
                        sum/= noyau.facteurMax;
                         // somme des coefficients de la matrice de convolution

                    }else{
                        sum/=noyau.facteurMax;
                    }

                    if (row == 192 && col == 211){
                        cout << "\n Valeur de la sum APRES division : " << sum << "\n" <<endl;
                    }

                    if (sum < 0){
                        sum = (sum*-1)%255;
                    }*/

                    //sum/= noyau.facteurMax;

                    if (sum < 0){
                        sum=0;
                    } else if(sum >255){
                        sum%=255;
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

int main(int n, char* params[])
{
    Mat m_in;
    if (n==2 || n==3){
        m_in = cv::imread(params[1], IMREAD_UNCHANGED );
    }else{
        m_in = cv::imread("in.jpeg", IMREAD_UNCHANGED );
    }

    uchar* rgb = m_in.data;

    auto cols = m_in.cols;
    auto rows = m_in.rows;
    auto sizeRGB = 3*(rows * cols);

    auto type = m_in.type();

    vector<string> convolutionList = {"blur3","blur5","blur11","gaussianBlur3", "nettete3", "detectEdges3", "maskBlur5"};

    uchar* g = new uchar[ 3*(rows * cols)]();


    for (int i=0; i< convolutionList.size(); i++){
        if (convolutionList[i]==("blur3")){
                matriceConvolution noyau = matriceConvolution(
                        vector<vector<int>>({ {1,1,1} , {1,1,1} , {1,1,1} })
                );
                if(sizeRGB%3==0){
                    pasAlpha(rgb,g,cols,rows, noyau);

                }
                if(sizeRGB%4==0){
                    //de l'alpha
                }

                cv::Mat m_out( rows, cols, type, g );
                if (n==3){
                    string res = "out_" + convolutionList[i] + "_";
                    res.append(params[2]);
                    cv::imwrite( res, m_out );
                }else if(n==2){
                    string res = "out_" +  convolutionList[i] + "_";
                    res.append(params[1]);
                    cv::imwrite( res, m_out );
                }else{
                    string res = "out_" + convolutionList[i];
                    res.append(".jpeg");
                    cv::imwrite( res, m_out );
                }

        }else if (convolutionList[i]==("blur5")){
                matriceConvolution noyau = matriceConvolution(
                        vector<vector<int>>({ {1,1,1,1,1} , {1,1,1,1,1} , {1,1,1,1,1}, {1,1,1,1,1}, {1,1,1,1,1} })
                );

                if(sizeRGB%3==0){
                    pasAlpha(rgb,g,cols,rows, noyau);

                }
                if(sizeRGB%4==0){
                    //de l'alpha
                }

                cv::Mat m_out( rows, cols, type, g );
                if (n==3){
                    string res = "out_" + convolutionList[i] + "_";
                    res.append(params[2]);
                    cv::imwrite( res, m_out );
                }else if(n==2){
                    string res = "out_" +  convolutionList[i] + "_";
                    res.append(params[1]);
                    cv::imwrite( res, m_out );
                }else{
                    string res = "out_" + convolutionList[i];
                    res.append(".jpeg");
                    cv::imwrite( res, m_out );
                }


        }else if (convolutionList[i]==("blur11")){
                matriceConvolution noyau = matriceConvolution(
                        vector<vector<int>>({ {1,1,1,1,1,1,1,1,1,1,1} , {1,1,1,1,1,1,1,1,1,1,1} , {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1,1,1} })
                );

                if(sizeRGB%3==0){
                    pasAlpha(rgb,g,cols,rows, noyau);

                }
                if(sizeRGB%4==0){
                    //de l'alpha
                }

                cv::Mat m_out( rows, cols, type, g );
                if (n==3){
                    string res = "out_" + convolutionList[i] + "_";
                    res.append(params[2]);
                    cv::imwrite( res, m_out );
                }else if(n==2){
                    string res = "out_" +  convolutionList[i] + "_";
                    res.append(params[1]);
                    cv::imwrite( res, m_out );
                }else{
                    string res = "out_" + convolutionList[i];
                    res.append(".jpeg");
                    cv::imwrite( res, m_out );
                }


        }else if (convolutionList[i]==("gaussianBlur3")){
                matriceConvolution noyau = matriceConvolution(
                        vector<vector<int>>({ {1,2,1} , {2,4,2} , {1,2,1} })
                );

                if(sizeRGB%3==0){
                    pasAlpha(rgb,g,cols,rows, noyau);

                }
                if(sizeRGB%4==0){
                    //de l'alpha
                }

                cv::Mat m_out( rows, cols, type, g );
                if (n==3){
                    string res = "out_" + convolutionList[i] + "_";
                    res.append(params[2]);
                    cv::imwrite( res, m_out );
                }else if(n==2){
                    string res = "out_" +  convolutionList[i] + "_";
                    res.append(params[1]);
                    cv::imwrite( res, m_out );
                }else{
                    string res = "out_" + convolutionList[i];
                    res.append(".jpeg");
                    cv::imwrite( res, m_out );
                }

        }else if (convolutionList[i]==("nettete3")){
                matriceConvolution noyau = matriceConvolution(
                        vector<vector<int>>({ {0,-1,0} , {-1,5,-1} , {0,-1,0} })
                );

                if(sizeRGB%3==0){
                    pasAlpha(rgb,g,cols,rows, noyau);

                }
                if(sizeRGB%4==0){
                    //de l'alpha
                }

                cv::Mat m_out( rows, cols, type, g );
                if (n==3){
                    string res = "out_" + convolutionList[i] + "_";
                    res.append(params[2]);
                    cv::imwrite( res, m_out );
                }else if(n==2){
                    string res = "out_" +  convolutionList[i] + "_";
                    res.append(params[1]);
                    cv::imwrite( res, m_out );
                }else{
                    string res = "out_" + convolutionList[i];
                    res.append(".jpeg");
                    cv::imwrite( res, m_out );
                }
        }else if (convolutionList[i]==("detectEdges3")){
                matriceConvolution noyau = matriceConvolution(
                        vector<vector<int>>({ {-1,-1,-1} , {-1,8,-1} , {-1,-1,-1} })
                );

                if(sizeRGB%3==0){
                    pasAlpha(rgb,g,cols,rows, noyau);

                }
                if(sizeRGB%4==0){
                    //de l'alpha
                }

                cv::Mat m_out( rows, cols, type, g );
                if (n==3){
                    string res = "out_" + convolutionList[i] + "_";
                    res.append(params[2]);
                    cv::imwrite( res, m_out );
                }else if(n==2){
                    string res = "out_" +  convolutionList[i] + "_";
                    res.append(params[1]);
                    cv::imwrite( res, m_out );
                }else{
                    string res = "out_" + convolutionList[i];
                    res.append(".jpeg");
                    cv::imwrite( res, m_out );
                }

        } else if (convolutionList[i]==("maskBlur5")){
                matriceConvolution noyau = matriceConvolution(
                        vector<vector<int>>({ {1,4,6,4,1} , {4,16,24,16,4} , {6,24,-476,24,6}, {4,16,24,16,4}, {1,4,6,4,1} })
                );

                if(sizeRGB%3==0){
                    pasAlpha(rgb,g,cols,rows, noyau);

                }
                if(sizeRGB%4==0){
                    //de l'alpha
                }

                cv::Mat m_out( rows, cols, type, g );
                if (n==3){
                    string res = "out_" + convolutionList[i] + "_";
                    res.append(params[2]);
                    cv::imwrite( res, m_out );
                }else if(n==2){
                    string res = "out_" +  convolutionList[i] + "_";
                    res.append(params[1]);
                    cv::imwrite( res, m_out );
                }else{
                    string res = "out_" + convolutionList[i];
                    res.append(".jpeg");
                    cv::imwrite( res, m_out );
                }
        }
    }





    return 0;
}
