#include "Convolution.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;

using ui32 = unsigned int;



struct matriceConvolution {
    vector<vector<int>> matrice;
    int cols;
    int rows;
    int sommeCoefficients;
    int facteurMax;

    __host__ __device__ matriceConvolution(vector<vector<int>> _matrice) : matrice(_matrice) {
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

    __device__ __host__  int getCols(){ return cols;}
    __device__ __host__ int getRows(){ return rows;}
    __device__ __host__ int getSommeCoefficients(){ return sommeCoefficients;}
    __device__ __host__ int getFacteurMax(){ return facteurMax;}
    __device__ __host__ vector<vector<int>> getMatrice(){ return matrice;};

};


__device__ unsigned char calculPixel(int x, int y, // le thread,
                                     size_t imgCols, size_t imgRows, // taille de l'image
                                     int limCols, int limRows, // la taille du noyay
                                     int couleur, // quelle couche de pixel
                                     unsigned char* rgb, matriceConvolution noyau){ // le tableau des pixel de l'image, la matrice de convolution
    auto sum=0;

    for (int decalageCol = -limCols; decalageCol < limCols+1; decalageCol++){
        for (int decalageRow = -limRows; decalageRow < limRows+1; decalageRow++){

            //coefficient de la matrice de convolution à l'indice associé, on fait la rotation en même temps par le calcul d'indice
            //sum += rgb[3*(( x + decalageRow )*imgCols+( y + decalageCol ))+couleur] * noyau.getMatrice()[ decalageRow + limRows ][ decalageCol + limCols ];
            sum += rgb[1]
        }
    }
    //normalisation en dehors de la boucle pour faire moins d'arrondis
    if (noyau.getSommeCoefficients()==noyau.getFacteurMax()){
        sum/= noyau.getFacteurMax();
    }

    if (sum < 0){
        sum=0;
    } else if(sum >255){
        sum=255;
    }
    return sum;
}

__global__ void pasAlpha(unsigned char* rgb, unsigned char* g, size_t imgCols,size_t imgRow, matriceConvolution noyau){
    int limCols = noyau.getCols()/2;
    int limRows = noyau.getRows()/2;


    int tidx = threadIdx.x/imgCols-1;
    int tidy = threadIdx.x -(imgCols*tidx);

    // si c'est pas un bord
    if( tidy >= limCols && tidy< imgCols-limCols && tidx >= limRows && tidy < imgRow-limRows){
        for( int i=0; i<3; i++){
            g[3*(tidy*imgCols+tidx)+i] = calculPixel(tidx,tidy,imgCols,imgRow,limCols,limRows,i,rgb,noyau);
        }
    }
    else{
        for(int i= 0; i<3;i++){
            g[3*((tidy)*imgCols+tidy)+i] = 0;
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

    auto bgr = m_in.data; // c'est pas du rgb c'est du bgr

    int cols = m_in.cols;
    int rows = m_in.rows;
    auto sizeBgr = 3*(cols*rows);

    auto type = m_in.type();

    std::vector<unsigned char > g(cols*rows);

    unsigned char * bgr_d;
    unsigned char * g_d;

    vector<string> convolutionList = {"blur3","blur5","blur11","gaussianBlur3", "nettete3", "detectEdges3"};
    cudaMalloc(&bgr_d, sizeBgr);
    cudaMalloc(&g_d, cols*rows);

    cudaMemcpy(bgr_d,bgr,sizeBgr, cudaMemcpyHostToDevice);

    int block =1;
    auto nbthread = cols *rows;

    for (int i=0; i< convolutionList.size(); i++){
        if (convolutionList[i]==("blur3")){
            matriceConvolution noyau = matriceConvolution(
                    vector<vector<int>>({ {1,1,1} , {1,1,1} , {1,1,1} })
            );


            if(sizeBgr%3==0){
                pasAlpha<<<block,nbthread>>>( bgr_d, g_d, cols,rows, noyau);
            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaMemcpy(g.data(),g_d,cols*rows,cudaMemcpyDeviceToHost);
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

            if(sizeBgr%3==0){
                pasAlpha<<<block,nbthread>>>( bgr_d, g_d, cols,rows, noyau);

            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaMemcpy(g.data(),g_d,cols*rows,cudaMemcpyDeviceToHost);
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

            if(sizeBgr%3==0){
                pasAlpha<<<block,nbthread>>>( bgr_d, g_d, cols,rows, noyau);

            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaMemcpy(g.data(),g_d,cols*rows,cudaMemcpyDeviceToHost);
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

            if(sizeBgr%3==0){
                pasAlpha<<<block,nbthread>>>( bgr_d, g_d, cols,rows, noyau);

            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaMemcpy(g.data(),g_d,cols*rows,cudaMemcpyDeviceToHost);
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

            if(sizeBgr%3==0){
                pasAlpha<<<block,nbthread>>>( bgr_d, g_d, cols,rows, noyau);

            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaMemcpy(g.data(),g_d,cols*rows,cudaMemcpyDeviceToHost);
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

            if(sizeBgr%3==0){
                pasAlpha<<<block,nbthread>>>( bgr_d, g_d, cols,rows, noyau);

            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaMemcpy(g.data(),g_d,cols*rows,cudaMemcpyDeviceToHost);
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
    cudaFree(bgr_d);
    cudaFree(g_d);


    return 0;
}