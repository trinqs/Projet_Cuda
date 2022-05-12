#include "Convolution.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;

using ui32 = unsigned int;



struct matriceConvolution {
    int* matrice;
    int cols;
    int rows;
    int sommeCoefficients;
    int facteurMax;

    __host__ __device__ matriceConvolution(int* _matrice,int tailleMatrice): matrice(_matrice), cols(tailleMatrice), rows(tailleMatrice){

        //this ->matrice = _matrice.data();
        this->sommeCoefficients = 0;
        int sommeNegative = 0;
        int sommePositive = 0;
        for (int i=0; i<rows; i++){
            for (int j=0; j< cols; j++){
                this->sommeCoefficients += _matrice[i*cols+j];
                if (_matrice[i*cols+j] < 0){
                    sommeNegative +=_matrice[i*cols+j];
                }else{
                    sommePositive += _matrice[i*cols+j];
                }
            }
        }
        this->facteurMax = max(sommePositive,(sommeNegative*-1));
    }

    __device__ __host__  int getCols(){ return cols;}
    __device__ __host__ int getRows(){ return rows;}
    __device__ __host__ int getSommeCoefficients(){ return sommeCoefficients;}
    __device__ __host__ int getFacteurMax(){ return facteurMax;}
    __device__ __host__ int* getMatrice(){ return matrice;};

};


__device__ unsigned char calculPixel(int x, int y, // le thread,
                                     int imgCols, int imgRows, // taille de l'image
                                     int limCols, int limRows, // la taille du noyau
                                     int couleur, // quelle couche de pixel
                                     unsigned char* rgb, matriceConvolution noyau,
                                     int * matriceNoyau){ // le tableau des pixel de l'image, la matrice de convolution
    int sum=0;
    //printf(" x :%d , y: %d \n", x, y);
    //printf(" couleur :%d  \n", couleur);
    for (int decalageCol = -limCols; decalageCol < limCols+1; decalageCol++){
        for (int decalageRow = -limRows; decalageRow < limRows+1; decalageRow++){

            //coefficient de la matrice de convolution à l'indice associé, on fait la rotation en même temps par le calcul d'indice
            sum += rgb[3*(( x + decalageRow )*imgCols+( y + decalageCol ))+couleur] * matriceNoyau[ (decalageRow + limRows) * noyau.getCols() + decalageCol + limCols ];
            if(x==6 && y==7){
                int indiceRGB = 3*(( x + decalageRow )*imgCols+( y + decalageCol ))+couleur;
                int indiceNoyau = (decalageRow + limRows) * noyau.getCols() + decalageCol + limCols;
                /*printf(" x :%d, y: %d, couleur: %d, sum : %d \n"
                       "indice dans rgb : %d\n"
                       "indice dans le noyau : %d\n"
                       "valeur du noyau : %d\n", x, y, couleur, sum, indiceRGB, indiceNoyau, matriceNoyau[indiceNoyau] );*/
                sum = 2;
            }
        }
    }

    if(x==6 && y==7){
        printf("sum avant div %d\n",sum);
    }
    //normalisation en dehors de la boucle pour faire moins d'arrondis
    if (noyau.getSommeCoefficients()==noyau.getFacteurMax()){
        sum/= noyau.getFacteurMax();
        if(x==6 && y==7){
            printf("facteur max = %d sum après div %d\n",noyau.getFacteurMax(),sum);
        }
    }

    // j'ai 0 si je print ici
    if (sum < 0){
        sum=0;
    } else if(sum >255){
        sum=255;
    }

    return sum;
}

__global__ void pasAlpha(unsigned char* rgb, unsigned char* g, int imgCol, int imgRow, matriceConvolution noyau, int* matriceNoyau){
    //printf("Dans le kernel, on comprend R, nb ligne : %d , nb cols : %d\n",imgRow, imgCol);
    int limCols = noyau.getCols()/2;
    int limRows = noyau.getRows()/2;

    //int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    //int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    int tidx = blockIdx.y;
    int tidy = threadIdx.y;

    // si c'est pas un bord
    if( tidy >= limCols && tidy< imgCol-limCols && tidx >= limRows && tidx < imgRow-limRows){
        for( int i=0; i<3; i++){
            g[3*(tidx*imgCol+tidy)+i] = calculPixel(tidx,tidy,imgCol,imgRow,limCols,limRows,i,rgb,noyau,matriceNoyau);
            //g[3*(tidx*imgCol+tidy)+i] = rgb[3*(tidx*imgCol+tidy)+i];


            /*int indice = 3*(tidx*imgCol+tidy)+i;
            //if((tidx==9 && tidy==1) || (tidx==0 && tidy==2)) {
            //if(tidx==88 && tidy==89){
            //if(131<=tidx && tidx<=141 && tidy==108){
                printf("\ntidx : %d , tidy : %d \n"
                       "non bord\n"
                       "couleur : %d \n"
                       "indice : %d\n"
                       "valeur du tableau rgb : %d\n"
                       "valeur du tableau g après : %d\n", tidx, tidy, i, indice, rgb[indice], g[indice]);
            */
        }
        }
    else{
        for(int i= 0; i<3;i++){
            //g[3*((tidx)*imgCols+tidy)+i] = 255;
            g[3*(tidx*imgCol+tidy)+i] = rgb[3*(tidx*imgCol+tidy)+i];
            //int indice = 3*(tidx*imgCol+tidy)+i;
            /*//if((tidx==9 && tidy==1) || (tidx==0 && tidy==2)) {
            //if(tidx==88 && tidy==89){
            //if(131<=tidx && tidx<=141 && tidy==108){

                printf("\ntidx : %d , tidy : %d \n"
                       "bord\n"
                       "couleur : %d \n"
                       "indice : %d\n"
                       "valeur du tableau rgb : %d\n"
                       "valeur du tableau g après : %d\n", tidx, tidy, i, indice, rgb[indice], g[indice]);
            //}*/
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

    printf("m_in data");
    for (int i=0 ; i<300;i+=3){
        if(i%30==0){
            printf("\n");
        }
        printf("{ %d, %d, %d} ", bgr[i],bgr[i+1],bgr[i+2]);
    }
    printf("\n");

    int cols = m_in.cols;
    int rows = m_in.rows;
    auto sizeBgr = 3*(cols*rows);

    auto type = m_in.type();

    std::vector<unsigned char > g(3*cols*rows);



    unsigned char * bgr_d;
    unsigned char * g_d;

    vector<string> convolutionList = {"blur3","blur5","blur11","gaussianBlur3", "nettete3", "detectEdges3"};
    cudaMalloc(&bgr_d, sizeBgr);
    cudaMalloc(&g_d, 3*cols*rows);

    cudaMemcpy(bgr_d,bgr,sizeBgr, cudaMemcpyHostToDevice);


    //int nbThreadMaxParBloc = 1024;
    //dim3 block( 32, 4 );
    //dim3 grid( (cols-1)/block.y+1,(rows-1)/block.x+1 );
    dim3 nbThreadParBlock(1,cols,1);
    dim3 nbBlock(1,rows,1);

    for (int i=0; i< convolutionList.size(); i++){
        if (convolutionList[i]==("blur3")){

            int tailleNoyau = 3;
            vector<int> matrice({1,1,1,
                                 1,1,1,
                                 1,1,1});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyau);

            int* noyau_d;
            cudaMalloc(&noyau_d, tailleNoyau*tailleNoyau);
            cudaMemcpy(noyau_d,matrice.data(),tailleNoyau*tailleNoyau, cudaMemcpyHostToDevice);

            /*for (int j=0;j <= tailleNoyau*tailleNoyau-1; j++){
                printf("\nindice du noyau : %d, valeur du noyau : %d\n", j, noyau.getMatrice()[j]);
            }*/
            if(sizeBgr%3==0){
                //printf("nb de colones : %d, nb de lignes : %d \n", cols, rows);

                pasAlpha<<< nbBlock, nbThreadParBlock >>>( bgr_d, g_d, cols, rows, noyau,noyau_d);
            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );

            cudaMemcpy(g.data(),g_d,3*cols*rows,cudaMemcpyDeviceToHost);


            auto gData = g.data();

            printf("m_out data");
            for (int pixel=0 ; pixel<300;pixel+=3){
                if(pixel%30==0){
                    printf("\n");
                }
                printf("{ %d, %d, %d} ", gData[pixel],gData[pixel+1],gData[pixel+2]);
            }
            printf("\n");


            if (n==3){
                string res = "out_cu_" + convolutionList[i] + "_";
                res.append(params[2]);
                cv::imwrite( res, m_out );
            }else if(n==2){
                string res = "out_cu_" +  convolutionList[i] + "_";
                res.append(params[1]);
                cv::imwrite( res, m_out );
            }else{
                string res = "out_cu_" + convolutionList[i];
                res.append(".jpeg");
                cv::imwrite( res, m_out );
            }

            cudaFree(noyau_d);
        /*}else if (convolutionList[i]==("blur5")){

            int tailleNoyaux = 5;
            vector<int> matrice({1,1,1,1,1,
                                 1,1,1,1,1,
                                 1,1,1,1,1,
                                 1,1,1,1,1,
                                 1,1,1,1,1});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyaux);


            if(sizeBgr%3==0){
                pasAlpha<<<block,grid>>>( bgr_d, g_d, cols,rows, noyau);

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

            int tailleNoyaux = 11;
            vector<int> matrice({1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyaux);

            if(sizeBgr%3==0){
                pasAlpha<<<block,grid>>>( bgr_d, g_d, cols,rows, noyau);

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

            int tailleNoyaux = 3;
            vector<int> matrice({1,2,1,
                                 2,4,2,
                                 1,2,1});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyaux);

            if(sizeBgr%3==0){
                pasAlpha<<<block,grid>>>( bgr_d, g_d, cols,rows, noyau);

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

            int tailleNoyaux = 3;
            vector<int> matrice({0,-1,0,
                                 -1,5,-1,
                                 0,-1,0});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyaux);


            if(sizeBgr%3==0){
                pasAlpha<<<block,grid>>>( bgr_d, g_d, cols,rows, noyau);

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

            int tailleNoyaux = 3;
            vector<int> matrice({-1,-1,-1,
                                 -1,8,-1,
                                 -1,-1,-1});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyaux);

            if(sizeBgr%3==0){
                pasAlpha<<<block,grid>>>( bgr_d, g_d, cols,rows, noyau);

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
            }*/
        }
    }
    cudaFree(bgr_d);
    cudaFree(g_d);


    return 0;
}