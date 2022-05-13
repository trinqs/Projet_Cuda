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

    for (int decalageCol = -limCols; decalageCol < limCols+1; decalageCol++){
        for (int decalageRow = -limRows; decalageRow < limRows+1; decalageRow++){
            //coefficient de la matrice de convolution à l'indice associé, on fait la rotation en même temps par le calcul d'indice
            sum += rgb[3*(( x + decalageRow )*imgCols+( y + decalageCol ))+couleur] * matriceNoyau[ (decalageRow + limRows) * noyau.getCols() + decalageCol + limCols ];
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

__global__ void pasAlpha(unsigned char* rgb, unsigned char* g, int imgCol, int imgRow, matriceConvolution noyau, int* matriceNoyau){
    printf("je viens d'entrer");
    int limCols = noyau.getCols()/2;
    int limRows = noyau.getRows()/2;

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    //int tidx = blockIdx.y;
    //int tidy = threadIdx.y;

    if (tidx == 0 && tidy==0){
        for (int i=0; i< noyau.getCols()*noyau.getRows(); i++){
            printf("\nindice du noyau : %d, valeur du noyau, %d\n", i, matriceNoyau[i]);
        }
    }

    // si c'est pas un bord
    if( tidy >= limCols && tidy< imgCol-limCols && tidx >= limRows && tidx < imgRow-limRows){
        for( int i=0; i<3; i++){
            g[3*(tidx*imgCol+tidy)+i] = calculPixel(tidx,tidy,imgCol,imgRow,limCols,limRows,i,rgb,noyau,matriceNoyau);
        }
    }
    else{
        for(int i= 0; i<3;i++){
            g[3*(tidx*imgCol+tidy)+i] = 0;
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

    cudaError_t cudaStatus;
    cudaError_t kernelStatus;

    printf("ligne %d colonne %d \n",rows,cols);

    auto sizeBgr = 3*(cols*rows);

    auto type = m_in.type();

    std::vector<unsigned char > g(3*cols*rows);

    unsigned char * bgr_d;
    unsigned char * g_d;

    vector<string> convolutionList = {"blur3","blur5","blur11","gaussianBlur3", "nettete3", "detectEdges3"};
    cudaStatus = cudaMalloc(&bgr_d, sizeBgr);
    if (cudaStatus != cudaSuccess) {
    	std::cout << "Error CudaMalloc bgr_d"  << std::endl;
    }
    cudaStatus = cudaMalloc(&g_d, sizeBgr);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error CudaMalloc g_d"  << std::endl;
    }

    cudaStatus = cudaMemcpy(bgr_d,bgr,sizeBgr, cudaMemcpyHostToDevice);
    if (cudaStatus  != cudaSuccess) {
        std::cout << "Error cudaMemcpy bgr_d - HostToDevice" << std::endl;
    }

    //int nbThreadMaxParBloc = 1024;
    dim3 nbThreadParBlock( 32, 4);
    dim3 nbBlock( ((cols-1)/nbThreadParBlock.x) +1,((rows-1)/nbThreadParBlock.y) +1 );
    //dim3 nbThreadParBlock(1,cols,1);
    //dim3 nbBlock(1,rows,1);

    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate( &start );
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventCreate start" << std::endl;
    }
    cudaStatus = cudaEventCreate( &stop );
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error cudaEventCreate stop" << std::endl;
    }


    for (int i=0; i< convolutionList.size(); i++){
        if (convolutionList[i]==("blur3")){

            int tailleNoyau = 3;
            vector<int> matrice({1,1,1,
                                 1,1,1,
                                 1,1,1});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyau);

            int* noyau_d;
            cudaStatus = cudaMalloc(&noyau_d, tailleNoyau*tailleNoyau*sizeof(int));
            if (cudaStatus != cudaSuccess) {
                std::cout << "Error (blur3) CudaMalloc noyau_d"  << std::endl;
            }
            cudaStatus = cudaMemcpy(noyau_d,matrice.data(),tailleNoyau*tailleNoyau*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (blur3) cudaMemcpy noyau_d - HostToDevice" << std::endl;
            }

            if(sizeBgr%3==0){
                //printf("nb de colones : %d, nb de lignes : %d \n", cols, rows);
                cudaStatus = cudaEventRecord( start );
                if (cudaStatus != cudaSuccess) {
                    std::cout << "Error (blur3) cudaEventRecord start" << std::endl;
                }
                printf("\njuste avant le kernel\n");
                pasAlpha<<< nbBlock, nbThreadParBlock >>>( bgr_d, g_d, cols, rows, noyau,noyau_d);
                printf("\njuste après le kernel\n");
                kernelStatus = cudaGetLastError();
                if ( kernelStatus != cudaSuccess ) {
                    std::cout << "CUDA Error (blur3) "<< cudaGetErrorString(kernelStatus) << std::endl;
                }

                cudaStatus = cudaEventRecord( stop );
                if (cudaStatus != cudaSuccess) {
                    std::cout << "Error (blur3) cudaEventRecord stop" << std::endl;
                }
                cudaStatus = cudaEventSynchronize( stop );
                if (cudaStatus != cudaSuccess) {
                    std::cout << "Error (blur3) cudaEventSynchronize stop" << std::endl;
                }

                float duration;
                cudaStatus = cudaEventElapsedTime( &duration, start, stop );
                if (cudaStatus != cudaSuccess) {
                    std::cout << "Error (blur3) cudaEventElapsedTime duration" << std::endl;
                }

                std::cout << "blur3 "<< duration <<" ms" <<std::endl;

            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaStatus = cudaMemcpy(g.data(),g_d,3*cols*rows,cudaMemcpyDeviceToHost);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (blur3) cudaMemcpy g_d - DeviceToHost" << std::endl;
            }


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
        }/*

        else if (convolutionList[i]==("blur5")){

            int tailleNoyau = 5;
            vector<int> matrice({1,1,1,1,1,
                                 1,1,1,1,1,
                                 1,1,1,1,1,
                                 1,1,1,1,1,
                                 1,1,1,1,1});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyau);

            int* noyau_d;
            cudaStatus = cudaMalloc(&noyau_d, tailleNoyau*tailleNoyau*sizeof(int));
            if (cudaStatus != cudaSuccess) {
                std::cout << "Error (blur5) CudaMalloc noyau_d"  << std::endl;
            }
            cudaStatus = cudaMemcpy(noyau_d,matrice.data(),tailleNoyau*tailleNoyau*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (blur5) cudaMemcpy noyau_d - HostToDevice" << std::endl;
            }



            if(sizeBgr%3==0){

                cudaStatus = cudaEventRecord( start );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (blur5) cudaEventRecord start " << std::endl;
                }

                pasAlpha<<<nbBlock, nbThreadParBlock>>>( bgr_d, g_d, cols,rows, noyau,noyau_d);
                kernelStatus = cudaGetLastError();
                if ( kernelStatus != cudaSuccess ){
                   std::cout << "CUDA Error (blur5) "<< cudaGetErrorString(kernelStatus) << std::endl;
                }

                cudaStatus = cudaEventRecord( stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (blur5) cudaEventRecord stop " << std::endl;
                }
                cudaStatus = cudaEventSynchronize( stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (blur5) cudaEventSynchronize stop " << std::endl;
                }

                float duration;
                cudaStatus = cudaEventElapsedTime( &duration, start, stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (blur5) cudaEventElapsedTime duration " << std::endl;
                }
                std::cout << "blur5 "<<duration<<" ms" <<std::endl;

            }
            if(sizeBgr%4==0){
                //de l'alpha
            }
            cv::Mat m_out( rows, cols, type, g.data() );

            cudaStatus = cudaMemcpy(g.data(),g_d, 3*cols*rows,cudaMemcpyDeviceToHost);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (blur5) cudaMemcpy g_d - DeviceToHost" << std::endl;
            }

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


        }

        else if (convolutionList[i]==("blur11")){

            int tailleNoyau = 11;
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

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyau);

            int* noyau_d;
            cudaStatus = cudaMalloc(&noyau_d, tailleNoyau*tailleNoyau*sizeof(int));
            if (cudaStatus != cudaSuccess) {
                std::cout << "Error (blur11) CudaMalloc noyau_d"  << std::endl;
            }
            cudaStatus = cudaMemcpy(noyau_d,matrice.data(),tailleNoyau*tailleNoyau*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (blur11) cudaMemcpy noyau_d - HostToDevice" << std::endl;
            }

            if(sizeBgr%3==0){

                cudaStatus = cudaEventRecord( start );
                if (cudaStatus != cudaSuccess) {
                    std::cout << "Error (blur11) cudaEventRecord start"  << std::endl;
                }

                pasAlpha<<<nbBlock, nbThreadParBlock>>>( bgr_d, g_d, cols,rows, noyau,noyau_d);
                kernelStatus = cudaGetLastError();
                if ( kernelStatus != cudaSuccess ) {
                    std::cout << "CUDA Error (blur11) "<< cudaGetErrorString(kernelStatus) << std::endl;
                }

                cudaStatus = cudaEventRecord( stop );
                if (cudaStatus != cudaSuccess) {
                    std::cout << "Error (blur11) cudaEventRecord stop"  << std::endl;
                }
                cudaStatus = cudaEventSynchronize( stop );
                if (cudaStatus != cudaSuccess) {
                    std::cout << "Error (blur11) cudaEventSynchronize stop"  << std::endl;
                }

                float duration;
                cudaStatus = cudaEventElapsedTime( &duration, start, stop );
                if (cudaStatus != cudaSuccess) {
                    std::cout << "Error (blur11) cudaEventElapsedTime duration"  << std::endl;
                }
                std::cout << "blur11 "<<duration<<" ms" <<std::endl;

            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaStatus = cudaMemcpy(g.data(),g_d,3*cols*rows,cudaMemcpyDeviceToHost);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (blur11) cudaMemcpy g_d - DeviceToHost" << std::endl;
            }
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
        }

        else if (convolutionList[i]==("gaussianBlur3")){

            int tailleNoyau = 3;
            vector<int> matrice({1,2,1,
                                 2,4,2,
                                 1,2,1});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyau);

            int* noyau_d;
            cudaStatus = cudaMalloc(&noyau_d, tailleNoyau*tailleNoyau*sizeof(int));
            if (cudaStatus != cudaSuccess) {
                std::cout << "Error (gaussianBlur3) CudaMalloc noyau_d"  << std::endl;
            }
            cudaStatus = cudaMemcpy(noyau_d,matrice.data(),tailleNoyau*tailleNoyau*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (gaussianBlur3) cudaMemcpy noyau_d - HostToDevice" << std::endl;
            }

            if(sizeBgr%3==0){

                cudaStatus = cudaEventRecord( start );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (gaussianBlur3) cudaEventRecord start" << std::endl;
                }

                pasAlpha<<<nbBlock, nbThreadParBlock>>>( bgr_d, g_d, cols,rows, noyau,noyau_d);
                kernelStatus = cudaGetLastError();
                if ( kernelStatus != cudaSuccess ) {
                   std::cout << "CUDA Error (gaussianBlur3) "<< cudaGetErrorString(kernelStatus) << std::endl;
                }

                cudaStatus = cudaEventRecord( stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (gaussianBlur3) cudaEventRecord stop" << std::endl;
                }
                cudaStatus = cudaEventSynchronize( stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (gaussianBlur3) cudaEventSynchronize stop" << std::endl;
                }

                float duration;
                cudaStatus = cudaEventElapsedTime( &duration, start, stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (gaussianBlur3) cudaEventElapsedTime duration" << std::endl;
                }
                std::cout << "gaussianBlur3 "<<duration<<" ms" <<std::endl;
            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaStatus = cudaMemcpy(g.data(),g_d,3*cols*rows,cudaMemcpyDeviceToHost);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (gaussianBlur3) cudaMemcpy g_d - DeviceToHost" << std::endl;
            }
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

        }

        else if (convolutionList[i]==("nettete3")){

            int tailleNoyau = 3;
            vector<int> matrice({0,-1,0,
                                 -1,5,-1,
                                 0,-1,0});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyau);

            int* noyau_d;
            cudaStatus = cudaMalloc(&noyau_d, tailleNoyau*tailleNoyau*sizeof(int));
            if (cudaStatus != cudaSuccess) {
                std::cout << "Error (nettete3) CudaMalloc noyau_d"  << std::endl;
            }
            cudaStatus = cudaMemcpy(noyau_d,matrice.data(),tailleNoyau*tailleNoyau*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (nettete3) cudaMemcpy noyau_d - HostToDevice" << std::endl;
            }

            if(sizeBgr%3==0){
                cudaStatus = cudaEventRecord( start );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (nettete3) cudaEventRecord start" << std::endl;
                }

                pasAlpha<<<nbBlock, nbThreadParBlock>>>( bgr_d, g_d, cols,rows, noyau,noyau_d);
                kernelStatus = cudaGetLastError();
                if ( kernelStatus != cudaSuccess ) {
                   std::cout << "CUDA Error (nettete3) "<< cudaGetErrorString(kernelStatus) << std::endl;
                }

                cudaStatus = cudaEventRecord( stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (nettete3) cudaEventRecord stop" << std::endl;
                }
                cudaStatus = cudaEventSynchronize( stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (nettete3) cudaEventSynchronize stop" << std::endl;
                }

                float duration;
                cudaStatus = cudaEventElapsedTime( &duration, start, stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (nettete3) cudaEventElapsedTime duration" << std::endl;
                }
                std::cout << "nettete3 "<<duration<<" ms" <<std::endl;
            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaStatus = cudaMemcpy(g.data(),g_d,3*cols*rows,cudaMemcpyDeviceToHost);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (nettete3) cudaMemcpy g_d - DeviceToHost" << std::endl;
            }
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
        }

        else if (convolutionList[i]==("detectEdges3")){

            int tailleNoyau = 3;
            vector<int> matrice({-1,-1,-1,
                                 -1,8,-1,
                                 -1,-1,-1});

            matriceConvolution noyau = matriceConvolution(matrice.data(),tailleNoyau);

            int* noyau_d;
            cudaStatus = cudaMalloc(&noyau_d, tailleNoyau*tailleNoyau*sizeof(int));
            if (cudaStatus != cudaSuccess) {
                std::cout << "Error (detectEdges3) CudaMalloc noyau_d"  << std::endl;
            }
            cudaStatus = cudaMemcpy(noyau_d,matrice.data(),tailleNoyau*tailleNoyau*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (detectEdges3) cudaMemcpy noyau_d - HostToDevice" << std::endl;
            }

            if(sizeBgr%3==0){

                cudaStatus = cudaEventRecord( start );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (detectEdges3) cudaEventRecord start" << std::endl;
                }

                pasAlpha<<<nbBlock, nbThreadParBlock>>>( bgr_d, g_d, cols,rows, noyau,noyau_d);
                kernelStatus = cudaGetLastError();
                if ( kernelStatus != cudaSuccess ) {
                    std::cout << "CUDA Error (detectEdges3) "<< cudaGetErrorString(kernelStatus) << std::endl;
                }

                cudaStatus = cudaEventRecord( stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (detectEdges3) cudaEventRecord stop" << std::endl;
                }
                cudaStatus = cudaEventSynchronize( stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (detectEdges3) cudaEventSynchronize stop" << std::endl;
                }

                float duration;
                cudaStatus = cudaEventElapsedTime( &duration, start, stop );
                if (cudaStatus  != cudaSuccess) {
                    std::cout << "Error (detectEdges3) cudaEventElapsedTime duration" << std::endl;
                }
                std::cout << "detectEdges3 "<<duration<<" ms" <<std::endl;
            }
            if(sizeBgr%4==0){
                //de l'alpha
            }

            cv::Mat m_out( rows, cols, type, g.data() );
            cudaStatus = cudaMemcpy(g.data(),g_d,3*cols*rows,cudaMemcpyDeviceToHost);
            if (cudaStatus  != cudaSuccess) {
                std::cout << "Error (detectEdges3) cudaMemcpy g_d - DeviceToHost" << std::endl;
            }
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
        }*/
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(bgr_d);
    cudaFree(g_d);


    return 0;
}