# Projet_Cuda

Pour lancer la version cpu :
faire les commandes suivantes 
-   make
-   ./Convolution <nom de l'image et extension> (l'argument est pour tester avec une image 
    particulière,par défaut cela execute le programme avec l'image in.jpeg)
-   toutes les images créer avec ce programme on pour nom out_<nom de la convolution>_<image d'origine>

Pour la version GPU :
-   make Convolution-cu
-   ./Convolution-cu <nom de l'image et extension> (l'argument est pour tester avec une image 
    particulière,par défaut cela execute le programme avec l'image in.jpeg)
-   toutes les images créer avec ce programme on pour nom out_cu_<nom de la convolution>_<image d'origine>