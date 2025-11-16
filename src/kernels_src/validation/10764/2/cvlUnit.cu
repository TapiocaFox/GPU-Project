#include "includes.h"
__global__ void cvlUnit(const char *imgR,const char *imgG,const char *imgB,const char *core, char *outR,char *outG,char *outB,int lenX,int lenY,int lenCore)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int index=y*lenX+x;
if(x>=lenX||y>=lenY)return;
if(x-lenCore/2<0||x+lenCore/2>=lenX||y-lenCore/2<0||y+lenCore/2>=lenY){
outR[index]=imgR[index];
outG[index]=imgG[index];
outB[index]=imgB[index];
return ;
}


int i,j,tmpX,tmpY;
int sumR=0;
int sumG=0;
int sumB=0;
for(i=0;i<lenCore;i++){
for(j=0;j<lenCore;j++){
tmpX = x-lenCore/2+i;
tmpY = y-lenCore/2+j;
//			if(x==8&&y==8){printf("tmpX=%d,tmpY=%d:\n",tmpX,tmpY);}
sumR+=imgR[tmpY*lenX+tmpX]*core[j*lenCore+i];
//			if(x==8&&y==8){
//				printf("\tR:\t %d*%d,new=%d\n",imgR[tmpY*lenX+tmpX],core[j*lenCore+i],sumR);
//			}
sumG+=imgG[tmpY*lenX+tmpX]*core[j*lenCore+i];
//			if(x==8&&y==8){
//				printf("\tG:\t %d*%d,new=%d\n",imgG[tmpY*lenX+tmpX],core[j*lenCore+i],sumG);
//			}
sumB+=imgB[tmpY*lenX+tmpX]*core[j*lenCore+i];
//			if(x==8&&y==8){
//				printf("\tB:\t %d*%d,new=%d\n",imgB[tmpY*lenX+tmpX],core[j*lenCore+i],sumB);
//			}
}
}
outR[index]=(char)(sumR*1.0/(lenCore*lenCore));
outG[index]=(char)(sumG*1.0/(lenCore*lenCore));
outB[index]=(char)(sumB*1.0/(lenCore*lenCore));
return;
}