#include "nsds.h"

bool isBigEndian()
{
    int num = 1;
    if(*(char *)&num == 1) return false;
    else return true;
}

//ret: -1: wrong parameter, then - parity value
qint8 parityCheck(char* data, qint32 dataLen, bool odd)
{
    if(data == nullptr) return -1;

    qint8 countOfOne = 0;
    for(int i=0;i<dataLen;i++){

        char byteData = data[i];
        for(int j=0;j<8;j++){
            if((byteData & (1<<j)) != 0) countOfOne++;
        }
    }

    qint8 ret = 0;
    if(odd){
        if(countOfOne % 2 == 0) ret = 1;
        else ret = 0;
    }else{
        if(countOfOne % 2 == 0) ret = 0;
        else ret = 1;
    }

    return ret;
}
