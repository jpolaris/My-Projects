#ifndef NSDS_H
#define NSDS_H

#include <QObject>
#include <QDebug>


#define BigtoLittle16(A) ((((uint16)(A) & 0xff00) >> 8) | (((uint16)(A) & 0x00ff) << 8))
#define BigtoLittle32(A) ((((qint32)(A) & 0xff000000) >> 24) | (((qint32)(A) & 0x00ff0000) >> 8) | \
             (((qint32)(A) & 0x0000ff00) << 8) | (((qint32)(A) & 0x000000ff) << 24))

bool isBigEndian();
qint8 parityCheck(char* data, qint32 dataLen, bool odd = true); //when odd = false, use even check

namespace nsds {

const static uint16_t tokTypelen = 1;
const static uint16_t fileNumlen = 2;
const static uint16_t runIdlen = 8;
const static uint16_t fileNamelen = 100;

enum TokenType{
    stx=0x01,
    ack,
    etx,
    nak
};

const static uint16_t filePathNameLen = 1;
const static uint16_t dataBytesLen = 4;
const static uint16_t parityLen = 1;

struct DATA_BLOCK
{
    QString relativeFilePath;
    QString fileName;
    QString fileFullName;
    qint32 fileLen;
    qint32 filePathLen;
    qint32 passedLen;
};


//template<typename T> bool toByte(T& nbr, char* pBuffer, bool bLittleEndian = true);
template<typename T>
bool toByte(T& nbr, char* pBuffer, bool bLittleEndian = true)
{
    if(pBuffer == NULL) return false;

    qint32 len = sizeof(T);

    for(int i=0;i<len;i++){
        if(bLittleEndian) pBuffer[i] = nbr>>(8*i);
        else pBuffer[i] = nbr>>(8*(len-1-i));
    }

//    qDebug()<<"toByte: "<<QString::fromLocal8Bit(pBuffer,len);

    return true;
}

//bool uint16ToByte(uint16_t nbr, char* pBuffer, bool bLittleEndian = true);
//bool uint16ToByte(uint16_t nbr, char* pBuffer, bool bLittleEndian = true)
//{
//    if(pBuffer == NULL) return false;

//    if(bLittleEndian){
//        pBuffer[1] = nbr>>8;
//        pBuffer[0] = nbr;
//    }else{
//        pBuffer[0] = nbr>>8;
//        pBuffer[1] = nbr;
//    }

//    return true;
//}

}

#endif // NSDS_H


