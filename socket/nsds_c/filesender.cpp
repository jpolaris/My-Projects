/*
 * Version: v0.1.0.0, build 4/16/2018
 * Author: Erling Si<elsi@cygnusbio.com>
 * Copyright: (c) Cygnus BioScience Co., Ltd.
*/

#include "filesender.h"
#include <QtNetwork>
#include <QDebug>
#include <QDir>
#include <QTime>

FileSender::FileSender(QObject *parent) : QObject(parent)
{
    qDebug()<<"Enter FileSender\n";
}

FileSender::FileSender(QString ip, int portNo):
    tcpSocket(new QTcpSocket(this)),
    mIp(ip), mPortNo(portNo),mSendCount(0)
{
    qDebug()<<"IP: "<<mIp<<" Port: "<<mPortNo<<endl;

    out.setDevice(tcpSocket);
    out.setVersion(QDataStream::Qt_5_10);

    connect(tcpSocket, &QAbstractSocket::connected, this, &FileSender::PrepareToSend);
    connect(tcpSocket, &QAbstractSocket::disconnected, this, [=](){
        qDebug()<<"disconnected from server!";});
    connect(tcpSocket, &QAbstractSocket::readyRead, this, &FileSender::readData);

    //in.setVersion(QDataStream::Qt_5_10);
}

void FileSender::Send(QString path)
{
    mPath = path;
    ConnectingServer();
}
void FileSender::ConnectingServer()
{
    qDebug()<<"Start to connect server..."<<endl;
    tcpSocket->abort();
    tcpSocket->connectToHost(mIp, mPortNo);
}
void FileSender::PrepareToSend()
{
    qDebug()<<"Connecting succeed. Start to send files..."<<endl;

    fileList = FindSeqImage(mPath);
    sendNsdsHeader(fileList.size());
}

bool FileSender::FileSender::int32ToByte(qint32 nbr, char* pBuffer)
{
    //little endian
    //char* pBuffer = new char[4];
    if(pBuffer == NULL) return false;
    pBuffer[3] = nbr>>24;
    pBuffer[2] = nbr>>16;
    pBuffer[1] = nbr>>8;
    pBuffer[0] = nbr;

    return true;

}

void FileSender::sendNsdsHeader(qint32 fileCount)
{
    char* header = new char[TOKEN_SIZE+FILE_COUNT_SIZE]();
    if(header == NULL){
        qDebug()<<"Faile to allocate memory for header"<<endl;
        return;
    }

    char* pBuffer = new char[FILE_COUNT_SIZE]();
    if(pBuffer == NULL){
        qDebug()<<"Faile to allocate memory for header"<<endl;
        return;
    }

    sprintf(header,"%s","NSDS");
    if(int32ToByte(fileCount, pBuffer)==false){
        qDebug()<<"Faile to change int32 to byte"<<endl;
        return;
    }

    memcpy(header+TOKEN_SIZE,pBuffer,FILE_COUNT_SIZE);
    out.writeRawData(header,TOKEN_SIZE+FILE_COUNT_SIZE);

    delete[] header;
    delete[] pBuffer;
}

bool FileSender::SendFile(QString filePath, QString runid)
{
    mSendCount++;
    //out<<QString("NSDS");

    //open file to read
    QFile file(filePath);
    if(!file.open(QIODevice::ReadOnly)){
        qDebug()<<"Fail to open file: "<<file.fileName()<<endl;
        return false;
    }

    QString relativePath = file.fileName().right(file.fileName().length()-mPath.length()-1);
    qint32 fileLen = (qint32)file.size();
    QDataStream fileIn(&file);
    QFileInfo fileInfo(file);

    //send file header, including file name and size
    qint32 fileHeaderSize = MAX_FILE_LENGTH+MAX_FILE_PATH_LENGTH+END_SIZE;
    char* fileHeader = new char[fileHeaderSize]();
    if(fileHeader == NULL){
        qDebug()<<"Faile to allocate memory for fileHeader"<<endl;
        return false;
    }

    char* pBuffer = new char[MAX_FILE_LENGTH]();
    if(pBuffer == NULL){
        qDebug()<<"Faile to allocate memory for header"<<endl;
        return false;
    }

    if(int32ToByte(fileLen, pBuffer)==false){
        qDebug()<<"Faile to change int32 to byte"<<endl;
        return false;
    }

    //qDebug()<<"Relative Path: "<<relativePath<<endl;
    sprintf(fileHeader,"%sEND",relativePath.toStdString().c_str());
    memcpy(fileHeader+MAX_FILE_PATH_LENGTH+END_SIZE, pBuffer, MAX_FILE_LENGTH);
    out.writeRawData(fileHeader,fileHeaderSize);

    delete[] fileHeader;
    delete[] pBuffer;

    //send file data
    int buffersize = 2048;
    char* buffer= 0;
    int passedlen = 0;
    while(!fileIn.atEnd()){

        int readln = 0;
        if(passedlen+buffersize>fileLen){
            buffersize = fileLen-passedlen;
        }

        buffer = new char[buffersize];
        readln = fileIn.readRawData(buffer,buffersize);



        out.writeRawData(buffer,readln);

        passedlen += readln;
        delete[] buffer;
    }

    //out<<QString("NSDS_END");
    file.close();
    qDebug()<<QTime::currentTime().toString("hh:mm:ss")<<": Finish sending file "<<file.fileName()<<endl;
    //qDebug()<<"File size: "<<file.size()<<endl;

    return true;
}

void FileSender::FindSeqImage(QString runid, QString pathName, QString ip, int port)
{
    QDir imgDir(pathName);
    if(!imgDir.exists()){
        qInfo()<<"Image path doesn't exist!\n";
        return;
    }

    QFileInfoList list = imgDir.entryInfoList(QDir::Dirs|QDir::Files| QDir::NoDotAndDotDot);
    QHash<QString, int>* fileMap = new QHash<QString, int>();
    int fileCount = 0;

    for (int i = 0; i < list.size(); ++i) {

        QFileInfo fileInfo = list.at(i);

        if(fileInfo.isDir()){
            qDebug()<<fileInfo.absoluteFilePath()<<endl;
            this->FindSeqImage(runid, fileInfo.absoluteFilePath(), ip, port);
        }else{
            QString fileName = fileInfo.fileName();
            if(!fileMap->contains(fileName)){
                fileMap->insert(fileName, fileCount);
                qDebug()<<qPrintable(QString("%1 %2").arg(fileInfo.size(), 10).arg(fileInfo.fileName()))<<endl;
                if(!SendFile(fileInfo.absoluteFilePath(),runid)){
                    qDebug()<<"Fail to send file: "<<fileInfo.fileName()<<endl;
                }
            }
        }
    }

}

QStringList FileSender::FindSeqImage(QString pathName)
{
    QStringList fileList;
    QDir imgDir(pathName);
    if(!imgDir.exists()){
        qInfo()<<"Directory doesn't exist!\n";
        return fileList;
    }

    //return all the files inside the target folder as a list
    QFileInfoList list = imgDir.entryInfoList(QDir::Dirs|QDir::Files| QDir::NoDotAndDotDot);
    for (int i = 0; i < list.size(); ++i){

        QFileInfo fileInfo = list.at(i);
        if(fileInfo.isDir()){
            fileList.append(FindSeqImage(fileInfo.absoluteFilePath()));
        }else{
            fileList.append(fileInfo.absoluteFilePath());
        }
    }

    return fileList;
}

void FileSender::readData()
{
    qDebug()<<"reading response..."<<endl;
    out.startTransaction();
    char* response=new char[TOKEN_SIZE]();
    if(out.readRawData(response,TOKEN_SIZE)==-1)
        qDebug()<<"readData: readRawData Error!"<<endl;

    qDebug()<<"show response..."<<endl;
    QString strResponse = QString::fromLocal8Bit(response,TOKEN_SIZE);
    out.commitTransaction();
    qDebug()<<"Response: "<<strResponse<<endl;

    if(strResponse.indexOf("NSDS_OK")!=-1){
        //you can send files to server now
        for(int i=0;i<fileList.size();i++){
            SendFile(fileList.at(i),"");
        }
        tcpSocket->disconnectFromHost();
    }

}

