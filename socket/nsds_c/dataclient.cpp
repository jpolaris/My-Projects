#include "dataclient.h"
#include <QtNetwork>
#include <QDebug>
#include <QFile>
#include <QTime>

#include <../nsds.h>

DataClient::DataClient():mDataClientSocket(new QTcpSocket(this))
{
    mMyName = this->metaObject()->className();

    out.setDevice(mDataClientSocket);
    out.setVersion(QDataStream::Qt_5_10);

    connect(this,&QThread::started, this, &DataClient::connectToServer);
    connect(this,&QThread::finished,this, &DataClient::threadFinished);

    connect(mDataClientSocket, &QAbstractSocket::connected, this, &DataClient::sendFiles);
    connect(mDataClientSocket, &QAbstractSocket::disconnected, this, [=](){
        qDebug()<<getFormattedOutput()<<"Disconnected from server!";});
}

DataClient::~DataClient()
{
    if(mDataClientSocket != nullptr) delete mDataClientSocket;
}

void DataClient::threadFinished()
{
    qDebug()<<endl<<getFormattedOutput()<<"Thread finished!";
//    if(returnCode == 0){
//        qDebug()<<"DataClient thread finished successfully!"<<endl;
//    }else{
//        qDebug()<<"DataClient thread finished because of errors!"<<endl;
//    }
}

void DataClient::sendFiles()
{
    qDebug()<<getFormattedOutput()<<"Connecting succeed. Start to send files..."<<endl;
    //qDebug()<<getFormattedOutput()<<"Start to send files"<<endl;

    for(int i=0;i<mFileList.size();i++){
        sendFile(mFileList.at(i));
    }

    //
    this->exit();
}

void DataClient::connectToServer()
{
    //qDebug()<<getFormattedOutput()<<"Start to connect server...";
    qDebug()<<getFormattedOutput()<<"Connecting to server: IP-"<<mIp<<" Port-"<<mPortNo;

    mDataClientSocket->abort();
    mDataClientSocket->connectToHost(mIp, mPortNo);
}

void DataClient::setIp(QString ip)
{
    mIp = ip;
}

void DataClient::setPortNo(int portNo)
{
    mPortNo = portNo;
}

void DataClient::setSrcPath(QString srcPath)
{
    mSrcPath = srcPath;
}

void DataClient::addFileList(QStringList& fileList)
{
    mFileList.clear();
    mFileList.append(fileList);
}

void DataClient::sendFile(QString filePath)
{
    //open file to read
    QFile file(filePath);
    if(!file.open(QIODevice::ReadOnly)){
        qDebug()<<getFormattedOutput()<<"Fail to open file: "<<file.fileName()<<endl;
        return;
    }

    QString relativePath = file.fileName().right(file.fileName().length()-mSrcPath.length()-1);
    qint32 fileLen = (qint32)file.size();
    QDataStream fileIn(&file);
    QFileInfo fileInfo(file);
    qint32 fileNameLen = relativePath.toStdString().length();

//    qDebug()<<"sendFile: fileNameLen="<<fileNameLen<<endl;
//    qDebug()<<"sendFile: fileLen="<<fileLen<<endl;
//    qDebug()<<"sendFile: relativePath="<<relativePath<<endl;

    qint32 headerln = nsds::filePathNameLen+fileNameLen+nsds::dataBytesLen+nsds::parityLen;
    qint32 dataBlockLen = 2048;
    char* pBuffer = new char[dataBlockLen]{};

    pBuffer[0] = fileNameLen;   //write FilePathNameLength
    sprintf(pBuffer+nsds::filePathNameLen,"%s", relativePath.toStdString().c_str()); //write FilePathName

    nsds::toByte(fileLen, pBuffer+nsds::filePathNameLen+fileNameLen); //write DataBytesLength
    qint8 parityVal = ::parityCheck(pBuffer,nsds::filePathNameLen+fileNameLen+nsds::dataBytesLen);
    pBuffer[nsds::filePathNameLen+fileNameLen+nsds::dataBytesLen] = parityVal; //TODOs: parity

    //qDebug()<<"File Header: "<<QString::fromLocal8Bit(pBuffer,nsds::filePathNameLen+fileNameLen+nsds::dataBytesLen+nsds::parityLen)<<endl;


    //send file data
    qint32 passedlen = 0;
    qint32 readln = 0;
    readln = fileIn.readRawData(pBuffer+headerln, qMin(dataBlockLen-headerln,fileLen));
    //out.writeRawData(pBuffer, dataBlockLen); //wrong!!!! only for test
    out.writeRawData(pBuffer, readln+headerln);
    passedlen += readln;
    delete[] pBuffer;

    while(!file.atEnd()){

        //resize buffer size
        if(passedlen+dataBlockLen>fileLen)
            dataBlockLen = fileLen - passedlen;

        pBuffer = new char[dataBlockLen]{};
        readln = fileIn.readRawData(pBuffer,dataBlockLen);

//        qDebug()<<"SendFile: buffersize="<<dataBlockLen<<endl;
//        qDebug()<<"SendFile: readln="<<readln<<endl;

        out.writeRawData(pBuffer,readln);

        passedlen += readln;
        delete[] pBuffer;
    }

    file.close();
    qDebug()<<getFormattedOutput()<<"Sending file "<<relativePath;

}

QString DataClient::getFormattedOutput()
{
    return QTime::currentTime().toString("hh:mm:ss ") + mMyName;
}
