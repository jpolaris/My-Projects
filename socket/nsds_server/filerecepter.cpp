/*
 * Version: v0.1.0.0, build 4/16/2018
 * Author: Erling Si<elsi@cygnusbio.com>
 * Copyright: (c) Cygnus BioScience Co., Ltd.
*/

#include "filerecepter.h"
#include <QtNetwork>
#include <QDebug>
#include <QTime>
#include <../nsds.h>

using namespace nsds;

FileRecepter::~FileRecepter()
{
    if(header != nullptr) delete[] header;
    if(fileHeader != nullptr) delete[] fileHeader;
}

FileRecepter::FileRecepter(QObject *parent) : QObject(parent)
{
    QNetworkConfigurationManager manager;
    if (manager.capabilities() & QNetworkConfigurationManager::NetworkSessionRequired) {
        // Get saved network configuration
        QSettings settings(QSettings::UserScope, QLatin1String("QtProject"));
        settings.beginGroup(QLatin1String("QtNetwork"));
        const QString id = settings.value(QLatin1String("DefaultNetworkConfiguration")).toString();
        settings.endGroup();

        // If the saved network configuration is not currently discovered use the system default
        QNetworkConfiguration config = manager.configurationFromIdentifier(id);
        if ((config.state() & QNetworkConfiguration::Discovered) !=
            QNetworkConfiguration::Discovered) {
            config = manager.defaultConfiguration();
        }

        networkSession = new QNetworkSession(config, this);
        connect(networkSession, &QNetworkSession::opened, this, &FileRecepter::sessionOpened);

        qDebug()<<"Opening network session."<<endl;
        networkSession->open();
    } else {
        sessionOpened();
    }

    connect(tcpServer, &QTcpServer::newConnection, this, &FileRecepter::getData);
    in.setVersion(QDataStream::Qt_5_10);

    header = new char[TOKEN_SIZE+FILE_COUNT_SIZE+END_SIZE]();
    fileHeader = new char[MAX_FILE_LENGTH+MAX_FILE_PATH_LENGTH+END_SIZE]();
}

void FileRecepter::setDestPath(QString& destPath){
    mDestPath = destPath;
}
void FileRecepter::sessionOpened()
{
    // Save the used configuration
    if (networkSession) {

        QNetworkConfiguration config = networkSession->configuration();
        QString id;
        if (config.type() == QNetworkConfiguration::UserChoice)
            id = networkSession->sessionProperty(QLatin1String("UserChoiceConfiguration")).toString();
        else
            id = config.identifier();

        QSettings settings(QSettings::UserScope, QLatin1String("QtProject"));
        settings.beginGroup(QLatin1String("QtNetwork"));
        settings.setValue(QLatin1String("DefaultNetworkConfiguration"), id);
        settings.endGroup();
    }

    tcpServer = new QTcpServer(this);
    if (!tcpServer->listen(QHostAddress::Any,52989)) {
        qDebug()<<tr("Unable to start the server: %1.")
                  .arg(tcpServer->errorString());
       // close();
        return;
    }

    QString ipAddress;
    QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();
    // use the first non-localhost IPv4 address
    for (int i = 0; i < ipAddressesList.size(); ++i) {
        if (ipAddressesList.at(i) != QHostAddress::LocalHost &&
            ipAddressesList.at(i).toIPv4Address()) {
            ipAddress = ipAddressesList.at(i).toString();
            break;
        }
    }
    // if we did not find one, use IPv4 localhost
    if (ipAddress.isEmpty())
        ipAddress = QHostAddress(QHostAddress::LocalHost).toString();

    QString ipInfo = tr("The server is running on IP: %1 port: %2")
                     .arg(ipAddress).arg(tcpServer->serverPort());
    qDebug()<<ipInfo<<endl;
}

void FileRecepter::getData()
{
    tcpSocket = tcpServer->nextPendingConnection();
    in.setDevice(tcpSocket);

    //获取对方的IP:port
    QString ipCli = tcpSocket->peerAddress().toString();
    qint16 portCli = tcpSocket->peerPort();
    QString temp = QString(" [%1:%2]:连接成功").arg(ipCli).arg(portCli);
    qDebug()<<QTime::currentTime().toString("hh:mm:ss")<<temp<<endl;

    connect(tcpSocket, &QTcpSocket::readyRead, this, &FileRecepter::readData);
    connect(tcpSocket, &QTcpSocket::disconnected, this, [=](){qDebug()<<"disconnected!";});
}

QString FileRecepter::findMyString(QString src)
{
    int pos = src.indexOf("END");
    if(pos != -1) return src.left(pos);
    else return QString("");
}

bool FileRecepter::readHeader()
{
    qint64 remainingDataLen = tcpSocket->bytesAvailable();
    qint64 headerSize = TOKEN_SIZE+sizeof(qint32);
    if(headerReadLen<headerSize){
        in.readRawData(header+headerReadLen, qMin(remainingDataLen, headerSize-headerReadLen));
        headerReadLen+= qMin(remainingDataLen, headerSize-headerReadLen);
    }

    if(headerReadLen == headerSize){
        QString nsds_token = QString::fromLocal8Bit(header, TOKEN_SIZE);
        if(nsds_token.indexOf(QString("NSDS"))==-1){
            qDebug()<<"NSDS_TOKEN: "<<nsds_token<<", not my data"<<endl;
            return false;
        }

        //qDebug()<<"Header: "<<QString::fromLocal8Bit(header, headerSize);

        if(::isBigEndian())
            fileCount = BigtoLittle32(*((int*)(header+TOKEN_SIZE)));
        else
            fileCount = *((int*)(header+TOKEN_SIZE));
        qDebug()<<"fileCount = "<<fileCount<<endl;


        return true;
    }

    return false;
}

bool FileRecepter::readFileHeader()
{
    if(fileHeaderReadLen == 0)
        memset(fileHeader,0,MAX_FILE_LENGTH+MAX_FILE_PATH_LENGTH+END_SIZE);

    qint64 remainingDataLen = tcpSocket->bytesAvailable();
    qint64 fileHeaderSize = MAX_FILE_LENGTH+MAX_FILE_PATH_LENGTH+END_SIZE;

    if(fileHeaderReadLen<fileHeaderSize){
        in.readRawData(fileHeader+fileHeaderReadLen, qMin(remainingDataLen, fileHeaderSize-fileHeaderReadLen));
        fileHeaderReadLen += qMin(remainingDataLen, fileHeaderSize-fileHeaderReadLen);
    }

    if(fileHeaderReadLen == fileHeaderSize){
        //qDebug()<<"File Header: "<<QString::fromLocal8Bit(fileHeader, fileHeaderSize);

        //parse file name and length
        imgInfo.name = findMyString(QString::fromLocal8Bit(fileHeader, MAX_FILE_PATH_LENGTH));
        if(::isBigEndian())
            imgInfo.fileLen = BigtoLittle32(*((int*)(fileHeader+MAX_FILE_PATH_LENGTH+END_SIZE)));
        else
            imgInfo.fileLen = *((int*)(fileHeader+MAX_FILE_PATH_LENGTH+END_SIZE));

        //get relative directory and file name
        int index = imgInfo.name.lastIndexOf('/');
        imgInfo.relativePath = index==-1?"":imgInfo.name.left(index);
        imgInfo.fileName = imgInfo.name.right(imgInfo.name.length()-1-index);

        return true;
    }

    return false;
}
void FileRecepter::readData()
{
    QTime t;
    t.start();

    in.startTransaction();
    qint64 remainingDataLen = tcpSocket->bytesAvailable();
    totalDataVolumn += remainingDataLen;
    //qDebug()<<"Incoming Data size: "<<remainingDataLen<<endl;
    while(remainingDataLen>0){

        //read token and file count
        if(!bReceive){
            bReceive = readHeader();
            if(bReceive){

                headerReadLen = 0;
                filesRead = 0;
                imgInfo.passedlen = 0;

                //Send response to client
                char* response = new char[TOKEN_SIZE]();
                sprintf(response, "%s", "NSDS_OK");
                in.writeRawData(response,TOKEN_SIZE);
                delete[] response;
            }

            in.commitTransaction();
            return;
        }

        //start to receive files
        if(imgInfo.passedlen==0){

            if(!readFileHeader()){
                in.commitTransaction();
                return;
            }
            fileHeaderReadLen = 0;
        }

        QString dstPath = QString("%1/%2").arg(mDestPath).arg(imgInfo.relativePath);
        QDir dstDir(dstPath);
        if(!dstDir.exists()) dstDir.mkpath(dstPath);

        QString filePath = dstPath+QDir::separator()+imgInfo.fileName;
        QFile file(filePath);
        if(!file.open(QFile::ReadWrite|QIODevice::Append)){
            qDebug()<<"Fail to create or open file: "<<filePath<<endl;
            return;
        }

        remainingDataLen = tcpSocket->bytesAvailable();
        if(remainingDataLen>imgInfo.fileLen-imgInfo.passedlen)
            remainingDataLen = imgInfo.fileLen - imgInfo.passedlen;

        char* buffer = new char[remainingDataLen];
        qint32 ret = in.readRawData(buffer,remainingDataLen);
        if(ret == -1){
            bReceive = false;
            qDebug()<<"Error during reading data..."<<endl;
        }

        QDataStream toFile(&file);
        toFile.setVersion(QDataStream::Qt_5_10);
        toFile.writeRawData(buffer,ret);
        file.close();

        delete[] buffer;

        imgInfo.passedlen += ret;
        if(imgInfo.passedlen>=imgInfo.fileLen){

            filesRead++;
            qDebug()<<QTime::currentTime().toString("hh:mm:ss")<<": Finish receiving file "<<imgInfo.name<<endl;

            if(filesRead == fileCount){
                qDebug()<<QTime::currentTime().toString("hh:mm:ss")<<" All the files are transferred!"<<endl;
                bReceive = false;
                duration += t.elapsed();
                qDebug()<<"Total file count: "<<fileCount<<endl;
                qDebug()<<"Total time used: "<<duration<<" ms"<<endl;
                qDebug()<<"Transfer speed: "<< totalDataVolumn*1000/(1024*1024*duration) <<" MB/s"<<endl;
            }

            imgInfo.passedlen = 0;
        }

        remainingDataLen = tcpSocket->bytesAvailable();
    }

    in.commitTransaction();
    duration += t.elapsed();
}
