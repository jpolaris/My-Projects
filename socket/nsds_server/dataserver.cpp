#include "dataserver.h"
#include <QtNetwork>
#include <QDebug>
#include <QTime>


//DataServer::DataServer(QObject *parent) : QObject(parent)
//{
//    connect(this,&QThread::started, this, &DataServer::initNetwork);
//}
DataServer::DataServer()
{
    mMyName = this->metaObject()->className();

    connect(this,&QThread::started, this, &DataServer::initNetwork);
}

DataServer::~DataServer()
{
    if(mpHeader != nullptr) delete[] mpHeader;
}

void DataServer::initNetwork()
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
        connect(networkSession, &QNetworkSession::opened, this, &DataServer::sessionOpened);

        qDebug()<<getFormattedOutput()<<"Opening network session."<<endl;
        networkSession->open();
    } else {
        sessionOpened();
    }

    connect(mDataServer, &QTcpServer::newConnection, this, &DataServer::getNewConnection);
    in.setVersion(QDataStream::Qt_5_10);
}

void DataServer::sessionOpened()
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

    mDataServer = new QTcpServer(this);
    //if (!mTcpServer->listen(QHostAddress::Any,52989)) {
    if (!mDataServer->listen(QHostAddress::Any,37690)) {
        qDebug()<<getFormattedOutput()<<tr("Unable to start the server: %1.")
                  .arg(mDataServer->errorString());
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
                     .arg(ipAddress).arg(mDataServer->serverPort());
    qDebug()<<getFormattedOutput()<<ipInfo<<endl;
}

void DataServer::getNewConnection()
{
    mDataServerSocket = mDataServer->nextPendingConnection();
    in.setDevice(mDataServerSocket);

    //获取对方的IP:port
    QString ipCli = mDataServerSocket->peerAddress().toString();
    qint16 portCli = mDataServerSocket->peerPort();
    QString temp = QString("[%1:%2]:连接成功").arg(ipCli).arg(portCli);
    qDebug()<<getFormattedOutput()<<temp<<endl;

    connect(mDataServerSocket, &QTcpSocket::readyRead, this, &DataServer::readData);
    connect(mDataServerSocket, &QTcpSocket::disconnected, this, [=](){qDebug()<<"DataServer:  disconnected!";});
}

void DataServer::readData()
{
    if(!mbReading){
        //qDebug()<<"DataServer: not ready for reading data"<<endl;
        return;
    }

    QTime t;
    t.start();

    in.startTransaction();
    qint64 remainingDataLen = mDataServerSocket->bytesAvailable();
    //qDebug()<<"remainingDataLen = "<<remainingDataLen<<endl;
    totalDataVolumn += remainingDataLen;

    while(remainingDataLen){
        if(!mbReceive){

            qint8 retReadHeader = readHeader();
            if(retReadHeader == 0) break;
            if(retReadHeader == -1){

                qDebug()<<getFormattedOutput()<<" Error during receiving files!"<<endl;
                duration += t.elapsed();
                qDebug()<<"Total file count: "<<mFilesToRead<<endl;
                qDebug()<<"File received: "<<mFilesRead<<endl;
                qDebug()<<"Total time used: "<<duration<<" ms"<<endl;
                qDebug()<<"Transfer speed: "<< totalDataVolumn*1000/(1024*1024*duration) <<" MB/s"<<endl;

                mbReading = false;
                break;
            }

            mbReceive = true;
            mReadLen = 0;
            mDataHeader.passedLen = 0;
        }

        //start to receive files
        QString dstPath = QString("%1/%2").arg(mDestPath).arg(mDataHeader.relativeFilePath);
        //qDebug()<<"dstPath="<<dstPath<<endl;
        QDir dstDir(dstPath);
        if(!dstDir.exists()) dstDir.mkpath(dstPath);

        QString filePath = dstPath+QDir::separator()+mDataHeader.fileName;
        QFile file(filePath);
        if(!file.open(QFile::ReadWrite|QIODevice::Append)){
            qDebug()<<getFormattedOutput()<<"Fail to create or open file: "<<filePath<<endl;
            return;
        }

        remainingDataLen = qMin(qint32(mDataServerSocket->bytesAvailable()),
                                mDataHeader.fileLen-mDataHeader.passedLen);

        char* buffer = new char[remainingDataLen]{};
        qint32 dataReadSize = in.readRawData(buffer,remainingDataLen);
        if(dataReadSize == -1){
            mbReceive = false;
            qDebug()<<"Error during reading data..."<<endl;
        }

        QDataStream toFile(&file);
        toFile.setVersion(QDataStream::Qt_5_10);
        toFile.writeRawData(buffer,dataReadSize);
        file.close();

        delete[] buffer;

        mDataHeader.passedLen += dataReadSize;
        if(mDataHeader.passedLen>=mDataHeader.fileLen){

            mFilesRead++;
            qDebug()<<getFormattedOutput()<<"Receiving file "<<mDataHeader.fileName;
            mLastSavedFile = mDataHeader.fileFullName;

            if(mFilesRead == mFilesToRead){

                duration += t.elapsed();
                qDebug()<<endl<<getFormattedOutput()<<"All files are transferred";
                qDebug()<<"    Total files:"<<mFilesToRead;
                qDebug()<<"    Total time used:"<<duration<<" ms";
                qDebug()<<"    Transfer speed:"<< totalDataVolumn*1000/(1024*1024*duration) <<" MB/s"<<endl;

                mbReading = false;
                emit(allFileRecved(mFilesToRead));
                break;
            }

            //start to read header
            mDataHeader.passedLen = 0;
            mbReceive = false;
            mReadLen = 0;
        }

        remainingDataLen = mDataServerSocket->bytesAvailable();
    }

    in.commitTransaction();
    duration += t.elapsed();
}

void DataServer::enableReading(bool bReading)
{
    mbReading = bReading;
}

//ret: -1: parity check error
//      0: not finished yet
//      1: finish reading header
qint8 DataServer::readHeader()
{
    qint32 remainingDataLen = (qint32)mDataServerSocket->bytesAvailable();

    //mReadLen means, start to receive a new file
    if(mReadLen == 0){
        //read filePathNameLength
        char temp;
        in.readRawData(&temp, nsds::filePathNameLen);
        mDataHeader.filePathLen = quint8(temp);

        mDataLen = mDataHeader.filePathLen+nsds::dataBytesLen+nsds::parityLen+nsds::filePathNameLen;

        if(mpHeader != nullptr){
            delete[] mpHeader;
            mpHeader = nullptr;
        }

        mpHeader = new char[mDataLen]{};
        mReadLen = nsds::filePathNameLen;
        mpHeader[0] = temp;
    }

    if(mReadLen<mDataLen){
        in.readRawData(mpHeader+mReadLen, qMin(remainingDataLen, mDataLen-mReadLen));
        mReadLen += qMin(remainingDataLen, mDataLen-mReadLen);
    }

    if(mReadLen == mDataLen){
        //parity check
        qint8 parityVal = ::parityCheck(mpHeader,nsds::filePathNameLen+mDataHeader.filePathLen+nsds::dataBytesLen);
        if(parityVal != mpHeader[nsds::filePathNameLen+mDataHeader.filePathLen+nsds::dataBytesLen]){
            qDebug()<<"!------------! parity check doesn't pass"<<endl;
            qDebug()<<"Header: "<<QString::fromLocal8Bit(mpHeader, mDataLen)<<endl;
            qDebug()<<"remainingDataLen="<<remainingDataLen<<endl;

            //send error message to msgServer
            emit(revDataErr(mLastSavedFile));
            return -1;
        }

        //read filePath
        mDataHeader.fileFullName = QString::fromLocal8Bit(mpHeader+nsds::parityLen,mDataHeader.filePathLen);

        //read fileLen
        if(::isBigEndian()){
            mDataHeader.fileLen = BigtoLittle32(*((int*)(mpHeader+nsds::parityLen+mDataHeader.filePathLen)));
        }else{
            mDataHeader.fileLen = *((int*)(mpHeader+nsds::parityLen+mDataHeader.filePathLen));
        }

//        qDebug()<<"readHeader: fileFullName="<<mDataHeader.fileFullName<<endl;
//        qDebug()<<"readHeader: headLen="<<mDataLen<<endl;
//        qDebug()<<"readHeader: fileLen="<<mDataHeader.fileLen<<endl;

        //get relative directory and file name
        int index = mDataHeader.fileFullName.lastIndexOf('/');
        mDataHeader.relativeFilePath = index==-1?"":mDataHeader.fileFullName.left(index);
        mDataHeader.fileName = mDataHeader.fileFullName.right(mDataHeader.fileFullName.length()-1-index);

        return 1;
    }

    return 0;
}

void DataServer::setDestPath(QString destPath)
{
    mDestPath = destPath;
    //qDebug()<<"setDestPath: "<<mDestPath<<endl;
}

void DataServer::setFilesCount(qint32 fileCount)
{
    mFilesToRead = fileCount;
}

void DataServer::setFilesRead(qint32 filesRead)
{
    mFilesRead = filesRead;
}

QString DataServer::getFormattedOutput()
{
    return QTime::currentTime().toString("hh:mm:ss ") + mMyName;
}
