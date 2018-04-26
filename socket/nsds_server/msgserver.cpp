#include "msgserver.h"
#include <QtNetwork>
#include <QDebug>
#include <QTime>

msgServer::msgServer(QObject *parent) : QObject(parent)
{
    mMyName = this->metaObject()->className();

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
        connect(networkSession, &QNetworkSession::opened, this, &msgServer::sessionOpened);

        //qDebug()<<"Opening network session."<<endl;
        networkSession->open();
    } else {
        sessionOpened();
    }

    connect(mTcpServer, &QTcpServer::newConnection, this, &msgServer::getNewConnection);
    in.setVersion(QDataStream::Qt_5_10);

    //qDebug()<<"mDestPath="<<mDestPath<<endl;

    connect(&mDataServer, &DataServer::revDataErr, this, &msgServer::handleRecvDataErr);
    connect(&mDataServer, &DataServer::allFileRecved, this, &msgServer::handleAllFileRecved);
    mDataServer.start();
}

void msgServer::sessionOpened()
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

    mTcpServer = new QTcpServer(this);
    //if (!mTcpServer->listen(QHostAddress::Any,52989)) {
    if (!mTcpServer->listen(QHostAddress::Any,37691)) {
        qDebug()<<tr("Unable to start the server: %1.")
                  .arg(mTcpServer->errorString());
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
                     .arg(ipAddress).arg(mTcpServer->serverPort());
    qDebug()<<getFormattedOutput()<<ipInfo;
}

void msgServer::getNewConnection()
{

    mMsgServerSocket = mTcpServer->nextPendingConnection();
    in.setDevice(mMsgServerSocket);

    //获取对方的IP:port
    QString ipCli = mMsgServerSocket->peerAddress().toString();
    qint16 portCli = mMsgServerSocket->peerPort();
    QString temp = QString("[%1:%2]:连接成功").arg(ipCli).arg(portCli);
    qDebug()<<getFormattedOutput()<<temp;

    connect(mMsgServerSocket, &QTcpSocket::readyRead, this, &msgServer::readData);
    connect(mMsgServerSocket, &QTcpSocket::disconnected, this, [=](){qDebug()<<getFormattedOutput()<<"disconnected!";});
}

void msgServer::readData()
{
    in.startTransaction();

    qint64 remainingDataLen = mMsgServerSocket->bytesAvailable();
    if(remainingDataLen==0){
        in.commitTransaction();
        return;
    }

    if(mDataReadLen == 0){
        //only read data with stx
        in.readRawData(&mMsgType, nsds::tokTypelen);
        if(mMsgType != nsds::TokenType::stx){
            qDebug()<<getFormattedOutput()<<"Unknown message from Client. Refuse to receive data"<<endl;
            in.commitTransaction();
            return;
        }

        msgLen = nsds::fileNumlen+nsds::runIdlen;
        if(mStrMessage != nullptr) delete[] mStrMessage;
        mStrMessage = new char[msgLen]{};
    }

    if(mDataReadLen<msgLen){
        in.readRawData(mStrMessage,qMin((qint32)remainingDataLen, msgLen-mDataReadLen));
        mDataReadLen += qMin((qint32)remainingDataLen, msgLen-mDataReadLen);
    }

    if(mDataReadLen == msgLen){
        //check filenum, runid or filename
        mDataReadLen = 0;
        checkMessage();
    }

    in.commitTransaction();
}

void msgServer::checkMessage()
{
    uint16_t filenum = *((uint16_t*)mStrMessage);
    QString runId = QString::fromLocal8Bit(mStrMessage+nsds::fileNumlen, nsds::runIdlen);
//    qDebug()<<"Check Message..."<<QString::fromLocal8Bit(mStrMessage,nsds::fileNumlen+nsds::runIdlen)<<endl;
//    qDebug()<<"Token type:"<<mMsgType<<endl;
//    qDebug()<<"filenum="<<filenum<<"; runId="<<runId<<endl;

    mFileNum = filenum;
    mRunId = runId;

    qDebug()<<getFormattedOutput()<<" Total files: "<<mFileNum;

    startDataServer();
    sendResponse();
}


void msgServer::sendResponse()
{
    char* runId = new char[nsds::runIdlen]{};
    char* pBuffer = new char[nsds::fileNumlen]{};
    char* stxMsg = new char[nsds::tokTypelen+nsds::runIdlen+nsds::fileNumlen]{};

    //write data
    //uint16ToByte(mFileNum,pBuffer);
    toByte(mFileNum,pBuffer);
    stxMsg[0] = nsds::TokenType::ack;
    memcpy(stxMsg+nsds::tokTypelen, pBuffer, nsds::fileNumlen);
    memcpy(stxMsg+nsds::tokTypelen+nsds::fileNumlen, runId, nsds::runIdlen);

    //send data
    in.writeRawData(stxMsg,nsds::tokTypelen+nsds::runIdlen+nsds::fileNumlen);

    //qDebug()<<"Sending response: "<<QString::fromLocal8Bit(stxMsg,nsds::tokTypelen+nsds::runIdlen+nsds::fileNumlen)<<endl;

    delete[] runId;
    delete[] pBuffer;
    delete[] stxMsg;
}

void msgServer::startDataServer()
{
    mDataServer.enableReading(true);
    mDataServer.setFilesCount(mFileNum);
    mDataServer.setDestPath(mDestPath);
    mDataServer.setFilesRead(0);

}

void msgServer::setDestPath(QString destPath)
{
    mDestPath = destPath;
}

void msgServer::handleRecvDataErr(QString lastSavedFile)
{
    qDebug()<<getFormattedOutput()<<"Get error message from dataserver"<<endl;
    qDebug()<<getFormattedOutput()<<"lastSavedFile="<<lastSavedFile<<endl;

    //send message to client
    //qint32 filelen = lastSavedFile.length();
    qint32 filelen = nsds::fileNamelen;

    char* nakMsg = new char[nsds::tokTypelen+filelen]{};
    nakMsg[0] = nsds::TokenType::nak;
    sprintf(nakMsg+nsds::tokTypelen, "%sEND", lastSavedFile.toStdString().c_str());

    in.writeRawData(nakMsg, nsds::tokTypelen+filelen);
    qDebug()<<getFormattedOutput()<<"Sending Error Message: "<<QString::fromLocal8Bit(nakMsg,nsds::tokTypelen+filelen)<<endl;

    delete[] nakMsg;
}

QString msgServer::getFormattedOutput()
{
    return QTime::currentTime().toString("hh:mm:ss ") + mMyName;
}

void msgServer::handleAllFileRecved(qint16 fileCount)
{
    qDebug()<<getFormattedOutput()<<"Sending ETX Message";

    char* runId = new char[nsds::runIdlen]{};
    char* pBuffer = new char[nsds::fileNumlen]{};
    char* etxMsg = new char[nsds::tokTypelen+nsds::runIdlen+nsds::fileNumlen]{};

    //write data
    //uint16ToByte(mFileNum,pBuffer);
    toByte(fileCount,pBuffer);
    etxMsg[0] = nsds::TokenType::etx;
    memcpy(etxMsg+nsds::tokTypelen, pBuffer, nsds::fileNumlen);
    memcpy(etxMsg+nsds::tokTypelen+nsds::fileNumlen, runId, nsds::runIdlen);

    //send data
    in.writeRawData(etxMsg,nsds::tokTypelen+nsds::runIdlen+nsds::fileNumlen);

    //qDebug()<<"Sending response: "<<QString::fromLocal8Bit(stxMsg,nsds::tokTypelen+nsds::runIdlen+nsds::fileNumlen)<<endl;

    delete[] runId;
    delete[] pBuffer;
    delete[] etxMsg;
}
