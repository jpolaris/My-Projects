#include "msgclient.h"
#include <QtNetwork>
#include <QDebug>
#include <QDir>
#include <QTime>

using namespace nsds;

MsgClient::MsgClient(QObject *parent) : QObject(parent)
{

}

MsgClient::~MsgClient()
{
    if(mStrResponse != nullptr) delete[] mStrResponse;
    if(mDataClient != nullptr) delete mDataClient;
    if(mMsgClientSocket != nullptr) delete mMsgClientSocket;
}

MsgClient::MsgClient(QString ip, int portNo, int dataPortNo):
    mIp(ip),mPortNo(portNo),mDataPortNo(dataPortNo),
    mMsgClientSocket(new QTcpSocket(this))
{
    mMyName = this->metaObject()->className();

    qDebug()<<getFormattedOutput()<<"Connecting to server: IP-"<<mIp<<" Port-"<<mPortNo;

    out.setDevice(mMsgClientSocket);
    out.setVersion(QDataStream::Qt_5_10);

    connect(mMsgClientSocket, &QAbstractSocket::connected, this, &MsgClient::sendToken);
    connect(mMsgClientSocket, &QAbstractSocket::disconnected, this, [=](){
        qDebug()<<"disconnected from server!";});
    connect(mMsgClientSocket, &QAbstractSocket::readyRead, this, &MsgClient::readData);

    mStrResponse = new char[tokTypelen+runIdlen+fileNumlen]{};


    connectToServer();
}


void MsgClient::sendToken()
{
    qDebug()<<getFormattedOutput()<<"Connecting succeed.";
    mFileList = getFileList(mSrcPath);
    uint16_t fileNum = mFileList.size();
    qDebug()<<getFormattedOutput()<<"Total files: "<<fileNum<<endl;

    char* runId = new char[runIdlen]{};
    char* pBuffer = new char[fileNumlen]{};
    char* stxMsg = new char[tokTypelen+runIdlen+fileNumlen]{};

    //write data
    //uint16ToByte(fileNum,pBuffer);
    toByte(fileNum,pBuffer);
    stxMsg[0] = TokenType::stx;
    memcpy(stxMsg+tokTypelen, pBuffer, fileNumlen);
    memcpy(stxMsg+tokTypelen+fileNumlen, runId, runIdlen);

    //send data
    out.writeRawData(stxMsg,tokTypelen+runIdlen+fileNumlen);

    //qDebug()<<"Send message: "<<QString::fromLocal8Bit(stxMsg,tokTypelen+runIdlen+fileNumlen)<<endl;

    delete[] runId;
    delete[] pBuffer;
    delete[] stxMsg;
}

void MsgClient::readData()
{
    //qDebug()<<"reading response from server..."<<endl;
    out.startTransaction();

    qint64 remainingDataLen = mMsgClientSocket->bytesAvailable();
    if(remainingDataLen==0){
        out.commitTransaction();
        return;
    }

    if(mDataReadLen == 0){

        out.readRawData(&mRespType,nsds::tokTypelen);
        if(mRespType != TokenType::ack && mRespType != TokenType::etx && mRespType != TokenType::nak){
            qDebug()<<getFormattedOutput()<<"Unknown message from server"<<endl;
            out.commitTransaction();
            return;
        }

        msgLen = mRespType==TokenType::nak?fileNamelen:fileNumlen+runIdlen;
        if(mStrResponse != nullptr) delete[] mStrResponse;
        mStrResponse = new char[msgLen]{};
    }

    if(mDataReadLen<msgLen){
        out.readRawData(mStrResponse,qMin((qint32)remainingDataLen, msgLen-mDataReadLen));
        mDataReadLen += qMin((qint32)remainingDataLen, msgLen-mDataReadLen);
    }

    if(mDataReadLen == msgLen){
        //check filenum, runid or filename
        mDataReadLen = 0;
        checkResponse();
    }

    out.commitTransaction();
}

void MsgClient::checkResponse()
{
    if(mRespType == TokenType::nak){
        //nak means something is wrong
        mRestartWithFileName = QString::fromLocal8Bit(mStrResponse, msgLen);
        qDebug()<<getFormattedOutput()<<"NAK message, filename = "<<mRestartWithFileName<<endl;

        //end data sending process
        if(mDataClient->isRunning())
            mDataClient->exit(-1);

    }else{

        uint16_t filenum = *((uint16_t*)mStrResponse);
        QString runId = QString::fromLocal8Bit(mStrResponse+fileNumlen, runIdlen);

        if(mRespType == TokenType::ack)
            startDataClientThread();
        else if(mRespType == TokenType::etx){
            qDebug()<<getFormattedOutput()<<"Received ETX message from server"<<endl;
        }
    }
}

void MsgClient::connectToServer()
{
    //qDebug()<<"Start to connect server..."<<endl;
    mMsgClientSocket->abort();
    mMsgClientSocket->connectToHost(mIp, mPortNo);
}

QStringList MsgClient::getFileList(QString srcPath)
{
    QStringList fileList;
    if(srcPath.isEmpty()) return fileList;

    QDir imgDir(srcPath);
    if(!imgDir.exists()){
        qInfo()<<"Directory doesn't exist!\n";
        return fileList;
    }

    //return all the files inside the target folder as a list
    QFileInfoList list = imgDir.entryInfoList(QDir::Dirs|QDir::Files| QDir::NoDotAndDotDot);
    for (int i = 0; i < list.size(); ++i){

        QFileInfo fileInfo = list.at(i);
        if(fileInfo.isDir()){
            fileList.append(getFileList(fileInfo.absoluteFilePath()));
        }else{
            fileList.append(fileInfo.absoluteFilePath());
        }
    }

    return fileList;
}

void MsgClient::setSrcPath(QString srcPath)
{
    mSrcPath = srcPath;
}

void MsgClient::setRunId(QString runId)
{
    mRunId = runId;
}

void MsgClient::startDataClientThread()
{
    if(mDataClient == nullptr) mDataClient = new DataClient();
    mDataClient->setIp(mIp);
    mDataClient->setPortNo(mDataPortNo);
    mDataClient->addFileList(mFileList);
    mDataClient->setSrcPath(mSrcPath);
    connect(mDataClient, &DataClient::finished,this,&MsgClient::dataSendingFinished);

    mDataClient->start();
}

void MsgClient::dataSendingFinished()
{
    qDebug()<<getFormattedOutput()<<"DataSending Finished"<<endl;
}

QString MsgClient::getFormattedOutput()
{
    return QTime::currentTime().toString("hh:mm:ss ") + mMyName;
}
