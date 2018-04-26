#ifndef MSGSERVER_H
#define MSGSERVER_H

#include <QObject>
#include <QDataStream>

#include <../nsds.h>
#include "dataserver.h"
using namespace nsds;

QT_BEGIN_NAMESPACE
class QTcpServer;
class QNetworkSession;
class QTcpSocket;
QT_END_NAMESPACE

class msgServer : public QObject
{
    Q_OBJECT
public:
    explicit msgServer(QObject *parent = nullptr);

signals:

public slots:
private slots:
    void sessionOpened();
    void getNewConnection();
    void readData();
    void handleRecvDataErr(QString lastSavedFile);
    void handleAllFileRecved(qint16 fileCount);

private:
    QTcpServer* mTcpServer = nullptr;
    QTcpSocket* mMsgServerSocket = nullptr;
    QNetworkSession *networkSession = nullptr;
    QDataStream in;
    DataServer mDataServer;

    qint32 mDataReadLen;
    char mMsgType = 0;
    qint32 msgLen = 0;

    char* mStrMessage = nullptr;

    QString mRunId;
    uint16_t mFileNum;
    QString mDestPath;

    void checkMessage();
    void sendResponse();
    void startDataServer();

    QString mMyName;

public:
    void setDestPath(QString destPath);

private:
    QString getFormattedOutput();

};

#endif // MSGSERVER_H
