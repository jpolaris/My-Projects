#ifndef MSGCLIENT_H
#define MSGCLIENT_H

#include<../nsds.h>

#include <QObject>
#include <QTcpSocket>
#include <QString>
#include <QDataStream>
#include <QList>

#include "dataclient.h"

class MsgClient : public QObject
{
    Q_OBJECT
public:
    explicit MsgClient(QObject *parent = nullptr);
    MsgClient(QString ip, int portNo, int dataPortNo);
    ~MsgClient();

signals:

public slots:
public:
    void setSrcPath(QString srcPath);

private:
    QString mIp;
    int mPortNo;
    int mDataPortNo;
    QTcpSocket *mMsgClientSocket = nullptr;
    QDataStream out;
    DataClient* mDataClient = nullptr;


    QString mSrcPath;
    QStringList mFileList;
    QString mRunId;

    QString mRestartWithFileName;

    char* mStrResponse = nullptr;
    qint32 msgLen = 0;
    qint32 mDataReadLen = 0;
    char mRespType = 0;

    QString mMyName;


    void connectToServer();
    void sendToken(); //sending first message after connected
    void readData();
    QStringList getFileList(QString srcPath);
    void startDataClientThread();
    void dataSendingFinished();


    void setRunId(QString runId);
    void checkResponse();
    QString getFormattedOutput();
};

#endif // MSGCLIENT_H
