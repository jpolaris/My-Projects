#ifndef DATASERVER_H
#define DATASERVER_H

#include <QObject>
#include <QThread>
#include <list>
#include <QDataStream>

#include <../nsds.h>
using namespace nsds;

QT_BEGIN_NAMESPACE
class QTcpServer;
class QNetworkSession;
class QTcpSocket;
QT_END_NAMESPACE

class DataServer : public QThread
{
    Q_OBJECT
public:
    DataServer();
    ~DataServer();

signals:
    void revDataErr(QString lastSavedFile);
    void allFileRecved(qint16 fileCount);

private slots:
    void sessionOpened();
    void getNewConnection();
    void readData();


private:
    QTcpServer* mDataServer = nullptr;
    QTcpSocket* mDataServerSocket = nullptr;
    QNetworkSession *networkSession = nullptr;
    QDataStream in;

    bool mbReading = false;
    qlonglong duration = 0;
    double totalDataVolumn = 0;
    char* mpHeader = nullptr;
    qint32 mReadLen = 0;
    qint32 mDataLen = 0;
    nsds::DATA_BLOCK mDataHeader;
    bool mbReceive = false;

    QString mDestPath;
    qint32 mFilesRead = 0;
    qint32 mFilesToRead = 0;
    QString mLastSavedFile; //save the file information last successfully received

    QString mMyName;   //class name

    void initNetwork();
    qint8 readHeader();
    QString getFormattedOutput();

public:
    void enableReading(bool bReading);
    void setDestPath(QString destPath);
    void setFilesCount(qint32 fileCount);
    void setFilesRead(qint32 filesRead);
};

#endif // DATASERVER_H
