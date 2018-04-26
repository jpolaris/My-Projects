/*
 * Version: v0.1.0.0, build 4/16/2018
 * Author: Erling Si<elsi@cygnusbio.com>
 * Copyright: (c) Cygnus BioScience Co., Ltd.
*/

#ifndef FILERECEPTER_H
#define FILERECEPTER_H

#include <QObject>
#include <QVector>
#include <QDataStream>

QT_BEGIN_NAMESPACE
class QTcpServer;
class QNetworkSession;
class QTcpSocket;
QT_END_NAMESPACE

struct IMAGE_INFO{
    QString name;
    QString runid;
    qint32 tile;
    qint32 cycle;
    qint64 fileLen;
    qint64 passedlen;
    QString relativePath;
    QString fileName;
};

class FileRecepter : public QObject
{
    Q_OBJECT
public:
    explicit FileRecepter(QObject *parent = nullptr);
    ~FileRecepter();

signals:

public slots:

private slots:
    void sessionOpened();
    void getData();
    void readData();

private:
    QTcpServer *tcpServer = nullptr;
    QTcpSocket *tcpSocket = nullptr;
    QNetworkSession *networkSession = nullptr;
    QDataStream in;
    bool bReceive = false;
    IMAGE_INFO imgInfo;
    QString mDestPath;
    char* header = nullptr;
    char* fileHeader = nullptr;
    qint64 headerReadLen = 0;
    qint64 fileHeaderReadLen = 0;
    qint32 fileCount = 0;
    qint32 filesRead = 0;

    const qint32 MAX_FILE_PATH_LENGTH = 100; //Byte
    const qint32 MAX_FILE_LENGTH = 4;
    const qint32 TOKEN_SIZE = 10;
    const qint32 FILE_COUNT_SIZE = 4;
    const qint32 END_SIZE = 3;
    //char header[TOKEN_SIZE+FILE_COUNT_SIZE];

    qlonglong duration = 0;
    double totalDataVolumn = 0;

public:
    void setDestPath(QString& destPath);
private:
    bool readHeader();
    bool readFileHeader();
    QString findMyString(QString src);
    //bool isBigEndian();
};

#endif // FILERECEPTER_H
