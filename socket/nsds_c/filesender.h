/*
 * Version: v0.1.0.0, build 4/16/2018
 * Author: Erling Si<elsi@cygnusbio.com>
 * Copyright: (c) Cygnus BioScience Co., Ltd.
*/

#ifndef FILESENDER_H
#define FILESENDER_H

#include <QObject>
#include <QTcpSocket>
#include <QString>
#include <QDataStream>
#include <QHash>
#include <QList>

class FileSender : public QObject
{
    Q_OBJECT
public:
    explicit FileSender(QObject *parent = nullptr);
    FileSender(QString ip, int portNo);
    void FindSeqImage(QString runid, QString pathName, QString ip, int port);
    QStringList FindSeqImage(QString pathName);
    void Send(QString path);
signals:

public slots:

private slots:
    void readData();

private:
    const QString pattern = "cyc_(\\d+)-T(\\d+)-\\w+-\\d+\\.\\d+-\\d+\\.\\d+-C\\w+-\\d+.*\\.tiff";
//    QHash<QString, int> fileMap = new QHash<QString, int>();

    const qint32 MAX_FILE_PATH_LENGTH = 100; //Byte
    const qint32 MAX_FILE_LENGTH = 4;
    const qint32 FILE_COUNT_SIZE = 4;       //Byte
    const qint32 TOKEN_SIZE = 10;
    const qint32 END_SIZE = 3;

    QTcpSocket *tcpSocket = nullptr;
    QDataStream out;
    QString mIp;
    int mPortNo;
    int mSendCount;
    QString mPath;
    QStringList fileList;


    bool SendFile(QString filePath, QString runid);
    void PrepareToSend();
    void ConnectingServer();
    void sendNsdsHeader(qint32 fileCount);
    bool int32ToByte(qint32 nbr, char* pBuffer);
};

#endif // FILESENDER_H
