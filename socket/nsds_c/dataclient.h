#ifndef DATACLIENT_H
#define DATACLIENT_H

#include <QObject>
#include <QThread>
#include <QTcpSocket>
#include <list>
#include <QDataStream>

class DataClient : public QThread
{
    Q_OBJECT
public:
    DataClient();
    ~DataClient();

    void setIp(QString ip);
    void setPortNo(int portNo);
    void setSrcPath(QString srcPath);
    void addFileList(QStringList& fileList);

private slots:
    void threadFinished();

private:
    void connectToServer();
    void sendFiles();
    void sendFile(QString filePath);
    QString getFormattedOutput();

private:
    QString mIp;
    int mPortNo;
    QString mSrcPath;
    QStringList mFileList;

    QTcpSocket *mDataClientSocket = nullptr;
    QDataStream out;

    QString mMyName;
};

#endif // DATACLIENT_H
