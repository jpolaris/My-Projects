#include <QCoreApplication>
#include <QDebug>
#include <QString>

#include "filerecepter.h"
#include "msgserver.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    a.setApplicationName(QString("NSDS Server"));
    qDebug()<<"Welcome to NSDS Server (C++ Version)"<<endl;
    qDebug()<<"Usage: argv[1]- Destination folder to save files\n"<<endl;

    QString destPath;

#ifdef Q_OS_LINUX
    destPath = QString("/home/erling/Documents/test/dest");
#else
    destPath = QString("E:/ErlingSi/image/dest");
#endif

    if(argc>1) destPath = QString(argv[1]);

    msgServer msgSvr;
    msgSvr.setDestPath(destPath);

   // qDebug()<<"destPath="<<destPath<<endl;

    while(1)
        a.exec();

    return 0;
}
