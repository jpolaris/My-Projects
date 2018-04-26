#include <QCoreApplication>
#include <QDebug>
#include "filesender.h"
#include "msgclient.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    a.setApplicationName(QString("NSDS Client"));

    qDebug()<<"Welcome to NSDS Client (C++ Version)\n";
    qDebug()<<"Usage: argv[1]-Server IP\n       argv[2]-Message Socket Port No\n       argv[3]-Data Socket Port No\n       argv[4]-Source file location"<<endl;

//    qDebug()<<"argc = "<<argc<<endl;
//    for(int i=0;i<argc;i++){
//        qDebug()<<"arg["<<i<<"]="<<argv[i]<<endl;
//    }

    QString ip = QString("10.0.32.238");
    int msgPortNo = 37691;
    int dataPortNo = 37690;
    QString srcPath;

#ifdef Q_OS_LINUX
    srcPath = QString("/home/erling/Documents/test/source");
#else
    srcPath = QString("E:/syncthing");
#endif

    //qDebug()<<"source folder: "<<srcPath<<endl;

    if(argc>5) srcPath = argv[4];
    if(argc>4) dataPortNo = QString(argv[3]).toInt();
    if(argc>3) msgPortNo = QString(argv[2]).toInt();
    if(argc>2) ip = QString(argv[1]);

    MsgClient msgClient(ip, msgPortNo, dataPortNo);
    msgClient.setSrcPath(srcPath);
//    FileSender fs(ip, portNo);
//    fs.Send(srcPath);
    //fs.FindSeqImage(QString(""),QString("E:/ErlingSi/image/test/cyc_0"),QString(""),0);

    return a.exec();
}
