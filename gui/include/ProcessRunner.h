#ifndef PROCESSRUNNER_H
#define PROCESSRUNNER_H

#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>

class ProcessRunner : public QObject
{
    Q_OBJECT

public:
    explicit ProcessRunner(QObject *parent = nullptr);
    ~ProcessRunner();

    void setExecutablePath(const QString &path);
    void setWorkingDirectory(const QString &dir);

    void run(const QStringList &arguments);
    void cancel();

    bool isRunning() const;

signals:
    void started();
    void finished(int exitCode);
    void error(const QString &errorMessage);
    void outputReceived(const QString &output);
    void progressReceived(const QString &progress);

private slots:
    void handleProcessError(QProcess::ProcessError error);
    void handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void handleStandardOutput();
    void handleStandardError();

private:
    QProcess *m_process;
    QString m_executablePath;
    QString m_workingDirectory;
};

#endif // PROCESSRUNNER_H