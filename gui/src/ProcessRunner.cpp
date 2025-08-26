#include "ProcessRunner.h"
#include <QDir>

ProcessRunner::ProcessRunner(QObject *parent)
    : QObject(parent), m_process(new QProcess(this))
{

    connect(m_process, &QProcess::errorOccurred,
            this, &ProcessRunner::handleProcessError);
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &ProcessRunner::handleProcessFinished);
    connect(m_process, &QProcess::readyReadStandardOutput,
            this, &ProcessRunner::handleStandardOutput);
    connect(m_process, &QProcess::readyReadStandardError,
            this, &ProcessRunner::handleStandardError);
    connect(m_process, &QProcess::started,
            this, &ProcessRunner::started);
}

ProcessRunner::~ProcessRunner()
{
    if (m_process->state() != QProcess::NotRunning)
    {
        m_process->terminate();
        if (!m_process->waitForFinished(5000))
        {
            m_process->kill();
        }
    }
}

void ProcessRunner::setExecutablePath(const QString &path)
{
    m_executablePath = path;
}

void ProcessRunner::setWorkingDirectory(const QString &dir)
{
    m_workingDirectory = dir;
}

void ProcessRunner::run(const QStringList &arguments)
{
    if (m_executablePath.isEmpty())
    {
        emit error("Executable path not set");
        return;
    }

    if (m_process->state() != QProcess::NotRunning)
    {
        emit error("Process is already running");
        return;
    }

    if (!m_workingDirectory.isEmpty())
    {
        m_process->setWorkingDirectory(m_workingDirectory);
    }

    m_process->start(m_executablePath, arguments);
}

void ProcessRunner::cancel()
{
    if (m_process->state() != QProcess::NotRunning)
    {
        m_process->terminate();
        if (!m_process->waitForFinished(5000))
        {
            m_process->kill();
        }
    }
}

bool ProcessRunner::isRunning() const
{
    return m_process->state() != QProcess::NotRunning;
}

void ProcessRunner::handleProcessError(QProcess::ProcessError error)
{
    QString errorString;
    switch (error)
    {
    case QProcess::FailedToStart:
        errorString = "Failed to start process. Check if the executable exists and is accessible.";
        break;
    case QProcess::Crashed:
        errorString = "Process crashed.";
        break;
    case QProcess::Timedout:
        errorString = "Process timed out.";
        break;
    case QProcess::WriteError:
        errorString = "Write error occurred.";
        break;
    case QProcess::ReadError:
        errorString = "Read error occurred.";
        break;
    case QProcess::UnknownError:
    default:
        errorString = "Unknown error occurred.";
        break;
    }
    emit this->error(errorString);
}

void ProcessRunner::handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::CrashExit)
    {
        emit error("Process crashed");
    }
    emit finished(exitCode);
}

void ProcessRunner::handleStandardOutput()
{
    QByteArray data = m_process->readAllStandardOutput();
    QString output = QString::fromUtf8(data);
    emit outputReceived(output);

    // Try to parse progress information
    // Look for patterns like "Frame: X" or "Progress: X%"
    QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    for (const QString &line : lines)
    {
        if (line.contains("Frame:") || line.contains("Progress:") ||
            line.contains("Step") || line.contains("Time"))
        {
            emit progressReceived(line.trimmed());
        }
    }
}

void ProcessRunner::handleStandardError()
{
    QByteArray data = m_process->readAllStandardError();
    QString errorOutput = QString::fromUtf8(data);

    // Some programs output progress to stderr
    emit outputReceived(errorOutput);

    // Check if it's actually an error
    if (errorOutput.contains("error", Qt::CaseInsensitive) ||
        errorOutput.contains("fatal", Qt::CaseInsensitive) ||
        errorOutput.contains("failed", Qt::CaseInsensitive))
    {
        emit error(errorOutput);
    }
}