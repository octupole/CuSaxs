#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include "OptionsData.h"

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
class QLabel;
class QProgressBar;
QT_END_NAMESPACE

class ProcessRunner;
class AdvancedOptionsDialog;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onRunRequested();
    void onAdvancedOptionsRequested();
    void onProcessStarted();
    void onProcessFinished(int exitCode);
    void onProcessError(const QString &error);
    void onProcessOutput(const QString &output);
    void onProcessProgress(const QString &progress);
    void onFormValidityChanged(bool isValid);

    // Menu actions
    void onNewProject();
    void onOpenProject();
    void onSaveProject();
    void onSaveProjectAs();
    void onAbout();
    void onPreferences();

    // Timer for status
    void updateElapsedTime();

private:
    void setupUI();
    void connectSignals();

    void loadProject(const QString &filename);
    void saveProject(const QString &filename);

    Ui::MainWindow *ui;

    // Process runner
    ProcessRunner *m_processRunner;

    // Advanced options dialog
    AdvancedOptionsDialog *m_advancedDialog;

    // Status bar widgets
    QLabel *m_statusLabel;
    QProgressBar *m_progressBar;
    QLabel *m_elapsedLabel;

    // Timer for elapsed time
    QTimer *m_elapsedTimer;
    qint64 m_startTime;

    // Current project file
    QString m_projectFile;
};

#endif // MAINWINDOW_H