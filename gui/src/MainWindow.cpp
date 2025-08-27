#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "InputForm.h"
#include "ProcessRunner.h"
#include "AdvancedOptionsDialog.h"
#include <QLabel>
#include <QProgressBar>
#include <QMessageBox>
#include <QFileDialog>
#include <QSettings>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QDateTime>
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      m_processRunner(new ProcessRunner(this)),
      m_advancedDialog(new AdvancedOptionsDialog(this)),
      m_elapsedTimer(new QTimer(this)),
      m_startTime(0)
{

    ui->setupUi(this);
    setupUI();
    connectSignals();

    // Load settings
    QSettings settings("cudaSAXS", "GUI");
    restoreGeometry(settings.value("geometry").toByteArray());
    restoreState(settings.value("windowState").toByteArray());

    // Set executable path (adjust as needed)
    QString execPath = settings.value("executablePath", "./cudaSAXS").toString();
    m_processRunner->setExecutablePath(execPath);
}

MainWindow::~MainWindow()
{
    // Save settings
    QSettings settings("cudaSAXS", "GUI");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("windowState", saveState());

    delete ui;
}

void MainWindow::setupUI()
{
    // Status bar widgets
    m_statusLabel = new QLabel("Ready");
    ui->statusbar->addWidget(m_statusLabel);

    m_progressBar = new QProgressBar;
    m_progressBar->setVisible(false);
    m_progressBar->setTextVisible(false);
    m_progressBar->setRange(0, 0); // Indeterminate
    ui->statusbar->addPermanentWidget(m_progressBar);

    m_elapsedLabel = new QLabel;
    ui->statusbar->addPermanentWidget(m_elapsedLabel);
}

void MainWindow::connectSignals()
{
    // Menu actions
    connect(ui->actionNew, &QAction::triggered, this, &MainWindow::onNewProject);
    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::onOpenProject);
    connect(ui->actionSave, &QAction::triggered, this, &MainWindow::onSaveProject);
    connect(ui->actionSaveAs, &QAction::triggered, this, &MainWindow::onSaveProjectAs);
    connect(ui->actionExit, &QAction::triggered, this, &QWidget::close);
    connect(ui->actionRun, &QAction::triggered, this, &MainWindow::onRunRequested);
    connect(ui->actionStop, &QAction::triggered, m_processRunner, &ProcessRunner::cancel);
    connect(ui->actionAdvancedOptions, &QAction::triggered, this, &MainWindow::onAdvancedOptionsRequested);
    connect(ui->actionPreferences, &QAction::triggered, this, &MainWindow::onPreferences);
    connect(ui->actionAbout, &QAction::triggered, this, &MainWindow::onAbout);

    // Input form signals
    connect(ui->inputForm, &InputForm::runRequested,
            this, &MainWindow::onRunRequested);
    connect(ui->inputForm, &InputForm::advancedOptionsRequested,
            this, &MainWindow::onAdvancedOptionsRequested);
    connect(ui->inputForm, &InputForm::formValidityChanged,
            this, &MainWindow::onFormValidityChanged);

    // Process runner signals
    connect(m_processRunner, &ProcessRunner::started,
            this, &MainWindow::onProcessStarted);
    connect(m_processRunner, &ProcessRunner::finished,
            this, &MainWindow::onProcessFinished);
    connect(m_processRunner, &ProcessRunner::error,
            this, &MainWindow::onProcessError);
    connect(m_processRunner, &ProcessRunner::outputReceived,
            this, &MainWindow::onProcessOutput);
    connect(m_processRunner, &ProcessRunner::progressReceived,
            this, &MainWindow::onProcessProgress);

    // Timer
    connect(m_elapsedTimer, &QTimer::timeout,
            this, &MainWindow::updateElapsedTime);
}

void MainWindow::onRunRequested()
{
    if (m_processRunner->isRunning())
    {
        QMessageBox::warning(this, "Process Running",
                             "A calculation is already running. Please wait for it to finish or stop it.");
        return;
    }

    OptionsData options = ui->inputForm->getOptionsData();
    if (!options.isValid())
    {
        QMessageBox::critical(this, "Invalid Options", options.validationError());
        return;
    }

    // Clear output
    ui->outputText->clear();

    // Get command line arguments
    QStringList args = options.toCommandLineArgs();

    // Log the command
    ui->outputText->append("Command: cudaSAXS " + args.join(" ") + "\n");
    ui->outputText->append(QString(80, '-') + "\n");

    // Set working directory to the directory of the trajectory file
    QFileInfo xtcInfo(options.xtcFile);
    m_processRunner->setWorkingDirectory(xtcInfo.absolutePath());

    // Run the process
    m_processRunner->run(args);
}

void MainWindow::onAdvancedOptionsRequested()
{
    OptionsData currentOptions = ui->inputForm->getOptionsData();
    m_advancedDialog->setOptionsData(currentOptions);

    if (m_advancedDialog->exec() == QDialog::Accepted)
    {
        OptionsData advancedOptions = m_advancedDialog->getOptionsData();
        // Merge with current options
        currentOptions.bsplineOrder = advancedOptions.bsplineOrder;
        currentOptions.scaleFactor = advancedOptions.scaleFactor;
        currentOptions.scaledGrid = advancedOptions.scaledGrid;
        currentOptions.waterModel = advancedOptions.waterModel;
        currentOptions.sodiumAtoms = advancedOptions.sodiumAtoms;
        currentOptions.chlorineAtoms = advancedOptions.chlorineAtoms;

        ui->inputForm->setOptionsData(currentOptions);
    }
}

void MainWindow::onProcessStarted()
{
    ui->actionRun->setEnabled(false);
    ui->actionStop->setEnabled(true);
    m_progressBar->setVisible(true);
    m_statusLabel->setText("Running...");

    // Start elapsed timer
    m_startTime = QDateTime::currentMSecsSinceEpoch();
    m_elapsedTimer->start(1000); // Update every second
    updateElapsedTime();
}

void MainWindow::onProcessFinished(int exitCode)
{
    ui->actionRun->setEnabled(true);
    ui->actionStop->setEnabled(false);
    m_progressBar->setVisible(false);
    m_elapsedTimer->stop();

    if (exitCode == 0)
    {
        m_statusLabel->setText("Finished successfully");
        ui->outputText->append("\n" + QString(80, '-'));
        ui->outputText->append("Process finished successfully");

        QMessageBox::information(this, "Success",
                                 "SAXS calculation completed successfully!");
    }
    else
    {
        m_statusLabel->setText("Finished with errors");
        ui->outputText->append("\n" + QString(80, '-'));
        ui->outputText->append(QString("Process finished with exit code: %1").arg(exitCode));

        QMessageBox::warning(this, "Process Failed",
                             QString("Process exited with code %1. Check the output for details.").arg(exitCode));
    }
}

void MainWindow::onProcessError(const QString &error)
{
    ui->outputText->append("\nERROR: " + error);
    m_statusLabel->setText("Error occurred");
}

void MainWindow::onProcessOutput(const QString &output)
{
    // Add output, but don't scroll if user is reading
    auto cursor = ui->outputText->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertText(output);
}

void MainWindow::onProcessProgress(const QString &progress)
{
    m_statusLabel->setText(progress);
}

void MainWindow::onFormValidityChanged(bool isValid)
{
    ui->actionRun->setEnabled(isValid && !m_processRunner->isRunning());
}

void MainWindow::updateElapsedTime()
{
    if (m_startTime > 0)
    {
        qint64 elapsed = QDateTime::currentMSecsSinceEpoch() - m_startTime;
        int seconds = elapsed / 1000;
        int minutes = seconds / 60;
        seconds %= 60;
        m_elapsedLabel->setText(QString("Elapsed: %1:%2")
                                    .arg(minutes, 2, 10, QChar('0'))
                                    .arg(seconds, 2, 10, QChar('0')));
    }
    else
    {
        m_elapsedLabel->clear();
    }
}

void MainWindow::onNewProject()
{
    if (m_processRunner->isRunning())
    {
        QMessageBox::warning(this, "Process Running",
                             "Cannot create new project while a calculation is running.");
        return;
    }

    m_projectFile.clear();
    ui->inputForm->resetToDefaults();
    ui->outputText->clear();
    setWindowTitle("cudaSAXS GUI");
}

void MainWindow::onOpenProject()
{
    if (m_processRunner->isRunning())
    {
        QMessageBox::warning(this, "Process Running",
                             "Cannot open project while a calculation is running.");
        return;
    }

    QString filename = QFileDialog::getOpenFileName(this,
                                                    "Open Project", QString(), "cudaSAXS Project (*.csp);;All Files (*)");

    if (!filename.isEmpty())
    {
        loadProject(filename);
    }
}

void MainWindow::onSaveProject()
{
    if (m_projectFile.isEmpty())
    {
        onSaveProjectAs();
    }
    else
    {
        saveProject(m_projectFile);
    }
}

void MainWindow::onSaveProjectAs()
{
    QString filename = QFileDialog::getSaveFileName(this,
                                                    "Save Project", QString(), "cudaSAXS Project (*.csp);;All Files (*)");

    if (!filename.isEmpty())
    {
        if (!filename.endsWith(".csp"))
        {
            filename += ".csp";
        }
        saveProject(filename);
    }
}

void MainWindow::loadProject(const QString &filename)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly))
    {
        QMessageBox::critical(this, "Error",
                              "Could not open project file: " + file.errorString());
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject root = doc.object();

    // Load options
    QJsonObject options = root["options"].toObject();
    OptionsData data;

    data.tprFile = options["tprFile"].toString();
    data.xtcFile = options["xtcFile"].toString();

    QJsonArray gridArray = options["gridSize"].toArray();
    for (const auto &val : gridArray)
    {
        data.gridSize.append(val.toInt());
    }

    data.startFrame = options["startFrame"].toInt();
    data.endFrame = options["endFrame"].toInt();
    data.frameInterval = options["frameInterval"].toInt(1);
    data.outputFile = options["outputFile"].toString("saxs.dat");
    data.bsplineOrder = options["bsplineOrder"].toInt(4);

    QJsonArray scaledArray = options["scaledGrid"].toArray();
    for (const auto &val : scaledArray)
    {
        data.scaledGrid.append(val.toInt());
    }

    data.scaleFactor = options["scaleFactor"].toDouble(2.5);
    data.binSize = options["binSize"].toDouble(0.05);
    data.qCutoff = options["qCutoff"].toDouble(4.0);
    data.waterModel = options["waterModel"].toString();
    data.sodiumAtoms = options["sodiumAtoms"].toInt();
    data.chlorineAtoms = options["chlorineAtoms"].toInt();
    data.simulationType = options["simulationType"].toString("npt");

    ui->inputForm->setOptionsData(data);

    // Load output if present
    if (root.contains("output"))
    {
        ui->outputText->setPlainText(root["output"].toString());
    }

    m_projectFile = filename;
    setWindowTitle(QString("cudaSAXS GUI - %1").arg(QFileInfo(filename).fileName()));
}

void MainWindow::saveProject(const QString &filename)
{
    QJsonObject root;

    // Save options
    OptionsData data = ui->inputForm->getOptionsData();
    QJsonObject options;

    options["tprFile"] = data.tprFile;
    options["xtcFile"] = data.xtcFile;

    QJsonArray gridArray;
    for (int val : data.gridSize)
    {
        gridArray.append(val);
    }
    options["gridSize"] = gridArray;

    options["startFrame"] = data.startFrame;
    options["endFrame"] = data.endFrame;
    options["frameInterval"] = data.frameInterval;
    options["outputFile"] = data.outputFile;
    options["bsplineOrder"] = data.bsplineOrder;

    QJsonArray scaledArray;
    for (int val : data.scaledGrid)
    {
        scaledArray.append(val);
    }
    options["scaledGrid"] = scaledArray;

    options["scaleFactor"] = data.scaleFactor;
    options["binSize"] = data.binSize;
    options["qCutoff"] = data.qCutoff;
    options["waterModel"] = data.waterModel;
    options["sodiumAtoms"] = data.sodiumAtoms;
    options["chlorineAtoms"] = data.chlorineAtoms;
    options["simulationType"] = data.simulationType;

    root["options"] = options;

    // Save output
    root["output"] = ui->outputText->toPlainText();

    // Write to file
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(this, "Error",
                              "Could not save project file: " + file.errorString());
        return;
    }

    QJsonDocument doc(root);
    file.write(doc.toJson());

    m_projectFile = filename;
    setWindowTitle(QString("cudaSAXS GUI - %1").arg(QFileInfo(filename).fileName()));
    m_statusLabel->setText("Project saved");
}

void MainWindow::onAbout()
{
    QMessageBox::about(this, "About cudaSAXS GUI",
                       "<h3>cudaSAXS GUI</h3>"
                       "<p>A graphical interface for cudaSAXS - GPU-accelerated SAXS calculations</p>"
                       "<p>This interface provides an easy way to configure and run SAXS calculations "
                       "on molecular dynamics trajectories using CUDA acceleration.</p>"
                       "<p>Version 1.0</p>");
}

void MainWindow::onPreferences()
{
    QSettings settings("cudaSAXS", "GUI");

    QString currentPath = settings.value("executablePath", "./cudaSAXS").toString();
    QString newPath = QFileDialog::getOpenFileName(this,
                                                   "Select cudaSAXS Executable", currentPath, "Executable Files (*)");

    if (!newPath.isEmpty())
    {
        settings.setValue("executablePath", newPath);
        m_processRunner->setExecutablePath(newPath);
        QMessageBox::information(this, "Preferences",
                                 "Executable path updated. The new path will be used for future calculations.");
    }
}