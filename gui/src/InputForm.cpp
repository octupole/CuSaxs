#include "InputForm.h"
#include "ui_InputForm.h"
#include <QFileDialog>
#include <QFileInfo>

InputForm::InputForm(QWidget *parent)
    : QWidget(parent),
      ui(new Ui::InputForm)
{
    ui->setupUi(this);
    connectSignals();
    resetToDefaults();
    validateForm();
}

InputForm::~InputForm()
{
    delete ui;
}

void InputForm::connectSignals()
{
    // File browse buttons
    connect(ui->tprBrowseBtn, &QPushButton::clicked, this, &InputForm::onBrowseTPR);
    connect(ui->xtcBrowseBtn, &QPushButton::clicked, this, &InputForm::onBrowseXTC);
    connect(ui->outputBrowseBtn, &QPushButton::clicked, this, &InputForm::onBrowseOutput);

    // Grid type change is already connected in the .ui file to the stacked widget
    connect(ui->gridTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InputForm::onGridTypeChanged);

    // Advanced and run buttons
    connect(ui->advancedBtn, &QPushButton::clicked, this, &InputForm::advancedOptionsRequested);
    connect(ui->runBtn, &QPushButton::clicked, this, &InputForm::runRequested);

    // Connect all inputs to validation
    connect(ui->tprFileEdit, &QLineEdit::textChanged, this, &InputForm::validateForm);
    connect(ui->xtcFileEdit, &QLineEdit::textChanged, this, &InputForm::validateForm);
    connect(ui->gridSingleSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &InputForm::validateForm);
    connect(ui->gridXSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &InputForm::validateForm);
    connect(ui->gridYSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &InputForm::validateForm);
    connect(ui->gridZSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &InputForm::validateForm);
    connect(ui->startFrameSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &InputForm::validateForm);
    connect(ui->endFrameSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &InputForm::validateForm);
}

void InputForm::onBrowseTPR()
{
    QString filename = QFileDialog::getOpenFileName(this,
                                                    "Select Topology File", QString(), "TPR Files (*.tpr);;All Files (*)");
    if (!filename.isEmpty())
    {
        ui->tprFileEdit->setText(filename);
    }
}

void InputForm::onBrowseXTC()
{
    QString filename = QFileDialog::getOpenFileName(this,
                                                    "Select Trajectory File", QString(), "XTC Files (*.xtc);;All Files (*)");
    if (!filename.isEmpty())
    {
        ui->xtcFileEdit->setText(filename);
    }
}

void InputForm::onBrowseOutput()
{
    QString filename = QFileDialog::getSaveFileName(this,
                                                    "Select Output File", QString(), "Data Files (*.dat);;All Files (*)");
    if (!filename.isEmpty())
    {
        ui->outputFileEdit->setText(filename);
    }
}

void InputForm::onGridTypeChanged(int index)
{
    // The stacked widget switching is handled by the connection in the .ui file
    // This is just for any additional logic needed
    validateForm();
}

void InputForm::validateForm()
{
    OptionsData data = getOptionsData();
    bool isValid = data.isValid();
    QString error = data.validationError();

    ui->runBtn->setEnabled(isValid);

    if (isValid)
    {
        ui->statusLabel->setText("Ready to run");
        ui->statusLabel->setStyleSheet("QLabel { color: green; }");
    }
    else
    {
        ui->statusLabel->setText(error);
        ui->statusLabel->setStyleSheet("QLabel { color: red; }");
    }

    emit formValidityChanged(isValid);
}

OptionsData InputForm::getOptionsData() const
{
    OptionsData data = m_advancedOptions; // Start with advanced options

    // Override with form values
    data.tprFile = ui->tprFileEdit->text();
    data.xtcFile = ui->xtcFileEdit->text();

    // Grid size
    data.gridSize.clear();
    if (ui->gridTypeCombo->currentIndex() == 0)
    {
        // Cubic grid
        data.gridSize.append(ui->gridSingleSpin->value());
    }
    else
    {
        // Custom grid
        data.gridSize.append(ui->gridXSpin->value());
        data.gridSize.append(ui->gridYSpin->value());
        data.gridSize.append(ui->gridZSpin->value());
    }

    data.startFrame = ui->startFrameSpin->value();
    data.endFrame = ui->endFrameSpin->value();
    data.frameInterval = ui->frameIntervalSpin->value();

    data.outputFile = ui->outputFileEdit->text();
    data.binSize = ui->binSizeSpin->value();
    data.qCutoff = ui->qCutoffSpin->value();
    data.simulationType = ui->simulationTypeCombo->currentText();

    return data;
}

void InputForm::setOptionsData(const OptionsData &data)
{
    ui->tprFileEdit->setText(data.tprFile);
    ui->xtcFileEdit->setText(data.xtcFile);

    // Grid size
    if (data.gridSize.size() == 1)
    {
        ui->gridTypeCombo->setCurrentIndex(0);
        ui->gridSingleSpin->setValue(data.gridSize[0]);
    }
    else if (data.gridSize.size() == 3)
    {
        ui->gridTypeCombo->setCurrentIndex(1);
        ui->gridXSpin->setValue(data.gridSize[0]);
        ui->gridYSpin->setValue(data.gridSize[1]);
        ui->gridZSpin->setValue(data.gridSize[2]);
    }

    ui->startFrameSpin->setValue(data.startFrame);
    ui->endFrameSpin->setValue(data.endFrame);
    ui->frameIntervalSpin->setValue(data.frameInterval);

    ui->outputFileEdit->setText(data.outputFile);
    ui->binSizeSpin->setValue(data.binSize);
    ui->qCutoffSpin->setValue(data.qCutoff);

    int simIndex = ui->simulationTypeCombo->findText(data.simulationType);
    if (simIndex >= 0)
    {
        ui->simulationTypeCombo->setCurrentIndex(simIndex);
    }

    m_advancedOptions = data;
    validateForm();
}

void InputForm::resetToDefaults()
{
    OptionsData defaults;
    defaults.gridSize.append(64);
    defaults.startFrame = 0;
    defaults.endFrame = 100;
    setOptionsData(defaults);
}