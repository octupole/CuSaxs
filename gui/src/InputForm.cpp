#include "InputForm.h"
#include "ui_InputForm.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>

InputForm::InputForm(QWidget *parent)
    : QWidget(parent),
      ui(new Ui::InputForm)
{
    ui->setupUi(this);
    setupValidation();
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
    // First run individual widget validation
    bool allWidgetsValid = true;
    for (auto& validator : m_validators) {
        validator->validate(); // This will update UI automatically
        if (!validator->isValid()) {
            allWidgetsValid = false;
        }
    }
    
    // Then run comprehensive form validation
    OptionsData data = getOptionsData();
    bool isDataValid = data.isValid();
    QString error = data.validationError();
    QStringList warnings = data.validationWarnings();

    bool isFormValid = allWidgetsValid && isDataValid;
    ui->runBtn->setEnabled(isFormValid);

    if (isFormValid)
    {
        if (warnings.isEmpty()) {
            ui->statusLabel->setText("✅ Ready to run");
            ui->statusLabel->setStyleSheet("QLabel { color: green; font-weight: bold; }");
        } else {
            ui->statusLabel->setText("⚠️ Ready to run (with warnings)");
            ui->statusLabel->setStyleSheet("QLabel { color: orange; font-weight: bold; }");
            ui->statusLabel->setToolTip("Warnings:\n" + warnings.join("\n"));
        }
    }
    else
    {
        ui->statusLabel->setText("❌ " + error);
        ui->statusLabel->setStyleSheet("QLabel { color: red; font-weight: bold; }");
        ui->statusLabel->setToolTip("");
    }

    emit formValidityChanged(isFormValid);
}

void InputForm::setupValidation()
{
    // Create error labels for validation feedback (assuming they exist in the UI)
    // If not in UI file, these could be added programmatically
    
    // TPR file validator
    auto tprValidator = std::make_unique<FileExistsValidator>(QStringList{"tpr"}, true);
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->tprFileEdit, std::move(tprValidator), nullptr, this
    ));
    
    // XTC file validator  
    auto xtcValidator = std::make_unique<FileExistsValidator>(QStringList{"xtc"}, true, 10LL * 1024 * 1024 * 1024); // 10GB max
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->xtcFileEdit, std::move(xtcValidator), nullptr, this
    ));
    
    // Output file validator
    auto outputValidator = std::make_unique<OutputFileValidator>();
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->outputFileEdit, std::move(outputValidator), nullptr, this
    ));
    
    // Grid size validators
    auto gridSingleValidator = std::make_unique<RangeValidator<int>>(8, 512, "grid points");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->gridSingleSpin, std::move(gridSingleValidator), nullptr, this
    ));
    
    auto gridXValidator = std::make_unique<RangeValidator<int>>(8, 512, "grid points");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->gridXSpin, std::move(gridXValidator), nullptr, this
    ));
    
    auto gridYValidator = std::make_unique<RangeValidator<int>>(8, 512, "grid points");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->gridYSpin, std::move(gridYValidator), nullptr, this
    ));
    
    auto gridZValidator = std::make_unique<RangeValidator<int>>(8, 512, "grid points");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->gridZSpin, std::move(gridZValidator), nullptr, this
    ));
    
    // Frame validators
    auto startFrameValidator = std::make_unique<RangeValidator<int>>(0, 1000000, "frames");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->startFrameSpin, std::move(startFrameValidator), nullptr, this
    ));
    
    // End frame validator that checks against start frame
    auto getStartFrame = [this]() { return ui->startFrameSpin->value(); };
    auto endFrameValidator = std::make_unique<FrameRangeValidator>(getStartFrame, 1000000);
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->endFrameSpin, std::move(endFrameValidator), nullptr, this
    ));
    
    // Frame interval validator
    auto frameIntervalValidator = std::make_unique<RangeValidator<int>>(1, 1000, "frames");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->frameIntervalSpin, std::move(frameIntervalValidator), nullptr, this
    ));
    
    // Bin size validator
    auto binSizeValidator = std::make_unique<RangeValidator<double>>(0.001, 1.0, "Å⁻¹");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->binSizeSpin, std::move(binSizeValidator), nullptr, this
    ));
    
    // Q cutoff validator
    auto qCutoffValidator = std::make_unique<RangeValidator<double>>(0.1, 20.0, "Å⁻¹");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->qCutoffSpin, std::move(qCutoffValidator), nullptr, this
    ));
    
    // Connect validation signals to form validation
    for (auto& validator : m_validators) {
        connect(validator.get(), &WidgetValidator::validationChanged,
                this, &InputForm::validateForm);
    }
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