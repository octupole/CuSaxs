#include "AdvancedOptionsDialog.h"
#include "ui_AdvancedOptionsDialog.h"
#include <QDialogButtonBox>
#include <QPushButton>

AdvancedOptionsDialog::AdvancedOptionsDialog(QWidget *parent)
    : QDialog(parent),
      ui(new Ui::AdvancedOptionsDialog),
      m_isValid(true)
{
    ui->setupUi(this);
    setupValidation();

    // Note: Most connections are already made in the .ui file:
    // - buttonBox accept/reject to dialog accept/reject
    // - scaledGridCheck toggled to scaledGridWidget setEnabled
    // - waterModelCheck toggled to waterModelWidget setEnabled
    // - scaledGridTypeCombo currentIndexChanged to scaledStackedWidget setCurrentIndex
    
    // Connect validation to dialog buttons
    connect(this, &AdvancedOptionsDialog::finished, this, &AdvancedOptionsDialog::validateForm);
}

AdvancedOptionsDialog::~AdvancedOptionsDialog()
{
    delete ui;
}

void AdvancedOptionsDialog::setOptionsData(const OptionsData &data)
{
    ui->orderSpin->setValue(data.bsplineOrder);
    ui->scaleFactorSpin->setValue(data.scaleFactor);

    // Scaled grid
    bool hasScaledGrid = !data.scaledGrid.isEmpty();
    ui->scaledGridCheck->setChecked(hasScaledGrid);

    if (hasScaledGrid)
    {
        if (data.scaledGrid.size() == 1)
        {
            ui->scaledGridTypeCombo->setCurrentIndex(0);
            ui->scaledGridSingleSpin->setValue(data.scaledGrid[0]);
        }
        else if (data.scaledGrid.size() == 3)
        {
            ui->scaledGridTypeCombo->setCurrentIndex(1);
            ui->scaledGridXSpin->setValue(data.scaledGrid[0]);
            ui->scaledGridYSpin->setValue(data.scaledGrid[1]);
            ui->scaledGridZSpin->setValue(data.scaledGrid[2]);
        }
    }

    // Water model
    bool hasWaterModel = !data.waterModel.isEmpty();
    ui->waterModelCheck->setChecked(hasWaterModel);

    if (hasWaterModel)
    {
        ui->waterModelEdit->setText(data.waterModel);
        ui->sodiumSpin->setValue(data.sodiumAtoms);
        ui->chlorineSpin->setValue(data.chlorineAtoms);
    }
}

OptionsData AdvancedOptionsDialog::getOptionsData() const
{
    OptionsData data;

    data.bsplineOrder = ui->orderSpin->value();
    data.scaleFactor = ui->scaleFactorSpin->value();

    // Scaled grid
    if (ui->scaledGridCheck->isChecked())
    {
        data.scaledGrid.clear();
        if (ui->scaledGridTypeCombo->currentIndex() == 0)
        {
            data.scaledGrid.append(ui->scaledGridSingleSpin->value());
        }
        else
        {
            data.scaledGrid.append(ui->scaledGridXSpin->value());
            data.scaledGrid.append(ui->scaledGridYSpin->value());
            data.scaledGrid.append(ui->scaledGridZSpin->value());
        }
    }

    // Water model
    if (ui->waterModelCheck->isChecked())
    {
        data.waterModel = ui->waterModelEdit->text();
        data.sodiumAtoms = ui->sodiumSpin->value();
        data.chlorineAtoms = ui->chlorineSpin->value();
    }

    return data;
}

void AdvancedOptionsDialog::setupValidation()
{
    // B-spline order validator
    auto orderValidator = std::make_unique<RangeValidator<int>>(1, 6, "order");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->orderSpin, std::move(orderValidator), nullptr, this
    ));
    
    // Scale factor validator
    auto scaleValidator = std::make_unique<RangeValidator<double>>(1.1, 10.0, "scaling factor");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->scaleFactorSpin, std::move(scaleValidator), nullptr, this
    ));
    
    // Scaled grid validators
    auto scaledGridSingleValidator = std::make_unique<RangeValidator<int>>(16, 1024, "grid points");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->scaledGridSingleSpin, std::move(scaledGridSingleValidator), nullptr, this
    ));
    
    auto scaledGridXValidator = std::make_unique<RangeValidator<int>>(16, 1024, "grid points");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->scaledGridXSpin, std::move(scaledGridXValidator), nullptr, this
    ));
    
    auto scaledGridYValidator = std::make_unique<RangeValidator<int>>(16, 1024, "grid points");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->scaledGridYSpin, std::move(scaledGridYValidator), nullptr, this
    ));
    
    auto scaledGridZValidator = std::make_unique<RangeValidator<int>>(16, 1024, "grid points");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->scaledGridZSpin, std::move(scaledGridZValidator), nullptr, this
    ));
    
    // Ion count validators
    auto sodiumValidator = std::make_unique<RangeValidator<int>>(0, 100000, "atoms");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->sodiumSpin, std::move(sodiumValidator), nullptr, this
    ));
    
    auto chlorineValidator = std::make_unique<RangeValidator<int>>(0, 100000, "atoms");
    m_validators.push_back(std::make_unique<WidgetValidator>(
        ui->chlorineSpin, std::move(chlorineValidator), nullptr, this
    ));
    
    // Connect validation signals
    for (auto& validator : m_validators) {
        connect(validator.get(), &WidgetValidator::validationChanged,
                this, &AdvancedOptionsDialog::validateForm);
    }
}

void AdvancedOptionsDialog::validateForm()
{
    bool allValid = true;
    for (auto& validator : m_validators) {
        validator->validate();
        if (!validator->isValid()) {
            allValid = false;
        }
    }
    
    m_isValid = allValid;
    
    // Enable/disable OK button based on validation
    if (ui->buttonBox) {
        ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(m_isValid);
    }
}

bool AdvancedOptionsDialog::isValid() const
{
    return m_isValid;
}

QString AdvancedOptionsDialog::getValidationError() const
{
    for (const auto& validator : m_validators) {
        if (!validator->isValid()) {
            return validator->getErrorMessage();
        }
    }
    return QString();
}

QStringList AdvancedOptionsDialog::getValidationWarnings() const
{
    QStringList warnings;
    for (const auto& validator : m_validators) {
        QString warning = validator->getWarningMessage();
        if (!warning.isEmpty()) {
            warnings.append(warning);
        }
    }
    return warnings;
}