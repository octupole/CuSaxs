#include "AdvancedOptionsDialog.h"
#include "ui_AdvancedOptionsDialog.h"

AdvancedOptionsDialog::AdvancedOptionsDialog(QWidget *parent)
    : QDialog(parent),
      ui(new Ui::AdvancedOptionsDialog)
{
    ui->setupUi(this);

    // Note: Most connections are already made in the .ui file:
    // - buttonBox accept/reject to dialog accept/reject
    // - scaledGridCheck toggled to scaledGridWidget setEnabled
    // - waterModelCheck toggled to waterModelWidget setEnabled
    // - scaledGridTypeCombo currentIndexChanged to scaledStackedWidget setCurrentIndex
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