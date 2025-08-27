#ifndef INPUTFORM_H
#define INPUTFORM_H

#include <QWidget>
#include "OptionsData.h"
#include "InputValidator.h"
#include <memory>

QT_BEGIN_NAMESPACE
namespace Ui
{
    class InputForm;
}
QT_END_NAMESPACE

class InputForm : public QWidget
{
    Q_OBJECT

public:
    explicit InputForm(QWidget *parent = nullptr);
    ~InputForm();

    // Get the current form data
    OptionsData getOptionsData() const;

    // Set form data
    void setOptionsData(const OptionsData &data);

    // Reset form to defaults
    void resetToDefaults();

signals:
    void advancedOptionsRequested();
    void runRequested();
    void formValidityChanged(bool isValid);

private slots:
    void onBrowseTPR();
    void onBrowseXTC();
    void onBrowseOutput();
    void onGridTypeChanged(int index);
    void validateForm();

private:
    void setupUI();
    void connectSignals();

    Ui::InputForm *ui;

    // Store advanced options
    OptionsData m_advancedOptions;
    
    // Validation system
    void setupValidation();
    std::vector<std::unique_ptr<WidgetValidator>> m_validators;
};

#endif // INPUTFORM_H