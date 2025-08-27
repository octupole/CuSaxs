#ifndef ADVANCEDOPTIONSDIALOG_H
#define ADVANCEDOPTIONSDIALOG_H

#include <QDialog>
#include "OptionsData.h"
#include "InputValidator.h"
#include <memory>

QT_BEGIN_NAMESPACE
namespace Ui
{
    class AdvancedOptionsDialog;
}
QT_END_NAMESPACE

class AdvancedOptionsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AdvancedOptionsDialog(QWidget *parent = nullptr);
    ~AdvancedOptionsDialog();

    void setOptionsData(const OptionsData &data);
    OptionsData getOptionsData() const;
    
    bool isValid() const;
    QString getValidationError() const;
    QStringList getValidationWarnings() const;

private:
    void setupValidation();
    void validateForm();
    
    Ui::AdvancedOptionsDialog *ui;
    std::vector<std::unique_ptr<WidgetValidator>> m_validators;
    bool m_isValid;
};

#endif // ADVANCEDOPTIONSDIALOG_H