#ifndef ADVANCEDOPTIONSDIALOG_H
#define ADVANCEDOPTIONSDIALOG_H

#include <QDialog>
#include "OptionsData.h"

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

private:
    Ui::AdvancedOptionsDialog *ui;
};

#endif // ADVANCEDOPTIONSDIALOG_H