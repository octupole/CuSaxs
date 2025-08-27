#ifndef INPUTVALIDATOR_H
#define INPUTVALIDATOR_H

#include <QString>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLabel>
#include <QWidget>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTimer>
#include <functional>

/**
 * @brief Validation result structure
 */
struct ValidationResult {
    bool isValid = true;
    QString errorMessage;
    QString warningMessage;
    
    ValidationResult() = default;
    ValidationResult(bool valid, const QString& error = "", const QString& warning = "")
        : isValid(valid), errorMessage(error), warningMessage(warning) {}
    
    static ValidationResult valid(const QString& warning = "") {
        return ValidationResult(true, "", warning);
    }
    
    static ValidationResult invalid(const QString& error) {
        return ValidationResult(false, error);
    }
};

/**
 * @brief Base validator interface
 */
class IValidator {
public:
    virtual ~IValidator() = default;
    virtual ValidationResult validate(const QString& value) const = 0;
    virtual QString getTooltip() const { return ""; }
};

/**
 * @brief File existence validator
 */
class FileExistsValidator : public IValidator {
private:
    QStringList m_allowedExtensions;
    bool m_checkReadable;
    qint64 m_maxFileSize;
    
public:
    explicit FileExistsValidator(const QStringList& extensions = {}, 
                               bool checkReadable = true, 
                               qint64 maxSize = -1)
        : m_allowedExtensions(extensions), m_checkReadable(checkReadable), m_maxFileSize(maxSize) {}
    
    ValidationResult validate(const QString& value) const override {
        if (value.isEmpty()) {
            return ValidationResult::invalid("File path cannot be empty");
        }
        
        QFileInfo fileInfo(value);
        
        if (!fileInfo.exists()) {
            return ValidationResult::invalid("File does not exist: " + value);
        }
        
        if (!fileInfo.isFile()) {
            return ValidationResult::invalid("Path is not a file: " + value);
        }
        
        if (m_checkReadable && !fileInfo.isReadable()) {
            return ValidationResult::invalid("File is not readable: " + value);
        }
        
        if (!m_allowedExtensions.isEmpty()) {
            QString suffix = fileInfo.suffix().toLower();
            if (!m_allowedExtensions.contains(suffix, Qt::CaseInsensitive)) {
                return ValidationResult::invalid(
                    QString("File must have one of these extensions: %1 (got .%2)")
                    .arg(m_allowedExtensions.join(", "))
                    .arg(suffix)
                );
            }
        }
        
        if (m_maxFileSize > 0 && fileInfo.size() > m_maxFileSize) {
            return ValidationResult::invalid(
                QString("File is too large: %1 MB (maximum: %2 MB)")
                .arg(fileInfo.size() / 1024.0 / 1024.0, 0, 'f', 1)
                .arg(m_maxFileSize / 1024.0 / 1024.0, 0, 'f', 1)
            );
        }
        
        return ValidationResult::valid();
    }
    
    QString getTooltip() const override {
        QString tip = "File must exist and be readable";
        if (!m_allowedExtensions.isEmpty()) {
            tip += QString("\nAllowed extensions: %1").arg(m_allowedExtensions.join(", "));
        }
        return tip;
    }
};

/**
 * @brief Numeric range validator
 */
template<typename T>
class RangeValidator : public IValidator {
private:
    T m_min, m_max;
    QString m_units;
    
public:
    RangeValidator(T min, T max, const QString& units = "")
        : m_min(min), m_max(max), m_units(units) {}
    
    ValidationResult validate(const QString& value) const override {
        bool ok;
        T numValue;
        
        if constexpr (std::is_integral_v<T>) {
            numValue = value.toInt(&ok);
        } else {
            numValue = value.toDouble(&ok);
        }
        
        if (!ok) {
            return ValidationResult::invalid("Invalid number format");
        }
        
        if (numValue < m_min || numValue > m_max) {
            return ValidationResult::invalid(
                QString("Value must be between %1 and %2%3")
                .arg(m_min).arg(m_max)
                .arg(m_units.isEmpty() ? "" : " " + m_units)
            );
        }
        
        return ValidationResult::valid();
    }
    
    QString getTooltip() const override {
        return QString("Valid range: %1 to %2%3")
            .arg(m_min).arg(m_max)
            .arg(m_units.isEmpty() ? "" : " " + m_units);
    }
};

/**
 * @brief Grid size validator
 */
class GridSizeValidator : public IValidator {
private:
    bool m_allowNonCubic;
    int m_minSize, m_maxSize;
    
public:
    GridSizeValidator(bool allowNonCubic = true, int minSize = 8, int maxSize = 512)
        : m_allowNonCubic(allowNonCubic), m_minSize(minSize), m_maxSize(maxSize) {}
    
    ValidationResult validate(const QString& value) const override {
        if (value.isEmpty()) {
            return ValidationResult::invalid("Grid size cannot be empty");
        }
        
        // Parse grid size (can be "64" or "64 64 128")
        QStringList parts = value.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        
        if (parts.size() != 1 && parts.size() != 3) {
            return ValidationResult::invalid("Grid size must be either 1 number (cubic) or 3 numbers (nx ny nz)");
        }
        
        QVector<int> sizes;
        for (const QString& part : parts) {
            bool ok;
            int size = part.toInt(&ok);
            if (!ok) {
                return ValidationResult::invalid("Invalid number in grid size: " + part);
            }
            sizes.append(size);
        }
        
        // Validate each dimension
        for (int size : sizes) {
            if (size < m_minSize || size > m_maxSize) {
                return ValidationResult::invalid(
                    QString("Grid dimension must be between %1 and %2 (got %3)")
                    .arg(m_minSize).arg(m_maxSize).arg(size)
                );
            }
            
            // Check if power of 2 (recommended for FFT performance)
            if ((size & (size - 1)) != 0) {
                QString warning = QString("Grid size %1 is not a power of 2. "
                                        "Powers of 2 (32, 64, 128, 256) provide better FFT performance.")
                                        .arg(size);
                return ValidationResult::valid(warning);
            }
        }
        
        // Check memory requirements
        qint64 totalMemory = 1;
        for (int size : sizes) {
            totalMemory *= size;
        }
        
        // Estimate memory usage (4 grids + FFT buffers + overhead)
        double memoryGB = totalMemory * (4 * 4 + 8) / 1e9; // 4 float grids + 1 complex grid
        
        if (memoryGB > 8.0) {
            return ValidationResult::valid(
                QString("Large grid will require approximately %.1f GB GPU memory. "
                        "Ensure your GPU has sufficient memory.").arg(memoryGB)
            );
        } else if (memoryGB > 4.0) {
            return ValidationResult::valid(
                QString("Grid will require approximately %.1f GB GPU memory.").arg(memoryGB)
            );
        }
        
        return ValidationResult::valid();
    }
    
    QString getTooltip() const override {
        return QString("Grid dimensions for FFT calculation\n"
                      "Range: %1 to %2 per dimension\n"
                      "Powers of 2 recommended for performance\n"
                      "Format: single number (cubic) or 'nx ny nz'")
            .arg(m_minSize).arg(m_maxSize);
    }
};

/**
 * @brief Frame range validator
 */
class FrameRangeValidator : public IValidator {
private:
    std::function<int()> m_getStartFrame;
    int m_maxFrames;
    
public:
    FrameRangeValidator(std::function<int()> getStartFrame, int maxFrames = 1000000)
        : m_getStartFrame(getStartFrame), m_maxFrames(maxFrames) {}
    
    ValidationResult validate(const QString& value) const override {
        bool ok;
        int endFrame = value.toInt(&ok);
        
        if (!ok) {
            return ValidationResult::invalid("Invalid frame number");
        }
        
        if (endFrame < 0) {
            return ValidationResult::invalid("Frame number cannot be negative");
        }
        
        if (m_getStartFrame) {
            int startFrame = m_getStartFrame();
            if (endFrame <= startFrame) {
                return ValidationResult::invalid(
                    QString("End frame (%1) must be greater than start frame (%2)")
                    .arg(endFrame).arg(startFrame)
                );
            }
            
            int frameCount = endFrame - startFrame + 1;
            if (frameCount > 10000) {
                return ValidationResult::valid(
                    QString("Processing %1 frames may take significant time. "
                            "Consider using frame interval (dt) to reduce computation.")
                    .arg(frameCount)
                );
            }
        }
        
        return ValidationResult::valid();
    }
    
    QString getTooltip() const override {
        return "Last frame to process (must be > start frame)";
    }
};

/**
 * @brief Output file validator
 */
class OutputFileValidator : public IValidator {
public:
    ValidationResult validate(const QString& value) const override {
        if (value.isEmpty()) {
            return ValidationResult::invalid("Output file path cannot be empty");
        }
        
        QFileInfo fileInfo(value);
        QFileInfo dirInfo(fileInfo.absolutePath());
        
        if (!dirInfo.exists()) {
            return ValidationResult::invalid("Output directory does not exist: " + dirInfo.absolutePath());
        }
        
        if (!dirInfo.isWritable()) {
            return ValidationResult::invalid("Output directory is not writable: " + dirInfo.absolutePath());
        }
        
        // Check if file exists and warn about overwriting
        if (fileInfo.exists()) {
            return ValidationResult::valid("File exists and will be overwritten");
        }
        
        return ValidationResult::valid();
    }
    
    QString getTooltip() const override {
        return "Output file path (.dat extension recommended)";
    }
};

/**
 * @brief Widget validator that connects to Qt widgets
 */
class WidgetValidator : public QObject {
    Q_OBJECT
    
private:
    QWidget* m_widget;
    QLabel* m_errorLabel;
    std::unique_ptr<IValidator> m_validator;
    QTimer* m_validationTimer;
    QString m_lastValue;
    ValidationResult m_lastResult;
    
public:
    explicit WidgetValidator(QWidget* widget, std::unique_ptr<IValidator> validator, 
                           QLabel* errorLabel = nullptr, QObject* parent = nullptr)
        : QObject(parent), m_widget(widget), m_errorLabel(errorLabel), 
          m_validator(std::move(validator)), m_validationTimer(new QTimer(this)) {
        
        m_validationTimer->setSingleShot(true);
        m_validationTimer->setInterval(500); // 500ms delay for real-time validation
        connect(m_validationTimer, &QTimer::timeout, this, &WidgetValidator::performValidation);
        
        connectWidgetSignals();
        
        if (m_validator && !m_validator->getTooltip().isEmpty()) {
            m_widget->setToolTip(m_validator->getTooltip());
        }
    }
    
    ValidationResult validate() {
        QString value = getWidgetValue();
        ValidationResult result = m_validator->validate(value);
        updateUI(result);
        m_lastResult = result;
        return result;
    }
    
    bool isValid() const { return m_lastResult.isValid; }
    QString getErrorMessage() const { return m_lastResult.errorMessage; }
    QString getWarningMessage() const { return m_lastResult.warningMessage; }

signals:
    void validationChanged(bool isValid);
    void validationResult(const ValidationResult& result);

private slots:
    void onValueChanged() {
        QString currentValue = getWidgetValue();
        if (currentValue != m_lastValue) {
            m_lastValue = currentValue;
            m_validationTimer->start(); // Restart timer for debounced validation
        }
    }
    
    void performValidation() {
        ValidationResult result = validate();
        emit validationChanged(result.isValid);
        emit validationResult(result);
    }

private:
    void connectWidgetSignals() {
        if (auto lineEdit = qobject_cast<QLineEdit*>(m_widget)) {
            connect(lineEdit, &QLineEdit::textChanged, this, &WidgetValidator::onValueChanged);
        } else if (auto spinBox = qobject_cast<QSpinBox*>(m_widget)) {
            connect(spinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
                   this, &WidgetValidator::onValueChanged);
        } else if (auto doubleSpinBox = qobject_cast<QDoubleSpinBox*>(m_widget)) {
            connect(doubleSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                   this, &WidgetValidator::onValueChanged);
        } else if (auto comboBox = qobject_cast<QComboBox*>(m_widget)) {
            connect(comboBox, &QComboBox::currentTextChanged, this, &WidgetValidator::onValueChanged);
        }
    }
    
    QString getWidgetValue() const {
        if (auto lineEdit = qobject_cast<QLineEdit*>(m_widget)) {
            return lineEdit->text();
        } else if (auto spinBox = qobject_cast<QSpinBox*>(m_widget)) {
            return QString::number(spinBox->value());
        } else if (auto doubleSpinBox = qobject_cast<QDoubleSpinBox*>(m_widget)) {
            return QString::number(doubleSpinBox->value());
        } else if (auto comboBox = qobject_cast<QComboBox*>(m_widget)) {
            return comboBox->currentText();
        }
        return "";
    }
    
    void updateUI(const ValidationResult& result) {
        if (m_errorLabel) {
            if (!result.isValid) {
                m_errorLabel->setText("❌ " + result.errorMessage);
                m_errorLabel->setStyleSheet("color: red; font-weight: bold;");
                m_errorLabel->setVisible(true);
            } else if (!result.warningMessage.isEmpty()) {
                m_errorLabel->setText("⚠️ " + result.warningMessage);
                m_errorLabel->setStyleSheet("color: orange;");
                m_errorLabel->setVisible(true);
            } else {
                m_errorLabel->setText("✅ Valid");
                m_errorLabel->setStyleSheet("color: green;");
                m_errorLabel->setVisible(true);
            }
        }
        
        // Update widget styling
        QString styleSheet;
        if (!result.isValid) {
            styleSheet = "border: 2px solid red; background-color: #ffe6e6;";
        } else if (!result.warningMessage.isEmpty()) {
            styleSheet = "border: 2px solid orange; background-color: #fff3e6;";
        } else {
            styleSheet = "border: 2px solid green; background-color: #e6ffe6;";
        }
        
        m_widget->setStyleSheet(styleSheet);
    }
};


#endif // INPUTVALIDATOR_H