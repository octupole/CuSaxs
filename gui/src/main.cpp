#include <QApplication>
#include "MainWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Set application info for QSettings
    app.setOrganizationName("cudaSAXS");
    app.setApplicationName("cudaSAXS GUI");

    // Create and show main window
    MainWindow window;
    window.show();

    return app.exec();
}