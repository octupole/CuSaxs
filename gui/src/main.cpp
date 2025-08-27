#include <QApplication>
#include "MainWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Set application info for QSettings
    app.setOrganizationName("CuSAXS");
    app.setApplicationName("CuSAXS GUI");

    // Create and show main window
    MainWindow window;
    window.show();

    return app.exec();
}