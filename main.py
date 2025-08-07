"""DubbingX主程序入口"""

import os
import signal
import sys

from PySide6.QtWidgets import QApplication

from core.gui.main_window import DubbingGUI


def main():
    """主函数"""
    try:
        app = QApplication(sys.argv)

        # 设置应用程序属性
        app.setApplicationName("DubbingX")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("DubbingX Team")

        # 设置应用程序退出时的行为
        app.setQuitOnLastWindowClosed(True)

        # 创建主窗口
        window = DubbingGUI()
        window.show()

        # 设置简单的信号处理
        import signal as sig

        def signal_handler(signum, frame):
            os._exit(0)

        sig.signal(sig.SIGINT, signal_handler)
        if hasattr(sig, "SIGTERM"):
            sig.signal(sig.SIGTERM, signal_handler)

        # 运行应用程序
        app.exec()

    except Exception as e:
        pass

    # 最终确保退出
    os._exit(0)


if __name__ == "__main__":
    main()