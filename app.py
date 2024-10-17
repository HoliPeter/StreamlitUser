import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QMessageBox


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口的标题和大小
        self.setWindowTitle("智能钢板堆垛系统")
        self.setGeometry(100, 100, 800, 600)

        # 创建界面布局
        layout = QVBoxLayout()

        # 添加标签
        self.label = QLabel("欢迎使用智能钢板堆垛系统", self)
        layout.addWidget(self.label)

        # 添加按钮
        self.upload_button = QPushButton("上传数据集", self)
        self.upload_button.clicked.connect(self.upload_dataset)
        layout.addWidget(self.upload_button)

        self.run_button = QPushButton("开始优化", self)
        self.run_button.clicked.connect(self.run_optimization)
        layout.addWidget(self.run_button)

        # 设置窗口布局
        self.setLayout(layout)

    def upload_dataset(self):
        # 文件上传对话框
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择数据集", "", "CSV Files (*.csv)", options=options)
        if file_name:
            self.label.setText(f"已选择数据集：{file_name}")

    def run_optimization(self):
        # 开始优化的处理逻辑
        QMessageBox.information(self, "提示", "优化开始...")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
