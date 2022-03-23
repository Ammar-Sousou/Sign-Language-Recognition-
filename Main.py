import numpy as np
import cv2
import sys
import typing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as ks
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap
from gtts import gTTS 
from gingerit.gingerit import GingerIt
from urllib import request
from playsound import playsound


labels = {}
model = None
latest_result = None
latest_result_acc = None
accuracy_threshold = 0.4
was_nothing = True
default_interval = "3"
RES = 224
SIZE = (RES, RES)
UPPER_LEFT = (376, 82)
BOTTOM_RIGHT = (600, 306)
NUMBERS_MODEL_PATH = r"G:\Users\new LAPTOP\Desktop\Desktop\eng\5th\GP\models\NumberModel.h5"
NUMBERS_LABELS = {
	0: "0",
	1: "1",
	2: "2",
	3: "3",
	4: "4",
	5: "5",
	6: "6",
	7: "7",
	8: "8",
	9: "9"
}
LETTERS_MODEL_PATH = r"G:\Users\new LAPTOP\Desktop\Desktop\eng\5th\GP\models\ASL_Model.h5"
LETTERS_LABELS = {
	0: "A",
	1: "B",
	2: "C",
	3: "D",
	4: "E",
	5: "F",
	6: "G",
	7: "H",
	8: "I",
	9: "J",
	10: "K",
	11: "L",
	12: "M",
	13: "N",
	14: "O",
	15: "P",
	16: "Q",
	17: "R",
	18: "S",
	19: "T",
	20: "U",
	21: "V",
	22: "W",
	23: "X",
	24: "Y",
	25: "Z",
	26: "Delete",
	27: "Nothing",
	28: "Space"
}


class Utilities:
	def __new__(cls) -> None:
		raise RuntimeError('%s should not be instantiated' % cls)

	@staticmethod
	def get_prediction(image_array: np.ndarray) -> typing.Tuple[float, str]:
		# Normalize and use the model to predict the sign
		normalized_image_array = Utilities.normalize_input(image_array)
		prediction = model.predict(normalized_image_array)

		# Get the accuracy of the detected sign
		accuracy = prediction[0][np.argmax(prediction[0])]
		prediction = labels.get(np.argmax(prediction[0]))
		return accuracy, prediction

	@staticmethod
	def normalize_input(image: np.ndarray) -> np.ndarray:
		image = cv2.resize(image, SIZE)
		image_array = image.reshape(1, RES, RES, 3) / 255.0
		return image_array

	@staticmethod
	def load_model(isLettersModel: bool) -> None:
		model_path = LETTERS_MODEL_PATH if isLettersModel else NUMBERS_MODEL_PATH
		global model
		model = ks.models.load_model(model_path, compile=False)
		global labels
		labels = LETTERS_LABELS if isLettersModel else NUMBERS_LABELS

	@staticmethod
	def convert_cv_qt(cv_img: np.ndarray, is_cam: bool) -> QPixmap:
		# Convert from an opencv image to QPixmap
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		convert_to_Qt_format = QtGui.QImage(
			rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

		wid = 480 if is_cam else 224
		hei = 360 if is_cam else 224

		p = convert_to_Qt_format.scaled(
			wid, hei, Qt.KeepAspectRatio)
		return QPixmap.fromImage(p)

	@staticmethod
	def test_connection():
		try:
			request.urlopen('http://google.com')
			return True
		except:
			return False


class Ui_MainWindow(QMainWindow):
	def __init__(self) -> None:
		super(Ui_MainWindow, self).__init__()
		self.setupUi(self)
		self.connect_signals()
		self.load_model()

	def setupUi(self, MainWindow) -> None:
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1250, 665)
		MainWindow.setMinimumSize(QtCore.QSize(700, 300))
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
		self.gridLayout_2.setObjectName("gridLayout_2")
		self.line = QtWidgets.QFrame(self.centralwidget)
		self.line.setFrameShape(QtWidgets.QFrame.VLine)
		self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
		self.line.setObjectName("line")
		self.gridLayout_2.addWidget(self.line, 0, 1, 1, 1)
		self.gridLayout_7 = QtWidgets.QGridLayout()
		self.gridLayout_7.setContentsMargins(10, 10, 10, 10)
		self.gridLayout_7.setObjectName("gridLayout_7")
		self.capture_char_button = QtWidgets.QPushButton(self.centralwidget)
		self.capture_char_button.setEnabled(False)
		self.capture_char_button.setObjectName("capture_char_button")
		self.gridLayout_7.addWidget(self.capture_char_button, 5, 0, 1, 1)
		self.gridLayout_8 = QtWidgets.QGridLayout()
		self.gridLayout_8.setObjectName("gridLayout_8")
		self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
		self.horizontalLayout_4.setContentsMargins(10, 10, 10, 10)
		self.horizontalLayout_4.setSpacing(10)
		self.horizontalLayout_4.setObjectName("horizontalLayout_4")
		self.predict_mode_groupBox = QtWidgets.QGroupBox(self.centralwidget)
		self.predict_mode_groupBox.setObjectName("predict_mode_groupBox")
		self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.predict_mode_groupBox)
		self.verticalLayout_4.setContentsMargins(10, 5, -1, 20)
		self.verticalLayout_4.setSpacing(2)
		self.verticalLayout_4.setObjectName("verticalLayout_4")
		self.predict_numbers_radioButton = QtWidgets.QRadioButton(self.predict_mode_groupBox)
		self.predict_numbers_radioButton.setObjectName("predict_numbers_radioButton")
		self.verticalLayout_4.addWidget(self.predict_numbers_radioButton)
		self.predict_letters_radioButton = QtWidgets.QRadioButton(self.predict_mode_groupBox)
		self.predict_letters_radioButton.setChecked(True)
		self.predict_letters_radioButton.setObjectName("predict_letters_radioButton")
		self.verticalLayout_4.addWidget(self.predict_letters_radioButton)
		self.horizontalLayout_4.addWidget(self.predict_mode_groupBox)
		self.verticalLayout_3 = QtWidgets.QVBoxLayout()
		self.verticalLayout_3.setContentsMargins(20, 20, 20, 20)
		self.verticalLayout_3.setObjectName("verticalLayout_3")
		self.start_button = QtWidgets.QPushButton(self.centralwidget)
		self.start_button.setObjectName("start_button")
		self.verticalLayout_3.addWidget(self.start_button)
		self.stop_button = QtWidgets.QPushButton(self.centralwidget)
		self.stop_button.setEnabled(False)
		self.stop_button.setObjectName("stop_button")
		self.verticalLayout_3.addWidget(self.stop_button)
		self.horizontalLayout_4.addLayout(self.verticalLayout_3)
		self.gridLayout_8.addLayout(self.horizontalLayout_4, 0, 1, 1, 1)
		self.gridLayout_7.addLayout(self.gridLayout_8, 0, 0, 1, 1)
		self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
		self.horizontalLayout_2.setObjectName("horizontalLayout_2")
		self.accent_comboBox = QtWidgets.QComboBox(self.centralwidget)
		self.accent_comboBox.setObjectName("accent_comboBox")
		self.accent_comboBox.addItem("English (United States)")
		self.accent_comboBox.addItem("English (Australia)")
		self.accent_comboBox.addItem("English (United Kingdom)")
		self.accent_comboBox.addItem("English (Canada)")
		self.accent_comboBox.addItem("English (India)")
		self.accent_comboBox.addItem("English (Ireland)")
		self.accent_comboBox.addItem("English (South Africa)")
		self.horizontalLayout_2.addWidget(self.accent_comboBox)
		self.correct_text_button = QtWidgets.QPushButton(self.centralwidget)
		self.correct_text_button.setEnabled(False)
		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("speaker_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		self.correct_text_button.setIcon(icon)
		self.correct_text_button.setIconSize(QtCore.QSize(15, 15))
		self.correct_text_button.setObjectName("correct_text_button")
		self.horizontalLayout_2.addWidget(self.correct_text_button)
		self.gridLayout_7.addLayout(self.horizontalLayout_2, 8, 0, 1, 1)
		self.predicted_text_plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
		self.predicted_text_plainTextEdit.setReadOnly(True)
		self.predicted_text_plainTextEdit.setObjectName("predicted_text_plainTextEdit")
		self.gridLayout_7.addWidget(self.predicted_text_plainTextEdit, 2, 0, 1, 1)
		self.horizontalLayout = QtWidgets.QHBoxLayout()
		self.horizontalLayout.setObjectName("horizontalLayout")
		self.add_captured_after_label = QtWidgets.QLabel(self.centralwidget)
		self.add_captured_after_label.setObjectName("add_captured_after_label")
		self.horizontalLayout.addWidget(self.add_captured_after_label)
		self.seconds_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
		self.onlyInt = QtGui.QIntValidator()
		self.seconds_lineEdit.setObjectName("seconds_lineEdit")
		self.seconds_lineEdit.setValidator(self.onlyInt)
		self.seconds_lineEdit.setText(default_interval)
		self.horizontalLayout.addWidget(self.seconds_lineEdit)
		self.timer = QtCore.QTimer()
		self.seconds_label = QtWidgets.QLabel(self.centralwidget)
		self.seconds_label.setObjectName("seconds_label")
		self.horizontalLayout.addWidget(self.seconds_label)
		spacerItem = QtWidgets.QSpacerItem(
			10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem)
		self.start_capture_button = QtWidgets.QPushButton(self.centralwidget)
		self.start_capture_button.setEnabled(False)
		self.start_capture_button.setObjectName("start_capture_button")
		self.horizontalLayout.addWidget(self.start_capture_button)
		self.stop_capture_button = QtWidgets.QPushButton(self.centralwidget)
		self.stop_capture_button.setEnabled(False)
		self.stop_capture_button.setObjectName("stop_capture_button")
		self.horizontalLayout.addWidget(self.stop_capture_button)
		spacerItem1 = QtWidgets.QSpacerItem(
			60, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem1)
		self.clear_text_button = QtWidgets.QPushButton(self.centralwidget)
		self.clear_text_button.setEnabled(False)
		self.clear_text_button.setObjectName("clear_text_button")
		self.horizontalLayout.addWidget(self.clear_text_button)
		self.gridLayout_7.addLayout(self.horizontalLayout, 4, 0, 1, 1)
		self.predicted_text_label = QtWidgets.QLabel(self.centralwidget)
		font = QtGui.QFont()
		font.setPointSize(10)
		self.predicted_text_label.setFont(font)
		self.predicted_text_label.setObjectName("predicted_text_label")
		self.gridLayout_7.addWidget(self.predicted_text_label, 1, 0, 1, 1)
		self.gridLayout_2.addLayout(self.gridLayout_7, 0, 2, 1, 1)
		self.gridLayout_3 = QtWidgets.QGridLayout()
		self.gridLayout_3.setObjectName("gridLayout_3")
		self.gridLayout_6 = QtWidgets.QGridLayout()
		self.gridLayout_6.setContentsMargins(5, 5, 5, 5)
		self.gridLayout_6.setHorizontalSpacing(20)
		self.gridLayout_6.setObjectName("gridLayout_6")
		self.roi_image_label = QtWidgets.QLabel(self.centralwidget)
		self.roi_image_label.setAutoFillBackground(False)
		self.roi_image_label.setStyleSheet("border-style: outset; border-width: 2px; border-radius: 5px; black;")
		self.roi_image_label.setText("Region of interest")
		self.roi_image_label.setAlignment(QtCore.Qt.AlignCenter)
		self.roi_image_label.setObjectName("roi_image_label")
		self.gridLayout_6.addWidget(self.roi_image_label, 0, 0, 1, 1)
		self.predicted_char_label = QtWidgets.QLabel(self.centralwidget)
		self.predicted_char_label.setAutoFillBackground(False)
		self.predicted_char_label.setStyleSheet("border-style: outset; border-width: 2px; border-radius: 5px; black;")
		self.predicted_char_label.setText("Predicted character")
		self.predicted_char_label.setAlignment(QtCore.Qt.AlignCenter)
		self.predicted_char_label.setObjectName("predicted_char_label")
		self.gridLayout_6.addWidget(self.predicted_char_label, 0, 1, 1, 1)
		self.gridLayout_3.addLayout(self.gridLayout_6, 2, 0, 1, 1)
		self.gridLayout_5 = QtWidgets.QGridLayout()
		self.gridLayout_5.setObjectName("gridLayout_5")
		self.camera_feed_label = QtWidgets.QLabel(self.centralwidget)
		self.camera_feed_label.setAutoFillBackground(False)
		self.camera_feed_label.setStyleSheet("border-style: outset; border-width: 2px; border-radius: 5px; black;")
		self.camera_feed_label.setText("Camera feed")
		self.camera_feed_label.setAlignment(QtCore.Qt.AlignCenter)
		self.camera_feed_label.setObjectName("camera_feed_label")
		self.gridLayout_5.addWidget(self.camera_feed_label, 0, 0, 1, 1)
		self.gridLayout_3.addLayout(self.gridLayout_5, 0, 0, 1, 1)
		self.line_2 = QtWidgets.QFrame(self.centralwidget)
		self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
		self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
		self.line_2.setObjectName("line_2")
		self.gridLayout_3.addWidget(self.line_2, 1, 0, 1, 1)
		self.gridLayout_3.setRowStretch(0, 5)
		self.gridLayout_3.setRowStretch(1, 1)
		self.gridLayout_3.setRowStretch(2, 3)
		self.gridLayout_2.addLayout(self.gridLayout_3, 0, 0, 1, 1)
		self.gridLayout_2.setColumnStretch(0, 2)
		self.gridLayout_2.setColumnStretch(1, 1)
		self.gridLayout_2.setColumnStretch(2, 3)
		MainWindow.setCentralWidget(self.centralwidget)
		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 1250, 26))
		self.menubar.setObjectName("menubar")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	def retranslateUi(self, MainWindow: QMainWindow) -> None:
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "SignToText"))
		self.predicted_text_label.setText(_translate("MainWindow", "Predicted text:"))
		self.capture_char_button.setText(_translate("MainWindow", "Add captured character"))
		self.predict_mode_groupBox.setTitle(_translate("MainWindow", "Predict:"))
		self.predict_numbers_radioButton.setText(_translate("MainWindow", "Numbers"))
		self.predict_letters_radioButton.setText(_translate("MainWindow", "Letters"))
		self.start_button.setText(_translate("MainWindow", "Start"))
		self.stop_button.setText(_translate("MainWindow", "Stop"))
		self.correct_text_button.setText(_translate("MainWindow", " Correct and read text"))
		self.add_captured_after_label.setText(_translate("MainWindow", "Add captured char each"))
		self.seconds_label.setText(_translate("MainWindow", "second(s)"))
		self.start_capture_button.setText(_translate("MainWindow", "Start"))
		self.stop_capture_button.setText(_translate("MainWindow", "Stop"))
		self.clear_text_button.setText(_translate("MainWindow", "Clear text"))
		self.accent_comboBox.setItemText(
			0, _translate("MainWindow", "English (United States)"))
		self.accent_comboBox.setItemText(
			1, _translate("MainWindow", "English (Australia)"))
		self.accent_comboBox.setItemText(
			2, _translate("MainWindow", "English (United Kingdom)"))
		self.accent_comboBox.setItemText(
			3, _translate("MainWindow", "English (Canada)"))
		self.accent_comboBox.setItemText(
			4, _translate("MainWindow", "English (India)"))
		self.accent_comboBox.setItemText(
			5, _translate("MainWindow", "English (Ireland)"))
		self.accent_comboBox.setItemText(
			6, _translate("MainWindow", "English (South Africa)"))

	def connect_signals(self) -> None:
		self.start_button.clicked.connect(self.start_button_click)
		self.stop_button.clicked.connect(self.stop_button_click)
		self.capture_char_button.clicked.connect(self.capture_char_button_click)
		self.predict_letters_radioButton.toggled.connect(self.load_model)
		self.correct_text_button.clicked.connect(self.correct_and_read_text)
		self.predicted_text_plainTextEdit.textChanged.connect(self.predicted_text_changed)
		self.timer.timeout.connect(self.capture_char_button_click)
		self.start_capture_button.clicked.connect(self.start_capture_button_click)
		self.stop_capture_button.clicked.connect(self.stop_capture_button_click)
		self.seconds_lineEdit.textChanged.connect(self.seconds_lineEdit_text_changed)
		self.clear_text_button.clicked.connect(self.clear_text_button_clicked)

	def start_button_click(self) -> None:
		if not self.seconds_lineEdit.text():
			self.seconds_lineEdit.setText(default_interval)
		self.predict_mode_groupBox.setDisabled(True)
		self.start_button.setDisabled(True)
		self.camera_feed_label.setText("Loading camera feed...")
		self.video_thread = VideoThread()
		self.video_thread.change_pixmap_signal.connect(self.update_labels)
		self.video_thread.camera_toggled_signal.connect(self.update_state)
		self.video_thread.start()

	def stop_button_click(self) -> None:
		self.start_button.setEnabled(True)
		if hasattr(self, 'video_thread'):
			self.video_thread.stop()
			delattr(self, 'video_thread')
		self.timer.stop()
		self.capture_char_button.setDisabled(True)
		self.stop_button.setDisabled(True)

	def capture_char_button_click(self) -> None:
		if model is not None and latest_result is not None and latest_result != "Nothing":
			if latest_result == "Space":
				self.predicted_text_plainTextEdit.insertPlainText(" ")
			elif latest_result == "Delete":
				old = self.predicted_text_plainTextEdit.toPlainText()
				self.predicted_text_plainTextEdit.setPlainText(old[:-1])
			else:
				if self.predicted_text_plainTextEdit.toPlainText():
					self.predicted_text_plainTextEdit.insertPlainText(latest_result.lower())
				else:
					self.predicted_text_plainTextEdit.insertPlainText(latest_result)

	def correct_and_read_text(self) -> None:
		current_accent = self.predicted_text_plainTextEdit.toPlainText()
		if current_accent:
			accent = "com"
			if current_accent == "English (Australia)":
				accent = "com.au"
			elif current_accent == "English (United Kingdom)":
				accent = "co.uk"
			elif current_accent == "English (Canada)":
				accent = "ca"
			elif current_accent == "English (India)":
				accent = "co.in"
			elif current_accent == "English (Ireland)":
				accent = "ie"
			elif current_accent == "English (South Africa)":
				accent = "co.za"

			corrected = GingerIt().parse(self.predicted_text_plainTextEdit.toPlainText())['result']
			self.predicted_text_plainTextEdit.setPlainText(corrected)
			myobj = gTTS(text=GingerIt().parse(corrected)[
		        	'result'], tld=accent, lang='en', slow=True)
			myobj.save("text.mp3") 
			playsound("text.mp3")
			os.remove("text.mp3")

	def predicted_text_changed(self) -> None:
		if self.predicted_text_plainTextEdit.toPlainText():
			if Utilities.test_connection:
				self.correct_text_button.setEnabled(True)
			self.clear_text_button.setEnabled(True)
		else:
			self.correct_text_button.setDisabled(True)
			self.clear_text_button.setDisabled(True)

	def get_timer_seconds(self) -> int:
		return int(self.seconds_lineEdit.text()) if self.seconds_lineEdit.text() else 1

	def start_capture_button_click(self) -> None:
		self.timer.start(self.get_timer_seconds() * 1000)
		self.start_capture_button.setDisabled(True)
		self.stop_capture_button.setEnabled(True)

	def stop_capture_button_click(self) -> None:
		self.timer.stop()
		self.start_capture_button.setEnabled(True)
		self.stop_capture_button.setDisabled(True)

	def seconds_lineEdit_text_changed(self) -> None:
		self.timer.stop()
		self.timer.start(self.get_timer_seconds() * 1000)

	def clear_text_button_clicked(self) -> None:
		self.predicted_text_plainTextEdit.clear()

	def load_model(self) -> None:
		if self.is_video_thread_running():
			self.predicted_char_label.setStyleSheet(
                    "font-size: 12px; border-style: outset; border-width: 2px; border-radius: 5px; black;")
			self.predicted_char_label.setText("Loading model...")
			Utilities.load_model(self.predict_letters_radioButton.isChecked())
			self.predicted_char_label.setStyleSheet(
				"font-size: 18px; border-style: outset; border-width: 2px; border-radius: 5px; black;")
		else:
			self.predicted_char_label.setText("Loading model...")
			Utilities.load_model(self.predict_letters_radioButton.isChecked())
			self.predicted_char_label.setText("Predicted character")

	@pyqtSlot(bool)
	def update_state(self, toggled_on: bool) -> None:
		if toggled_on:
			self.stop_button.setEnabled(True)
			self.capture_char_button.setEnabled(True)
			self.predict_mode_groupBox.setEnabled(True)
			self.predicted_char_label.setStyleSheet(
				"font-size: 18px; border-style: outset; border-width: 2px; border-radius: 5px; black;")
			self.start_capture_button.setEnabled(True)
		else:
			global latest_result, latest_result_acc
			latest_result, latest_result_acc = None, None
			self.camera_feed_label.clear()
			self.roi_image_label.clear()
			self.camera_feed_label.setText("Camera feed")
			self.roi_image_label.setText("Region of interest")
			self.predicted_char_label.setText("Predicted character")
			self.predicted_char_label.setStyleSheet(
				"font-size: 12px; border-style: outset; border-width: 2px; border-radius: 5px; black;")
			self.start_capture_button.setDisabled(True)
			self.stop_capture_button.setDisabled(True)

	@pyqtSlot(np.ndarray, np.ndarray)
	def update_labels(self, cv_cam: np.ndarray, cv_roi: np.ndarray) -> None:
		# Updates the camera_feed_label with a new opencv image
		qt_cam_img = Utilities.convert_cv_qt(cv_cam, True)
		qt_roi_img = Utilities.convert_cv_qt(cv_roi, False)
		self.camera_feed_label.setPixmap(qt_cam_img)
		self.roi_image_label.setPixmap(qt_roi_img)
		
		global latest_result, was_nothing
		if latest_result is not None and latest_result_acc is not None:
			if latest_result_acc > accuracy_threshold and latest_result != "Nothing":
				self.predicted_char_label.setText(
					f'{latest_result}, {latest_result_acc:.2%}')
			else:
				latest_result = "Nothing"
				self.predicted_char_label.setText(
					f'{latest_result}')

	def is_video_thread_running(self) -> bool:
		return hasattr(self, 'video_thread') and self.video_thread.isRunning()

	def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
		if event.key() in [QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter] and self.capture_char_button.isEnabled():
			self.capture_char_button_click()
		elif event.key() == QtCore.Qt.Key_Escape:
			self.stop_button_click()
		elif event.key() == QtCore.Qt.Key_F5:
			self.start_button_click()
		else:
			super().keyPressEvent(event)

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:
		if hasattr(self, 'video_thread'):
			self.video_thread.stop()
			delattr(self, 'video_thread')
		return super().closeEvent(event)


class VideoThread(QThread):
	change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray)
	camera_toggled_signal = pyqtSignal(bool)

	def __init__(self) -> None:
		super().__init__()
		self._run_flag = True

	def run(self) -> None:
		# capture from web cam
		cap = cv2.VideoCapture(0)
		self.camera_toggled_signal.emit(True)
		while self._run_flag:
			ret, frame = cap.read()
			if ret:
				frame = cv2.flip(frame, 1)
				cv2.rectangle(frame, UPPER_LEFT, BOTTOM_RIGHT,
							color=(100, 50, 200), thickness=5)
				sliced_frame = frame[82:306, 376:600]

				if model is not None:
					flipped_sliced_frame = cv2.flip(sliced_frame, 1)
					prediction_acc, prediction = Utilities.get_prediction(
						flipped_sliced_frame)
					if prediction is not None:
						global latest_result, latest_result_acc
						latest_result, latest_result_acc = prediction, prediction_acc

				self.change_pixmap_signal.emit(frame, sliced_frame)

		# shut down capturing system
		cap.release()
		self.camera_toggled_signal.emit(False)

	def stop(self) -> None:
		# Sets run flag to False and waits for thread to finish
		self._run_flag = False
		self.wait()


if __name__ == "__main__":
	app = QApplication(sys.argv)
	ui = Ui_MainWindow()
	ui.show()
	sys.exit(app.exec())

