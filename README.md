# GUIUVVIS
The UVVIS user interface was implemented using QtDesigner for Python, which is a tool with a graphical environment that through drag-and-drop actions provides the possibility to easily create a GUI. It was implemented for a prototype that allows making inferences using an object detection model known as YOLOv5 and calculating distances with a ZED camera using UVC mode. At the moment it has two versions, the Windows version allows to perform the inference using PyTorch and for the Raspberry Pi 4 TensorFlow Lite. The project is in UVVIS_GUI and to run the program is through the file UVVIS_Callback.py for Windows and UVVIS_CallbackRPI.py, both use the same graphical interface.
