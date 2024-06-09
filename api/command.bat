@echo off
rem Activate the virtual environment
 "E:\00_git_thesis\cv_process\.venv\Scripts"
call activate

rem Define the path to the Python executable within the virtual environment
set PYTHON_EXEC=E:\00_git_thesis\cv_process\.venv\Scripts\python.exe

rem Define the path to the Python script
set PYTHON_SCRIPT=E:\00_git_thesis\cv_process\py_classes\ImageEngine.py

rem Define the arguments
set FOLDER_ARG=--folder
set FOLDER_PATH=C:\Users\plame\Documents\__camera__
set OCR_ARG=--ocr
set OCR_OPTION=easyocr

rem Run the Python script with arguments
%PYTHON_EXEC% %PYTHON_SCRIPT% %FOLDER_ARG% %FOLDER_PATH% %OCR_ARG% %OCR_OPTION%