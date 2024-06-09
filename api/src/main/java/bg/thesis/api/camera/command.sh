#!/bin/sh
cd /e/00_git_thesis/
. /e/00_git_thesis/cv_process/.venv/Scripts/activate

# Define the path to the Python executable within the virtual environment
PYTHON_EXEC="/e/00_git_thesis/cv_process/.venv/Scripts/python.exe"

# Define the path to the Python script
PYTHON_SCRIPT="/e/00_git_thesis/cv_process/py_classes/ImageEngine.py"

# Define the arguments
FOLDER_ARG="--folder"
FOLDER_PATH="C:/Users/plame/Documents/__camera__"
OCR_ARG="--ocr"
OCR_OPTION="easyocr"

# Run the Python script with arguments
$PYTHON_EXEC $PYTHON_SCRIPT $FOLDER_ARG $FOLDER_PATH $OCR_ARG $OCR_OPTION

read -p "Press Enter to continue"