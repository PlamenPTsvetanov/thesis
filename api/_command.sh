#!/bin/sh
cd /d/Thesis/thesis
. /d/Thesis/thesis/cv_process/venv/Scripts/activate

# Define the path to the Python executable within the virtual environment
PYTHON_EXEC="/d/Thesis/thesis/cv_process/venv/Scripts/python.exe"

# Define the path to the Python script
PYTHON_SCRIPT="/d/Thesis/thesis/cv_process/py_classes/ImageEngine.py"

# Define the arguments
FOLDER_ARG="--folder"
FOLDER_PATH=$1
OCR_ARG="--ocr"
OCR_OPTION=$2

# Run the Python script with arguments
start $PYTHON_EXEC $PYTHON_SCRIPT $FOLDER_ARG "$FOLDER_PATH" $OCR_ARG "$OCR_OPTION" PAUSE