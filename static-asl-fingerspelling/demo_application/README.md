# American sign language fingerspelling demo

The demo uses static images to predict asl letters.

## How to use
The code was tested on windows with python [3.9.13](https://www.python.org/downloads/release/python-3913/) installed

To run the program install dependencies from `requirements.txt`
(For apple computers running on M1 chipset, mediapipe 0.9.0.1 will not be available, choose the closest edition possible!)
I recommend using a clean virtual environment, but its not mandatory.

Windows:
```
python -m venv .venv
.\.venv\Scripts\activate.bat
```

Mac:
```
python3 -m venv my_env
source my_env/bin/activate
```

Install the dependencies and run the code:
```
pip install -r requirements.txt
python main.py
```

By default the program uses the default webcam, but sou can specify a source:
```
python main.py -s "path/to/file"
```

## Details

The demo detects letters a-z in the English alphabet, except fot j and z, as these letters require motion, and the model used in the demo operates on still frames.

There are 2 modes available:

1. Continuous mode, where each frame the detection result is displayed.
2. Connect words mode, which allows you to sign complete words.

By default the program starts in continuous mode, but you can change that with:

```
python main.py -cw
```
