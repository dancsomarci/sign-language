# American sign language fingerspelling demo

The demo consists of two parts: 
    - Static fingerspelling
    - Sequence to Sequence translation of short video samples

## How to use

Before using any of the demos, one must follow all the steps below.

The code was tested on windows 10 with python [3.9.13](https://www.python.org/downloads/release/python-3913/) installed.

To run the program install dependencies from `requirements.txt`
(For apple computers running arm chipset, mediapipe 0.9.0.1 will not be available, choose the closest edition possible!)
I recommend using a clean virtual environment:

Windows:
```
python -m venv .venv
.\.venv\Scripts\activate.bat
```

Mac:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:
```
pip install -r requirements.txt
```

## Static fingerspelling demo

To run the demo:
```
python static.py
```

To stop the demo press Esc while the window has focus.

By default the program uses the default webcam, but you can specify another source (like *.mp4):
```
python static.py -s "path/to/file"
```

### Details

The demo detects letters in the English alphabet, except for j and z, as these letters require motion, and the model used in the demo operates on still frames.

There are 2 modes available:

1. Continuous mode: where each frame the detection result is displayed.
2. Connect words mode: which allows you to sign complete words.

By default the program starts in continuous mode, but you can change that with:

```
python static.py -cw
```

## Sequence to Sequence translation

To run the demo:
```
python translate.py alligator.mp4
```

You can refer to any video placed inside the `test_videos` folder.

### Details

First the referred video is processed with the model and is translated only at the end. The result will be printed to the console once it is ready. (The "<", ">" characters denote the starting and ending token to avoid confusion with possible whitespace characters.) Along with the result the architecture of the model is also printed.

`Note` that I have tested the code on an intel i7 7700, and the processing of the translated video was real-time.


