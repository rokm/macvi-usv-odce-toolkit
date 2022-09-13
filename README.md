# USV Obstacle Detection Challenge Evaluation Toolkit

This repository provides source code of the evaluation toolkit for the
*USV Obstacle Detection Challenge*, hosted at the *1st Workshop on Maritime
Computer Vision (MaCVi)* as part of the WACV2023.

The official site for the challenge can be found [here](https://seadronessee.cs.uni-tuebingen.de/wacv23_MODS_od).

The evaluation protocol is based on the paper by *Bovcon et al.*:

Bovcon Borja, Muhovič Jon, Vranac Duško, Mozetič Dean, Perš Janez and Kristan Matej,
*"MODS -- A USV-oriented object detection and obstacle segmentation benchmark"*,
IEEE Transactions on Intelligent Transportation Systems, 2021.
[Pre-print version available on arXiv](https://arxiv.org/abs/2105.02359).


The evaluation code is based on the implementation provided by
the authors in https://github.com/bborja/mods_evaluation
in `object_detection` sub-directory in `bbox_obstacle_detection` branch
([here](https://github.com/bborja/mods_evaluation/tree/bbox_obstacle_detection/object_detection)).


## Getting started

### 1. Download the MODS dataset

Download and unpack [the MODS dataset](https://vision.fe.uni-lj.si/public/mods).

### 2. Process the dataset with your detection method

Use your algorithm to process the whole MODS dataset. The MODS dataset
does not provide training data, and should be used only for evaluation.

For training data, you can use any other dataset that is available to
you, including the [MODD2 dataset](https://box.vicos.si/borja/viamaro/index.html)
and the older [MODD dataset](https://www.vicos.si/resources/modd).

The algorithm should output the detections with rectangular axis-aligned
bounding boxes of waterborne objects belonging to the following semantic
classes: *vessel*, *person*, and *others*. The results should be stored
in a single JSON file using the format described below.

#### Results file format

The results JSON file, expected by the evaluation tool, is very similar
to the `mods.json` file from the MODS dataset, except that each frame
object provides a `detections` array describing detections:

```json
{
  "dataset": {
    "name": "myalgorithm",
    "num_seq": 94,
    "sequences": [
      {
        "id": 0,
        "path": "/kope100-00006790-00007090/frames/",
        "num_frames": 31
        "frames": [
          {
            "id": 0,
            "image_file_name": "00006790L.jpg",
            "detections": []
          },
          <...>
          {
            "id": 30,
            "image_file_name": "00007090L.jpg",
            "detections": []
          }
        ]
      },
      {
        "id": 1,
        "path": "/kope100-00011830-00012500/frames/",
        "num_frames": 68,
        "frames": [
          {
            "id": 0,
            "image_file_name": "00011830L.jpg",
            "detections": [
              {
                "id": 0,
                "type": 2,
                "bbox": [366, 544, 16, 22]
              }
            ]
          },
          {
            "id": 1,
            "image_file_name": "00011840L.jpg",
            "detections": [
              {
                "id": 0,
                "type": 2,
                "bbox": [156, 575, 14, 14]
              },
              {
                "id": 1,
                "type": 2,
                "bbox": [270,555,15,20]
              }
            ]
          },
          <...>
        ],
      },
      <...>
    ]
  }
}
```

The JSON file must contain a root `dataset` object, which must contain
a `sequences` array. Each element is a sequence object that corresponds
to the sequence object from in the dataset (`mods.json`) file. Each
sequence object must contain an `id` field (with a value that matches
the `id` of the sequence in the dataset), and a `detections` array.
If there are no detections in the frame, the `detections` should be
an empty array (or alternatively, it can be omitted altogether).
Otherwise, it should contain one object per detection, each consisting
of an`id` (which must be unique within the image), `type` (denoting the
detection type; see below), and `bbox` (bounding box; `[x, y, width, height]`).

The `type` field denotes the detection's category, and can be either
a string or an integer, with following values being recognized:
 * `"ship"` or `0`
 * `"person"` or `1`
 * `"other"` or `2`

In the above example, we included additional fields to make it easier
to compare the structure to that of the `mods.json` file. The easiest
way to generate the results file is, in fact, taking the data from the
`mods.json` file and adding the `detections` arrays to the sequence
objects.

For reference, we provide exemplar result JSON files for the methods
evaluated in the *Bovcon et al.* paper: MaskRCNN, FCOS, YOLOv4, and SSD:
* [detection-results-original.zip](https://rokm.dynu.net/macvi2023_detection/detection-results-original.zip):
  this archive contains original JSON files, as provided by the authors.
* [detection-results-minimal.zip](https://rokm.dynu.net/macvi2023_detection/detection-results-minimal.zip):
  this archive contains JSON files with minimum content required by the evaluation toolkit.

These reference detection JSON files can also be used in the subsequent
steps to verify that the evaluation toolkit has been properly installed
and is functioning as expected. They also illustrate various options
discussed above (for example, results for SSD omit empty `detections`
array; results for FCOS and SSD use numeric class `type`, while MaskRCNN
and YOLOv4 use string-based class `type`).

### 3. Install the evaluation toolkit

The evaluation toolkit requires a recent version of python3 (>= 3.6)
depends on `pycocotools`, `numpy`, and `opencv-python-headless` (or
a "regular" `opencv-python`).

To prevent potential conflicts with python packages installed in the
global/base python environment, it is recommended to use a separate
python virtual environment using python's `venv` module:

1. Create the virtual environment:

```python3 -m venv venv-usv```

This will create a new virtual environment called `venv-usv` in the
current working directory.

2. Activate the virtual environment:

On Linux and macOS (assuming `bash` shell), run:
```
. venv-usv/bin/activate
```

On Windows, run:
```
venv-usv/Scripts/activate
```

3. Once virtual environment is available, update `pip`, `wheel`, and `setuptools`:
```
python3 -m pip install --upgrade pip wheel setuptools
```

4a. Install the toolkit (recommended approach)

The toolkit can be installed directly from the git repository, using the
following command

```
python3 -m pip install git+https://github.com/rokm/macvi-usv-odce-toolkit.git
```

This will automatically check out the source code from the repository,
and install it into your (virtual) environment. It should also create
an executable called ``macvi-usv-odce-tool`` in your environment's
scripts directory. Running

˙˙˙macvi-usv-odce-tool --help```

should display the help message:

```
usage: macvi_usv_odce_tool [-h] [--version] command ...

MaCVi USV Obstacle Detection Challenge Evaluation Toolkit

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

valid commands:
  command               command description
    evaluate (e)        Evaluate the results.
    prepare-submission (s)
                        Evaluate the results and prepare archive for submission.
    unpack-submission (u)
                        Unpack the submission archive.
```

The tool provides three commands (`evaluate`, `prepare-submission`,
and `unpack-submission`); the help for each can be obtained by adding
`--help` argument *after* the command name:

```
macvi_usv_odce_tool evaluate --help
macvi_usv_odce_tool prepare-submission --help
```

NOTE: Runing the tool via the ``macvi-usv-odce-tool`` requires your
environment's scripts directory to be in `PATH`. This is usually the
case when using virtual environments, but may not be the case if you
are using your base python environment (especially on Windows). If
the system cannot find the ``macvi-usv-odce-tool`` command, try
using

```
python3 -m macvi_usv_odce_toolkit
```

instead. If neither works, the toolkit was either not installed, or
you have forgotten to activate your virtual environment.


4b. Install the toolkit (alternative approach)

Alternatively, you can also check out the source code from the repository,
and run the ``macvi_usv_odce_tool.py`` script to launch the evaluation
tool from within the check-out directory:

```
git clone https://github.com/rokm/macvi-usv-odce-toolkit.git
cd macvi-usv-odce-toolkit
python3 -m pip install --requirement requirements.txt
python3 macvi_usv_odce_tool.py --help
```

### 4. Evaluate the results

While testing your algorithm locally, you can use the `evaluate` command
to perform evaluation and receive immediate feedback. Assuming that your
current working directory contains unpacked MODS dataset in `mods`
sub-directory and the results JSON file called `results.json`,
run:

```
macvi-usv-odce-tool evaluate mods/mods.json results.json
```

This should run the evaluation using all three detection evaluation
setups from the *Bovcon et al.* paper:
* Setup 1: evaluation using sea-edge based mask, taking into account the
  class labels of ground truth and detections.
* Setup 2: evaluation using sea-edge based mask, ignoring the class
  labels (detection without recognition).
* Setup 3: evaluation using danger-zone based mask (the radial area
  with radius 15 meters in front of the USV), ignoring the class
  labels.

```
MaCVi USV Obstacle Detection Challenge Evaluation Toolkit

Settings:
 - mode: 'evaluate'
 - dataset JSON file: 'mods/mods.json'
 - results JSON file: 'results.json'
 - output file: None
 - sequence(s): None

Evaluating Setup 1...
Evaluation complete in 16.37 seconds!
Evaluating Setup 2...
Evaluation complete in 15.39 seconds!
Evaluating Setup 3...
Evaluation complete in 16.37 seconds!

Results: F_all F_small F_medium F_large
Setup_1: 0.122 0.065 0.209 0.260
Setup_2: 0.172 0.090 0.385 0.522
Setup_3: 0.964 0.976 0.958 0.968

Challenge results (F_avg, F_s1, F_s2, F_s3):
0.419 0.122 0.172 0.964

Done!
```

The ranking metric for the challenge is the average of the overall
F-score values obtained in each of the three setups (in the above
example, `0.419 = (0.122 + 0.172 + 0.964) / 3`. In the case of the
tie, the overall F-score from Setup 1 is used as the tie-breaker
(in the above example, `0.122`).


### 5. Prepare submission

Having obtained the results, you can prepare the submission archive.
To do so, use the `macvi-usv-odce-tool` and `prepare-submission`
command. Its behavior is similar to the `evaluate` command, except
that it requires an additional argument - the path to the source
code of the algorithm, which needs to be supplied as part of the
submission.

The tool performs the evaluation, and generates the archive that
contains raw detection results (the results JSON file that was used
for evaluation), the evaluation results, and the collected source code.

If the source code path points to a directory, its while contents are
recursively collected into the submission archive. If the source code
path points to a file (a single-file source, or a pre-generated archive
containing the whole source code), the file is collected into archive
as-is.

To continue the example from the previous step, assuming that your
current working directory contains unpacked MODS dataset in `mods`
sub-directory, the results JSON file called `results.json`, and source
code archive called `source-code.zip`, run:

The output of the tool should look similar to:

and it should generate a file called `submission.zip˙ in the current
working directory.

To use a different name or a different target directory, you can provide
a custom path via the `--output-file <filename>` command-line argument.


### 6. Submit the archive

Once the submission archive is generated, you can submit it on the
challenge's web page.

Once the archive is submitted, the submission server backend will
unpack its contents using the `unpack-submission` command, and
optionally perform re-evaluation of the results using the local copy
of the toolkit and the dataset annotations.
