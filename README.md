# Event Classification
Classify events into the four categories: "non-event", "change-of-state", "process" and "stative-event".
The spans of events are inferred using a rule-based system on top of a dependency parser.

Three entry points to perform different tasks exist:
* `main.py`, perform classification training on gold-span data from a CATMA project
    * e.g: `python main.py batch_size=16`
* `preprocess.py` perform event segmentation, saving a JSON file of event spans suitable for inference using the predict script
    * e.g. `python preprocess.py text_1.txt text_2.txt all_texts.json`
* `predict.py`, perform classification inference on an existing dataset JSON file using a pretrained model
    * e.g.: `python predict.py all_texts.json all_texts_classified.json path_to_model`

To run this project you will need to install all dependencies in `requirements.txt`, additionally you will need to install PyTorch.

## Setup

Initialize content of CATMA submodule: `git submodule update --init --recursive`

(Optionally) create and activate a virtual environment to not install the dependencies globally:
```
python -m virtualenv venv
source venv/bin/activate
```

Install all depenencies:
```
pip -r requirements.txt
```

If your machine does not have a cuda device you will first have to comment out the line containing "cupy" in requirements.txt

## Usage

### Inference
You can easily process a single text file:
```
python predict.py plain-text-file <model_path> <input_txt_file> <output_json_file>
```

If your system does not have a cuda device pass `--device=cpu` as the script currently does not properly recognize this by itself.

The JSON data will contain information besides the event types, these predictions are however not of good quality and should not be used for any purposes.


### Training Configuration

The training script `main.py` can be configured via `conf/config.yaml`,
individual parameters can be overridden using command line parameters like this: `python main.py label_smoothing=false`.

Model weights and logs are saved to `outputs/<date>/<time>`, tensorboard logs are created in `runs`.
Start the tensorboard like this: `tensorboard --logdir runs`.
