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

## Training Configuration

The training script `main.py` can be configured via `conf/config.yaml`,
individual parameters can be overridden using command line parameters like this: `python main.py label_smoothing=false`.

Model weights and logs are saved to `outputs/<date>/<time>`, tensorboard logs are created in `runs`.
Start the tensorboard like this: `tensorboard --logdir runs`.
