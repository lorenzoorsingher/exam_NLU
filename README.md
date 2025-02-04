# NLU Course Project

This project is part of the Natural Language Understanding course at the University of Trento. The goal is to build different systems leveraging architectures like RNNs, LSTMs and Transformers to perform intent classification, slot filling and sentiment analysis tasks.

## Project Structure

```bash

├── LM
│   ├── part_1
│   │   ├── bin
│   │   │   └── ...
│   │   ├── dataset
│   │   │   └── ...
│   │   ├── functions.py
│   │   ├── main.py
│   │   ├── model.py
│   │   └── utils.py
│   ├── part_2
│   │   └── ...
│   └── README.md
├── NLU
│   └── ...
├── SA
│   └── ...
├── LABS_README.md
├── nlu_env.yaml
├── README.md
└── requirements.txt
```

## Getting started (from LABS_README.md)

We suggest you install [Anaconda](https://www.anaconda.com/download) on your machine and import the conda environment that we have prepared for you. The reason for this is to give you the same library versions that we used to test the labs. However, if you are not a conda lover, you can manually install on your favourite virtual env the libraries listed in the `requirements.txt` file.

```bash
conda env create -f nlu_env.yaml -n nlu24
conda activate nlu24
```

## How to run the code

### Testing

For each part of the project, you can find a `main.py` file in the corresponding directory. You can run the code by executing the following command:

```bash
python main.py
```

The script will automatically execute the testing phase and print the results, the test executed are reported in each `main.py` file. Make sure the checkpoint files are present in the `bin` directory, otherwise the test will be skipped.

### Training

In order to train the model, you can add the flag `--train` to the command. The training will be executed for the experiments defined in the `main.py` file.

```bash
python main.py --train
```

It's possible to define the experiments in a `json` format by adding the flag `-j` followed by the path to the file. Some examples are aviable in the various `assignment_*` directories.

```bash
python main.py --train -j assignment_1/test.json
```

It's also possible to log on `wandb` by adding the flag `-L true`, loggin is disabled by default. A `wandb` key must be provided in a `.env` file in the root directory:

```bash
WANDB_SECRET="{YOUR_KEY}"
```

```bash
python main.py --train -L true
```

### Authors

- [Lorenzo Orsingher](https://github.com/lorenzoorsingher)
- [GitHub repo](https://github.com/lorenzoorsingher/exam_NLU)
