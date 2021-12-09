# HealthGen: Realistic EHR Time Series Generation

![Code Coverage](https://img.shields.io/badge/Python-3.7-blue)

This repository contains the implementation of the HealthGen model, a generative
model to synthesize realistic EHR time series data with missingness.

## Installation

1. Clone the repo with: 
```git clone --recurse-submodules git@github.com:simonbing/HealthGen.git```.

2. Navigate to the `/healthgen` directory and install the dependencies by running:
```pip install requirements.txt```.

3. Add the `HealthGen` module to your `PYTHONPATH` by running 
`export PYTHONPATH=$PYTHONPATH:/path/to/HealthGen/healthgen`.

4. Optionally, setup [wandb](https://wandb.ai/), a useful tool for experiment tracking, 
which is integrated into our pipeline. After setting up a free account, add your
credentials and the desired project name for the placeholders `wandb_user` and 
`wandb_project` in the code.

## Data Access

We utilize the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) data set 
for the training and evaluation of our generative model, which is publicly available
to credentialed users.

To extract an intermediate representation of the EHR time series data, we utilize
a slightly modified version of [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract),
which is automatically cloned if you followed the instructions for installation.
To extract the intermediate tables of the data required for our pipeline, follow
the steps 1-4 in the [instructions](https://github.com/MLforHealth/MIMIC_Extract/blob/master/README.md) 
of MIMIC-Extract. In addition to the standard flags, you can set the sampling frequency (e.g. to 15 minutes)
by calling: `python mimic_direct_extract.py --time_step 15 ...`

After the extraction has finished (extracting all patients can take several hours
on a machine with around 50 GB of memory), you should obtain four tables with the
extracted patient data. This is the input data for our experimental pipeline.

## Use

The main components of the pipeline can be run independently: 
data querying and processing from the database, training a generative model,
and evaluation. 

To run the entire experimental pipeline, i.e. extract the time series from the 
intermediate tables, train a generative model and run the resulting evaluation, run:

```
main.py 
--input_vitals /path/to/vitals/table 
--input_outcomes /path/to/outcomes/table
--input_static /path/to/static/table
--gen_model healthgen
--evaluation grud
--out_path /path/to/save/results
```
For more information on all available flags, run `main.py --helpfull`, and see
the comments in the code for additional information.

### License

[MIT License](LICENSE)

### Authors

Simon Bing, Andrea Dittadi, Stefan Bauer, Patrick Schwab