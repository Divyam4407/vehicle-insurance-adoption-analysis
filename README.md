# vehicle-insurance-adoption-analysis

# Replication Package: Online Vehicle Insurance Adoption in Bengaluru
## Overview
This repository contains the dataset and Python analytical pipeline for the academic research project: **"A Study on Adoption of Online Vehicle Insurance in Bengaluru."** The study investigates the transition from traditional offline agents to digital-first aggregators and insurer portals, extending the UTAUT2 framework with Perceived Risk and Trust variables.

## Contents
* `survey_data_anonymized.csv`: Primary dataset (N=232 valid respondents) collected via convenience sampling in Bengaluru.
* `insurance_adoption_analysis.py`: Python pipeline executing:
  * Harman's Single Factor Test for Common Method Bias (CMB)
  * Exploratory Factor Analysis (PCA)
  * Construct Reliability (Cronbach's Alpha)
  * Multicollinearity Diagnostics (VIF)
  * Econometric Modelling (Logistic Regression & Linear Probability Model)

## Usage
To reproduce the findings, clone this repository and install the required dependencies:
```bash
pip install -r requirements.txt
python insurance_adoption_analysis.py
