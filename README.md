
Customer Churn at Telco Inc.
===

## Table of Conents

I.   [Project Goals        ](#i-project-goals)<br>
II.  [Project Deliverables ](#ii-project-deliverables)<br>
III. [Data Dictionary      ](#iii-data-dictionary)<br>
IV.  [Process              ](#iv-process)
1.   [Project Planning     ](#1-project-planning)
2.   [Data Acquisition     ](#2-data-acquisition)
3.   [Data Preparation     ](#3-data-preparation)
4.   [Data Exploration     ](#4-data-exploration)
5.   [Modeling & Evaluation](#5-modeling--evaluation)
6.   [Product Delivery     ](#6-product-delivery)

V.   [Project Reproduction ](#v-project-reproduction)

---

## I. Project Goals

This project holds the intent of predicting and reducing churn at Telco Inc., a telecommunication company that provides telephony and internet services to members of the consumer class. Churn in this context refers to the act of customer services and subscriptions being terminated, also known as attrition or turnover. Our goal is to find drivers of churn in the existing data and use machine learning models to predict further incidence in test samples.

## II. Project Deliverables

- Jupyter Notebook Report which contains the process of exploring, modeling, and testing
- This README which contains:
  + Project goals, findings, and takeaways
  + Data dictionary
  + Data science pipeline process
  + Instructions on reproducing
- CSV with `customer_id`, `probability_of_churn`, and `prediction_of_churn`
- Modules as `.py` files containing functions to acquire and prepare data
- Jupyter Notebook Presentation with high-level overview of project

## III. Data Dictionary

WIP

## IV. Process

Plan ➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver

This section serves as step-by-step project documentation of the data science pipeline shown above- from planning stages through final product delivery. Each checkmark indicates a completed step in each sub-process. It may also serve as a guide for project reproduction in conjunction with Section V of this README; however, it does not serve to limit the process to strict definitions.

#### 1. Project Planning
🟢***Plan***🟢 ➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver

- [ ] Describe project goals and product
- [ ] Set task list for working through pipeline
- [ ] Create data dictionary to explain data and context
- [ ] State clearly the starting hypothesis

#### 2. Data Acquisition
Plan ➜ 🟢***Acquire***🟢 ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver <br>

- [ ] Create `acquire.py` with:
  - Function(s) needed to fetch data into pandas DataFrame
  - Required imports to perform tasks
  - Ensured security of personal credentials
- [ ] In Jupyter Notebook:
  - Import function(s) from `acquire.py` module
  - Perform data summarization
  - Plot variable distributions

#### 3. Data Preparation
Plan ➜ Acquire ➜ 🟢***Prepare***🟢 ➜ Explore ➜ Model & Evaluate ➜ Deliver

- [ ] Create `prepare.py` with function(s) to:
  - Split data into train, validate, test sets
  - Address missing values
  - Encode variables
  - Create new features if needed
- [ ] In Jupyter Notebook:
  - Import function(s) from `prepare.py` module
  - Explore missing values and document how to address them
  - Explore dtypes and values to ensure numeric representation
  - Create new features for use in modeling

#### 4. Data Exploration
Plan ➜ Acquire ➜ Prepare ➜ 🟢***Explore***🟢 ➜ Model & Evaluate ➜ Deliver

In Jupyter Notebook:
- [ ] Answer key questions about hypotheses and find drivers of churn
  - Run at least two statistical tests
  - Document findings
- [ ] Create visualizations with intent to discover variable relationships
  - Identify variables related to churn
  - Identify any potential data integrity issues
- [ ] Summarize conclusions, provide clear answers, and summarize takeaways
  - Explain plan of action as deduced from work to this point


#### 5. Modeling & Evaluation
Plan ➜ Acquire ➜ Prepare ➜ Explore ➜ 🟢***Model & Evaluate***🟢 ➜ Deliver

In Jupyter Notebook:
- [ ] Establish baseline accuracy
- [ ] Train and fit multiple (3+) models with varying algorithims and/or hyperparametes
- [ ] Compare evaluation metrics across models
- [ ] Remove undeeded features
- [ ] Evaluate best performing models using validate set
- [ ] Choose best performing validation model for use on test set
- [ ] Test final model on out-of-sample testing dataset
  - Summarize performance
  - Interpret and document findings

#### 6. Product Delivery
Plan ➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ 🟢***Deliver***🟢

- [ ] Prepare five minute presentation using Jupyter Notebook
- [ ] Include introduction of project and goals
- [ ] Provide executive summary of findings, key takeaways, and recommendations
- [ ] Create walkthrough of analysis 
  - Visualize relationships
  - Dcoument takeaways
  - Explicitly define questions asked during initial analysis
- [ ] Provide final takeaways, recommend course of action, and next steps
- [ ] Be prepared to answer questions following presentation

## V. Project Reproduction

WIP
