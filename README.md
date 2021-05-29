
Customer Churn at Telco Inc.
===

## Table of Conents

&nbsp;&nbsp;                         I.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Project Goals            ](#i-project-goals)<br>
&nbsp;&nbsp;                         II.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      [Project Deliverables     ](#ii-project-deliverables)<br>
&nbsp;&nbsp;                         III.&nbsp;&nbsp;&nbsp;&nbsp;           [Data Context             ](#iii-data-context)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.&nbsp;                               [Database Relationship Map](#1-database-relationship)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.&nbsp;                               [Data Dictionary          ](#2-data-dictionary)<br>
&nbsp;&nbsp;                         IV.&nbsp;&nbsp;&nbsp;&nbsp;            [Process                  ](#iv-process)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.&nbsp;                               [Project Planning         ](#1-project-planning)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.&nbsp;                               [Data Acquisition         ](#2-data-acquisition)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.&nbsp;                               [Data Preparation         ](#3-data-preparation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.&nbsp;                               [Data Exploration         ](#4-data-exploration)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.&nbsp;                               [Modeling & Evaluation    ](#5-modeling--evaluation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.&nbsp;                               [Product Delivery         ](#6-product-delivery)<br>
&nbsp;&nbsp;                         V.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       [Project Reproduction     ](#v-project-reproduction)<br>

---

## I. Project Goals

This project holds the intent of predicting and reducing churn at Telco Inc., a telecommunication company that provides telephony and internet services to members of the consumer class. Churn in this context refers to the act of customer services and subscriptions being terminated, also known as attrition or turnover. Our goal is to find drivers of churn in the existing data and use machine learning models to predict further incidence in test samples. From there we will recommend actions to improve customer retention in these areas of high churn.

## II. Project Deliverables

- Jupyter Notebook Report which contains the process of exploring, modeling, and testing
- This README which contains:
  + Project goals, findings, and takeaways
  + Data context
  + Data science pipeline process
  + Instructions on reproducing
- CSV with `customer_id`, `probability_of_churn`, and `prediction_of_churn`
- Modules as `.py` files containing functions to acquire and prepare data
- Jupyter Notebook Presentation with high-level overview of project

## III. Data Context


#### 1. Database Relationship

The Codeup `telco_churn` SQL database contains four tables: `customers`, `contract_types`, `internet_service_types`, and `payment_types`. These tables are connected in the manner defined in the following image with foreign key links being represented by connecting arrows to two highlighted keys. This database is read into a pandas DataFrame and prepared in the manner described in the data dictionary below.

![](https://i.ibb.co/G5F1k2w/Screen-Shot-2021-05-28-at-17-41-18.png)

#### 2. Data Dictionary

Following acquisition and preparation of the initial SQL database, the DataFrames used in this project contain the following variables. Contained values are defined along with their respective data types.

|  Feature               |  Definition                                |  Data Type             |
| :--------------------: | :----------------------------------------  | :--------------------: |
|  is_female             |  binary gender identity is female          |  integer (boolean)     |
|  is_senior             |  qualifies as senior citizen (65+)         |  integer (boolean)     |
|  has_partner           |  has spouse, partner, or significant other |  integer (boolean)     |
|  has_dependent         |  has dependent(s), children or otherwise   |  integer (boolean)     |
|  has_phone             |  is or was a phone customer                |  integer (boolean)     |
|  one_line              |  has or had one phone line *               |  integer (boolean)     |
|  multiple_lines        |  has or had multiple phone lines           |  integer (boolean)     |
|  has_internet          |  is or was an internet customer            |  integer (boolean)     |
|  dsl                   |  is or was a dsl internet customer *       |  integer (boolean)     |
|  fiber                 |  had or has fiber internet service         |  integer (boolean)     |
|  streaming_tv          |  internet option: has or had service addon |  integer (boolean)     |
|  streaming_movies      |  internet option: has or had service addon |  integer (boolean)     |
|  online_security       |  internet option: has or had service addon |  integer (boolean)     |
|  online_backup         |  internet option: has or had service addon |  integer (boolean)     |
|  device_protection     |  internet option: has or had service addon |  integer (boolean)     |
|  tech_support          |  internet option: has or had service addon |  integer (boolean)     |
|  mailed_check          |  payment type is or was check via post     |  integer (boolean)     |
|  electronic_check      |  payment type is or was electronic check   |  integer (boolean)     |
|  bank_transfer         |  payment type is or was bank transfer      |  integer (boolean)     |
|  credit_card           |  payment type is or was credit card        |  integer (boolean)     |
|  paperless_billing     |  customer bill is or was paperless         |  integer (boolean)     |
|  autopay               |  customer payment is or was automatic      |  integer (boolean)     |
|  no_contract           |  customer has or had month-to-month term   |  integer (boolean)     |
|  one_year_contract     |  customer has or had one year service term |  integer (boolean)     |
|  two_year_contract     |  customer has or had two year service term |  integer (boolean)     |
|  monthly_charges       |  current monthly charges in USD            |  float                 |
|  total_charges         |  sum of all charges for tenure in USD      |  float                 |
|  tenure                |  length of customer service in months      |  integer               |
|  churn (target)        |  customer services are terminated          |  integer (boolean)     |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Feature was only used in exploration DataFrames

## IV. Process

This section serves as step-by-step project documentation of the data science pipeline—from planning stages through final product delivery. Each checkmark indicates a completed step in each sub-process. It may also serve as a guide for project reproduction in conjunction with [Section V](#v-project-reproduction) of this README; however, it does not serve to limit the process to strict definitions.

#### 1. Project Planning
🟢 **Plan** ➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver

- [ ] Describe project goals and product
- [ ] Set task list for working through pipeline
- [ ] Create data dictionary to explain data and context
- [ ] State clearly the starting hypothesis

#### 2. Data Acquisition
Plan ➜ 🟢 **Acquire** ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver <br>

- [ ] Create `acquire.py` with:
  - Function(s) needed to fetch data into pandas DataFrame
  - Required imports to perform tasks
  - Ensured security of personal credentials
- [ ] In Jupyter Notebook:
  - Import function(s) from `acquire.py` module
  - Perform data summarization
  - Plot variable distributions

#### 3. Data Preparation
Plan ➜ Acquire ➜ 🟢 **Prepare** ➜ Explore ➜ Model & Evaluate ➜ Deliver

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
Plan ➜ Acquire ➜ Prepare ➜ 🟢 **Explore** ➜ Model & Evaluate ➜ Deliver

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
Plan ➜ Acquire ➜ Prepare ➜ Explore ➜ 🟢 **Model & Evaluate** ➜ Deliver

In Jupyter Notebook:
- [ ] Establish baseline accuracy
- [ ] Train and fit multiple (3+) models with varying algorithms and/or hyperparameters
- [ ] Compare evaluation metrics across models
- [ ] Remove unnecessary features
- [ ] Evaluate best performing models using validate set
- [ ] Choose best performing validation model for use on test set
- [ ] Test final model on out-of-sample testing dataset
  - Summarize performance
  - Interpret and document findings

#### 6. Product Delivery
Plan ➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ 🟢 **Deliver**

- [ ] Prepare five minute presentation using Jupyter Notebook
- [ ] Include introduction of project and goals
- [ ] Provide executive summary of findings, key takeaways, and recommendations
- [ ] Create walkthrough of analysis 
  - Visualize relationships
  - Document takeaways
  - Explicitly define questions asked during initial analysis
- [ ] Provide final takeaways, recommend course of action, and next steps
- [ ] Be prepared to answer questions following presentation

## V. Project Reproduction

WIP
