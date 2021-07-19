<div align = "center">

<h3>Becode AI training

group assignment: Regressions</h3>


<img width = "200" src = /assets/BeCode_Logo.png>
</div>

# ImmoEliza-Regressions
Machine learning model to predict prices on Belgium's real estate sales.

## Table of contents
[Description](#Description)  
[Installation](#Installation)  
[Usage](#Usage)  
[Output](#Output)  
[How it works](#How-it-works)  
[Examples](#Examples)  
[Authors](#Authors)

## Description
The model predicts the prices of properties in Belgium, based on data that were gathered in a previous scraping project.
In relation with the postal code, the state of the construction, the property subtype (apartment, studio, villa, chalet, ...),
and existance of a fireplace, terrace, garden and/or fully equiped kitchen, an estimate of the asking price is made.

The accuracy of the model is 0.89, which means that there is always a possibility for outliers (less then 11 %). More importantly: in 89 %
of the cases the prediction will be within a respectable range.

## Installation
Clone the repository:
```
git clone https://github.com/jejobueno/ImmoEliza-Regressions
``` 

## Usage
Run main.py.

## Output
When you run the program, you will get: 

- a print of the train and test scores (without and with boost),
- a print of the regressor score,
- a list of data that are to be predicted (based on the test dataframe),
- a list of the predictions themselves,
- four plots (the normalized data, the predictions vs the y_test database to show the accuracy, 
the boosted predictions, and the normalized data of the test dataframe)

## How it works
