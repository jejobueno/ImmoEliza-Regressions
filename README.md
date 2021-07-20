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
- some useful plots (the normalized data, the predictions vs the y_test database to show the accuracy, 
the boosted predictions, and the normalized data of the test dataframe, ...)

## How it works
1. DataCleaner
First, the data are cleaned. That means that we drop all the entirely empty rows, string values
are cleaned up, outliers and properties without price and area indication are dropped, duplicates
and columns with the lowest correlation rate are deleted, and some other minor riddances.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To put everything ready for the rest of the process, the variables that remain are transformed into
features.

2. DataRegressor
In the second step, the prediction is prepared. Firstly, the price, area, outside space and land
surface are rescaled. This is done in order to limit the differences and make the model more
effective.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Secondly, the database is split and into a train and test dataframe. The former is used to train the
model. A gradient boost is implemented.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the final step, predictions are made using the test dataset.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As an important addendum, we created a function that will prepare any new dataset to be pushed through
the program and make predictions about the price.

## Examples

**Correlation map:**

<img width = "400" src = /assets/Correlation%20map.png> 


**Rescaled area (square root) and price (logarhythmic):**

<img width = "400" src = /assets/Rescaled%20sqrtArea%20vs%20logPrice.png> 


**Predictions vs. expected values**

<img width = "400" src = /assets/predictions%20VS%20y.png>

## Authors
Jesús Bueno - Project Manager/dev & doc  
Pauwel De Wilde - dev & doc  
Camille de Neef & Hugo Pradier