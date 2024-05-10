run django app as an ASGI to enable asynchronous in a server

1. create virtual environment  
`virtualenv venv`  
2. activate virtual environment  
`source venv/bin/activate`  
3. install requirements    
`pip install -r requirements.txt`  
4. run server  
`uvicorn --reload modeling.asgi:application --port 7000`

# ML Models
Lumba.ai provides several Machine Learning models that can be used by user depends on their needs.
## 1. Linear Regression
Simple linear regression is used to estimate the relationship between two quantitative variables. You can use simple linear regression when you want to know:
1. How strong the relationship is between two variables (e.g., the relationship between rainfall and soil erosion).
2. The value of the dependent variable at a certain value of the independent variable (e.g., the amount of soil erosion at a certain level of rainfall)
   
**Regression models** describe the relationship between variables by fitting a line to the observed data. Linear regression models use a straight line, while logistic and nonlinear regression models use a curved line. Regression allows you to estimate how a dependent variable changes as the independent variable(s) change.

> *Reference: https://www.scribbr.com/statistics/simple-linear-regression/*

## 2. Decision Tree
A decision tree is a support tool with a tree-like structure that models probable outcomes, cost of resources, utilities, and possible consequences. Decision trees provide a way to present algorithms with conditional control statements. They include branches that represent decision-making steps that can lead to a favorable result.

Decision trees can boost predictive models with accuracy, ease in interpretation, and stability. The tools are also effective in fitting non-linear relationships since they can solve data-fitting challenges, such as regression and classifications.

Types of decision trees are based on the type of target variable we have. It can be of two types:

* **Categorical Variable Decision Tree**: Decision Tree which has a categorical target variable then it called a Categorical variable decision tree.
* **Continuous Variable Decision Tree**: Decision Tree has a continuous target variable then it is called Continuous Variable Decision Tree.

> *References*:
> * https://corporatefinanceinstitute.com/resources/data-science/decision-tree/
> * https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html
