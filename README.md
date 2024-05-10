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

* **Categorical Variable Decision Tree**: Decision Tree which has a categorical target variable then it called a Categorical variable decision tree.
* **Continuous Variable Decision Tree**: Decision Tree has a continuous target variable then it is called Continuous Variable Decision Tree.

> *References*:
> * https://corporatefinanceinstitute.com/resources/data-science/decision-tree/
> * https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html
