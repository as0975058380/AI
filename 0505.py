#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[ ]:





# In[3]:


plt.plot([9,9.2,9.6,7.5,6.7,7],[9.4,9.2,9.2,9.2,7.1,7.4],'yx')
plt.plot([9,9.2,9.6,7.5,6.7,7],[9.4,9.2,9.2,9.2,7.1,7.4],'yx')

plt.plot([7.2,7.3,7.2,7.3,7.2,7.3,7.3],[10.3,10.5,9.2,10.2,9.7,10.1,10.1],'gx')
plt.plot([6.5,9.0],[7.8,12.5],'b--')
plt.ylabel(['H cm'])
plt.xlabel(['W cm'])
plt.legend(('Orange','Lemons'),loc='upper right')
plt.show()


# In[5]:


import matplotlib.pyplot as plt

plt.plot([1,2,3,4],[0,0.3,0.6,0.9],'gx')
plt.plot([1,2,3,4],[0,0.3,0.6,0.9],'r--')

plt.axis([0,5,0,1])
plt.ylabel(['Y'])
plt.xlabel(['X'])
plt.legend(('price','passenger'),loc='upper right')
plt.show()


# In[8]:


import matplotlib.pyplot as plt
import numpy as np
plt.plot([1,2,3,4],[0,0.3,0.6,0.9],'gx')
plt.plot([1,2,3,4],[0,0.3,0.6,0.9],'r--')
X = 1+np.arange(30)/10
delta = np.random.uniform(low=-0.1,high=0.1, size=(30,))
Y=0.3*X-0.3 +delta
plt.plot(X,Y,'bo')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[13]:


import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Body']]
y_values = dataframe[['Brain']]
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values,y_values)
pre = body_reg.predict(x_values)
print(body_reg.predict( pd.DataFrame(data=[[170]])))
plt.scatter(x_values,y_values)
plt.plot(x_values,pre)
plt.show()


# In[16]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
diabetes = datasets.load_diabetes()
diabetes_X=diabetes.data[:,np.newaxis,2]
diabetes_X_train =diabetes_X[:-20]
diabetes_X_test =diabetes_X[-20:]

diabetes_y_train =diabetes.target[:-20]
diabetes_y_test =diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print('Coefficients: \n',regr.coef_)
print("Mean squared error: %.2f" 
      %np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))
print('Variance score: %.2f'% regr.score(diabetes_X_test,diabetes_y_test))
plt.scatter(diabetes_X_test,diabetes_y_test,color='black')
plt.plot(diabetes_X_test,regr.predict(diabetes_X_test),color='blue',
        linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


# In[ ]:




