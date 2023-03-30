# %% [markdown]
# ## IMPORT REQUIRED LIBRARIES

# %%

import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## LOAD THE DATA

# %%
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=RELIANCE.BSE&outputsize=full&apikey=AUPPYZ3PTCAM6OO3&datatype=csv'
data = pd.read_csv(url)

# %% [markdown]
# #### We used the API provided by www.alphavantage.co and imported it using pandas

# %% [markdown]
# ## DATA DESCRIPTION

# %%
data.head()

# %%
data.describe()

# %% [markdown]
# ## PREPARING THE DATA TO FIT INTO THE MODEL

# %%
data = data.drop(['adjusted_close'], axis=1)
data['timestamp'] = data['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').toordinal())

# %% [markdown]
# ## SPLIT INTO INPUT AND TARGET

# %%
X = data.drop(['close'], axis=1)
y = data['close']

# %% [markdown]
# ## USING GRIDSEARCHCV TO FIND THE BEST PARAMETERS

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# %%
rf = RandomForestRegressor()

# %%
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

# %% [markdown]
# ## FITTING THE MODEL

# %%
rf = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'],
                            max_depth=grid_search.best_params_['max_depth'],
                            min_samples_split=grid_search.best_params_['min_samples_split'],
                            min_samples_leaf=grid_search.best_params_['min_samples_leaf'])

rf.fit(X, y)

# %% [markdown]
# ## PLOTTING PREDICTED AGAINST THE ACTUAL DATA

# %%
y_pred = rf.predict(X)
data['timestamp'] = data['timestamp'].apply(lambda x: datetime.fromordinal(x))
df = pd.DataFrame({'timestamp': data['timestamp'], 'actual': y, 'predicted': y_pred})
fig = px.line(df, x='timestamp', y=['actual', 'predicted'], title='Actual vs Predicted Close')
fig.show()

# %% [markdown]
# ## PREDICTING FOR A RANDOM VALUE

# %%
ordinal = datetime.strptime('2023-03-27', '%Y-%m-%d').toordinal()
predicted_close = rf.predict([[ordinal, data['open'][0], data['high'][0], data['low'][0], data['volume'][0], data['dividend_amount'][0], data['split_coefficient'][0]]])[0]
print('Predicted Close:', predicted_close,ordinal)

actual_close = data.iloc[0]['close']
print('Actual Close:', actual_close)


