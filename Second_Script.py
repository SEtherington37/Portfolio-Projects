#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import kagglehub
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#Data Loading

path = kagglehub.dataset_download("mikhail1681/walmart-sales")
path = path+"/Walmart_Sales.csv"

df = pd.read_csv(path)

plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x:f'{x:,.2f}')

print("Dataset Preview:")
df.head()

#Data Cleaning

df["Date"] = pd.to_datetime(df['Date'], 
            format='%d-%m-%Y', 
            errors='coerce')

print("Missing Values Before Cleaning")
print(df.isnull().sum())

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Day'] = df['Date'].dt.day
df['Quarter'] = df['Date'].dt.quarter
df['Month_Name'] = df['Date'].dt.month_name()

def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Spring'
    elif month in [6,7,8]:
        return 'Summer'
    else:
        return 'Fall'
    
df['Season'] = df['Month'].apply(assign_season)

scaler = MinMaxScaler()
df['Weekly_Sales_Scaled'] = scaler.fit_transform(df[['Weekly_Sales']])

print("\nData After Cleaning and Feature Engineering:")
df[['Date', 'Year', 'Month', 'Week', 'Quarter', 'Season', 'Weekly_Sales_Scaled']].head()

#show a brief summary of the data before performing analysis
store_summary = df.groupby('Store')['Weekly_Sales'].agg(['sum', 'mean', 'std']).reset_index()
store_summary.columns = ['Store', 'Total_Sales', 'Average_Sales', 'Sales_Std_Dev',]
print("\nTop 5 Stores by Total Sales:")
print(store_summary.sort_values(by='Total_Sales',ascending=False).head())

#total sales over time graph
sales_over_time = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
plt.figure(figsize=(12,6))
plt.plot(sales_over_time['Date'], sales_over_time['Weekly_Sales'], color='steelblue', linewidth=2)

plt.title('Total Weekly Sales Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()

##Total Sales by Week

print(sales_over_time.sort_values(by="Weekly_Sales", ascending=False))

print(sales_over_time[sales_over_time["Weekly_Sales"].isin([sales_over_time["Weekly_Sales"].median()])])

import matplotlib.cm as cm
import matplotlib.colors as mcolors

##Average Sales by Store
avg_sales_store = df.groupby('Store')['Weekly_Sales'].mean().reset_index()
avg_sales_store = avg_sales_store.sort_values(by='Weekly_Sales', ascending=False)

norm = mcolors.Normalize(vmin=avg_sales_store['Weekly_Sales'].min(),
                         vmax=avg_sales_store['Weekly_Sales'].max())
cmap=cm.get_cmap('viridis')

colors = cmap(norm(avg_sales_store['Weekly_Sales']))

fig, ax = plt.subplots(figsize=(14,6))

x_positions = range(len(avg_sales_store))
bars = ax.bar(x_positions,
              avg_sales_store['Weekly_Sales'],
              color=colors)

ax.set_title('Average Weekly Sales by Store', fontsize=16)
ax.set_xlabel('Store ID', fontsize=12)
ax.set_ylabel('Average Sales ($)', fontsize=12)
ax.set_xticks(x_positions)
ax.set_xticklabels(avg_sales_store['Store'].astype(str), rotation=45)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Average Weekly Sales ($)', fontsize=12)
plt.tight_layout()
plt.show()

##Standard Deviation of Sales by Store
sales_stdev_store = df.groupby('Store')['Weekly_Sales'].std().reset_index()
sales_stdev_store = sales_stdev_store.sort_values(by='Weekly_Sales', ascending=False)

norm = mcolors.Normalize(vmin=sales_stdev_store['Weekly_Sales'].min(),
                         vmax=sales_stdev_store['Weekly_Sales'].max())
cmap=cm.get_cmap('viridis')

colors = cmap(norm(sales_stdev_store['Weekly_Sales']))

fig, ax = plt.subplots(figsize=(14,6))

x_positions = range(len(sales_stdev_store))
bars = ax.bar(x_positions,
              sales_stdev_store['Weekly_Sales'],
              color=colors)

ax.set_title('Standard Deviation of Sales by Store', fontsize=16)
ax.set_xlabel('Store ID', fontsize=12)
ax.set_ylabel('Standard Deviation ($)', fontsize=12)
ax.set_xticks(x_positions)
ax.set_xticklabels(sales_stdev_store['Store'].astype(str), rotation=45)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Standard Deviation', fontsize=12)
plt.tight_layout()
plt.show()

##Average Sales by Month
monthly_sales = df.groupby('Month_Name')['Weekly_Sales'].mean().reindex([
    'January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December'
]).reset_index()

sales_values = monthly_sales['Weekly_Sales']
month_names = monthly_sales['Month_Name']

norm = mcolors.Normalize(vmin=sales_values.min(), vmax=sales_values.max())
cmap = cm.PuBuGn
bar_colors = cmap(norm(sales_values))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(month_names, sales_values, color=bar_colors)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Average Sales ($)')

ax.set_title('Average Weekly Sales by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Average Sales ($)')

plt.xticks(rotation=90)
plt.tight_layout()

plt.show()

##Average Sales by Season
seasonal_sales = df.groupby('Season')['Weekly_Sales'].mean().reindex([
    'Winter', 'Spring', 'Summer', 'Fall'
]).reset_index()

sales_values = seasonal_sales['Weekly_Sales']
season_names = seasonal_sales['Season']

norm = mcolors.Normalize(vmin=sales_values.min(), vmax=sales_values.max())
cmap = cm.Oranges
bar_colors = cmap(norm(sales_values))

fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(season_names, sales_values, color=bar_colors)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Average Sales ($)')

ax.set_title('Average Weekly Sales by Season')
ax.set_xlabel('Season')
ax.set_ylabel('Average Sales ($)')

plt.xticks(rotation = 0)
plt.tight_layout()

plt.show()

#fixing the holiday flag column
df['Holiday_Flag_Old'] = df['Holiday_Flag']

if 'holiday_flag_old' not in locals():
    holiday_flag_old = df['Holiday_Flag_Old']
else:
    print('holiday_flag_old already exists')

df['Holiday_Flag'] = 0
list_of_dates = ['2010-12-31', '2010-11-26', '2011-11-25',
                 '2010-09-10', '2010-12-24', '2011-12-23', 
                 '2011-12-30', '2011-09-09', '2012-09-07', 
                 '2010-02-12', '2011-02-11', '2012-02-12']
list_of_dates = pd.to_datetime(list_of_dates)
df.loc[df['Date'].isin(list_of_dates), 'Holiday_Flag'] = 1

df = df.drop(columns="Holiday_Flag_Old")

#use this if something goes wrong to reset the holiday
# flag column to original values
#df['Holiday_Flag_Old'] = holiday_flag_old
#df['Holiday_Flag'] = df['Holiday_Flag_Old']

##Average Sales by Holiday vs Non-Holiday Week
holiday_sales = df.groupby('Holiday_Flag')['Weekly_Sales'].mean().reset_index()
holiday_sales['Holiday_Label'] = holiday_sales['Holiday_Flag'].map({
    0: 'Non-Holiday Week', 1: 'Holiday Week'
})

sales_values = holiday_sales['Weekly_Sales']
holiday_names = holiday_sales['Holiday_Label']

norm = mcolors.Normalize(vmin=sales_values.min(), vmax=sales_values.max())
cmap = cm.Blues
bar_colors = cmap(norm(sales_values))

fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(holiday_names, sales_values, color=bar_colors)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Average Sales ($)')

ax.set_title('Holiday vs Non-Holiday Sales')
ax.set_xlabel('Week Type')
ax.set_ylabel('Average Sales ($)')

plt.xticks(rotation = 0)
plt.tight_layout()

plt.show()

##Stores with Highest Holiday Uplift
#raw value
diff = (
    df[df["Holiday_Flag"] == 1].groupby("Store")["Weekly_Sales"].mean()
    - df[df["Holiday_Flag"] == 0].groupby("Store")["Weekly_Sales"].mean()
)

print(diff.sort_values(ascending=False))

plt.figure(figsize=(14,6))

diff.plot(kind="bar")

plt.axhline(0, color="black", linewidth=0.8)
plt.title("Holiday Uptick by Store (Raw Number)")
plt.xlabel("Store")
plt.ylabel("Uptick (# of Sales)")
plt.tight_layout()
plt.show()

#percentage
diff_perc= (
    (df[df["Holiday_Flag"] == 1].groupby("Store")["Weekly_Sales"].mean()
    - df[df["Holiday_Flag"] == 0].groupby("Store")["Weekly_Sales"].mean()) /
    df[df["Holiday_Flag"] == 0].groupby("Store")["Weekly_Sales"].mean() * 100
)

print(diff_perc.sort_values(ascending=False))

plt.figure(figsize=(14,6))

diff_perc.plot(kind="bar")

plt.axhline(0, color="black", linewidth=0.8)
plt.title("Holiday Uptick by Store (Percent)")
plt.xlabel("Store")
plt.ylabel("Uptick (% of Sales)")
plt.tight_layout()
plt.show()

##Scatter Plot of Sales and Temperature
temperature_sales = df.groupby('Temperature')['Weekly_Sales'].mean().reset_index()
temp_sales_vals = temperature_sales['Weekly_Sales']
temp_vals = temperature_sales['Temperature']

plt.figure(figsize=(12,6))
plt.scatter(temp_vals, temp_sales_vals, color='red', linewidth=2)

plt.title('Average Total Weekly Sales by Temperature', fontsize=16)
plt.xlabel('Temperature', fontsize=12)
plt.ylabel('Average Total Sales ($)', fontsize=12)

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()

##Scatter Plot of Sales and Fuel Price
fuel_price_sales = df.groupby('Fuel_Price')['Weekly_Sales'].mean().reset_index()
fuel_sales_vals = fuel_price_sales['Weekly_Sales']
fuel_vals = fuel_price_sales['Fuel_Price']

plt.figure(figsize=(12,6))
plt.scatter(fuel_vals, fuel_sales_vals, color='blue', linewidth=2)

plt.title('Average Total Weekly Sales by Fuel Price', fontsize=16)
plt.xlabel('Fuel Price', fontsize=12)
plt.ylabel('Average Total Sales ($)', fontsize=12)

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()

##Scatter Plot of Sales and CPI
CPI_sales = df.groupby('CPI')['Weekly_Sales'].mean().reset_index()
CPI_sales_vals = CPI_sales['Weekly_Sales']
CPI_vals = CPI_sales['CPI']

plt.figure(figsize=(12,6))
plt.scatter(CPI_vals, CPI_sales_vals, color='blue', linewidth=2)

plt.title('Average Total Weekly Sales by CPI', fontsize=16)
plt.xlabel('CPI', fontsize=12)
plt.ylabel('Average Total Sales ($)', fontsize=12)

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()

##Scatter Plot of Sales and Unemployment
unemployment_sales = df.groupby('Unemployment')['Weekly_Sales'].mean().reset_index()
unemployment_sales_vals = unemployment_sales['Weekly_Sales']
unemployment_vals = unemployment_sales['Unemployment']

plt.figure(figsize=(12,6))
plt.scatter(unemployment_vals, unemployment_sales_vals, color='red', linewidth=2)

plt.title('Average Total Weekly Sales by Unemployment', fontsize=16)
plt.xlabel('Unemployment', fontsize=12)
plt.ylabel('Average Total Sales ($)', fontsize=12)

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()

##Correlation Map
correlation_matrix = df[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0)

plt.title("Correlation Heatmap: Sales vs External Factors", fontsize=14)
plt.tight_layout()
plt.show()

##Lag Analysis
lag_df = df.copy()
lag_df = df.sort_values(["Store", "Year", "Week"])

for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
    for lag in [1, 2, 4]:
        lag_df[f"{col}_lag{lag}"] = lag_df.groupby("Store")[col].shift(lag)

lag_df_corr = lag_df[[
              "Weekly_Sales", 
              "Temperature", "Temperature_lag1", "Temperature_lag2", "Temperature_lag4",
              "Fuel_Price", "Fuel_Price_lag1", "Fuel_Price_lag2", "Fuel_Price_lag4",
              "CPI", "CPI_lag1", "CPI_lag2", "CPI_lag4",
              "Unemployment", "Unemployment_lag1", "Unemployment_lag2", "Unemployment_lag4"
              ]].corr()

print(lag_df_corr["Weekly_Sales"].sort_values(ascending=False))

##Sensitivity Analysis
import statsmodels.api as sm
import statsmodels.formula.api as smf

lag_df1 = lag_df.copy()

if 'Store' not in lag_df1.columns:
    lag_df1 = lag_df1.reset_index()
lag_df1["Store"] = lag_df1["Store"].astype("category")

lag_df1 = lag_df1.sort_values(['Store', 'Date'])

for col in ["Fuel_Price", "CPI", "Unemployment", "Temperature"]:
    lag_df1[f"{col}_pctchg"] = lag_df1[col].pct_change()

lag_df1["CPI_pctchg_lag1"] = lag_df1.groupby('Store')["CPI_pctchg"].shift(1)
lag_df1["CPI_pctchg_lag2"] = lag_df1.groupby('Store')["CPI_pctchg"].shift(2)
lag_df1["CPI_pctchg_lag4"] = lag_df1.groupby('Store')["CPI_pctchg"].shift(4)

lag_df1['log_sales'] = np.log(lag_df1['Weekly_Sales']+1)

panel_df = lag_df1.dropna(subset=[
    'log_sales',
    'CPI_pctchg', 'CPI_pctchg_lag1', 'CPI_pctchg_lag2', 'CPI_pctchg_lag4',
    'Fuel_Price_pctchg', 'Unemployment_pctchg', 'Temperature', 'Store'
]).copy()

panel_df = panel_df.sort_values(["Store", "Date"]).copy()

panel_df["log_sales"] = np.log(panel_df["Weekly_Sales"])

for col in ["Fuel_Price", "CPI", "Unemployment", "Temperature"]:
    panel_df[f"{col}_pctchg"] = panel_df.groupby("Store")[col].pct_change() * 100

def winsorize(series, lower=0.01, upper=0.99):
    return series.clip(
        lower = series.quantile(lower),
        upper = series.quantile(upper)
    )

for col in ["Fuel_Price_pctchg", "CPI_pctchg", "Unemployment_pctchg", "Temperature_pctchg"]:
    panel_df[col] = winsorize(panel_df[col])

for col in ["Fuel_Price_pctchg", "CPI_pctchg", "Unemployment_pctchg", "Temperature_pctchg"]:
    panel_df[f"{col}_lag1"] = panel_df.groupby("Store")[col].shift(1)

panel_df = panel_df.dropna()

lag_periods = [1, 2, 4]
base_factors = ["Fuel_Price_pctchg", "CPI_pctchg", "Unemployment_pctchg", "Temperature_pctchg"]

for factor in base_factors:
    for lag in lag_periods:
        panel_df[f"{factor}_lag{lag}"] = panel_df.groupby("Store")[factor].shift(lag)

panel_df = panel_df.dropna().copy()

formula = (
    "log_sales ~ Fuel_Price_pctchg + Fuel_Price_pctchg_lag1 + Fuel_Price_pctchg_lag2 + Fuel_Price_pctchg_lag4 "
    "+ CPI_pctchg + CPI_pctchg_lag1 + CPI_pctchg_lag2 + CPI_pctchg_lag4 "
    "+ Unemployment_pctchg + Unemployment_pctchg_lag1 + Unemployment_pctchg_lag2 + Unemployment_pctchg_lag4 "
    "+ Temperature_pctchg + Temperature_pctchg_lag1 + Temperature_pctchg_lag2 + Temperature_pctchg_lag4"
)

model = smf.mixedlm(
    formula,
    data=panel_df,
    groups=panel_df["Store"],
    re_formula="~ Fuel_Price_pctchg + Fuel_Price_pctchg_lag1 + Fuel_Price_pctchg_lag2 + Fuel_Price_pctchg_lag4 "
               "+ CPI_pctchg + CPI_pctchg_lag1 + CPI_pctchg_lag2 + CPI_pctchg_lag4 "
               "+ Unemployment_pctchg + Unemployment_pctchg_lag1 + Unemployment_pctchg_lag2 + Unemployment_pctchg_lag4 "
               "+ Temperature_pctchg + Temperature_pctchg_lag1 + Temperature_pctchg_lag2 + Temperature_pctchg_lag4"
)
res = model.fit() #this line takes 3 minutes to run, be patient
print(res.summary())

fixed = res.fe_params
random = res.random_effects

store_elasticities = []
for store, rand in random.items():
    coeffs = fixed + rand
    coeffs["Store"] = store
    store_elasticities.append(coeffs)

elasticity_df = pd.DataFrame(store_elasticities).set_index("Store")

factors = ["Fuel_Price_pctchg", "CPI_pctchg", "Unemployment_pctchg", "Temperature_pctchg"]
lags = ["", "_lag1", "_lag2", "_lag4"]

for factor in factors:
    cols_to_plot = [f"{factor}{lag}" for lag in lags if f"{factor}{lag}" in elasticity_df.columns]

    ax = elasticity_df[cols_to_plot].plot(
        kind="bar",
        figsize=(14,6),
        width=0.8,
        colormap="BrBG"
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"Store-level Elasticities to {factor.replace('_pctchg','')} (Current and Lags)")
    plt.ylabel("Elasticity")

    handles, _ = ax.get_legend_handles_labels()
    lag_labels = [lag.replace("_lag","") if lag else "0" for lag in lags if f"{factor}{lag}" in elasticity_df.columns]
    plt.legend(handles, lag_labels, title="Lag (weeks)")
    plt.show()

for factor in factors:
    cols_to_plot = [f"{factor}{lag}" for lag in lags if f"{factor}{lag}" in elasticity_df.columns]
    for col in cols_to_plot:
        elasticity_df[f"{factor}_SumofSens"] = elasticity_df[factor] if col == factor else elasticity_df[f"{factor}_SumofSens"]+elasticity_df[col]

    ax = elasticity_df[f"{factor}_SumofSens"].plot(
        kind="bar",
        figsize=(14,6),
        width=0.8,
        colormap="BrBG"
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"Net % Change after 4 weeks due to {factor.replace('_pctchg', '').replace('_', ' ')}")
    plt.ylabel("Net % Change")
    plt.show()

##Machine Learning
from sklearn.linear_model import LinearRegression

ml_df = df.copy()
ml_df = ml_df.drop(columns=['Date', 'Weekly_Sales_Scaled', 'Month_Name'])

ml_df = pd.get_dummies(ml_df, columns=['Season'], drop_first=True)

X = ml_df.drop(['Weekly_Sales'], axis=1)
y = ml_df['Weekly_Sales']

imputer = SimpleImputer(strategy='mean')
feature_columns=X.columns
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression Performance")
print(f"R² Score: {lr_r2:.3f}")
print(f"RMSE: {lr_rmse:,.2f}")

#Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None]
}
rf_model = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='r2', n_jobs=1)
rf_grid.fit(X_train, y_train) #this line takes about 2.5 minutes to run, be patient
y_pred_rf = rf_grid.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("\nRandom Forest Performance (Best Params: {})".format(rf_grid.best_params_))
print(f"R² Score: {rf_r2:.3f}")
print(f"RMSE: {rf_rmse:,.2f}")

#Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
gb_r2 = r2_score(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
print("\nGradient Boosting Performance")
print(f"R² Score: {gb_r2:.3f}")
print(f"RMSE: {gb_rmse:,.2f}")

#Importance Plots
importances = rf_grid.best_estimator_.feature_importances_
features = ml_df.drop(['Weekly_Sales'], axis=1).columns
feat_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

norm = plt.Normalize(feat_importance_df['Importance'].min(), feat_importance_df['Importance'].max())
colors = cm.Blues(norm(feat_importance_df['Importance']))

plt.figure(figsize=(10,6))
bars = plt.barh(feat_importance_df['Feature'], feat_importance_df['Importance'], color=colors)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

stores_lr = []
sales_lr = []
preds_lr = []

for i in range(0, len(X_test)):
    stores_lr.append(X_test[i][0])
    sales_lr.append(y_test.reset_index()["Weekly_Sales"][i])
    preds_lr.append(y_pred_lr[i])

test_stores_df_lr = pd.DataFrame({
    'Store': stores_lr,
    'Weekly_Sales': sales_lr,
    'Predictions': preds_lr
})

test_stores_df_lr = test_stores_df_lr.sort_values(by='Store', ascending=True)
rmse_list = []

for store in test_stores_df_lr["Store"].unique():
    rmse = np.sqrt(mean_squared_error(test_stores_df_lr[test_stores_df_lr["Store"]==store]["Weekly_Sales"], test_stores_df_lr[test_stores_df_lr["Store"]==store]["Predictions"]))
    rmse_list.append(rmse)

plt.figure(figsize=(14,6))
plt.bar(test_stores_df_lr["Store"].unique(), rmse_list)
plt.xlabel('Store')
plt.ylabel('Root Mean Squared Error')
plt.title('Linear Regression Model Accuracy By Store')
plt.tight_layout()
plt.show()

r2_list = []

for store in test_stores_df_lr["Store"].unique():
    r2 = r2_score(test_stores_df_lr[test_stores_df_lr["Store"]==store]["Weekly_Sales"], test_stores_df_lr[test_stores_df_lr["Store"]==store]["Predictions"])
    r2_list.append(r2)

plt.figure(figsize=(14,6))
plt.bar(test_stores_df_lr["Store"].unique(), r2_list)
plt.xlabel('Store')
plt.ylabel('R² Score')
plt.title('Linear Regression Model Accuracy By Store')
plt.tight_layout()
plt.show()

stores_rf = []
sales_rf = []
preds_rf = []

for i in range(0, len(X_test)):
    stores_rf.append(X_test[i][0])
    sales_rf.append(y_test.reset_index()["Weekly_Sales"][i])
    preds_rf.append(y_pred_rf[i])

test_stores_df_rf = pd.DataFrame({
    'Store': stores_rf,
    'Weekly_Sales': sales_rf,
    'Predictions': preds_rf
})

test_stores_df_rf = test_stores_df_rf.sort_values(by='Store', ascending=True)
rmse_list = []

for store in test_stores_df_rf["Store"].unique():
    rmse = np.sqrt(mean_squared_error(test_stores_df_rf[test_stores_df_rf["Store"]==store]["Weekly_Sales"], test_stores_df_rf[test_stores_df_rf["Store"]==store]["Predictions"]))
    rmse_list.append(rmse)

plt.figure(figsize=(14,6))
plt.bar(test_stores_df_rf["Store"].unique(), rmse_list)
plt.xlabel('Store')
plt.ylabel('Root Mean Squared Error')
plt.title('Random Forest Model Accuracy By Store')
plt.tight_layout()
plt.show()

r2_list = []

for store in test_stores_df_rf["Store"].unique():
    r2 = r2_score(test_stores_df_rf[test_stores_df_rf["Store"]==store]["Weekly_Sales"], test_stores_df_rf[test_stores_df_rf["Store"]==store]["Predictions"])
    r2_list.append(r2)

plt.figure(figsize=(14,6))
plt.bar(test_stores_df_rf["Store"].unique(), r2_list)
plt.xlabel('Store')
plt.ylabel('R² Score')
plt.title('Random Forest Model Accuracy By Store')
plt.tight_layout()
plt.show()

stores_gb = []
sales_gb = []
preds_gb = []

for i in range(0, len(X_test)):
    stores_gb.append(X_test[i][0])
    sales_gb.append(y_test.reset_index()["Weekly_Sales"][i])
    preds_gb.append(y_pred_gb[i])

test_stores_df_gb = pd.DataFrame({
    'Store': stores_gb,
    'Weekly_Sales': sales_gb,
    'Predictions': preds_gb
})

test_stores_df_gb = test_stores_df_gb.sort_values(by='Store', ascending=True)
rmse_list = []

for store in test_stores_df_gb["Store"].unique():
    rmse = np.sqrt(mean_squared_error(test_stores_df_gb[test_stores_df_gb["Store"]==store]["Weekly_Sales"], test_stores_df_gb[test_stores_df_gb["Store"]==store]["Predictions"]))
    rmse_list.append(rmse)

plt.figure(figsize=(14,6))
plt.bar(test_stores_df_gb["Store"].unique(), rmse_list)
plt.xlabel('Store')
plt.ylabel('Root Mean Squared Error')
plt.title('Gradient Boosting Model Accuracy By Store')
plt.tight_layout()
plt.show()

r2_list = []

for store in test_stores_df_gb["Store"].unique():
    r2 = r2_score(test_stores_df_gb[test_stores_df_gb["Store"]==store]["Weekly_Sales"], test_stores_df_gb[test_stores_df_gb["Store"]==store]["Predictions"])
    r2_list.append(r2)

plt.figure(figsize=(14,6))
plt.bar(test_stores_df_gb["Store"].unique(), r2_list)
plt.xlabel('Store')
plt.ylabel('R² Score')
plt.title('Gradient Boosting Model Accuracy By Store')
plt.tight_layout()
plt.show()

trend_results = []

for store, group in df.groupby("Store"):
    group = group.sort_values("Date")
    X = np.arange(len(group)).reshape(-1, 1)
    y = group["Weekly_Sales"].values

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    trend_results.append({"Store": store, "Trend_Slope": slope})

trend_df_base = pd.DataFrame(trend_results)

threshold = trend_df_base["Trend_Slope"].std() * 0.5
trend_df_base['Trend_Category'] = np.where(trend_df_base["Trend_Slope"] > trend_df_base["Trend_Slope"].mean() + threshold, "Growth",
                                      np.where(trend_df_base["Trend_Slope"] < trend_df_base["Trend_Slope"].mean() - threshold, "Decline", "Stable"))

stores = sorted(test_stores_df_rf["Store"].unique())
store_metrics = []

for store in stores:
    g = test_stores_df_rf[test_stores_df_rf["Store"] == store]
    r2 = r2_score(g["Weekly_Sales"], g["Predictions"])
    rmse = np.sqrt(mean_squared_error(g["Weekly_Sales"], g["Predictions"]))
    mean_sales = g["Weekly_Sales"].mean()
    volatility = g["Weekly_Sales"].std() / mean_sales
    store_metrics.append({"Store": store, "R2": r2, "RMSE": rmse, "Volatility": volatility})

store_metrics_df = pd.DataFrame(store_metrics)

r2_threshold = store_metrics_df["R2"].quantile(0.75)
r2_threshold_low = store_metrics_df["R2"].quantile(0.25)
vol_threshold = store_metrics_df["Volatility"].quantile(0.25)
vol_threshold_high = store_metrics_df["Volatility"].quantile(0.75)

store_metrics_df["Category"] = store_metrics_df.apply(
    lambda row: "Highly Predictable" if (row["R2"] >= r2_threshold and row["Volatility"] <= vol_threshold)
    else "Volatile" if (row["R2"] <= r2_threshold_low or row["Volatility"] >= vol_threshold_high)
    else "Moderate",
    axis = 1
)

import matplotlib.patheffects as path_effects

plt.figure(figsize=(10,6))

moderate = store_metrics_df[store_metrics_df["Category"] == "Moderate"]
plt.scatter(moderate["Volatility"], moderate["R2"], color="gray", alpha=0.5, label="Moderate")

predictable = store_metrics_df[store_metrics_df["Category"] == "Highly Predictable"]
plt.scatter(predictable["Volatility"], predictable["R2"], color="green", s=100, label="Highly Predictable")

volatile = store_metrics_df[store_metrics_df["Category"] == "Volatile"]
plt.scatter(volatile["Volatility"], volatile["R2"], color="red", s=100, label="Highly Volatile")

plt.axhline(r2_threshold, color="green", linestyle="--", label="Hgh R² Threshold")
plt.axvline(vol_threshold, color="green", linestyle=":", label="Low Volatility Threshold")
plt.axvline(vol_threshold_high, color="red", linestyle=":", label="High Volatility Threshold")
plt.axhline(r2_threshold_low, color="red", linestyle="--", label="Low R² Threshold")

for _, row in predictable.iterrows():
    t = plt.text(row["Volatility"], row["R2"] + 0.01, str(int(row["Store"])),
                 fontsize=8, color="green", ha="center", va="bottom", fontweight="bold", zorder=4)
    t.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                       path_effects.Normal()])

for _, row in volatile.iterrows():
    t = plt.text(row["Volatility"], row["R2"] - 0.01, str(int(row["Store"])),
                 fontsize=8, color="red", ha="center", va="top", fontweight="bold", zorder=4)
    t.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                        path_effects.Normal()])

plt.xlabel("Sales Volatility (Std / Mean)")
plt.ylabel("Model R² (Predictability)")
plt.title("Store Predictablility vs. Volatility")
plt.legend(loc="best", frameon=True)
plt.tight_layout()
plt.show()

##Forecasting
#Synthesizing Future Data Based on Historical Data
import holidays as hds

us_holidays = hds.US(observed=False)
us_hds_2010_12 = hds.US(observed=False, years=range(2010,2013))

WalMart_holidays = [
    # Super Bowl (first Sunday in February or last Sunday in Jan)
    pd.Timestamp('2012-02-05'),
    pd.Timestamp('2013-02-03'),
    pd.Timestamp('2014-02-02'),
    pd.Timestamp('2015-02-01'),
    pd.Timestamp('2016-02-07'),
    pd.Timestamp('2017-02-05'),
    pd.Timestamp('2018-02-04'),
    pd.Timestamp('2019-02-03'),
    pd.Timestamp('2020-02-02'),
    pd.Timestamp('2021-02-07'),
    pd.Timestamp('2022-02-06'),

    # Labor Day (first Monday in September)
    pd.Timestamp('2012-09-03'),
    pd.Timestamp('2013-09-02'),
    pd.Timestamp('2014-09-01'),
    pd.Timestamp('2015-09-07'),
    pd.Timestamp('2016-09-05'),
    pd.Timestamp('2017-09-04'),
    pd.Timestamp('2018-09-03'),
    pd.Timestamp('2019-09-02'),
    pd.Timestamp('2020-09-07'),
    pd.Timestamp('2021-09-06'),
    pd.Timestamp('2022-09-05'),

    # Thanksgiving (fourth Thursday in November)
    pd.Timestamp('2012-11-22'),
    pd.Timestamp('2013-11-28'),
    pd.Timestamp('2014-11-27'),
    pd.Timestamp('2015-11-26'),
    pd.Timestamp('2016-11-24'),
    pd.Timestamp('2017-11-23'),
    pd.Timestamp('2018-11-22'),
    pd.Timestamp('2019-11-28'),
    pd.Timestamp('2020-11-26'),
    pd.Timestamp('2021-11-25'),
    pd.Timestamp('2022-11-24'),

    # Christmas (fixed)
    pd.Timestamp('2012-12-25'),
    pd.Timestamp('2013-12-25'),
    pd.Timestamp('2014-12-25'),
    pd.Timestamp('2015-12-25'),
    pd.Timestamp('2016-12-25'),
    pd.Timestamp('2017-12-25'),
    pd.Timestamp('2018-12-25'),
    pd.Timestamp('2019-12-25'),
    pd.Timestamp('2020-12-25'),
    pd.Timestamp('2021-12-25'),
    pd.Timestamp('2022-12-25')
]

def contains_holiday_in_past_week(date, holiday_calendar):
    week_range = pd.date_range(end=date, periods=7)  # 7-day window ending with `date`
    return int(any(day in holiday_calendar for day in week_range))

fc_df = df.copy()

fc_df = fc_df.sort_values("Date")
fc_df.set_index("Date", inplace = True)
fc_df = fc_df.drop(columns=['Month_Name'])
fc_df = pd.get_dummies(fc_df, columns=['Season'], drop_first=True)

temperature_means = fc_df.groupby(['Store', 'Month'])['Temperature'].mean()
fuel_means = fc_df.groupby(['Store', 'Month'])['Fuel_Price'].mean()
cpi_means = fc_df.groupby('Month')['CPI'].mean()
unemployment_means = fc_df.groupby('Month')['Unemployment'].mean()

forecast_horizon = 104
last_date = fc_df.index.max()

future_dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(weeks=1), periods=forecast_horizon, freq='W')

future_df = pd.DataFrame(index=future_dates)
stored_df1 = future_df.copy()

for store in fc_df.sort_values('Store')['Store'].unique():
    stored_df = stored_df1
    stored_df['Store'] = store
    stored_df['Holiday_Flag'] = stored_df.index.to_series().apply(
        lambda d: contains_holiday_in_past_week(d, WalMart_holidays)
    )
    stored_df['Month'] = stored_df.index.month
    stored_df['Year'] = stored_df.index.year
    stored_df['Week'] = stored_df.index.isocalendar().week
    stored_df['Day'] = stored_df.index.day
    stored_df['Quarter'] = stored_df.index.quarter
    stored_df['Temperature'] = stored_df['Month'].map(
        lambda m: temperature_means.loc[(store, m)]
    )
    stored_df['Fuel_Price'] = stored_df['Month'].map(
        lambda m: fuel_means.loc[(store, m)]
    )
    stored_df['CPI'] = stored_df['Month'].map(
        lambda m: cpi_means[m]
    )
    stored_df['Unemployment'] = stored_df['Month'].map(
        lambda m: unemployment_means.loc[m]
    )
    stored_df['Season'] = stored_df['Month'].apply(assign_season)

    desired_column_order = [
        'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'Year', 'Month', 'Week', 'Day', 'Quarter', 'Season'
    ]
    stored_df = stored_df[desired_column_order]
    future_df = pd.concat([future_df, stored_df])

future_df = future_df.dropna()

future_df = pd.get_dummies(future_df, columns=['Season'], drop_first=True)
expected_columns = feature_columns
for col in expected_columns:
    if col not in future_df.columns:
        future_df[col] = 0
future_df = future_df.reindex(columns=['Store', 'Holiday_Flag', 'Temperature',
                            'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month',
                            'Week', 'Day', 'Quarter', 'Season_Spring', 
                            'Season_Summer', 'Season_Winter'], fill_value=0)

future_X = imputer.transform(future_df)

#Predicting Based on the Future Data
future_lr_preds = lr_model.predict(future_X)
future_rf_preds = rf_grid.predict(future_X)
future_gb_preds = gb_model.predict(future_X)

#Plotting the Forecasts
store_list = future_df.sort_values('Store')['Store'].unique()
future_preds_df = pd.DataFrame({
    'Store': np.repeat(store_list, forecast_horizon),
    'Date': np.tile(future_dates, len(store_list)),
    'Forecast_LR': future_lr_preds,
    'Forecast_RF': future_rf_preds,
    'Forecast_GB': future_gb_preds
})

store_means = future_preds_df.groupby('Store')['Forecast_RF'].mean()
top_5_stores = store_means.sort_values(ascending=False).head(5).index
bottom_5_stores = store_means.sort_values(ascending=True).head(5).index
selected_stores = top_5_stores.union(bottom_5_stores)
future_preds_by_mean = future_preds_df[future_preds_df['Store'].isin(selected_stores)]

include_all = False #Whether to include all stores or just top and bottom 5
                    #Use True for all store, False for top/bottom
desired_model = 'RF' #Choose which model's forecast you wish to plot
                    #options are 'LR' for linear regression,
                    #'RF' for random forest, and 'GB' for gradient boosting

plt.figure(figsize=(14, 8))

if include_all == True:
    sorted_stores = future_preds_df.groupby('Store')['Forecast_'+desired_model].mean().sort_values(ascending=False).index
    for store in sorted_stores:
        store_df = future_preds_df[future_preds_df['Store'] == store]
        plt.plot(store_df['Date'], store_df['Forecast_'+desired_model], label=f'Store {store}')

if include_all == False:
    sorted_stores = future_preds_by_mean.groupby('Store')['Forecast_'+desired_model].mean().sort_values(ascending=False).index
    for store in sorted_stores:
        store_df = future_preds_by_mean[future_preds_by_mean['Store'] == store]
        plt.plot(store_df['Date'], store_df['Forecast_'+desired_model], label=f'Store {store}')

if desired_model == 'LR':
    plt.title("Forecasted Weekly Sales per Store (Linear Regression)")
elif desired_model == 'RF':
    plt.title("Forecasted Weekly Sales per Store (Random Forest)")
elif desired_model == 'GB':
    plt.title("Forecasted Weekly Sales per Store (Gradient Boosting)")
else:
    plt.title("Forecasted Weekly Sales per Store")

plt.xlabel("Date")
plt.ylabel("Predicted Sales (Millions)")
plt.legend(title="Store", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()
plt.show()

forecast_df = pd.DataFrame({
    'Store': future_preds_df['Store'],
    'Date': future_preds_df['Date'],
    'Forecast_'+desired_model: future_preds_df['Forecast_'+desired_model]
})

trend_results = []

for store, group in forecast_df.groupby("Store"):
    group = group.sort_values("Date")
    X = np.arange(len(group)).reshape(-1, 1)
    y = group["Forecast_"+desired_model].values

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    trend_results.append({"Store": store, "Trend_Slope": slope})

trend_df = pd.DataFrame(trend_results)

threshold = trend_df["Trend_Slope"].std() * 0.5
trend_df["Trend_Category"] = np.where(trend_df["Trend_Slope"] > threshold, "Growth",
                                      np.where(trend_df["Trend_Slope"] < -threshold, "Decline", "Stable"))

for store in trend_df.sort_values("Trend_Slope", ascending=False)["Store"].head(5):
    subset = forecast_df[forecast_df["Store"] == store]
    plt.plot(subset["Date"], subset["Forecast_"+desired_model], label=f"Store {store}")
plt.legend()
plt.title("Example of Growth Stores")
plt.show()

for store in trend_df_base.sort_values("Trend_Slope", ascending=False)["Store"].head(5):
    subset = df[df["Store"] == store]
    plt.plot(subset["Date"], subset["Weekly_Sales"], label=f"Store {store}")
plt.legend()
plt.title("Example of Growth Stores")
plt.show()

#putting all the forecast stores into one dataframe for PowerBI purposes
big_subset=pd.DataFrame(columns=["Store", "Date", "Weekly_Sales"])
for store in trend_df_base.sort_values(by="Store", ascending=True)["Store"]:
    subset1 = df[df["Store"] == store]
    subset1 = subset1[["Store", "Date", "Weekly_Sales"]]
    subset = forecast_df[forecast_df["Store"] == store]
    subset.columns = ["Store", "Date", "Weekly_Sales"]
    subset = pd.concat([subset1, subset], ignore_index=True)
    big_subset = pd.concat([big_subset, subset], ignore_index=True)

##Highest Growth Based on Historical Data
for store in trend_df_base.sort_values(by="Trend_Slope", ascending=False).head(5)["Store"]:
    subset1 = df[df["Store"] == store]
    subset1 = subset1[["Store", "Date", "Weekly_Sales"]]
    subset = forecast_df[forecast_df["Store"] == store]
    subset.columns = ["Store", "Date", "Weekly_Sales"]
    subset = pd.concat([subset1, subset], ignore_index=True)
    plt.plot(subset["Date"], subset["Weekly_Sales"], label=f"Store {store}")
plt.axvline(forecast_df.head(1)["Date"], color="black", linestyle=":", label="")
plt.legend()
plt.title("Example of Growth Stores (Trend Based on Historical Data)")
plt.show()

##Highest Growth Based on Forecast Data
for store in trend_df.sort_values(by="Trend_Slope", ascending=False).head(5)["Store"]:
    subset = df[df["Store"] == store]
    subset1 = forecast_df[forecast_df["Store"] == store]
    subset = subset[["Store", "Date", "Weekly_Sales"]]
    subset1.columns = ["Store", "Date", "Weekly_Sales"]
    subset = pd.concat([subset, subset1], ignore_index=True)
    plt.plot(subset["Date"], subset["Weekly_Sales"], label=f"Store {store}")
plt.axvline(forecast_df.head(1)["Date"], color="black", linestyle=":", label="")
plt.legend()
plt.title("Example of Growth Stores (Trend Based on Forecast Data)")
plt.show()

# %%
