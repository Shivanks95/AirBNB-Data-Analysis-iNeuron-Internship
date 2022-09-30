#!/usr/bin/env python
# coding: utf-8

# ![imageedit_12_3637837732.png](attachment:imageedit_12_3637837732.png)

# # AIRBNB PYTHON VISUALIZATION

# ## IMPORTATION OF PYTHON LIBRARIES 

# In[82]:


# First, Let's Import the "Libraries" -

import pandas as pd # For Data Frame

import numpy as np # For operation on data

from matplotlib import pyplot as plt # For Visuallization of Graphs

import seaborn as sns # For Visualization Of Graphs 

import plotly.express as px # For Visualization Of Graphs.

import cufflinks as cf # For Visualization Of Graphs.
cf.set_config_file(offline=True)

import dataprep # EDA Report Library 
from dataprep.eda import create_report

import dtale # EDA Report Library

import folium # For Visualization of Latitude and Longitude on MAP



import warnings # For ignorance of any kind of unnecessary warnings 
warnings.filterwarnings(action="ignore")


# ## READING CSV FILE 

# In[2]:


df_airbnb = pd.read_csv('airbnb prices.csv')


# In[3]:


df_airbnb.shape


# In[4]:


df_airbnb.head()


# In[5]:


df_airbnb.columns


# In[6]:


df_airbnb.dtypes


# # Exploratory Data Analysis(EDA) AND Feature engineering (FE):

# ## REPORT CREATION USING DATAPREP LIBRARY

# In[83]:


create_report(df_airbnb)


# # SAVING THE REPORT IN HTML FORMAT

# In[84]:


report = create_report(df_airbnb)
report.save('AirBnB_EDA_DataPrep_Final_Report_Internship_iNeuron')


# In[85]:


df_airbnb.head()


# In[86]:


df_airbnb.shape


# # DATA CLEANING PROCESS (MISSING VALUE HANDLE)

# In[7]:


plt.figure(figsize=(10,6))
sns.heatmap(df_airbnb.isnull(), yticklabels = False, cmap = 'viridis', cbar = False)


# In[8]:


df_airbnb.isnull().sum()


# ### Findings:

# ##### In the above plot and output we can clearly see that there are major missing of values in features i.e. "country", "borough", "bathroom" and "minstay" which counts 18723 also we can see that the "name" feature is missing almost 52 values. Further we will try to provide values where necessary. 
# 

# In[9]:


Mode = df_airbnb["country"].mode()
Mode


# In[10]:


df_airbnb['country'].fillna(value = 'Netherlands', axis = 'index', inplace = True)


# In[11]:


df_airbnb['country'].isnull().sum()


# In[12]:


Mode = df_airbnb["borough"].mode()
Mode
df_airbnb['borough'].fillna(value = 'Centrum', axis = 'index', inplace = True)
df_airbnb['borough'].isnull().sum()


# In[13]:


Mode = df_airbnb["bathrooms"].mode()
Mode
df_airbnb['bathrooms'].fillna(value = '1', axis = 'index', inplace = True)
df_airbnb['bathrooms'].isnull().sum()


# In[14]:


Mode = df_airbnb["minstay"].mode()
Mode
df_airbnb['minstay'].fillna(value = '1 Day', axis = 'index', inplace = True)
df_airbnb['minstay'].isnull().sum()


# In[15]:


df_airbnb['name'].isnull().sum()


# In[16]:


df_airbnb['name'].value_counts()


# In[17]:


df_airbnb['name'].fillna(value = 'Private Room/ Shared Room/ Apartment', axis = 'index', inplace = True)
df_airbnb['name'].isnull().sum()


# 
# ##### With the help of the above codes for all the features having missing values we have provided the certain values without effecting the dataset so that we can have zero missing values without dropping any feature for better analysis.
# 
# 
# 
# 

# ##### lets move forward.

# # DATA CLEANING PROCESS BY ADDING VALUES SUCCESSFULLY IMPLIMENTED

# In[18]:


plt.figure(figsize=(10,6))
sns.heatmap(df_airbnb.isnull(), yticklabels = False, cmap = 'viridis', cbar = False)


# In[19]:


df_airbnb


# ##### Above we can see there is no values left having null or NaN values.

# # DATA ANALYSIS 

# In[20]:


df_airbnb.dtypes


# ### Lets find out the Top Earners:
# ### We will find out the top 20 earners on the basis of names as it represent names of the properties available in AirBNB Netherlands.

# In[21]:


df_TopEarners = df_airbnb.groupby(['name'])['price'].mean().reset_index().sort_values(by='price', ascending=False)[0:20]
df_TopEarners = df_TopEarners.round(decimals = 2)
df_TopEarners


# ### We will now visualize the above dataset into a simple bar chart using Plotly Express library

# In[22]:


fig = px.bar(data_frame = df_TopEarners, 
             y = df_TopEarners['price'], 
             x = df_TopEarners['name'],
             color = 'price',
            text = 'price',
            labels = {'name':'Name of Host', 'price':'Average Earnings'})

fig.update_layout(template = 'plotly_dark', title_text = "TOP EARNERS WITH RESPECT TO NAME AND PRICE")
fig.show()


# # Conclusion:

# ###### We can clearly see that the Top 3 Earners are: 
# ###### 1. "Zonnige woonboot,centraal en rustig"	with the highest earning of € 6000
# ###### 2. "One public bedroom" being on second highest with the earning of € 3770
# ###### 3. "AmsterdamBase" retains the third position with the earning of € 1920
# 
# ###### There is no feature of Monthly Earning in the dataframe so we are unable to fetch any relation between Monthly Earnings and Price. 

# # Maximum Number Of Booking

# ### In this Block of code we are going to perform analysis on 2 factors:  
# 
# #### 1. Any particular location getting maximum number of bookings.
# 
# #### 2. Price relation with respect to location

# ### Lets see which location gets the maximum reviews which will help us understand the maximum number of booking on  a perticular location.

# In[23]:


df_airbnb['location'].unique
# To check weather the data having unique values or not.


# In[24]:


df_airbnb


# In[25]:


df_MaxBooking = df_airbnb.groupby(['location'])['reviews'].mean().reset_index().sort_values(by='reviews', ascending=False)[0:20]
df_MaxBooking = df_MaxBooking.round(decimals = 2)
df_MaxBooking


# In[26]:


df_airbnb['name'].unique


# ### Now we will plot the simple bar chart for the above output using Plotly Express library.

# In[27]:


fig = px.bar(df_MaxBooking, x="location", y="reviews", color="reviews", barmode="group")
fig.show()


# #### Since, it is not possible to understand the chart as on the x-axis location are not readable we will now try to plot the chart by using name feature for better understanding.

# In[28]:


df_MaxBooking_name = df_airbnb.groupby(['name'])['reviews'].mean().reset_index().sort_values(by='reviews', ascending=False)[0:20]
df_MaxBooking_name = df_MaxBooking_name.round(decimals = 2)
df_MaxBooking_name


# In[29]:


df_MaxBooking_name,df_MaxBooking[0:20]


# In[30]:


fig = px.bar(df_MaxBooking_name, x="name", y="reviews", color="reviews", barmode="group")
fig.show()


# ###  Price relation with respect to location

# In[31]:


df_airbnb


# In[32]:


df_price_location = df_airbnb.groupby(['location'])['price'].mean().reset_index().sort_values(by='price', ascending=False)[0:20]
df_price_location = df_price_location.round(decimals = 2)
df_price_location


# In[33]:


fig = px.scatter(data_frame = df_price_location,
           y='price',
           x='location',
          color='price',
          size = 'price')

fig.update_layout(template = 'plotly_dark', title_text = "Relationship between Location and Prices")
fig.show()


# ### As we need more clearity on the price vs location we will going to perform analysis on the basis of negiborhood and price.

# In[34]:


df_price_neighborhood = df_airbnb.groupby(['neighborhood'])['price'].mean().reset_index().sort_values(by='price', ascending=False)[0:20]
df_price_neighborhood = df_price_neighborhood.round(decimals = 2)
df_price_neighborhood


# #### We will now plot a Scatter graph for better visuallization.

# In[35]:


fig = px.scatter(data_frame = df_price_neighborhood,y='price',x='neighborhood',color='price',size = 'price')
fig.update_layout(template = 'plotly_dark', title_text = "Relationship between Neighborhood(location) and Average Prices")
fig.show()


# #### As we can see that the above Scatter Plot is understandable we will now try one more visuallization for convinient analysis. Lets try to plot the above output using Line Graph of Cufflinks Library (Plotly).

# In[36]:


df_price_neighborhood.iplot(x="neighborhood",y="price",
               xTitle="Negibourhood", yTitle="Price", title="Open Price From Apr,2019 - Mar,2020")


# ## Conclusion: 

# ### In the above analysis we have found that the price and location have a unique relation as the prices are unique on the basis of location or name .

# ### On the other hand we can clearly see the anlysis of negibourhood vs price in the Scatter Plot and Line Graph. In which it came to our knowledge that the "Centrum West"  has the highest average price i.e. "€ 208.31" also "Centrum Oost" is on second position with the average price of "€ 201.22"	while "Noord-West / Noord-Midden" retains the 3rd position in terms of average price of "€ 182.73". 

# ## Lets move further and try to find out the Relationship between Quality and Price.

# In[37]:


df_airbnb


# ### As we can see the dataset do not have any Quality feature, we will now try to find relationship on the basis of Overall Satisfacton which is in other words represents the Quality feature of the dataset.

# In[38]:


df_min_price = df_airbnb.price.min(),df_airbnb.overall_satisfaction.max()
df_min_price


# In[39]:


df_max_price = df_airbnb.price.max(),df_airbnb.overall_satisfaction.min()
df_max_price


# In[40]:


df_minmax_price=df_min_price,df_max_price
df_minmax_price


# In[41]:


df_price_overall_satisfaction = df_airbnb.groupby(['price'])['overall_satisfaction'].mean().reset_index().sort_values(by='overall_satisfaction')[0:400]
df_price_overall_satisfaction = df_price_overall_satisfaction.round(decimals = 2)
df_price_overall_satisfaction


# #### We have already found the relationship of Quality and price, lets now see the relationship on the basis of Scatter Plot. 

# In[42]:


fig = px.scatter(data_frame = df_price_overall_satisfaction,
           y='overall_satisfaction',
           x='price',
          color='overall_satisfaction',
          size = 'price')

fig.update_layout(template = 'plotly_dark', title_text = "Relationship between Overall Satisfaction and Prices")
fig.show()


# ## Conclusion:

# ### In the above scatter plot we can clearly see that if the "Price" is high than the "Overall Satisfaction (Quality)" is less and where "Price" is low the "Overall Satisfaction (Quality)" is High, for an instance lets take the example where Price = "€ 112.0" the Overall Satisfaction is "5.0"  where if we talk about the Price = "€ 6000.0" the Overall Satisfaction is "0.0"
# 

# # Price VS Amenitites

# In[43]:


df_price_amenitites = df_airbnb.groupby(['name'])['price'].mean().reset_index().sort_values(by='price', ascending=False)[0:20]
df_price_amenitites = df_price_amenitites.round(decimals = 2)
df_price_amenitites


# In[44]:


fig = px.scatter(data_frame = df_price_amenitites,
           y='price',
           x='name',
          color='price',
          size = 'price')

fig.update_layout(template = 'plotly_dark', title_text = "Prices Vs Amenities(Name)")
fig.show()


# # Price VS Location

# In[45]:


df_price_neighborhood1 = df_airbnb.groupby(['neighborhood'])['price'].mean().reset_index().sort_values(by='price', ascending=False)[0:20]
df_price_neighborhood1 = df_price_neighborhood1.round(decimals = 2)
df_price_neighborhood


# #### From the above output we can see the Average Price and Neighborhood(Location) in table format, lets try to create a donut chart for better visualization. For this we will use the cufflinks library.

# In[46]:


df_price_neighborhood1.iplot(kind="pie",
                             labels="neighborhood",
                             values="price",
                             textinfo='percent+label', hole=.4,
                             )


# ## Conclusion:

# ### Price Vs Amenities

# ### In the Price Vs Amenities, we can see that the if amenities is of 5-Star grade like Boat-House, Bunglow, and Resort, etc. the prices are high, where the amenities are 2-Star grade like Room, Shared-room, Dormatries, Hostel, etc. the prices are low.

# ### Price Vs Location

# ### if we talk about the Price Vs Location where we are taking the average price for a perticular location to know about the grade of area the price is high if we talk about the major city location or negibourhood.  
# 
# ### For better understanding lets take an view and analyse the above donut chart. We can clearly see that the percentage of average price in  "Centrum West" holds 6.83%, which is higher than all the other negibourhood. 
# 
# ### And also, we can see the share of "Gaasperdam / Driemond" is only 3.98% which is less as compare to other negibourhood.

# # Other Findings:

# In[47]:


df_airbnb


# ### Lets Find out the relationship between Room_type and negibourhood

# In[48]:


data_room = df_airbnb.groupby(['neighborhood','room_type'])['price'].agg('mean').to_frame()
data_room.sort_values( by='price', ascending=True, inplace=False)
data_room.head()


# In[49]:


df_data_room = df_airbnb.groupby(['room_type','neighborhood'], as_index=False)['price'].agg('mean')
df_data_room.sort_values(by='price', ascending=False, inplace=True)
df_data_room = df_data_room.round(decimals = 2)
df_data_room.head(40)


# #### We will now try to plot the above output in a Bar Plot for better understanding using Seaborn library.

# In[50]:


plt.figure(figsize=(22,8))
sns.barplot(data = df_data_room, x = 'neighborhood', y = 'price', hue = 'room_type')
plt.xticks(rotation = 90)
plt.title('Avg. Price vs Neighborhood & Room Type', fontsize=16, fontweight='bold', fontstyle='italic')
plt.xlabel('Neighborhood', fontsize=12, fontweight='bold')
plt.ylabel('Avg. Price', fontsize=12, fontweight='bold')
plt.legend(title = 'Room Type', loc='upper right')
plt.show()


# ## Conclusion:

# ### From the above Bar Plot Visuallization we can clearly see that the "Centrum West" Neigbourhood is having average price for all the 3 Room Types available in the dataset is very high in comperison with any other neibourhood. 

# #### Lets find out the preference of guest in comperison with Room Type.

# In[51]:


df_roomtype = df_airbnb['room_type'].value_counts()
df_roomtype = df_roomtype.reset_index()#(inplace=True)
df_roomtype.columns = ['Room_type', 'Count']
df_roomtype.head()


# In[52]:


fig = px.density_contour(df_roomtype, y = 'Room_type', x ='Count', title='Preference of Guests w.r.t. Room Type', color="Room_type")

fig.show()


# ## Conclusion:

# ### From the above Contour Map representation we can clearly see that the "Entire Home/ Apartment" is prefered the most by the guest where the preference of "Private and Shared Room" is less.	

# ### Lets find out the Cheapest AirBNB

# In[53]:


df_cheapest = df_airbnb[['name', 'price']].sort_values(by = 'price').nsmallest(20, columns = 'price')
df_cheapest = df_cheapest.round()
df_cheapest


# In[54]:


fig = px.bar(data_frame = df_cheapest, y = df_cheapest['name'][0:20], x = df_cheapest['price'][0:20], 
             color = df_cheapest['price'][0:20], 
             text = df_cheapest['price'][0:20], 
             labels = {'x':'Cheapest Price', 'y':'Names of AirBNB'}, orientation= 'h' )

fig.update_layout(template = 'seaborn', title_text = "Affordable/Budgeted AirBnB")
fig.show()


# ## Conclusion:

# ### From the above Bar Chart we can clearly see that in the top 20 cheapest AirBNB "Kattenoppas gezocht" is having the lowest price of  "€ 12". 

# ### Lets find out the Most Expensive AirBNB

# In[55]:



df_expensive = df_airbnb.groupby(['name', 'room_type'])['price'].mean().reset_index().sort_values(by='price', ascending=False)[0:20]
df_expensive


# In[56]:


fig = px.bar(data_frame = df_expensive, 
             x = df_expensive['price'], 
             y = df_expensive['name'],
             color = 'price',
            text = 'price',
            labels = {'name':'Name of the AirBnB', 'price':'Avg. Price'})

fig.update_layout(template = 'seaborn', title_text = "Top 20 Most Expensive AirBnB's")
fig.show()


# ## Conclusion:

# ### From the above Bar Chart we can clearly see that in the top 20 most expensive AirBNB  "Zonnige woonboot,centraal en rustig" is having the most expensive price of  "€ 6000". 

# ### Lets find out the top 5 Location having most booking on AirBNB.

# In[57]:


df_airbnb['neighborhood'].unique()


# In[58]:


df_airbnb['neighborhood(Pre-processed)'] = df_airbnb['neighborhood']


# In[59]:


df_airbnb


# In[60]:


data_neighborhood = df_airbnb['neighborhood(Pre-processed)'].value_counts()[0:5]
data_neighborhood


# In[61]:


fig = px.pie(data_frame = data_neighborhood, 
             names = data_neighborhood.index, 
             values = data_neighborhood, 
             title = 'Top 5 Location (Negibourhood) having Maximum Number of Bookings')

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[62]:


df_airbnb["neighborhood"].value_counts()


# In[63]:


fig = plt.subplots(figsize = (20,8))
sns.set_style = "darkgrid"

sns.countplot(x = df_airbnb["neighborhood"], order = df_airbnb["neighborhood"].value_counts().index, palette = "plasma")

plt.xticks(rotation = 90)
plt.title("Location (Neighbourhood) w.r.t Maximum Number of Bookings", fontsize = 14, fontweight = 'bold', fontstyle = 
          'italic')
plt.xlabel('Neighbourhood Location')
plt.ylabel('Number of Bookings')

plt.show()


# ## Conclusion:

# ### From the above two Visualization i.e. A Pie Chart and A Bar Plot we can do analysis that "De Baarsjes / Oud West" is the place having maximum number of booking i.e. "29.8%" on Pie Chart and "3289 (Count)" on Bar Plot representation.

# ### Price Relationship with respect to negibourhood

# In[64]:


df_price_relation  = df_airbnb[['neighborhood', 'price']].sort_values(by = 'price', ascending = False)

df_price_relation


# In[65]:


plt.figure(figsize=(20,12))
sns.scatterplot(data = df_price_relation,x = 'neighborhood', y = 'price', hue = 'neighborhood', size = 'price', legend = False)

plt.xticks(rotation = 'vertical')

plt.title('Price relation w.r.t. Location', fontsize = 16, fontweight = 'bold', fontstyle = 'italic')
plt.xlabel('Neighbourhood Location', fontsize = 12, fontweight = 'bold')
plt.ylabel('Price', fontsize = 12, fontweight = 'bold')
plt.show()


# ## Conclusion:

# ###  In the above Scatter Plot we can see the relationship between Price and Location an it came to our knowledge that the prime location of amsterdam having higher rates as compare to other locations.
# 

# ### Lets try to Map the Latitude and Longitude Features of the data set Using Folium Library.

# In[66]:


df_airbnb


# In[67]:


df_lat_long=df_airbnb.groupby(['latitude','longitude'], as_index=True)
df_lat_long


# In[80]:


AirBNB_Map = folium.Map(prefer_canvas=True)

def plotDot(point,axis):
    df_lat_long
    folium.CircleMarker(location=[point.latitude, point.longitude],
                        radius=2,
                        weight=5).add_to(AirBNB_Map)
df_lat_long.apply(plotDot, axis = 1)
AirBNB_Map.fit_bounds(AirBNB_Map.get_bounds())
AirBNB_Map


# ## Conclusion:

# ### On the above output we can see a Beautifull map of Neatherlands, Amsterdam which with the help of Bule dots we can see all the 18723 AirBNB of Amsterdam in the above visualization. 

# ## FINAL REPORT CREATION USING D-TALE LIBRARY (EDA)

# In[81]:


dtale_eda=dtale.show(df_airbnb)
dtale_eda
#dtale_eda.open_browser()


# In[69]:


df_last_Modified_DateTime=df_airbnb['last_modified']


# In[70]:


df_airbnb['last_modified_datetime'] = pd.to_datetime(df_airbnb['last_modified'])


# In[71]:


df_airbnb


# In[72]:


df_airbnb ['last_modified_time'] = df_airbnb ['last_modified_datetime'].dt.time


# In[73]:


df_airbnb


# In[74]:


df_airbnb['last_modified_time']


# In[75]:


df_lastmodified_location  = df_airbnb[['neighborhood', 'last_modified_time']].sort_values(by = 'last_modified_time', ascending = False)[0:20]

df_lastmodified_location


# In[76]:


df_last_Modified_roomtype = df_airbnb.groupby(['room_type','neighborhood'], as_index=False)['last_modified_time'].agg('count')
df_last_Modified_roomtype.sort_values(by='last_modified_time', ascending=False, inplace=True)
df_last_Modified_roomtype.head(40)


# In[78]:


df_lastmodified_location.iplot(x="last_modified_time",y="neighborhood",
                               xTitle="last_modified_time", 
                               yTitle="neighborhood", 
                               title="Most Last Modified Location")


# In[ ]:




