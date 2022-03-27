Project: No_Show_Appointments Data Analysis
Table of Contents
Introduction
Data Wrangling
Exploratory Data Analysis
Conclusions

Introduction
Dataset Description
Tip: This data contains informations for more than 100k patient in brazil and is focused on the question of whether the patient show or not show on thier appointment based on some factors: 1- Gender (male or female) 2- Schedule day (The day of the appointment) 3- Age 4- Nighbourhood (where patient live) 5- Scholarship (Brasilian welfare) 6- Sms recieved

Question(s) for Analysis
1- is the gender has a correlation with patient show !!

2- is the age has a correlation with patient show !!

3- is the neighbourhood has a correlation with patient show !!

4- is the scholarship has a correlation with patient show !!

5- if there is a certain disease has a correlation with patient show !!

6- is the sms has a correlation with patient show !!

# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import numpy as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
# Upgrade pandas to use dataframe.explode() function. 
!pip install --upgrade pandas==0.25.0
Requirement already up-to-date: pandas==0.25.0 in /opt/conda/lib/python3.6/site-packages (0.25.0)
Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /opt/conda/lib/python3.6/site-packages (from pandas==0.25.0) (2017.3)
Requirement already satisfied, skipping upgrade: numpy>=1.13.3 in /opt/conda/lib/python3.6/site-packages (from pandas==0.25.0) (1.19.5)
Requirement already satisfied, skipping upgrade: python-dateutil>=2.6.1 in /opt/conda/lib/python3.6/site-packages (from pandas==0.25.0) (2.6.1)
Requirement already satisfied, skipping upgrade: six>=1.5 in /opt/conda/lib/python3.6/site-packages (from python-dateutil>=2.6.1->pandas==0.25.0) (1.11.0)

Data Wrangling
Tip: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.

#load data to dataframe
df=pd.read_csv('noshow.csv')
df.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
PatientId	AppointmentID	Gender	ScheduledDay	AppointmentDay	Age	Neighbourhood	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	No-show
0	2.987250e+13	5642903	F	2016-04-29T18:38:08Z	2016-04-29T00:00:00Z	62	JARDIM DA PENHA	0	1	0	0	0	0	No
1	5.589978e+14	5642503	M	2016-04-29T16:08:27Z	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	0	0	0	0	0	0	No
2	4.262962e+12	5642549	F	2016-04-29T16:19:04Z	2016-04-29T00:00:00Z	62	MATA DA PRAIA	0	0	0	0	0	0	No
3	8.679512e+11	5642828	F	2016-04-29T17:29:31Z	2016-04-29T00:00:00Z	8	PONTAL DE CAMBURI	0	0	0	0	0	0	No
4	8.841186e+12	5642494	F	2016-04-29T16:07:23Z	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	0	1	1	0	0	0	No
#Lets discover the number of patients and thier characteristics
df.shape
(110527, 14)
There is 110527 patients and 14 columns

#Lets see if there is a missing data in our dataset
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 110527 entries, 0 to 110526
Data columns (total 14 columns):
PatientId         110527 non-null float64
AppointmentID     110527 non-null int64
Gender            110527 non-null object
ScheduledDay      110527 non-null object
AppointmentDay    110527 non-null object
Age               110527 non-null int64
Neighbourhood     110527 non-null object
Scholarship       110527 non-null int64
Hipertension      110527 non-null int64
Diabetes          110527 non-null int64
Alcoholism        110527 non-null int64
Handcap           110527 non-null int64
SMS_received      110527 non-null int64
No-show           110527 non-null object
dtypes: float64(1), int64(8), object(5)
memory usage: 11.8+ MB
##There is no missing data

df.groupby("Gender")
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f0048e0ab70>
#check for dublicated values 
df.duplicated().sum()
0
There is no duplicated values

df.describe()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
PatientId	AppointmentID	Age	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received
count	1.105270e+05	1.105270e+05	110527.000000	110527.000000	110527.000000	110527.000000	110527.000000	110527.000000	110527.000000
mean	1.474963e+14	5.675305e+06	37.088874	0.098266	0.197246	0.071865	0.030400	0.022248	0.321026
std	2.560949e+14	7.129575e+04	23.110205	0.297675	0.397921	0.258265	0.171686	0.161543	0.466873
min	3.921784e+04	5.030230e+06	-1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
25%	4.172614e+12	5.640286e+06	18.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
50%	3.173184e+13	5.680573e+06	37.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
75%	9.439172e+13	5.725524e+06	55.000000	0.000000	0.000000	0.000000	0.000000	0.000000	1.000000
max	9.999816e+14	5.790484e+06	115.000000	1.000000	1.000000	1.000000	1.000000	4.000000	1.000000
The mean of ages is 37 years

25% of patients are below 18 and most of them are below 55

The maximum age is 115 years

The minimum age is negative which doesnt make anysense so its probably a mistake.

# Converting the date information in string to datetime type:

df.AppointmentDay=pd.to_datetime(df.AppointmentDay)
df.ScheduledDay=pd.to_datetime(df.ScheduledDay)
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 110527 entries, 0 to 110526
Data columns (total 14 columns):
PatientId         110527 non-null float64
AppointmentID     110527 non-null int64
Gender            110527 non-null object
ScheduledDay      110527 non-null datetime64[ns, UTC]
AppointmentDay    110527 non-null datetime64[ns, UTC]
Age               110527 non-null int64
Neighbourhood     110527 non-null object
Scholarship       110527 non-null int64
Hipertension      110527 non-null int64
Diabetes          110527 non-null int64
Alcoholism        110527 non-null int64
Handcap           110527 non-null int64
SMS_received      110527 non-null int64
No-show           110527 non-null object
dtypes: datetime64[ns, UTC](2), float64(1), int64(8), object(3)
memory usage: 11.8+ MB
Data Cleaning
Tip: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).

#now we need to filter the data we will work on it so lets drop the columns we wont use.
df.drop(labels=["PatientId", "AppointmentID", 'AppointmentDay', 'ScheduledDay'],axis=1, inplace=True)
df.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Gender	Age	Neighbourhood	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	No-show
0	F	62	JARDIM DA PENHA	0	1	0	0	0	0	No
1	M	56	JARDIM DA PENHA	0	0	0	0	0	0	No
2	F	62	MATA DA PRAIA	0	0	0	0	0	0	No
3	F	8	PONTAL DE CAMBURI	0	0	0	0	0	0	No
4	F	56	JARDIM DA PENHA	0	1	1	0	0	0	No
#transform No-show to int (1,0) to better deal with it
df['No-show'].replace({'No': 0, 'Yes': 1}, inplace = True)
#We noticed some mistakes in columns names so lets fix it.
df.rename(columns={"Hipertension": "Hypertension","Handcap": "Handicap","No-show": "No_show"},inplace=True)
df.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Gender	Age	Neighbourhood	Scholarship	Hypertension	Diabetes	Alcoholism	Handicap	SMS_received	No_show
0	F	62	JARDIM DA PENHA	0	1	0	0	0	0	0
1	M	56	JARDIM DA PENHA	0	0	0	0	0	0	0
2	F	62	MATA DA PRAIA	0	0	0	0	0	0	0
3	F	8	PONTAL DE CAMBURI	0	0	0	0	0	0	0
4	F	56	JARDIM DA PENHA	0	1	1	0	0	0	0

Exploratory Data Analysis
Tip: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. Compute statistics and create visualizations with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must compare and show trends in the varied visualizations.

Number of people who showed up vs number of people who didnt )
#lets take a look at the histograms
df.hist(figsize=(15,12));
png

#value counts for NO-SHOW 
df['No_show'].value_counts()
0    88208
1    22319
Name: No_show, dtype: int64
Number of people who came to their appointment = 88208

Number of people who didnt came to their appointment = 22319

Aproximately around 20% of appointments has been missed.

ax = sns.countplot(x=df.No_show, data=df)
ax.set_title("Show/No_Show Patients")
plt.show()
png

Question No-1 :- is the gender has a correlation with showing up??
print("Unique Values in `Gender` ",  df.Gender.unique())
Unique Values in `Gender`  ['F' 'M']
#plot the number of females vs males that show for appointment
ax = sns.countplot(x=df.Gender, hue=df.No_show, data=df)
ax.set_title("Show/No_Show for Females and Males")
x_ticks_labels=['Female', 'Male']
ax.set_xticklabels(x_ticks_labels)
plt.show()
png

#plot the percentage of females vs males that show for appointment
ax = df.Gender.value_counts().plot(kind='pie',labels=['Female %','Male %']);
ax.set_title("Gender No Shows",fontsize=16);
png

As we can see that of the 88,000 patients that appeared, about 57,000 were female and 31,000 were male.

The ratio of the males who attended with the females seems to be the same

and therefore gender does not affect on showing up ( 24% females for 26% males ) didnt showed up.

Question No-2 :- is the age has a correlation with showing up??
#Age range for patients.
plt.figure(figsize=(16,2))
plt.xticks(rotation=90)
N = sns.boxplot(x=df.Age)
png

#This figure shows how many patients for each age.
plt.figure(figsize=(16,4))
plt.xticks(rotation=90)
ax = sns.countplot(x=df.Age)
ax.set_title("No of appointments by age")
plt.show()
png

The above graph shows a peak for the infants.

Then it starts to be uniform for higher ages.

Then starts to be skewed right at the age value 60.

Question No-3 :- is the neighborhood has a correlation with showing up??
#Let's see the patients count for each neighborhood.
plt.figure(figsize=(16,4))
plt.xticks(rotation=90)
ax = sns.countplot(x=df.Neighbourhood, hue=df.No_show)
ax.set_title("Show/No_Show by Neighbourhood")
plt.show()
png

From the graph we can see that the number of patients for some neighbourhood's is very high but nearly the same for the other neighbourhood's.

Question No-4 :- is the scholarship has a correlation with showing up??
df['Scholarship'].value_counts()
0    99666
1    10861
Name: Scholarship, dtype: int64
x = sns.countplot(x=df.Scholarship, hue=df.No-show, data=df)
ax.set_title("Show/No-Show for Scholarship")
x_ticks_labels=['No Scholarship', 'Scholarship']
ax.set_xticklabels(x_ticks_labels)
plt.show()
---------------------------------------------------------------------------

AttributeError                            Traceback (most recent call last)

<ipython-input-24-4034eaa642d0> in <module>()
----> 1 x = sns.countplot(x=df.Scholarship, hue=df.No-show, data=df)
      2 ax.set_title("Show/No-Show for Scholarship")
      3 x_ticks_labels=['No Scholarship', 'Scholarship']
      4 ax.set_xticklabels(x_ticks_labels)
      5 plt.show()


/opt/conda/lib/python3.6/site-packages/pandas/core/generic.py in __getattr__(self, name)
   5178             if self._info_axis._can_hold_identifiers_and_holds_name(name):
   5179                 return self[name]
-> 5180             return object.__getattribute__(self, name)
   5181 
   5182     def __setattr__(self, name, value):


AttributeError: 'DataFrame' object has no attribute 'No'
From visualization we can see that there are around 100,000 patients without Scholarship and out of them around 80% have come for the visit. Out of the 10,800 patients with Scholarship around 75% of them have come for the visit. So, Scholarship feature could help us in determining if a patient will turn up for the visit after an appointment.

Question No-5 :- if there is a certain disease has a correlation with patient show !!
#lets see the patients diagnostic relation with show for appointment
def disease(i):
    if i == "Hypertension":
        df["Hypertension"].value_counts()
        ax = sns.countplot(x=df.Hypertension, hue=df.No_show, data=df)
        ax.set_title("Show/No_Show for Hypertension")
        x_ticks_labels=['No Hypertension', 'Hypertension']
        ax.set_xticklabels(x_ticks_labels)
        plt.show()
    elif i == "Diabetes":
        df["Diabetes"].value_counts()
        ax = sns.countplot(x=df.Diabetes, hue=df.No_show, data=df)
        ax.set_title("Show/No_Show for Diabetes")
        x_ticks_labels=['No Diabetes', 'Diabetes']
        ax.set_xticklabels(x_ticks_labels)
        plt.show()
    elif i == "Alcoholism":
        df["Alcoholism"].value_counts()
        ax = sns.countplot(x=df.Alcoholism, hue=df.No_show, data=df)
        ax.set_title("Show/No_Show for Alcoholism")
        x_ticks_labels=['No Alcoholism', 'Alcoholism']
        ax.set_xticklabels(x_ticks_labels)
        plt.show()
    else:
        df["Handicap"].value_counts()
        ax = sns.countplot(x=df.Handicap, hue=df.No_show, data=df)
        ax.set_title("Show/No_Show for Handicap")
        x_ticks_labels=['No Handicap', 'Handicap']
        ax.set_xticklabels(x_ticks_labels)
        plt.show()
disease("Hypertension")
disease("Diabetes")
disease("Alcoholism")
disease("Handicap")
1- By visualizing, we can see that there are about 88,000 patients suffering from high blood pressure and about 78% of them attended the visit. Of 21801 patients with no high blood pressure, about 85% came to visit. Therefore, the high blood pressure feature can help us determine whether a patient will show up on a post-appointment visit.

2- From visualization we can see that there are about 102,000 diabetics and about 80% of them attended the visit. Of the 7943 diabetic patients, about 83% came to visit. Therefore, the diabetes feature can help us determine whether a patient will attend the post-appointment visit.

3- By visualizing, we can see that there are about 107,000 patients who do not suffer from alcoholism and about 80% of them attended the visit. Of the 3360 patients with alcohol addiction, about 80% attended the visit. Since the rate of visits for non-alcoholic patients is the same, this may not help us determine whether or not the patient is coming for a visit.

4- Through visualization, we can see that there are about 108,000 unobstructed patients and about 80% of them have come for a visit. Since we see a clear distinction between different levels of disability, this feature will help us determine if a patient will come for a visit after making an appointment.

Question No-6 :- is the SMS_Received has a correlation with showing up??
#Number of patients recieved sms
df['SMS_received'].value_counts()
#the plot of patients that recieved sms vs patients who didn't
ax = sns.countplot(x=df.SMS_received, hue=df.No_show, data=df)
ax.set_title("Show/No_Show for SMS_received")
x_ticks_labels=['No SMS_received', 'SMS_received']
ax.set_xticklabels(x_ticks_labels)
plt.show()
Through visualization, we can see that there are about 75,000 patients who did not receive text messages, and about 84% of them attended the visit. Of the 35,500 patients who received text messages, about 72% attended the visit. This feature will help us determine if a patient will come for a visit after scheduling an appointment.


Conclusions:
there are a very strong relation between sms_reminder and people that showed_up so it suppose to send sms-message regulary.

men are more persistent to go to the appoinments.

age is the most important factor.

The patients are 37 years on average. 25% of patients are below 18 and most of them are below 55.

Most of the patients are not alcoholics.

Most patients do not have hypertension diagnosed.

On average, 20% of appointments were missed.

Finally we can say that there is no significant factor that we can define to know the reason of missing appointments.

Limitations:
Missing features that could be useful to get more sure what is the most feature that impacts showing to the appointment such as if the patient is employeed or not , or whether the patient have a series medical issue or not.
there we some illogical data such as patients with age 0 or less
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
