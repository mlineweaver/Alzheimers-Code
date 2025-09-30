import csv
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import warnings

#To open csv files
with open("UpdatedMetaData (2).csv", newline="") as f:
    reader = csv.reader(f)
    headers = next(reader) #Get the first row
    for h in headers:
        print(h)
with open("UpdatedLuminex.csv", newline="") as f:
    reader = csv.reader(f)
    headers = next(reader)
    for h in headers:
        print(h)

#1) BUILD OUR CLASS BY COMBINING TWO .csv FILES OF DATA

class Patient: 

    all_patients = []

    death_age = []

    education_lvl = {}

    def __init__(self, DonorID, ABeta40: float , ABeta42: float, tTau: float, pTau: float):

        self.DonorID = DonorID
        self.ABeta40 = ABeta40
        self.ABeta42 = ABeta42
        self.tTau = tTau
        self.pTau = pTau
        self.sex = None
        self.death_age = None
        self.ed_lvl = None
        self.cog_stat = None
        self.age_symp_on = None
        self.age_diag = None 
        self.head_inj = None
        self.thal_score = None
        self.mmse = None
        Patient.all_patients.append(self)

    def __repr__(self):
        return f"{self.DonorID} | sex: {self.sex} | ABeta40 {self.ABeta40} | tTau {self.tTau} | pTau {self.pTau} | Death Age {self.death_age} | Thal Score {self.thal_score}"

    def get_id(self):
        return self.DonorID

    def get_ABeta42(self):
        return self.ABeta42
    
    def get_thal(self):
        return self.thal_score
    
    def get_death_age(self):
        return self.death_age


    @classmethod
    def combine_data(cls, filename: str):
            with open(filename, encoding="utf8") as f:
                reader = csv.DictReader(f)
                rows_of_patients = list(reader)
                #for line in csv create object
                for row in range(len(rows_of_patients)):
                    if Patient.all_patients[row].DonorID == rows_of_patients[row]["Donor ID"]:
                        if rows_of_patients[row]["Sex"] != "":
                            Patient.all_patients[row].sex = rows_of_patients[row]["Sex"]

                        if rows_of_patients[row]["Age at Death"] != "":
                            Patient.all_patients[row].death_age = int(rows_of_patients[row]["Age at Death"])

                        if rows_of_patients[row]["Highest level of education"] != "":
                            Patient.all_patients[row].ed_lvl = rows_of_patients[row]["Highest level of education"]

                        if rows_of_patients[row]["Cognitive Status"] != "":
                            Patient.all_patients[row].cog_stat = rows_of_patients[row]["Cognitive Status"]

                        if rows_of_patients[row]["Age of onset cognitive symptoms"] != "":
                            Patient.all_patients[row].age_symp_on = int(rows_of_patients[row]["Age of onset cognitive symptoms"])

                        if rows_of_patients[row]["Age of Dementia diagnosis"] != "":
                            Patient.all_patients[row].age_diag = int(rows_of_patients[row]["Age of Dementia diagnosis"])

                        if rows_of_patients[row]["Known head injury"] != "":
                            Patient.all_patients[row].head_inj = rows_of_patients[row]["Known head injury"]

                        if rows_of_patients[row]["Thal"] != "":
                            Patient.all_patients[row].thal_score = int(rows_of_patients[row]["Thal"].split()[1])

                        if rows_of_patients[row]["Last MMSE Score"] != "":
                            Patient.all_patients[row].mmse = rows_of_patients[row]["Last MMSE Score"]
                    else:
                        warnings.warn("IDs do not match.")
   
    @classmethod
    def instantiate_from_csv(cls, filename: str, other_file: str):
    #open csv and create list of all rows
        with open(filename, encoding="utf8") as f:
            reader = csv.DictReader(f)
            rows_of_patients = list(reader)
            #for line in csv create object
            for row in rows_of_patients:
                Patient(
                    DonorID = row['Donor ID'],
                    ABeta40 = float(row['ABeta40 pg/ug']),
                    ABeta42 = float(row['ABeta42 pg/ug']),
                    tTau = float(row['tTAU pg/ug']),
                    pTau = float(row['pTAU pg/ug'])
                )
            Patient.all_patients.sort(key = Patient.get_id)
            Patient.combine_data(other_file)
    @classmethod
    def sort_ed(cls):
        for patient in Patient.all_patients:
            Patient.education_lvl.update({patient.ed_lvl: []})
        for patient in Patient.all_patients:
            Patient.education_lvl[patient.ed_lvl].append(patient)

    @classmethod
    def subsort_thal(cls):
        for key in Patient.education_lvl:
            values = Patient.education_lvl.get(key)
            values.sort(key = Patient.get_thal)
            Patient.education_lvl.update({key: values})
    @classmethod
    def filter(cls, list, ABeta40:float ="any", ABeta42:float ="any", tTau:float= "any", pTau:float ="any", sex:str ="any", death_age:int ="any", ed_lvl:str ="any", cog_stat:str ="any", age_symp_on:int ="any", age_diag:int ="any", head_inj:str ="any", thal_score:int ="any"):
        all_patients = list
        remove_list = []
        attr_list = (
                    ABeta40,
                    ABeta42,
                    tTau,
                    pTau,
                    sex,
                    death_age,
                    ed_lvl,
                    cog_stat,
                    age_symp_on,
                    age_diag,
                    head_inj,
                    thal_score
                    )
        attr_name = (
                    "ABeta40",
                    "ABeta42",
                    "tTau",
                    "pTau",
                    "sex",
                    "death_age",
                    "ed_lvl",
                    "cog_stat",
                    "age_symp_on",
                    "age_diag",
                    "head_inj",
                    "thal_score"
                    )
        for attr in range(len(attr_list)):
            if attr_list[attr] != "any":
                for patient in all_patients:
                    if getattr(patient,attr_name[attr]) != attr_list[attr]:
                        remove_list.append(patient)
                all_patients = [patient for patient in all_patients if patient not in remove_list]
                remove_list.clear()
        
        return all_patients
#2) PRINT OUR LIST OF PATIENTS

Patient.instantiate_from_csv('UpdatedLuminex.csv', 'UpdatedMetaData (2).csv')

for patient in Patient.all_patients:
    print(patient)
#3) SORT OUR LIST OF PATIENTS

Patient.all_patients.sort(key=Patient.get_ABeta42, reverse=False)

for patient in Patient.all_patients:
    print(patient)
#4) BUILD DICTIONARIES TO DO SORTING AND SUB-SORTING

@classmethod
def sort_ed(cls):
    for patient in Patient.all_patients:
        Patient.education_lvl.update({patient.ed_lvl: []})
    for patient in Patient.all_patients:
        Patient.education_lvl[patient.ed_lvl].append(patient)

@classmethod
def subsort_thal(cls):
    for key in Patient.education_lvl:
        values = Patient.education_lvl.get(key)
        values.sort(key = Patient.get_thal)
        Patient.education_lvl.update({key: values})
#5) SORT THE SUB-LISTS WITHIN THE LIST AND PRINT IT


Patient.sort_ed()
Patient.subsort_thal()

for key in Patient.education_lvl:
    for patient in Patient.education_lvl.get(key):
        print(patient)
    print()


#7) USE OUR FILTER TO PULL OUT AND COUNT SUB-SETS OF PATIENTS THAT ARE FILTERED

fem_healty_patients = range(len(Patient.filter(Patient.all_patients, sex = "Female", cog_stat = "No dementia")))
male_healthy_patients = range(len(Patient.filter(Patient.all_patients, sex = "Male", cog_stat = "No dementia")))
fem_diseased_patients = range(len(Patient.filter(Patient.all_patients, sex = "Female", cog_stat = "Dementia")))
male_diseased_patients = range(len(Patient.filter(Patient.all_patients, sex = "Male", cog_stat = "Dementia")))

print(f'Female Healthy Patients = {len(fem_healty_patients)} | Male Healthy Patients = {len(male_healthy_patients)}')
print(f'Female Diseased Patients = {len(fem_diseased_patients)} | Male Diseased Patients = {len(male_diseased_patients)}')

#7) PLOT A BAR GRAPH OF THE SORTED DATA
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import statistics 


# Patient.instantiate_from_csv('UpdatedLuminex.csv', 'UpdatedMetaData.csv')


ABeta42_fem_vals = []
ABeta42_male_vals = []

for patient in Patient.filter(Patient.all_patients, sex = "Female"):
     ABeta42_fem_vals.append(patient.ABeta42)
for patient in Patient.filter(Patient.all_patients, sex = "Male"):
     ABeta42_male_vals.append(patient.ABeta42)

x_fem_bar = (statistics.mean(ABeta42_fem_vals))
x_male_bar = (statistics.mean(ABeta42_male_vals))

ABeta_fem_stdev = (statistics.stdev(ABeta42_fem_vals))
ABeta_male_stdev = (statistics.stdev(ABeta42_male_vals))

print(f'x_fem_bar = {x_fem_bar}, ABeta_fem_stdev {ABeta_fem_stdev}')
print(f'x_male_bar = {x_male_bar}, ABeta_male_stdev {ABeta_male_stdev}')

x_fem_vals = range(len(Patient.filter(Patient.all_patients, sex = "Female")))
x_male_vals = range(len(Patient.filter(Patient.all_patients, sex = "Male")))

sex_cols = ['Female', 'Male']
mean_sex_ABeta42 = [x_fem_bar, x_male_bar]
stdev_sex_ABeta42 = [ABeta_fem_stdev, ABeta_male_stdev]
colors = ["pink", "blue"]
yerr = [np.zeros(len(mean_sex_ABeta42)), stdev_sex_ABeta42]

# Equal variance assumed
t_stat, p_val = stats.ttest_ind(ABeta42_fem_vals, ABeta42_male_vals)
print("t-statistic:", t_stat)
print("p-value:", p_val)

plt.bar(sex_cols, mean_sex_ABeta42, yerr=yerr, capsize=10, color=["red", "blue"])
plt.title("ABeta42 Levels")
plt.xlabel("Sex")
plt.ylabel("Abeta42")
plt.show()

#8) PLOT A BAR GRAPH OF THE FURTHER SORTED DATA

ABeta42_fem_diseased_vals = []
ABeta42_male_diseased_vals = []

for patient in Patient.filter(Patient.all_patients, sex = "Female", cog_stat = "Dementia"):
    ABeta42_fem_diseased_vals.append(patient.ABeta42)
for patient in Patient.filter(Patient.all_patients, sex = "Male", cog_stat = "Dementia"):
     ABeta42_male_diseased_vals.append(patient.ABeta42)

x_fem_diseased_bar = (statistics.mean(ABeta42_fem_diseased_vals))
x_male_diseased_bar = (statistics.mean(ABeta42_male_diseased_vals))

ABeta42_fem_diseased_stdev = (statistics.stdev(ABeta42_fem_diseased_vals))
ABeta42_male_diseased_stdev = (statistics.stdev(ABeta42_male_diseased_vals))

print(f'x_fem_diseased_bar = {x_fem_diseased_bar}, ABeta42_fem_diseased_stdev {ABeta42_fem_diseased_stdev}')
print(f'x_male_diseased_bar = {x_male_diseased_bar}, ABeta42_male_diseased_stdev {ABeta42_male_diseased_stdev}')

x_fem_diseased_vals = range(len(Patient.filter(Patient.all_patients, sex = "Female")))
x_male_diseased_vals = range(len(Patient.filter(Patient.all_patients, sex = "Male")))

sex_diseased_cols = ['Diseased Females', 'Diseased Males']
mean_sex_diseased_ABeta42 = [x_fem_diseased_bar, x_male_diseased_bar]
stdev_sex_diseased_ABeta42 = [ABeta42_fem_diseased_stdev, ABeta42_male_diseased_stdev]
colors = ["pink", "blue"]
yerr = [np.zeros(len(mean_sex_diseased_ABeta42)), stdev_sex_diseased_ABeta42]

# Equal variance assumed
t_stat, p_val = stats.ttest_ind(ABeta42_fem_diseased_vals, ABeta42_male_diseased_vals)
print("t-statistic:", t_stat)
print("p-value:", p_val)

plt.bar(sex_diseased_cols, mean_sex_diseased_ABeta42, yerr=yerr, capsize=10, color=["pink", "skyblue"])
plt.title("ABeta42 Levels in Diseased Patients")
plt.xlabel("Sex")
plt.ylabel("Abeta42")
plt.show()

#9) GET PATIENT ATTRIBUTES THAT WE WANT TO COMPARE ON A SCATTER PLOT

death_age_list = []
MMSE_score= []

for patient in Patient.all_patients:
       death_age_list.append(patient.death_age)

for patient in Patient.all_patients:
       MMSE_score.append(patient.mmse)

X = death_age_list  # Independent variable
y = MMSE_score   # Dependent variable

print(X)
print(y)
#10) VISUALIZE DATA ON A SCATTER PLOT

plt.scatter(X, y)
plt.xlabel('Age of Death')
plt.ylabel('MMSE Score')
plt.title('Scatter Plot of Age of Death vs MMSE Score')
plt.show()

#11) EXPORT DATA TO A .csv FILE

import pandas as pd

print(death_age_list)
print(MMSE_score)

# Create a DataFrame
df = pd.DataFrame({
    'Age of Death': death_age_list,
    'Last MMSE Score': MMSE_score
})

# Write to CSV
df.to_csv('patient_data.csv', index=False)

print("CSV file 'patient_data.csv' has been created.")

#12) LOAD LIBRARIES FOR A LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#13) LOAD DATA SET FOR A LINEAR REGRESSION

df = pd.read_csv("patient_data.csv")

#14) Update these variable names to match EXACTLY your .csv file headers

x = df["Age of Death"].values.reshape(-1, 1)
y = df["MMSE Score"].values
#15) Perform the linear regression

model = LinearRegression()
model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(x, y)
#16) Make scatterplot
plt.scatter(x, y, label="Data")
plt.plot(x, model.predict(x), color="red")

# Annotate equation
equation = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.2f}"
plt.text(x.min(), y.max(), equation, color="red", fontsize=12, verticalalignment='top')

# Annotate scatterplot with labeles and title
plt.xlabel("Age of Death")
plt.ylabel("Last MMSE Score")
plt.title("Age of Death vs. Last MMSE Score")
plt.show()