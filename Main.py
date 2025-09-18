import csv
import warnings
import matplotlib.pyplot as plt


class Patient: 
    all_patients = []

    def __init__(self, DonorID, sex=None, death_age=None, cog_stat=None, 
                 consensus_dx=None, brain_weight=None):
        self.DonorID = DonorID
        self.sex = sex
        self.death_age = death_age
        self.cog_stat = cog_stat
        self.consensus_dx = consensus_dx  # list of diagnoses or None
        self.brain_weight = brain_weight
        Patient.all_patients.append(self)

    def __repr__(self):
        dx_display = self.consensus_dx if self.consensus_dx else "None"
        return (f"{self.DonorID} | sex: {self.sex} | Death Age {self.death_age} | "
                f"Cognitive Status {self.cog_stat} | Consensus Dx {dx_display} | "
                f"Brain Wt {self.brain_weight}")

    def get_id(self):
        return self.DonorID

    @classmethod
    def instantiate_from_csv(cls, filename: str):
        with open(filename, encoding="utf8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            for row in reader:
                donor_id = row["Donor ID"].strip()
                sex = row["Sex"].strip() if row["Sex"] else None
                death_age = int(row["Age at Death"]) if row["Age at Death"] else None
                cog_stat = row["Cognitive Status"].strip() if row["Cognitive Status"] else None

                # collect diagnoses marked "Checked"
                consensus_cols = [h for h in headers if h.startswith("Consensus Clinical Dx")]
                diagnoses = []
                for col in consensus_cols:
                    val = row[col].strip()
                    if val == "Checked":
                        dx_name = col.replace("Consensus Clinical Dx (choice=", "").replace(")", "")
                        diagnoses.append(dx_name)
                if not diagnoses:  # if no "Checked" diagnoses
                    diagnoses = None

                # handle brain weight safely
                brain_weight = None
                if row["Fresh Brain Weight"]:
                    try:
                        brain_weight = float(row["Fresh Brain Weight"])
                    except ValueError:
                        brain_weight = None

                # create patient object
                Patient(
                    DonorID=donor_id,
                    sex=sex,
                    death_age=death_age,
                    cog_stat=cog_stat,
                    consensus_dx=diagnoses,
                    brain_weight=brain_weight
                )

        # print all patients after reading
        for patient in Patient.all_patients:
            print(patient)
Patient.all_patients.sort(key=lambda p: p.consensus_dx or [], reverse=False)

# Example usage:
if __name__ == "__main__":
    Patient.instantiate_from_csv("UpdatedMetaData.csv")


#3) SORT OUR LIST OF PATIENTS

#) Make a bar graph
import matplotlib.pyplot as plt

# Count patients with and without diagnoses
with_dx = sum(1 for p in Patient.all_patients if p.consensus_dx is not None)
without_dx = sum(1 for p in Patient.all_patients if p.consensus_dx is None)

# Prepare data for bar plot
labels = ['With Diagnoses', 'Without Diagnoses']
counts = [with_dx, without_dx]

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color=['skyblue', 'lightcoral'])
plt.ylabel("Number of Patients")
plt.title("Patients With vs Without Diagnoses")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()



