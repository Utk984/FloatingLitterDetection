import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("./data/TUD-GV Dataset/TUD-GV.csv")

df_litter = df[
    ["No litter", "Little litter", "Moderate litter", "Lots of litter"]
].sum()
df_litter.plot(kind="bar", color=["green", "orange", "red", "purple"], figsize=(8, 5))
plt.title("Distribution of Litter Classes")
plt.xlabel("Litter Classes")
plt.ylabel("Number of Images")
plt.show()

df["Weather conditions"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", colors=["lightblue", "lightgray"], figsize=(6, 6)
)
plt.title("Distribution of Weather Conditions")
plt.ylabel("")
plt.show()


def categorize_time(time_str):
    hour = int(time_str.split(":")[0])
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


# Apply the function to the 'Collecting time' column
df["Time of Day"] = df["Collecting time"].apply(categorize_time)

time_of_day_litter = df.groupby("Time of Day")[
    ["No litter", "Little litter", "Moderate litter", "Lots of litter"]
].sum()
time_of_day_litter.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=["green", "orange", "red", "purple"],
)
plt.title("Litter Class Distribution Based on Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.show()
