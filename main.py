# Import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import classification_report

# Pandas option
pd.set_option('future.no_silent_downcasting', True)

# Open DataSet
dataset = pd.read_csv("./archive/penguins_size.csv", sep=",", encoding="utf-8")

# Label Encoding
dataset["island"] = dataset["island"].replace({"Biscoe": 0, "Dream": 1, "Torgersen": 2})
dataset["sex"] = dataset["sex"].replace({"MALE": 1, "FEMALE": 0})
dataset["species"] = dataset["species"].replace({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})

# Ordering columns
new_columns = ["island", "sex", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "species"]
dataset = dataset.reindex(columns=new_columns)

# Correcting errors in the dataset
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.dropna()

# Separating the parts of the dataset for training
y = dataset["species"]
x = dataset.drop(["species"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Loading model
model = MultinomialNB()

# Training
model.fit(x_train, y_train)

# Evaluating the model
y_predict = model.predict(x_test)

result = classification_report(y_test, y_predict)

print(f"""########## Avaliação do modelo  ##########\n
{result}""")

# Inferences 
print(f"""########## Inferência de um caso ##########\n""")
island = int(input("Island: "))
culmen_length_mm = float(input("culmen_length_mm: "))
culmen_depth_mm = float(input("culmen_depth_mm: "))
flipper_length_mm = float(input("flipper_length_mm: "))
body_mass_g = float(input("body_mass_g: "))
sex = int(input("sex: "))

line = pd.Series({
    "island": island,
    "sex": sex,
    "culmen_length_mm": culmen_length_mm,
    "culmen_depth_mm": culmen_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g
})

# Convert a Series into a DataFrame
line_df = pd.DataFrame([line])

# Model inference
line_result = model.predict(line_df)

if line_result == 0: print("Especie: Adelie")
elif line_result == 1: print("Especie: Chinstrap")
else: print("Especie: Gentoo")
