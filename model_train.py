import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_excel("complete_material_dataset.xlsx")

le_env = LabelEncoder()
le_proj = LabelEncoder()
le_avail = LabelEncoder()
le_toxic_lvl = LabelEncoder()
le_toxic_flag = LabelEncoder()
le_recycle = LabelEncoder()

df['Suitable_Environment'] = le_env.fit_transform(df['Suitable_Environment'])
df['Project_Type'] = le_proj.fit_transform(df['Project_Type'])
df['Availability'] = le_avail.fit_transform(df['Availability'])
df['Toxicity_Level'] = le_toxic_lvl.fit_transform(df['Toxicity_Level'])
df['Toxicity'] = le_toxic_flag.fit_transform(df['Toxicity'])
df['Recyclability'] = le_recycle.fit_transform(df['Recyclability'])

X = df.drop("Material_Name", axis=1)
y = df["Material_Name"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "material_recommender.pkl")
joblib.dump(le_env, "env_encoder.pkl")
joblib.dump(le_proj, "proj_encoder.pkl")
joblib.dump(le_avail, "avail_encoder.pkl")
joblib.dump(le_toxic_lvl, "toxic_lvl_encoder.pkl")
joblib.dump(le_toxic_flag, "toxic_flag_encoder.pkl")
joblib.dump(le_recycle, "recycle_encoder.pkl")
