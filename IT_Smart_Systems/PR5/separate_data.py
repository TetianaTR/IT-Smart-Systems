import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('E:/KN-305/IT_Smart_Systems/PR5/data/variant_3.csv')

train, test = train_test_split(df, train_size=0.9, test_size=0.1, stratify=df['AdoptionSpeed'])

train.to_csv('E:/KN-305/IT_Smart_Systems/PR5/data/train.csv', index=False)
test.to_csv('E:/KN-305/IT_Smart_Systems/PR5/data/new_input.csv', index=False)