import pandas as pd
import math, json
from df_transform import find_non_NAs, transform_values
from generate_aml_schema import generate_aml_schema, generate_aml_recipe
from aml_train import create_datasource, create_model, produce_batch_prediction
pd.options.mode.chained_assignment = None  # default='warn'

doawscall = True
keyname = 'kaggle-houses-reg'

# load data
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

# remove outliers > $700k
#df_train = df_train.drop(df_train.index[691])
#df_train = df_train.drop(df_train.index[1182])

# load data features definition
features = pd.read_csv('features.csv')

with open('transform_keys.json', 'r') as tfile:
    transmap = json.loads(tfile.read())
    for col in transmap:
        newName = transmap[col]
        print 'Renaming column ' + col + ' to ' + newName
        df_train.rename(columns={col: newName}, inplace=True)
        df_test.rename(columns={col: newName}, inplace=True)
        features['ColName'][features['ColName'] == col] = newName

# transform data
cols = find_non_NAs(df_train)
df_train = df_train[cols]
#df_train['SalePrice']=df_train['SalePrice'].apply(lambda x:math.log(x))

df_test['SalePrice'] = 0
df_test = df_test[cols]

transform_values('transform_binaries.json', df_train)
transform_values('transform_binaries.json', df_test)

#cols = transform('transform_categoricals0.json', df_train)
#transform('transform_categoricals0.json', df_test)
#features['Type'][features.ColName.isin(cols)] = 'NUMERIC'

# save data
final_train_file = 'train_transformed.csv'
df_train.to_csv(final_train_file, sep=',', index=False)

final_test_file = 'test_transformed.csv'
df_test.to_csv(final_test_file, sep=',', index=False)

# generate AML schema
aml_schema_file = 'aml_schema.json'
generate_aml_schema(aml_schema_file, df_train, features)

# generate AML recipe
aml_recipe_file = 'aml_recipe.json'
generate_aml_recipe(aml_recipe_file, df_train, features)

if (doawscall):
    # create AML datasources
    ds_id = create_datasource(final_train_file, aml_schema_file,
                              keyname + '-train', 0, 100)
    test_ds_id = create_datasource(final_test_file, aml_schema_file,
                                   keyname + '-test')

    model_id = create_model(keyname, ds_id, aml_recipe_file)

    produce_batch_prediction(keyname, model_id, test_ds_id)
