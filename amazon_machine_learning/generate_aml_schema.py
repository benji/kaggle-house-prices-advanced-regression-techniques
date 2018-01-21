# https://github.com/awslabs/machine-learning-samples/blob/master/targeted-marketing-python/build_model.py

import pandas as pd
import io, json

base_aml_schema = """ {
  "excludedAttributeNames": [], 
  "version": "1.0", 
  "dataFormat": "CSV", 
  "rowId": "Id", 
  "dataFileContainsHeader": true, 
  "attributes": [], 
  "targetAttributeName": "SalePrice"
}"""


def generate_aml_schema(schema_filename, df, features):
    aml_schema = json.loads(base_aml_schema)

    for col in df.columns:
        t = features[features.ColName == col]['Type'].iloc[0]
        aml_schema['attributes'].append({
            "attributeName": col,
            "attributeType": t
        })
    #print json.dumps(aml_schema, indent=4, sort_keys=True)

    with io.open(schema_filename, 'w', encoding='utf-8') as f:
        f.write(
            json.dumps(
                aml_schema, indent=4, sort_keys=True, ensure_ascii=False))


def generate_aml_recipe(recipe_filename, df, features):
    aml_recipe = {"outputs": []}

    for col in df.columns:
        t = features[features.ColName == col]['Type'].iloc[0]

        if (col == 'SalePrice' or col == 'Id'):
            print "Skipping " + col
        elif (t == 'NUMERIC'):
            aml_recipe['outputs'].append("quantile_bin(" + col + ",50)")
        else:
            aml_recipe['outputs'].append(col)

    with io.open(recipe_filename, 'w', encoding='utf-8') as f:
        f.write(
            unicode(
                json.dumps(
                    aml_recipe, indent=4, sort_keys=True, ensure_ascii=False)))
