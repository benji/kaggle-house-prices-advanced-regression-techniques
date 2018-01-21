import json,math


def find_non_NAs(df):
    nulls = df.isnull().sum()
    print "NAs columns stats:"
    print nulls[nulls > 0].sort_values(ascending=False)
    return nulls[nulls == 0].keys()

def apply_transform_lambda(transmap, col, x):
    if (x == x):
      return transmap[col][x]

def transform_values(transform_file, df):
    with open(transform_file, 'r') as myfile:
        transmap = json.loads(myfile.read())

        cols = []

        for col in df.columns:
            if col in transmap:
                print 'Transform column ' + col
                df[col] = df[col].apply(
                    lambda x: apply_transform_lambda(transmap, col, x))
                cols.append(col)

        return cols
