import base64
import boto3
import json
import os
import sys

bucket = 'kaggles'

boto3.setup_default_session(profile_name='personal', region_name='us-east-1')

s3_client = boto3.client('s3')
ml = boto3.client('machinelearning')


def create_datasource(from_file, schema_file, keyname, begin=0, end=100):
    to_file = 'ds-data-' + keyname + '.csv'
    data_s3_url = 's3://' + bucket + '/' + to_file

    print 'uploading data file ' + from_file + ' to ' + data_s3_url + '...'
    s3_client.upload_file(from_file, bucket, to_file)

    ds_id = 'ds-' + keyname + '-' + base64.b32encode(os.urandom(10))
    print 'Creating AML datasource ' + ds_id + '...'

    spec = {
        "DataLocationS3": data_s3_url,
        "DataSchema": open(schema_file).read()
    }

    if (begin != 0 | end != 100):
        spec['DataRearrangement'] = json.dumps({
            "splitting": {
                "percentBegin": begin,
                "percentEnd": end
            }
        })

    ml.create_data_source_from_s3(
        DataSourceId=ds_id,
        DataSpec=spec,
        DataSourceName="ds-" + keyname,
        ComputeStatistics=True)

    return ds_id


def create_model(keyname, train_ds_id, aml_recipe_file):
    model_id = 'ml-' + keyname + '-' + base64.b32encode(os.urandom(10))
    print 'Creating AML model ' + model_id + '...'

    with open(aml_recipe_file, 'r') as recipefile:
        ml.create_ml_model(
            MLModelId=model_id,
            MLModelName="model-" + keyname,
            MLModelType="REGRESSION",  # we're predicting True/False values
            Parameters={
                # Refer to the "Machine Learning Concepts" documentation
                # for guidelines on tuning your model
                "sgd.maxPasses": "100",
                "sgd.maxMLModelSizeInBytes": "104857600",  # 100 MiB
                "sgd.l2RegularizationAmount": "1e-6",
                "sgd.shuffleType": "auto"
            },
            Recipe=recipefile.read(),
            TrainingDataSourceId=train_ds_id)

    print("Created ML Model %s" % model_id)
    return model_id


def create_evaluation(model_id, test_ds_id, name):
    eval_id = 'ev-' + base64.b32encode(os.urandom(10))
    ml.create_evaluation(
        EvaluationId=eval_id,
        EvaluationName=name + " evaluation",
        MLModelId=model_id,
        EvaluationDataSourceId=test_ds_id)
    print("Created Evaluation %s" % eval_id)
    return eval_id


def produce_batch_prediction(keyname, model_id, ds_id):
    bp_id = 'bp-' + keyname + '-' + base64.b32encode(os.urandom(10))
    print 'Creating AML batch prediction ' + bp_id + '...'

    data_s3_url = 's3://' + bucket + '/aml_predictions.csv'

    response = ml.create_batch_prediction(
        BatchPredictionId=bp_id,
        BatchPredictionName='bp-' + keyname,
        MLModelId=model_id,
        BatchPredictionDataSourceId=ds_id,
        OutputUri=data_s3_url)
