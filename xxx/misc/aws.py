"""
misc AWS
"""

from snowflake import connector

conn = connector.connect(
    user='russell.burdt',
    database='dp_prod_db',
    warehouse='labs_prod_vwh',
    role='lytx_labs_engineer_prod',
    authenticator='externalbrowser',
    account='lytx')

from snowflake import connector

conn = connector.connect(
    user='SVC_REST_API',
    database='dp_prod_db',
    warehouse='APP_REST_PROD_VWH',
    role='LYTX_REST_APP_INTERNAL_PROD',
    password='!G3titT0W0rk!',
    account='lytx')


# import boto3

# # get AWS account id
# account = boto3.client('sts').get_caller_identity()['Account']

# # get available s3 buckets
# s3 = boto3.resource(service_name='s3')
# buckets = [x.name for x in s3.buckets.all()]

# # get dictionary of AML public keys
# pks = {}
# bucket = 'amldev-pubkeys'
# assert bucket in buckets
# for key in [x.key for x in s3.Bucket(bucket).objects.all()]:
#     pks[key] = s3.Object(bucket_name=bucket, key=key).get()['Body'].read().decode()
