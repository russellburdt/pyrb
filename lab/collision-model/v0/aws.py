
"""
post-process data from cost_cur_parquet
"""

import os
import lytx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
from ipdb import set_trace
plt.style.use('bmh')


# aws costs raw data
fn = r'/mnt/home/russell.burdt/data/aws.p'
if not os.path.isfile(fn):
    conn = lytx.get_conn('dataplatform')
    sql = f"""
        SELECT x.*, y."account name" AS account_name
        FROM cur_costs.cost_cur_parquet AS x
            INNER JOIN cur_costs.cur_accounts AS y
            ON CAST(x.line_item_usage_account_id AS BIGINT) = y."account id"
        WHERE y."account name" IN ('AML', 'Lab')
        AND x.line_item_product_code IN ('ElasticMapReduce', 'AmazonS3', 'AmazonRDS', 'AmazonEC2')
        AND x.line_item_usage_start_date >= CAST('2023-03-15 00:00' AS TIMESTAMP)
        AND x.line_item_usage_end_date <= CAST('2023-03-30 00:00' AS TIMESTAMP)
        AND x.line_item_line_item_type LIKE '%Usage'"""
    now = datetime.now()
    df = pd.read_sql_query(sql, conn)
    sec = (datetime.now() - now).total_seconds()
    print(f'{sec:.0f}sec to read {df.shape[0]} rows from aws costs table')
    df.to_pickle(fn)
else:
    df = pd.read_pickle(fn)

# filter by resources used in GPS enrichment
rids = (
    # S3
    'russell-s3',
    # EMR + EC2
    'arn:aws:elasticmapreduce:us-west-2:573123634485:cluster/j-38HL5ZCIKPINI',
    'i-01c341affdd253c8f',
    'i-096e979a33c8a0b2a',
    'i-049f1a957916ae6b6',
    'i-05724c9dee45a9d19',
    'i-0dc10af697d23e738',
    # RDS
    'arn:aws:rds:us-west-2:315456707986:db:dev-mapping-domain-a-cluster',
    'arn:aws:rds:us-west-2:315456707986:cluster:cluster-midwuepzrchg4czhzzx3ep6r5m',
    'arn:aws:rds:us-west-2:315456707986:db:dev-mapping-domain-b-cluster',
    'arn:aws:rds:us-west-2:315456707986:cluster:cluster-6bscutvu6nhc3xwwh4t6iehhqq',
    'arn:aws:rds:us-west-2:315456707986:db:dev-mapping-domain-c-cluster',
    'arn:aws:rds:us-west-2:315456707986:cluster:cluster-euq2hpvhkthwb3id4prq4lxioy',
    'arn:aws:rds:us-west-2:315456707986:db:dev-mapping-domain-d-cluster',
    'arn:aws:rds:us-west-2:315456707986:cluster:cluster-yavbodgfuax3kuo3shekcebf4y')
for rid in rids:
    assert (df['line_item_resource_id'] == rid).sum() > 0
ok = np.logical_or.reduce([df['line_item_resource_id'] == rid for rid in rids])
df = df.loc[ok].reset_index(drop=True)

# filter by time bounds of EMR resource
rid = 'arn:aws:elasticmapreduce:us-west-2:573123634485:cluster/j-38HL5ZCIKPINI'
tmin = df.loc[df['line_item_resource_id'] == rid, 'line_item_usage_start_date'].min()
tmax = df.loc[df['line_item_resource_id'] == rid, 'line_item_usage_end_date'].max()
df = df.loc[(df['line_item_usage_start_date'] >= tmin) & (df['line_item_usage_end_date'] <= tmax)]

# convert aws costs raw data
now = datetime.now()
df0, df, sv, mv, ds, dn = lytx.process_aws_cost_data(df, timezone=None, convert_dates_only=False)
sec = (datetime.now() - now).total_seconds()
print(f'{sec:.0f}sec to convert aws costs raw data')

# cost, usage, time, and metadata metrics
cost = 'pricing_public_on_demand_cost'
usage = 'line_item_usage_amount'
time = 'line_item_usage_end_date'
metadata = [
    'line_item_usage_type',
    'line_item_resource_id',
    # 'line_item_operation',
    'account_name',
    # 'line_item_line_item_description',
    'pricing_unit']
    # 'product_servicecode',
    # 'resource_tags_user_name',
    # 'resource_tags_user_component']
assert df[cost].dtype.type == np.float64
assert df[usage].dtype.type == np.float64
assert df[time].dtype.type == np.datetime64
assert all([df[x].dtype.type == np.object_ for x in metadata])

# total cost by service
df['service'] = df.pop('line_item_product_code')
services = pd.unique(df['service'])
dc0 = df.groupby(['account_name', 'service'])[cost].sum().reset_index(drop=False)
dc0[f'total cost'] = dc0.pop(cost)

# remove common prefix from 'line_item_resource_id'
if False:
    if 'line_item_resource_id' in metadata:
        for service in services:
            rid = df.loc[df['service'] == service, 'line_item_resource_id'].values
            ustr = np.array([x for x in pd.unique(rid) if x != ''])
            prefix = os.path.commonprefix(ustr.tolist())
            if prefix:
                df.loc[df['service'] == service, 'line_item_resource_id'] = [x.replace(prefix, '') for x in rid]

# total cost and usage by service and metadata
dc1 = df.groupby(['service'] + metadata)[[cost, usage]].sum().reset_index(drop=False)
dc1 = pd.merge(dc0, dc1, on='service', how='left')
dc1['percentage of total cost'] = 100 * dc1[cost] / dc1[f'total cost'].values
dc1 = dc1.sort_values(['service', 'percentage of total cost'], ascending=False).reset_index(drop=True)
dc1 = dc1[['service'] + metadata + [usage, cost, 'total cost', 'percentage of total cost']]

# grouped costs for display
dx = dc1[['account_name', 'service', 'line_item_usage_type', 'line_item_usage_amount', 'pricing_unit', 'pricing_public_on_demand_cost', 'percentage of total cost']]
dx = dx.groupby(['account_name', 'service', 'line_item_usage_type', 'pricing_unit']).sum()
dc = {}
for sx in services:
    dc[sx] = dx.loc[dx.index.isin([sx], level='service')].sort_values('percentage of total cost', ascending=False)
    dc[sx] = dc[sx].loc[dc[sx]['percentage of total cost'] > 1]
    dc[sx]['usage amount'] = dc[sx].pop('line_item_usage_amount')
    dc[sx]['cost'] = dc[sx].pop('pricing_public_on_demand_cost')
    dc[sx]['percentage of service cost'] = dc[sx].pop('percentage of total cost')



# group cost vs time
# x = 3
# dx = dc1.loc[x]
# ok = np.logical_and.reduce([(df[field] == dx[field]).values for field in metadata])
# assert np.isclose(df.loc[ok, cost].sum(), dx[cost])
# dfx = df.loc[ok].groupby(time)[cost].sum().sort_index()
# # title = '\n'.join([f"""-- {field} --\n   {dx[field]}""" for field in metadata])
# title = '\n'.join([dx[field] for field in metadata])
# fig, ax = open_figure(f'{cost} vs {time}', figsize=(12, 5))
# ax.plot(dfx.index.to_numpy(), dfx.values, 'x-', ms=8, lw=3)
# format_axes('', 'cost', title, apply_concise_date_formatter=True)
# ax.title.set_position((0, 1))
# ax.title.set_horizontalalignment('left')
# largefonts(16)
# fig.tight_layout()

# plt.show()
