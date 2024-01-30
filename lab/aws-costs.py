
"""
cost summary for specified resources in Lab account
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


# costs parameters
ta = '2023-12-01 00:00:00'
tb = '2024-02-01 00:00:00'
services = ['AmazonEC2', 'AmazonS3', 'AmazonAthena']
resources = {
    'AmazonS3': ['russell-lab-s3', 'russell-athena'],
    'AmazonEC2': ['i-04dd08ed578c4f986', 'vol-01f2d468c2b894952']}

# costs data
fn = r'/mnt/home/russell.burdt/data/aws-costs.p'
# fn = r'c:\Users\russell.burdt\Downloads\aws-costs.p'
if not os.path.isfile(fn):
    conn = lytx.get_conn('dataplatform')
    sql = f"""
        SELECT x.*, y."account name" AS account_name
        FROM cur_costs.cost_cur_parquet AS x
            INNER JOIN cur_costs.cur_accounts AS y
            ON CAST(x.line_item_usage_account_id AS BIGINT) = y."account id"
        WHERE y."account name" = 'Lab'
        AND x.line_item_usage_start_date >= CAST('{ta}' AS TIMESTAMP)
        AND x.line_item_usage_end_date <= CAST('{tb}' AS TIMESTAMP)
        AND x.line_item_line_item_type LIKE '%Usage'"""
    if services:
        sql += '\n\tAND (\n'
        for service in services:
            if resources.get(service, None):
                rs = ','.join([f"""'{x}'""" for x in resources[service]])
                sql += f"""\t\t((x.line_item_product_code = '{service}') AND (line_item_resource_id IN ({rs})))"""
            else:
                sql += f"""\t\t(x.line_item_product_code = '{service}')"""
            sql += '\n\t\tOR\n'
        sql = sql[:-6] + ')'
    now = datetime.now()
    df = pd.read_sql_query(sql, conn)
    sec = (datetime.now() - now).total_seconds()
    print(f'{sec:.0f}sec to read {df.shape[0]} rows from aws costs table')
    df.to_pickle(fn)
else:
    df = pd.read_pickle(fn)

# summary objects
df0, df, sv, mv, ds, dn = lytx.process_aws_cost_data(df, timezone='US/Pacific', convert_dates_only=False)
assert sv.size + mv.shape[1] <= df.shape[1]
assert ds.shape[1] + dn.shape[1] == mv.shape[1]
assert df0.shape[0] == df.shape[0] == mv.shape[0] == ds.shape[0] == dn.shape[0]

# cost, usage, time, and metadata metrics
cost = 'pricing_public_on_demand_cost'
usage = 'line_item_usage_amount'
time = 'line_item_usage_end_date'
metadata = ['line_item_usage_type', 'line_item_resource_id', 'pricing_unit']
# 'line_item_operation', 'account_name', 'line_item_line_item_description', 'product_servicecode', 'resource_tags_user_name', 'resource_tags_user_component'
assert df[cost].dtype.type == np.float64
assert df[usage].dtype.type == np.float64
assert df[time].dtype.type == np.datetime64
assert all([df[x].dtype.type == np.object_ for x in metadata])
df['service'] = df.pop('line_item_product_code')

# min and max usage datetime by service
dc0 = df.groupby('service')['line_item_usage_start_date'].min().reset_index(drop=False).rename(columns={'line_item_usage_start_date': 'min usage datetime'})
dc1 = df.groupby('service')['line_item_usage_end_date'].max().reset_index(drop=False).rename(columns={'line_item_usage_end_date': 'max usage datetime'})
dcx = pd.merge(dc0, dc1, on='service', how='inner')

# total cost by service
dc0 = df.groupby('service')[cost].sum().reset_index(drop=False)
dc0 = pd.merge(left=dc0, right=dcx, on='service', how='inner')
dc0[f'total cost'] = dc0.pop(cost)

# total cost and usage by service and metadata
dc1 = df.groupby(['service'] + metadata)[[cost, usage]].sum().reset_index(drop=False)
dc1 = pd.merge(dc0, dc1, on='service', how='left')
dc1['percentage of total cost'] = 100 * dc1[cost] / dc1[f'total cost'].values
dc1 = dc1.sort_values(['service', 'percentage of total cost'], ascending=False).reset_index(drop=True)
dc1 = dc1[['service'] + metadata + ['min usage datetime', 'max usage datetime'] + [usage, cost, 'total cost', 'percentage of total cost']]

# top cost item by service
dc2 = dc1.groupby('service').first().reset_index(drop=False)

# charts of daily cost by service
dc3 = df.groupby(['service', time])[cost].sum().reset_index(drop=False).sort_values(['service', time])
dc4 = dc3.copy()
dc4[time] = [x.date() for x in dc4.pop(time)]
days = np.array([x.date() for x in pd.date_range(dc4[time].min(), dc4[time].max(), freq='D')])
for service in services:
    dc4 = pd.concat((dc4, pd.DataFrame(data={time: days, cost: np.zeros(days.size), 'service': np.full(days.size, service)})))
dc4 = dc4.groupby(['service', time])[cost].sum().reset_index(drop=False).sort_values(['service', time])
fig, ax = open_figure(f'aws-costs, {cost} vs day', len(services), 1, figsize=(14, 8), sharex=True)
for x, service in enumerate(services):
    title = f'{service}, {cost} vs day'
    ax[x].plot(dc4.loc[dc4['service'] == service, time].values, dc4.loc[dc4['service'] == service, cost].values, 'x-', ms=12, lw=4)
    format_axes('', '', title, ax[x], apply_concise_date_formatter=True)
largefonts(18)
fig.tight_layout()
save_pngs(r'/mnt/home/russell.burdt/data')
assert os.path.isfile(r'/mnt/home/russell.burdt/data/image.png')
os.rename(src=r'/mnt/home/russell.burdt/data/image.png', dst=r'/mnt/home/russell.burdt/data' + os.sep + f'aws-costs, {cost} vs day' + '.png')
