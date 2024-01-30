
"""
coach-notes summarization app
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div, Button, Select
from pyrb.bokeh import console_str_objects
from ipdb import set_trace


# datadir and valid experiments
datadir = r'/mnt/home/russell.burdt/data/gen-ai'
assert os.path.isdir(datadir)
exp = glob(os.path.join(datadir, 'exp*.parquet'))
desc = glob(os.path.join(datadir, 'desc*.txt'))
assert len(exp) == len(desc)
nx = len(exp)
for x in range(nx):
    assert os.path.isfile(os.path.join(datadir, f'exp{x}.parquet'))
    assert os.path.isfile(os.path.join(datadir, f'desc{x}.txt'))
exps = [f'experiment{x}' for x in range(nx)]

# width formats
w0, w1 = 300, 500

def update_experiment(attr, old, new):

    global ds

    # data and desc for experiment
    ds = pd.read_parquet(os.path.join(datadir, f'exp{exp_select.value[10:]}.parquet'))
    with open(os.path.join(datadir, f'desc{exp_select.value[10:]}.txt'), 'r') as fid:
        desc = fid.readlines()
    desc = desc[1:]
    desc = ''.join(desc).replace('\n', '<br>')
    c0, c1 = console_str_objects(w0)
    exp_desc.text = c0 + desc + c1

    # update company and month select
    company.remove_on_change('value', update_company_month)
    month.remove_on_change('value', update_company_month)
    companies = sorted(pd.unique(ds['company-name']))
    months = ['Apr23', 'May23', 'Jun23']
    company.value, company.options = companies[0], companies
    month.value, month.options = months[0], months
    company.on_change('value', update_company_month)
    month.on_change('value', update_company_month)
    update_company_month(None, None, None)

def update_company_month(attr, old, new):

    global ds

    # load data for selection
    dx = ds.loc[(ds['company-name'] == company.value) & (ds['month'] == month.value)]
    assert dx.shape[0] == 1
    dx = dx.squeeze()

    # prompt metadata
    c0, c1 = console_str_objects(w0)
    cs = f"""
        company-id, {dx['company-id']}<br>
        company-name, {dx['company-name']}<br>
        month, {dx['month']}<br>
        num-notes, {dx['num-notes']}<br>
        num-chars-prompt, {dx['num-chars-prompt']}<br>"""
    prompt_metadata.text = c0 + cs + c1

    # prompt data
    c0, c1 = console_str_objects(w1)
    cs = dx['prompt'].replace('\n', '<br>')
    cs = cs.replace('<text>', '&lt;text&gt;')
    cs = cs.replace('</text>', '&lt;&sol;text&gt;')
    prompt.text = c0 + cs + c1

    # response data
    cs = dx['response'].replace('\n', '<br>')
    cs = cs.replace('<text>', '&lt;text&gt;')
    cs = cs.replace('</text>', '&lt;&sol;text&gt;')
    response.text = c0 + cs + c1

# experiment select objects
exp_select = Select(title='select experiment', value=exps[0], options=exps, width=w0)
exp_select.on_change('value', update_experiment)
c0, c1 = console_str_objects(w0)
exp_desc_title = Div(text='<strong>Experiment Description</strong>', width=w0)
exp_desc = Div(text=c0 + c1)

# company and month select objects
company = Select(title='select company-name', width=w0)
month = Select(title='select month', width=w0)
company.on_change('value', update_company_month)
month.on_change('value', update_company_month)

# prompt metadata objects
c0, c1 = console_str_objects(w0)
prompt_metadata_title = Div(text='<strong>Prompt Metadata</strong>', width=w0)
prompt_metadata = Div(text=c0 + c1)

# prompt and response objects
c0, c1 = console_str_objects(w1)
prompt_title = Div(text='<strong>Prompt</strong>', width=w1)
prompt = Div(text=c0 + c1)
response_title = Div(text='<strong>Response</strong>', width=w1)
response = Div(text=c0 + c1)

# app layout, document object, initialize
layout = row(
    column(exp_select, exp_desc_title, exp_desc, company, month, prompt_metadata_title, prompt_metadata),
    column(prompt_title, prompt),
    column(response_title, response))
doc = curdoc()
doc.add_root(layout)
doc.title = 'coach-notes summary app'
update_experiment(None, None, None)
