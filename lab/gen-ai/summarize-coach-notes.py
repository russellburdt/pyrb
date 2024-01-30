
"""
summarize coaching notes, currently supported models
- amazon.titan-tg1-large
- anthropic.claude-v1
- anthropic.claude-v2
all model param values copied from AWS Bedrock text playground
"""

import os
import json
import boto3
import numpy as np
import pandas as pd
from collections import defaultdict
from bedrock.utils import bedrock
from glob import glob
from tqdm import tqdm
from ipdb import set_trace


# datadir
datadir = r'/mnt/home/russell.burdt/data/gen-ai'
assert os.path.isdir(datadir)
dm = pd.read_parquet(os.path.join(datadir, 'coach-notes', 'metadata.parquet'))

# bedrock objects
client = bedrock.get_bedrock_client(
    region='us-east-1',
    endpoint_url='https://bedrock.us-east-1.amazonaws.com',
    assumed_role='arn:aws:iam::315456707986:role/EC2_Role_Bedrock')
models = [x['modelId'] for x in client.list_foundation_models()['modelSummaries']]
# modelId = 'amazon.titan-tg1-large'
# modelId = 'anthropic.claude-v1'
modelId = 'anthropic.claude-v2'
assert modelId in models

# initialize summaries, scan over companies
ds = defaultdict(list)
for _, row in tqdm(dm.iterrows(), desc='summaries', total=dm.shape[0]):
    for month in ['Apr23', 'May23', 'Jun23']:

        # load and validate notes DataFrame for company / month
        fn = glob(os.path.join(datadir, 'coach-notes', f"""CompanyId={row['CompanyId']}""", f'desc={month}', '*.parquet'))
        assert len(fn) == 1
        dn = pd.read_parquet(fn[0])
        notes = pd.unique(dn['Note'])
        assert row[f'notes-{month}'] == notes.size

        # prompt metadata
        ds['company-id'].append(row['CompanyId'])
        ds['company-name'].append(row['CompanyName'])
        ds['month'].append(month)
        ds['num-notes'].append(notes.size)

        # create and save prompt
        prompt = f"""Human: You will be acting as a content summarizer of driver coaching notes.\n"""
        prompt += f"""Identify the common instructions provided to the driver.\n"""
        prompt += f"""Focus on the common mistakes drivers are committing and how drivers are suggested to improve safety.\n"""
        prompt += f"""The rules for the content summarization are below.\n\n"""
        prompt += """- Provide a summary of the following text as no more than three concise bullet points.\n"""
        prompt += """- Only use bullet points in the reponse.\n"""
        prompt += """- Never use more than three bullet points in the reponse.\n"""
        prompt += """- Do not include personal names in the response.\n"""
        prompt += """- Do not add any information that is not mentioned in the text below.\n\n"""
        prompt += """Rewrite sections of the content based on these rules with clarity.\n\n"""
        for note in [x.strip() for x in notes]:
            prompt += f"""<text>{note}</text>\n"""
        ds['prompt'].append(prompt)
        ds['num-chars-prompt'].append(len(prompt))

        # titan model response
        if modelId == 'amazon.titan-tg1-large':
            body = json.dumps({'inputText': prompt,
                'textGenerationConfig': {'maxTokenCount': 512, 'stopSequences': [], 'temperature':0, 'topP':0.9}})
            try:
                response = client.invoke_model(body=body, modelId=modelId, accept='application/json', contentType='application/json')
                response = json.loads(response.get('body').read())
                ds['response'].append(response['results'][0]['outputText'].strip())
            except:
                ds['response'].append('model did not run')

        # claude model response
        elif (modelId == 'anthropic.claude-v1') or (modelId == 'anthropic.claude-v2'):
            prompt += """\nAssistant:"""
            body = json.dumps({'prompt': prompt,
                'max_tokens_to_sample': 300, 'temperature': 1, 'top_k': 250, 'top_p': 0.999})
            try:
                response = client.invoke_model(body=body, modelId=modelId, accept='application/json', contentType='application/json')
                response = json.loads(response.get('body').read())
                ds['response'].append(response['completion'].strip())
            except:
                ds['response'].append('model did not run')
ds = pd.DataFrame(ds)

# save data
exp = glob(os.path.join(datadir, 'exp*.parquet'))
desc = glob(os.path.join(datadir, 'desc*.txt'))
assert len(exp) == len(desc)
x = len(exp)
x0 = ds.loc[ds['response'] != 'model did not run', 'num-chars-prompt'].max()
x1 = ds.loc[ds['response'] == 'model did not run', 'num-chars-prompt'].min()
assert x0 < x1
px = 100 * (ds['response'] == 'model did not run').sum() / ds.shape[0]
ds.to_parquet(os.path.join(datadir, f'exp{x}.parquet'))
with open(os.path.join(datadir, f'desc{x}.txt'), 'w') as fid:
    fid.write(f"""
        - summarize coach-notes as up to 3 concise bullet points without personal names
        - {modelId}, all default model parameters
        - model did not run for {px:.1f}% of cases, due to too many chars in prompt""")
