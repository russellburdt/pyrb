
import pandas as pd
import numpy as np
import prospector as prs
from datetime import datetime


# validate active device
device = prs.Device('MV00249435')
assert device.ping()['status'] == 'ok'
assert device.is_request_data_supported()

# validate time-window and datatypes for data request
t0 = pd.Timestamp(datetime.fromisoformat('2023-10-06 18:41:15'))
ta = t0 - pd.Timedelta(seconds=15)
tb = t0 + pd.Timedelta(seconds=60)
dts = [prs.DataType.AUDIO, prs.DataType.GPS, prs.DataType.FORWARD, prs.DataType.REAR]
timeline = device.timeline()
ok = np.array([(ta > pd.Timestamp(x.start_time)) and (tb < pd.Timestamp(x.end_time)) for x in timeline])
assert ok.sum() == 1
tok = timeline[ok.argmax()]
assert [x in tok.datatypes for x in dts]

# data-request
data_request = prs.DataRequest(start_time=ta.to_pydatetime(), end_time=tb.to_pydatetime(), datatypes=dts, usecase='lab:cr')
data = device.request_data(data_request)
