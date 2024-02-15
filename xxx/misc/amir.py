
import pandas as pd
from dceutils import dceinfo4


fn = r'/mnt/home/russell.burdt/data/2023-08-19 video_tag_v2_neo - Inside.dce'
dce = dceinfo4.DCEFile(fn)
gps = dce.getForms('UGPS')
gps = pd.DataFrame(dceinfo4.FormParser(gps[0]).parse())

gps.to_csv(r'/mnt/home/russell.burdt/data/amir.csv')
