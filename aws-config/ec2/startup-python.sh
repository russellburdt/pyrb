
# update conda
source /mnt/home/russell.burdt/miniconda3/etc/profile.d/conda.sh
conda update -n base conda -y

# spark 3.4.1, hadoop 3.3.6, 11-14-2023
# pandas 2.1.3 - snowflake packages incompatible, many pandas/pyspark warnings, try again with later versions
conda create -n s34 -y
conda activate s34
conda install -c conda-forge pyspark=3.4.1 pandas=1.5 bokeh=2.4.3 matplotlib numpy pyarrow -y
conda install -c conda-forge ipython pip sqlalchemy pyodbc psycopg2 pyathena -y
conda install -c conda-forge pyproj ipdb tqdm scikit-learn -y
pip install shap
pip install boto3
pip install geopandas
pip install snowflake-connector-python
pip install snowflake-sqlalchemy
pip install awswrangler
pip install asn1crypto
git clone ssh://dctfs4:22/tfs/Lytx/_git/Lytx.AML.DCEUtils
pip install --no-deps ./Lytx.AML.DCEUtils
rm -r -f Lytx.AML.DCEUtils/
conda deactivate

# # Python environment for collision-prediction model, spark 3.2.1, 3-21-2023
# conda install -c conda-forge ipython -y
# conda create -n cpm -c conda-forge python=3.9 -y
# conda activate cpm
# conda install -c conda-forge matplotlib=3.7 numpy=1.23 pyspark=3.2.1 pyarrow=8.0 pandas=1.5 scikit-learn=1.2 shap=0.41 numba=0.56 bokeh=2.4.3 -y
# conda install -c conda-forge boto3=1.26 sqlalchemy=2.0 pyodbc=4.0 pyproj=3.4 reverse_geocoder=1.5 psycopg2=2.9 -y
# conda install -c conda-forge dask ipython tqdm ipdb pyathena pip -y
# pip install snowflake-connector-python
# pip install timezonefinder
# pip install geopandas
# conda deactivate

# # Python environment for ffmepg3, 9-1-2023
# conda create -n ffmpeg3 -c conda-forge ffmpeg=3
# conda activate ffmpeg3
# conda install -c conda-forge pip
# git clone ssh://dctfs4:22/tfs/Lytx/_git/Lytx.AML.DCEUtils
# pip install --no-deps ./Lytx.AML.DCEUtils
# rm -r -f Lytx.AML.DCEUtils/
# pip install python-dateutil
# pip install pytz
# conda deactivate

# # Python environment for eye-witness, 7-17-2023
# conda create -n eye -y
# conda activate eye
# conda install -c conda-forge ipython sqlalchemy pytz boto3 numpy pandas pyarrow pyspark pyproj geopandas ipdb tqdm pyodbc dask -y
# conda install -c conda-forge bokeh=2.4.3 -y
# pip install snowflake-connector-python
# pip install opencv-python
# git clone ssh://dctfs4:22/tfs/Lytx/_git/Lytx.AML.DCEUtils
# pip install --no-deps ./Lytx.AML.DCEUtils
# rm -r -f Lytx.AML.DCEUtils/
# git clone ssh://dctfs4:22/tfs/Lytx/_git/Lytx.AML.Prospector
# pip install --no-deps ./Lytx.AML.Prospector
# rm -r -f Lytx.AML.Prospector/
# conda deactivate

# # Python environment for collision-reconstruction, 9-27-2023
# conda create -n cr -y
# conda activate cr
# conda install -c conda-forge pyspark=3.4.1 pandas=1.5.3 numpy=1.24 matplotlib=3.8 pyarrow -y
# conda install -c conda-forge ipython sqlalchemy snowflake-sqlalchemy pyodbc psycopg2 -y
# conda install -c conda-forge boto3 bokeh pyproj geopandas ipdb tqdm -y
# git clone ssh://dctfs4:22/tfs/Lytx/_git/Lytx.AML.DCEUtils
# pip install --no-deps ./Lytx.AML.DCEUtils
# rm -r -f Lytx.AML.DCEUtils/
# git clone ssh://dctfs4:22/tfs/Lytx/_git/Lytx.AML.Prospector
# pip install --no-deps ./Lytx.AML.Prospector
# rm -r -f Lytx.AML.Prospector/
# conda install -c conda-forge smart_open ffmpeg ffmpeg-python -y
# pip install labelbox
# pip install opencv-python
# conda deactivate

# # Python environment for bedrock, 8-8-2023
# # following https://github.com/aws-samples/amazon-bedrock-workshop/blob/7e3c85b4db3c4c2229aeeddfb382e8236c6e87ad/00_Intro/bedrock_boto3_setup.ipynb
# conda create -n bk -y
# conda activate bk
# conda install -c conda-forge pip -y
# pip install --no-build-isolation --force-reinstall bedrock-dependencies/awscli*.whl
# pip install --no-build-isolation --force-reinstall bedrock-dependencies/boto3*.whl
# pip install --no-build-isolation --force-reinstall bedrock-dependencies/botocore*.whl
# conda install -c conda-forge ipython tqdm ipdb pandas numpy pyarrow -y
# pip install langchain==0.0.249
# pip install faiss-cpu
# pip install bs4
# conda install -c conda-forge bokeh -y
