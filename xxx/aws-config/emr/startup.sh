#!/usr/bin/bash

echo "-- system update --"
sudo yum update -y
sudo yum install git htop -y
sudo yum install postgresql unixODBC postgresql-odbc -y

echo "-- conda application --"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /mnt1/miniconda.sh
bash /mnt1/miniconda.sh -b -p /mnt1/miniconda
rm /mnt1/miniconda.sh
/mnt1/miniconda/condabin/conda init
source ~/.bashrc
source /mnt1/miniconda/bin/activate

echo "-- lab Python environment for EMR 6.7.0, spark 3.2.1, 3-21-2023 --"
conda update -n base conda -y
conda create -n lab -c conda-forge python=3.9 -y
conda activate lab
conda install -c conda-forge matplotlib=3.7 numpy=1.23 pyspark=3.2.1 pyarrow=8.0 pandas=1.5 scikit-learn=1.2 shap=0.41 numba=0.56 -y
conda install -c conda-forge boto3=1.26 sqlalchemy=2.0 pyodbc=4.0 pyproj=3.4 reverse_geocoder=1.5 psycopg2=2.9 -y
conda install -c conda-forge dask ipython tqdm ipdb pip conda-pack -y
pip install snowflake-connector-python
pip install timezonefinder
conda pack -o /mnt1/environment.tar.gz

echo "-- clone RussellB Python repo --"
git clone http://dctfs4.drivecaminc.loc:8080/tfs/Lytx/_git/Lytx.AML.RussellB/
mv Lytx.AML.RussellB/ /mnt1/
echo 'export PYTHONPATH="/mnt1/Lytx.AML.RussellB"' >> /home/hadoop/.bashrc
cd /mnt1/Lytx.AML.RussellB
git config --global user.email "russell.burdt@lytx.com"
git config --global user.name "Russell Burdt"
cd /home/hadoop

echo "-- spark startup --"
echo 'function session() {' >> /home/hadoop/.bashrc
echo '  pyspark --archives /mnt1/environment.tar.gz#environment --py-files /mnt1/Lytx.AML.RussellB/lytx.py' >> /home/hadoop/.bashrc
echo '  }' >> /home/hadoop/.bashrc

echo "-- exec bash --"
exec bash
