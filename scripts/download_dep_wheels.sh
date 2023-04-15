# install pipreqs
pip install pipreqs --upgrade

# save required dependency into requirements.txt
pipreqs ./src --force

# download and save dependency wheels
pip wheel -r ./src/requirements.txt -w ./dep_wheels

# download extra pip
pip wheel pip==22.0.4 -w ./dep_wheels

# download extra gpustat
pip wheel gpustat -w ./dep_wheels
pip wheel tensorboard -w ./dep_wheels
pip wheel pysqlite3 -w ./dep_wheels