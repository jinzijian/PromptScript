# install pip
pip install pip --upgrade --no-index --find-links ./dep_wheels

# install all dependent wheels
pip install -r ./src/requirements.txt --no-index --find-links ./dep_wheels

# # install extra pkg for nlpaug
# pip install sacremoses --upgrade --no-index --find-links ./dep_wheels

# # install jupyter wheels
# pip install jupyter --no-index --find-links ./dep_wheels

# install extra gpustat
pip install gpustat --no-index --find-links ./dep_wheels
pip install tensorboard --no-index --find-links ./dep_wheels

# # uninstall local cuda (with pytorch >=1.3.0)
# pip uninstall nvidia_cublas_cu11 -y