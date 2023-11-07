# Dockerfile for mybinder.org
# 
# You can build a Docker image locally using
# >>> docker build --rm -t cfspopcon .
# 
# To start the docker image and launch a Jupyter notebook, use
# >>> docker run -it -p 8888:8888 -t cfspopcon
# 
# To start the docker image with a bash shell, use
# >>> docker run --entrypoint /bin/bash -it -p 8888:8888 -t cfspopcon

# 1. Start by using a base image from
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-scipy-notebook
# 
# N.b. for mybinder.org, we need to pin to a specific tag
FROM jupyter/scipy-notebook:848a82674792

# 2. Install a fortran compiler into this image, so that we
# can compile the radas headers
RUN conda install -c conda-forge gfortran -y
# 3. Install poetry, which is what we use to build cfspopcon
RUN pip install poetry

# 4. Copy in the files from the local directory
COPY --chown=$NB_USER:$NB_GID . ./

# 5. Tell poetry to install in the global python environment,
# so we don't have to worry about custom kernels or venvs.
RUN poetry config virtualenvs.create false
# 6. Install cfspopcon in the global python environment.
RUN poetry install

# 7. Download the radas project
RUN git clone https://github.com/cfs-energy/radas.git
# 8. Download atomic data files from OpenADAS
RUN PYTHONPATH=radas python radas/adas_data/fetch_adas_data.py
# 9. Run radas to compute Lz and <Z> curves from the atomic data
RUN PYTHONPATH=radas python radas/run_radas.py

# 10. Copy the results from radas into cfspopcon
RUN cp radas/cases/*/output/*.nc cfspopcon/atomic_data
