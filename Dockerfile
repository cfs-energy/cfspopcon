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
FROM jupyter/minimal-notebook:python-3.11

# 2. Install poetry, which is what we use to build cfspopcon
RUN pip install poetry==1.8.2

# 3. Copy in the files from the local directory
COPY --chown=$NB_USER:$NB_GID . ./

# 4. Tell poetry to install in the global python environment,
# so we don't have to worry about custom kernels or venvs.
RUN poetry config virtualenvs.create false
# 5. Install cfspopcon in the global python environment.
RUN poetry install --without dev

# 6. Run radas to get the atomic data files
RUN poetry run radas -c radas_config.yaml
