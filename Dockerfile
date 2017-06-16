FROM ubuntu:16.04

ADD . /vagrant
WORKDIR /vagrant
RUN  apt-get update \
  && apt-get install -y bzip2 \
  gcc \
  wget
RUN wget http://repo.continuum.io/miniconda/Miniconda-3.7.0-Linux-x86_64.sh -O miniconda.sh
RUN [ "/bin/bash", "miniconda.sh -b -p /vagrant/miniconda"]

ENV PATH=/vagrant/miniconda/bin:${PATH}
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda

RUN conda create -q -n test-environment python=2.7 cytoolz numpy pandas pip pytables pyyaml toolz setuptools
RUN [ "/bin/bash", "-c", "source activate test-environment && pip install orca openmatrix zbox pytest pytest-cov coveralls pep8 pytest-xdist sphinx numpydoc psutil && easy_install -v -U --user ortools && pip --no-cache-dir install https://github.com/RSGInc/activitysim/zipball/master" ]

RUN [ "/bin/bash", "-c", "source activate test-environment && pip install ."]
