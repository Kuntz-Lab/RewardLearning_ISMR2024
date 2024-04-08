#!/bin/bash

set -e

MINICONDA_DIR=$HOME/.miniconda3


install_conda () {
    # Download and execute Miniconda installation script, creates Miniconda
    # environment in MINICONDA_DIR (variable should be set already).
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $MINICONDA_DIR
    rm Miniconda3-latest-Linux-x86_64.sh
    
    source $MINICONDA_DIR/etc/profile.d/conda.sh
    conda init
}


# Try to install conda if it's not installed and user wants it, otherwise exit
CONDA_INSTALLED=false
if ! type "conda" &> /dev/null ; then
    echo -e "\nCould not find 'conda'."
    while true; do
        read -p "Do you wish to install and initialize conda ('yes' or 'no')? " yn
        case $yn in
            [Yy]* ) install_conda; break;;
            [Nn]* ) echo "Cannot continue installation without conda. Exiting."; exit 1;;
            * ) echo "Please answer 'yes' or 'no'.";;
        esac
    done
    CONDA_INSTALLED=true
    source $MINICONDA_DIR/etc/profile.d/conda.sh
else
    source $MINICONDA_DIR/etc/profile.d/conda.sh
    conda deactivate
    conda deactivate
fi


conda update -y -n base -c defaults conda
conda install mamba -n base -c conda-forge
