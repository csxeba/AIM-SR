# AIM-SR

*Road sign classification with the BrainForge Deep Learning library*

**Athor**: Csaba Gór

The demo notebook is `experiment.ipynb`. Check the `env.yml` file for an appropriate
conda environment. Running the experiment with an Anaconda distribution is advised,
because the *Anaconda NumPy* package is built with *Intel MKL* and provides a 2-fold
acceleration compared to the default *NumPy* available through `pip`.

Please consider downloading the dataset before running the experiment. There is the
`get.sh` script which automates downloading and extracting the dataset. The script should
be ran with its parent `data` directory as the working directory.
