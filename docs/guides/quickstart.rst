Quickstart Guide
===============

Installation
-----------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/[username]/oocr.git
   cd oocr

2. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Basic Usage
----------

Training
^^^^^^^^

Train a model using the default configuration:

.. code-block:: bash

   python train.py --config configs/datasets-configs/sroie.yaml

Inference
^^^^^^^^

Run inference on an image:

.. code-block:: bash

   python inference.py --image_path path/to/image.jpg --model_path path/to/model

Configuration
------------

The project uses YAML configuration files located in ``configs/``. Example configuration:

.. code-block:: yaml

   # Data Configuration
   data: "data/datasets/SROIE2019"
   dataset: "SROIE"
   
   # Model Configuration
   model: "microsoft/trocr-base-printed"
   
   # Training Configuration
   epochs: 10
   batchsize: 8
   lr: 1e-6