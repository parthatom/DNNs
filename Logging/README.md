# Logging

A simple python library to log your Machine Learning experiments.

## Prerequisites

1. Make sure that you have recent versions installed of:
	- Python (version 3.6 or higher)
	- Numpy (e.g. `pip install numpy`)
	- Pandas (e.g. `pip install pandas`)

3. Clone this repository, e.g.:
```sh
git clone https://github.com/parthatom/Logging.git
```

## How to use

```py
from Logging import Logger
import os

model_logger = Logger(df_path = data_path, create = False)
#Loop for various models
  model_dir = "Path.to.model"
  epoch_logger = Logger(df_path = os.path.join(data_path, model_dir, "epochs.csv"), create = False)
  batch_logger = Logger(df_path = os.path.join(data_path, model_dir,"batches,csv"), create = False)

#  #Epoch Loop

#     #Minibatch Loop
          batch_logger.log("batch_train_loss", batch_loss)
          batch_logger.log("batch_accuracy", batch_accuracy)
          batch_logger.log("batch_val_loss", batch_val_loss)
          batch_logger.log("batch_val_accuracy", batch_val_accuracy)
          batch_logger.next()

      epoch_logger.log("train_loss", loss)
      epoch_logger.log("accuracy", accuracy)
      epoch_logger.log("val_loss", val_loss)
      epoch_logger.log("val_accuracy", val_accuracy)
      epoch_logger.next()

  model_logger.log('key', value)
  batch_logger.save()
  epoch_logger.save()
  model_logger.next()

model_logger.save()
```

## Why to Use

Stores data in a pandas compatible csv file. Pandas enables you to view and vizualise model results and experiments easily.
