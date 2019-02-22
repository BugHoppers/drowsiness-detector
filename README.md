# Drowsiness Detector

An application to detect drowsiness of drivers and alert them.

### Requirements:

- Python 3.6 +
- Keras
- OpenCV


### Training the Model

Run - 

```
    $ python3 model.py
```

The trained model is saved as 'eyeblink.hdf5'

* NOTE - The model is trained with the Eye images(24x24) of [Closed Eyes In The Wild (CEW) dataset](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html)


### Future Implementation

The application can be loaded into a raspberry pi and using its camera module it can be used in cars for detecting drowsiness in drivers.
This could help in preventing many car accidents cause due to drowsiness while driving.