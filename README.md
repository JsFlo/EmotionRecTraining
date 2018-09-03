# EmotionRecTraining

Python script using Keras + TensorFlow to train a custom machine learning model
to recognize faces using a dataset with human faces and labeled with emotions.

I wrote an article explaining this script and model architecture and that can be
found here: https://medium.com/@jsflo.dev/training-a-tensorflow-model-to-recognize-emotions-a20c3bcd6468

## Training
Script: emotionRecTrain.py

Required args:
  * `csv_file` - path to the fer2013.csv file
  * `export_path` - path to save the trained model artifacts

Optional args:
  * `batch_size` - batch size during training
  * `n_epochs` - number of training epochs
  * `debug` - Will override script arguments for batch_size and n_epocs to 10 and 1

Running
```
# required
python3 emotionRecTrain.py --csv_file=data/fer2013.csv --export_path=model_out/

python3 emotionRecTrain.py --csv_file=data/fer2013.csv --export_path=model_out/ --batch_size=50
python3 emotionRecTrain.py --csv_file=data/fer2013.csv --export_path=model_out/ --n_epochs=5000
python3 emotionRecTrain.py --csv_file=data/fer2013.csv --export_path=model_out/ --debug

```
