
from app.oz_rnn.rnn_manager import OzRNNManager


rnnEx = OzRNNManager()
rnnEx.make_dataset()
# model load
#rnnEx.loadModel('./FirstrnnModel.hdf5')
model_filenames = ['./rnnModel.hdf5', './FirstrnnModel.hdf5']
sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
for fname in model_filenames:
    rnnEx.loadModel('./rnnModel.hdf5')
    # rnnEx.evaluate()
    rnnEx.print_model_summary()

    predictions = rnnEx.sample_predict(sample_text, pad=False)
    print(predictions)
