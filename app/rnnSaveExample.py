
from app.oz_rnn.rnn_manager import OzRNNManager


rnnEx = OzRNNManager()
rnnEx.make_dataset()
# model make
rnnEx.make_model(2)
# model fit
rnnEx.fit()
# chart print
rnnEx.plot_graphs( 'accuracy')
rnnEx.plot_graphs( 'loss')

# test
sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
print(rnnEx.sample_predict(sample_pred_text, True))

