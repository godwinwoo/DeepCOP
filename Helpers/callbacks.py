from keras.callbacks import Callback
import datetime

class NEpochLogger(Callback):
    def __init__(self,display=100):
        '''
        display: Number of epochs to wait before outputting loss
        '''
        self.seen = 0
        self.display = display

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.display == 0:
            print('{0}: {1}/{2} - Epoch Train Loss: {3:8.7f} Val Loss: {4:8.7f}'
                  .format(datetime.datetime.now(), epoch, self.params['epochs'], logs['loss'], logs['val_loss']))
