import math
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

__all__ = ['FindLr', 'Lr1CycleSchedule', 'LRTensorBoard', 'tsne']


class FindLr(object):
    # usage: 1. get a instance
    #        2. get a callback and add to callback list for training
    #        3. use visual method and find the appropriate lr

    # if val data set is specified, the visual will use val loss instead of train loss

    def __init__(self, steps_per_epoch, x_val=None, y_val=None, lr_l=1e-8, lr_h=10., beta=0.98):
        assert steps_per_epoch > 0

        self.steps_per_epoch = steps_per_epoch
        self.x_val = x_val
        self.y_val = y_val
        if (x_val is not None) and (y_val is not None):
            self.loss_type = 'val_acc'
        else:
            self.loss_type = 'train_loss'
        print('find %s mode.' %(self.loss_type))
        self.lr_l = lr_l
        self.lr_h = lr_h
        self.beta = beta

        self.untitled_count = 0
        self.lr_dict = {}
        self.loss_dict = {}

    def get_callback(self, name):
        class LRSchedule(keras.callbacks.Callback):
            def __init__(self, findlr, name=None):
                super(LRSchedule, self).__init__()
                self.findlr = findlr
                self.name = name
                self.loss_type = findlr.loss_type
                self.lr = findlr.lr_l
                self.mult = (findlr.lr_h / findlr.lr_l) ** (1. / findlr.steps_per_epoch)
                self.avg_loss = 0.
                self.best_loss = 0.

                if name is None:
                    self.name = 'untitled_%d' %(self.findlr.untitled_count)
                    self.findlr.untitled_count += 1
                while self.name in self.findlr.lr_dict.keys():
                    print('error: name conflict in LRSchedule(%s).' %self.name)
                    self.name += '_1'

                self.findlr.lr_dict[self.name] = []
                self.findlr.loss_dict[self.name] = []

            def on_batch_end(self, batch, logs=None):
                super(LRSchedule, self).on_batch_end(batch=batch, logs=logs)

                print('\niter:%d, lr:%f' %(batch, self.lr))
                if 'train_loss' == self.loss_type:
                    loss = logs['loss']
                else:
                    loss = self.model.evaluate(self.findlr.x_val, self.findlr.y_val, batch_size=logs['size'], verbose=0)
                    if isinstance(loss, list):
                        loss = loss[1]#0:loss   1:acc

                # compute the smoothed loss
                self.avg_loss = self.findlr.beta * self.avg_loss + (1. - self.findlr.beta) * loss
                smoothed_loss = self.avg_loss / (1. - self.findlr.beta**(batch + 1))

                # stop record if the loss is exploding, note the training still goes on
                if batch and smoothed_loss > 4 * self.best_loss:
                    return

                # update the best loss
                if smoothed_loss < self.best_loss or 0 == batch:
                    self.best_loss = smoothed_loss

                # record lr and loss
                self.findlr.loss_dict[self.name].append(smoothed_loss)
                self.findlr.lr_dict[self.name].append(math.log10(self.lr))

                self.lr *= self.mult

            def on_batch_begin(self, batch, logs=None):
                super(LRSchedule, self).on_batch_begin(batch=batch, logs=logs)

                keras.backend.set_value(self.model.optimizer.lr, self.lr)

        lrschedule = LRSchedule(self, name=name)

        return lrschedule

    def visual(self):
        x_axis_min = int(math.log10(self.lr_l))
        x_axis_max = int(math.log10(self.lr_h))
        x_axis = list(range(x_axis_min, x_axis_max))

        for test_name in self.lr_dict.keys():
            lrs = self.lr_dict[test_name]
            losses = self.loss_dict[test_name]
            plt.plot(lrs, losses, label=test_name)
        plt.xticks(x_axis)
        # plt.ylim(0.9116, 0.9120)
        plt.xlabel('log_lr')
        plt.ylabel(self.loss_type)
        plt.legend(loc='lower right')
        # plt.savefig('./w_dropx0.3_0.25_wdy04.png')
        plt.savefig('./week8_test.png')
        plt.show()


class Lr1CycleSchedule(keras.callbacks.Callback):
    def __init__(self, lr_max, epochs, annihilation_period_scale):
        self.lr_1 = lr_max
        self.lr_0 = self.lr_1 / 10.
        self.lr_2 = self.lr_0
        self.lr_3 = self.lr_2 / 100.

        self.momentum_0 = 0.95
        self.momentum_1 = 0.85
        self.momentum_2 = 0.95
        self.momentum_3 = 0.95

        self.epochs = epochs
        self.period_1 = range((int(epochs - epochs*annihilation_period_scale)) // 2)
        self.period_2 = range(max(self.period_1) + 1, int(epochs - epochs*annihilation_period_scale))
        self.period_3 = range(max(self.period_2) + 1, epochs)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.period_1:
            lr = (self.lr_1 - self.lr_0) * (float(epoch - min(self.period_1)) / len(self.period_1)) + self.lr_0
            momentum = (self.momentum_1 - self.momentum_0) * (float(epoch - min(self.period_1)) / len(self.period_1)) + self.momentum_0
        elif epoch in self.period_2:
            lr = (self.lr_2 - self.lr_1) * (float(epoch - min(self.period_2)) / len(self.period_2)) + self.lr_1
            momentum = (self.momentum_2 - self.momentum_1) * (float(epoch - min(self.period_2)) / len(self.period_2)) + self.momentum_1
        elif epoch in self.period_3:
            lr = (self.lr_3 - self.lr_2) * (float(epoch - min(self.period_3) + 1) / len(self.period_3)) + self.lr_2
            momentum = (self.momentum_3 - self.momentum_2) * (float(epoch - min(self.period_3)) / len(self.period_3)) + self.momentum_2
        else:
            print('\nwarning: epochs over 1cycle policy.')
            lr = self.lr_3
            momentum = self.momentum_3

        keras.backend.set_value(self.model.optimizer.lr, lr)
        keras.backend.set_value(self.model.optimizer.momentum, momentum)

        print('\nlr:%f, momentum:%f' %(lr, momentum))


class LRTensorBoard(keras.callbacks.TensorBoard):

    def __init__(self, log_dir='./logs', **kwargs):
        super(LRTensorBoard, self).__init__(log_dir, **kwargs)

        self.lr_log_dir = log_dir

    def set_model(self, model):
        self.lr_writer = tf.summary.FileWriter(self.lr_log_dir)
        super(LRTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        lr = keras.backend.get_value(self.model.optimizer.lr)
        if hasattr(self.model.optimizer, 'momentum'):
            momentum = keras.backend.get_value(self.model.optimizer.momentum)
            summary = tf.Summary(value=[tf.Summary.Value(tag='lr', simple_value=lr),
                                        tf.Summary.Value(tag='momentum', simple_value=momentum)])
        else:
            summary = tf.Summary(value=[tf.Summary.Value(tag='lr', simple_value=lr)])
        self.lr_writer.add_summary(summary, epoch)
        self.lr_writer.flush()

        super(LRTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(LRTensorBoard, self).on_train_end(logs)
        self.lr_writer.close()


def tsne(samples, labels=None, lr=100, figure=None, save_path=None):
    """
    :param samples: ndarray, first dim is samples
    :param labels: if specified, visual img will use diff colours to mark each classes.
        ndarray with shape(n_samples)
    :param lr: lr used in TSNE
    :return np image of TSNE
    """

    from sklearn.manifold import TSNE

    import matplotlib.pyplot as plt
    from moviepy.video.io.bindings import mplfig_to_npimage as figage
    import numpy as np

    suppressed_samples = TSNE(learning_rate=lr, random_state=8).fit_transform(samples.reshape(samples.shape[0], -1))
    plt.scatter(suppressed_samples[:, 0], suppressed_samples[:, 1], c=labels.squeeze())

    plt.axis('off')
    lower_bound = np.min(suppressed_samples)
    lower_bound = lower_bound * 0.8 if lower_bound > 0 else lower_bound * 1.2
    higher_bound = np.max(suppressed_samples)
    higher_bound = higher_bound * 1.2 if higher_bound > 0 else higher_bound * 0.8
    plt.xlim(lower_bound, higher_bound)
    plt.ylim(lower_bound, higher_bound)

    if figure is not None:
        img = figage(figure)
    else:
        my_figure = plt.figure(dpi=800)
        img = figage(my_figure)

    if save_path is not None:
        import cv2
        cv2.imwrite(save_path, img)

    return img


class DynamicTsne(object):
    def __init__(self, layer, samples, labels=None):
        """
        generate dynamic tsne visualize
        :param layer: 1. str of layer name
            or 2. var of keras layer tensor
            or 3. int of layer id
        :param samples: small set of smaples
        :param labels: optional, sparse idx
        """
        self.imgs = []
        self.embeddings = []

        self._layer = layer
        self._samples = samples
        self._labels = labels

        self._figure = plt.figure(dpi=800)

    def get_callback(self, batchsize=64, epoch_end=True):
        class CollectEmbeddings(keras.callbacks.Callback):
            def __init__(self, dynamic_tsne, layer, samples, labels=None, batchsize=64, epoch_end=True, figure=None):
                super(CollectEmbeddings, self).__init__()
                self._dynamic_tsne = dynamic_tsne
                self._layer = layer
                self._samples = samples
                self._labels = labels
                self._batchsize = batchsize
                self._epoch_end = epoch_end
                self._figure = figure

            def _build_layer_predict_function(self):
                input_layer = self.model.layers[0].input
                if isinstance(self._layer, int):
                    output_layer = self.model.layers[self._layer].output
                elif isinstance(self._layer, str):
                    output_layer = self.model.get_layer(self._layer).output
                else:
                    output_layer = self._layer

                self._layer_model = keras.models.Model(input_layer, output_layer)

            def set_model(self, model):
                super(CollectEmbeddings, self).set_model(model)
                self._build_layer_predict_function()

            def on_epoch_begin(self, epoch, logs=None):
                super(CollectEmbeddings, self).on_batch_begin(epoch, logs)
                if self._epoch_end: return

                layer_output = self._layer_model.predict(self._samples, batchsize)
                self._dynamic_tsne.embeddings.append(layer_output.reshape(layer_output.shape[0], -1))

            def on_epoch_end(self, epoch, logs=None):
                super(CollectEmbeddings, self).on_epoch_end(epoch, logs)
                if not self._epoch_end: return

                layer_output = self._layer_model.predict(self._samples, batchsize)
                self._dynamic_tsne.embeddings.append(layer_output.reshape(layer_output.shape[0], -1))

        collect_embeddings = CollectEmbeddings(self, self._layer, self._samples, self._labels,
                                          batchsize=batchsize, epoch_end=epoch_end,figure=self._figure)
        return collect_embeddings

    def _make_frame(self, t):
        frame = self.imgs[self._frameid]
        self._frameid = min(len(self.imgs)-1, self._frameid + 1)
        return frame

    def save_embeddings(self, pkl_path):
        import pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.embeddings, f)

    def load_embeddings(self, pkl_path):
        import pickle
        with open(pkl_path, 'rb') as f:
            self.embeddings = pickle.load(f)

    def _generate_imgs(self):
        import matplotlib.pyplot as plt
        from moviepy.video.io.bindings import mplfig_to_npimage as figage
        import numpy as np
        from tsne_model.dynamic_tsne import dynamic_tsne


        self.imgs = []
        suppressed_samples = dynamic_tsne(self.embeddings, perplexity=70, lmbda=0.1, verbose=1, sigma_iters=50)
        for samples in suppressed_samples:
            plt.scatter(samples[:, 0], samples[:, 1], c=self._labels.squeeze())

            plt.axis('off')
            lower_bound = np.min(suppressed_samples)
            lower_bound = lower_bound * 0.8 if lower_bound > 0 else lower_bound * 1.2
            higher_bound = np.max(suppressed_samples)
            higher_bound = higher_bound * 1.2 if higher_bound > 0 else higher_bound * 0.8
            plt.xlim(lower_bound, higher_bound)
            plt.ylim(lower_bound, higher_bound)

            img = figage(self._figure)
            self.imgs.append(img)

    def generate_video(self, save_path, fps=30):
        import moviepy.editor as mpe

        self._generate_imgs()
        self._frameid = 0
        video = mpe.VideoClip(self._make_frame, duration=len(self.imgs) / float(fps))

        video.write_videofile(save_path, codec='mpeg4', fps=fps)




