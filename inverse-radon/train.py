import tensorflow as tf
from losses import *
from matplotlib import pyplot as plt
from radon import radon, sino2multi
from generator import norm, im_norm

class Train():
    def __init__(self, models):
        self.models = models
        self.losses = Losses(self.models)
        self.train_loss_results = []
        self.test_loss_results = []
        self.g_epoch = 1
    
    def reset(self):
        self.train_loss_results = []
        self.test_loss_results = []
        self.g_epoch = 1
    
    def get_results(self):
        return self.train_loss_results, self.test_loss_results

    def train(self, num_epochs, dataset, im_shape, lr=1e-7):
        slice_net = self.models['slice_net']

        #optimizer_disc_realfake = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer_generator = tf.keras.optimizers.Adam(learning_rate=lr)

        slice_net.trainable = True

        for epoch in range(num_epochs):
            epoch_loss_avg_slice = tf.keras.metrics.Mean()

            epoch_loss_test_avg_slice = tf.keras.metrics.Mean()

            for x in dataset:
                # Optimize the model

                with tf.GradientTape() as tape:
                    slice_loss = self.losses.loss_f(x['sino'], x['pov'], im_shape)
                slice_grads = tape.gradient(slice_loss, slice_net.trainable_variables)
                optimizer_generator.apply_gradients(zip(slice_grads, slice_net.trainable_variables))
                epoch_loss_avg_slice(slice_loss)

            self.train_loss_results.append(epoch_loss_avg_slice.result())

#             dataset.switchMode('test')

#             for x in dataset:
#                 test_loss = self.losses.generator_loss_f(x['frame1'], x['frame2'], x['input_intrinsics'], weights_generator)

#                 epoch_loss_test_avg_generator(test_loss[0])

#             self.test_loss_results.append(epoch_loss_test_avg_generator.result())

            if epoch % 1 == 0:
                print('Epoch ', self.g_epoch + epoch,
                      '; Loss: ', self.train_loss_results[-1].numpy(), 
                      #'; Depth loss: ', self.train_loss_results['generator'][-1][1].numpy(), 
                      #'; Disc loss: ', self.train_loss_results['disc'][-1][0].numpy(),
                      )
                #print('; gen_rf: ', self.train_loss_results['encdec'][-1][7].numpy())

            if epoch % 1 == 0:
                #dataset.switchMode('train')
                x = dataset[0]
                
                pred_slice = slice_net(sino2multi(x['sino'], im_shape))

                fig=plt.figure(figsize=(10, 10))

                fig.add_subplot(1,5,1)
                plt.imshow(x['sino'][0].numpy())
                fig.add_subplot(1,5,2)
                plt.imshow(x['gt'][0].numpy())
                fig.add_subplot(1,5,3)
                plt.imshow(x['rec'][0].numpy())
                fig.add_subplot(1,5,4)
                plt.imshow(pred_slice[0].numpy())
                fig.add_subplot(1,5,5)
                plt.imshow(norm(radon(pred_slice, x['pov']), im_shape[1])[0].numpy())
                
                plt.show()


                #print(x['attrs'])

            dataset.on_epoch_end()