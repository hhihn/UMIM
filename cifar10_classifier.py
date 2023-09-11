from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Flatten
from VariationalLayer import VariationalLayer
from VariationalConvLayer import VariationalConv2D
from helper import *
from global_settings import *



class ConvClassifier(Model):
    def __init__(self, inputdim, n_classes, filter_factor=4, n_latent_units=0, temperature=0.5, n_proj_latent_units=128,
                 proj_act_fun=tf.nn.tanh, act_fun=tf.keras.layers.LeakyReLU, dkl_weight=0.0, name="t_0_e_0",
                 norm_fun=tf.keras.layers.LayerNormalization, pool_fun=tf.keras.layers.GlobalMaxPooling2D,
                 layer_params=[1, 1, 1, 2]):
        super(ConvClassifier, self).__init__()
        self.expert_name = "expert_%s" % name
        self.n_latent_units = n_latent_units
        self.filter_factor = filter_factor
        self.n_proj_latent_units = n_proj_latent_units
        self.temperature = temperature
        self.proj_act_fun = proj_act_fun
        self.act_fun = act_fun
        self.norm_fun = norm_fun
        self.pool_fun = pool_fun

        if dkl_weight > 0:
            self.dkl_weight = tf.constant(1.0)
            self.loss_w = tf.constant(10.0 ** -dkl_weight)
        else:
            self.dkl_weight = tf.constant(0.0)
            self.loss_w = tf.constant(1.0)
        self.inputdim = inputdim
        self.seed = SEED
        print("created expert %s with dklw %.3f, lossw %f" % (self.expert_name, self.dkl_weight, self.loss_w))
        input_layer = Input(inputdim)
        x = VariationalConv2D(16 * self.filter_factor, (3, 3),
                              strides=(1, 1),
                              padding='valid',
                              kernel_initializer='he_normal')(input_layer)
        x = self.act_fun()(x)
        x = self.norm_fun()(x)
        x = VariationalConv2D(16 * self.filter_factor, (3, 3),
                              strides=(1, 1),
                              padding='valid',
                              kernel_initializer='he_normal')(x)
        x = self.act_fun()(x)
        x = self.norm_fun()(x)
        x = tf.keras.layers.MaxPooling2D((3, 3))(x)
        x = VariationalConv2D(32 * self.filter_factor, (3, 3),
                              strides=(1, 1),
                              padding='valid',
                              kernel_initializer='he_normal')(x)
        x = self.act_fun()(x)
        x = self.norm_fun()(x)
        x = VariationalConv2D(32 * self.filter_factor, (3, 3),
                              strides=(1, 1),
                              padding='valid',
                              kernel_initializer='he_normal')(x)
        x = self.act_fun()(x)
        x = self.norm_fun()(x)
        x = tf.keras.layers.MaxPooling2D((3, 3))(x)
        x = VariationalConv2D(64 * self.filter_factor, (3, 3),
                              strides=(1, 1),
                              padding='same',
                              kernel_initializer='he_normal')(x)
        x = self.act_fun()(x)
        x = self.norm_fun()(x)
        x = VariationalConv2D(64 * self.filter_factor, (3, 3),
                              strides=(1, 1),
                              padding='same',
                              kernel_initializer='he_normal')(x)
        x = self.act_fun()(x)
        x = self.norm_fun()(x)
        flat_x = Flatten()(x)
        latent_embedding = VariationalLayer(units=self.n_latent_units, name='embedding_encoder_%s' % name)(flat_x)
        latent_embedding = self.act_fun()(latent_embedding)
        latent_embedding = self.norm_fun()(latent_embedding)
        proj_latent_embedding = VariationalLayer(units=self.n_proj_latent_units,
                                                 activation=self.proj_act_fun,
                                                 name='proj_embedding_encoder_%s' % name,
                                                 seed=self.seed)(flat_x)
        output_layer = VariationalLayer(units=n_classes, activation=tf.nn.softmax)(latent_embedding)
        self.model_layers = Model(input_layer, [output_layer, proj_latent_embedding])
        self.model_layers.compile()
        if "t_0" in self.expert_name:
            print(self.model_layers.summary())

    def make_basic_block_base(self, inputs, filter_num, stride=1, act_fun=tf.keras.layers.LeakyReLU,
                              norm_fun=tf.keras.layers.LayerNormalization):
        # https://github.com/acoadmarmon/resnet18-tensorflow/blob/master/resnet_18.py
        x = VariationalConv2D(filter_num,
                              kernel_size=(3, 3),
                              strides=stride,
                              kernel_initializer='he_normal',
                              padding="same")(inputs)
        x = norm_fun()(x)
        x = VariationalConv2D(filter_num,
                              kernel_size=(3, 3),
                              strides=1,
                              kernel_initializer='he_normal',
                              padding="same")(x)
        x = norm_fun()(x)

        shortcut = inputs
        if stride != 1:
            shortcut = VariationalConv2D(filter_num,
                                         kernel_size=(1, 1),
                                         strides=stride,
                                         kernel_initializer='he_normal')(inputs)
            shortcut = norm_fun()(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = act_fun()(x)
        return x

    def make_basic_block_layer(self, inputs, filter_num, blocks, stride=1, act_fun=tf.keras.layers.LeakyReLU,
                               norm_fun=tf.keras.layers.LayerNormalization):
        x = self.make_basic_block_base(inputs, filter_num, stride=stride, act_fun=act_fun, norm_fun=norm_fun)

        for _ in range(1, blocks):
            x = self.make_basic_block_base(x, filter_num, stride=1, act_fun=act_fun, norm_fun=norm_fun)

        return x

    # @tf.function(experimental_relax_shapes=True)
    def loss_call(self, input, training=True):
        if input is None:
            return None, None, None
        out, proj_latent_embedding = self.model_layers(input, training=training)
        return out, proj_latent_embedding

    @tf.function(experimental_relax_shapes=True)
    def latent_call(self, input, training=True):
        if input is None:
            return None, None, None
        out, proj_latent_embedding = self.model_layers(input, training=training)
        prediction_entropy = -tf.reduce_sum(tf.multiply(out, tf.math.log(tf.maximum(1e-3, out))), axis=-1)
        mean_prediction_entropy = tf.reduce_mean(prediction_entropy, axis=0)
        return out, proj_latent_embedding, mean_prediction_entropy

    @tf.function(experimental_relax_shapes=True)
    def call(self, input, training=True):
        if input is None:
            return None, None, None
        return self.model_layers(input, training=training)[0]

    def prior_call(self, input, training=True):
        if input is None:
            return None, None, None
        out, logprob = self.model_layers(input, training=training)
        return out, logprob

    @tf.function(experimental_relax_shapes=True)
    def add_contrastive_loss(self, hidden,
                             hidden_norm=True):
        """Compute loss for model.
        https://github.com/google-research/simclr/blob/master/tf2/objective.py
        Args:
          hidden: hidden vector (`Tensor`) of shape (bsz, dim).
          hidden_norm: whether to use normalization on the hidden vector.
          temperature: a `floating` number for temperature scaling.
          strategy: context information for tpu.
        Returns:
          A loss scalar.
          The logits for contrastive prediction task.
          The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        LARGE_NUM = 1e9
        if hidden_norm:
            hidden = tf.math.l2_normalize(hidden, -1)
        hidden1, hidden2 = tf.split(hidden, 2, 0)
        batch_size = tf.shape(hidden1)[0]

        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / self.temperature
        sm_xent = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                          reduction=losses_utils.ReductionV2.NONE)
        pred_a = tf.concat([logits_ab, logits_aa], 1)
        pred_b = tf.concat([logits_ba, logits_bb], 1)
        loss_a = sm_xent(labels, pred_a)
        loss_b = sm_xent(labels, pred_b)
        loss = loss_a + loss_b
        # loss = sm_xent(y_true=labels, y_pred=logits_ab)
        return loss  # , tf.linalg.diag_part(logits_ab)

    @tf.function(experimental_relax_shapes=True)
    def similarity(self, hidden, hidden_norm=True):
        if hidden_norm:
            hidden = tf.math.l2_normalize(hidden, -1)
        hidden1, hidden2 = tf.split(hidden, 2, 0)
        logits = tf.matmul(hidden1, hidden2, transpose_b=True)  # + tf.matmul(hidden2, hidden1, transpose_b=True)
        probs_ab = tf.nn.softmax(logits=logits / self.temperature, axis=-1)
        return tf.linalg.diag_part(probs_ab)

    @tf.function(experimental_relax_shapes=True)
    def total_similarity(self, hidden_anchor, hidden_pos, hidden_neg):
        contrastive_loss = self.add_contrastive_loss(hidden=tf.concat([hidden_anchor, hidden_pos], axis=0),
                                                     hidden_norm=True)
        contrastive_loss_neg = self.add_contrastive_loss(hidden=tf.concat([hidden_anchor, hidden_neg], axis=0),
                                                         hidden_norm=True)
        # contrastive_loss_neg_2 = self.add_contrastive_loss(hidden=tf.concat([hidden_anchor, hidden_neg_2], axis=0),
        #                                                     hidden_norm=True)
        contrastive_loss = contrastive_loss[:, tf.newaxis]
        contrastive_loss_neg = contrastive_loss_neg[:, tf.newaxis]
        # contrastive_loss_neg_2 = contrastive_loss_neg_2[:, tf.newaxis]
        return -tf.reduce_mean(tf.concat([contrastive_loss, contrastive_loss_neg], axis=-1),
                               axis=-1, keepdims=True)

    @tf.function(experimental_relax_shapes=True)
    def total_contrastive_loss(self, hidden_anchor, hidden_pos, hidden_neg):
        contrastive_loss = self.add_contrastive_loss(hidden=tf.concat([hidden_anchor, hidden_pos], axis=0),
                                                     hidden_norm=True)
        contrastive_loss_neg = self.add_contrastive_loss(hidden=tf.concat([hidden_anchor, hidden_neg], axis=0),
                                                         hidden_norm=True)
        # contrastive_loss_neg_2 = self.add_contrastive_loss(hidden=tf.concat([hidden_anchor, hidden_neg_2], axis=0),
        #                                                      hidden_norm=True)
        contrastive_loss = contrastive_loss[:, tf.newaxis]
        contrastive_loss_neg = contrastive_loss_neg[:, tf.newaxis]
        # contrastive_loss_neg_2 = contrastive_loss_neg_2[:, tf.newaxis]
        return tf.reduce_sum(tf.concat([contrastive_loss, contrastive_loss_neg], axis=-1),
                             axis=-1, keepdims=True)

    @tf.function(experimental_relax_shapes=True)
    def update(self, inputs, pos_pair, neg_pair, neg_pair_2, train_y, opt):
        train_class_loss = tf.keras.losses.CategoricalCrossentropy(reduction=losses_utils.ReductionV2.NONE)

        with tf.GradientTape() as tape:
            all_samples = tf.concat([inputs, pos_pair, neg_pair], axis=0)
            y_pred, embeddings = self.loss_call(all_samples, training=True)
            y_pred = tf.split(y_pred, 3, 0)
            proj_latent_anchor, proj_latent_pos, proj_latent_neg = tf.split(embeddings, 3, 0)
            dkl_loss = tf.reduce_sum(self.model_layers.losses)
            class_loss = tf.reduce_mean(train_class_loss(y_pred=y_pred[0], y_true=train_y))
            class_loss += tf.reduce_mean(train_class_loss(y_pred=y_pred[1], y_true=train_y))
            class_loss += tf.reduce_mean(train_class_loss(y_pred=y_pred[2], y_true=train_y))
            # class_loss += tf.reduce_mean(train_class_loss(y_pred=y_pred[3], y_true=train_y))
            total_constr_loss = tf.reduce_mean(
                self.total_contrastive_loss(proj_latent_anchor, proj_latent_pos, proj_latent_neg))
            total_loss = self.loss_w * class_loss + self.loss_w * total_constr_loss + self.dkl_weight * dkl_loss
        trainable_class_weights = self.trainable_variables
        class_grads = tape.gradient(total_loss, trainable_class_weights)
        opt.apply_gradients(zip(class_grads, trainable_class_weights))
        return class_loss, dkl_loss, total_constr_loss  # + tf.reduce_mean(contrastive_loss_pos_neg)
# sdf
