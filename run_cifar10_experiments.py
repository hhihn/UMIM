import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import logging
from pickle import dump, load
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
from tensorflow.python.util import deprecation
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler

deprecation._PRINT_DEPRECATION_WARNINGS = False
import matplotlib.pyplot as plt
from collections import deque
from helper import *
from global_settings import *
from cifar10_classifier import ConvClassifier

tf.compat.v1.enable_eager_execution()
tf.get_logger().setLevel('ERROR')
tfd = tfp.distributions
from datetime import datetime, timedelta

tf.autograph.set_verbosity(0)
tf.random.set_seed(SEED)
np.random.seed(SEED)
num_classes = 10
input_shape = (32, 32, 3)

f, ax = plt.subplots(1, 6)


def run_experiment(params, x_train, y_train, x_test, y_test):
    tf.keras.backend.clear_session()
    pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    input_dim = (32, 32, 3)
    dkl_weight = params["dkl_weight"]
    batch_size = params["batch_size"]
    eval_batch_size = 64
    n_hidden_units = params["n_hidden_units"]
    lr = params["lr"]
    verbose = params["verbose"]
    epochs = params["epochs"]
    temperature = params["temperature"]
    proj_act_fun = params["proj_act_fun"]
    act_fun = params["act_fun"]
    norm_fun = params["norm_fun"]
    pool_fun = params["pool_fun"]
    n_latent_units = params['n_latent_units']
    filter_factor = params['filter_factor']
    new_model_threshold = 5000
    old_data = []
    old_models = []
    num_models = 1
    max_num_models = params["num_models"]
    accuracy_matrix = np.zeros(shape=(len(pairs), len(pairs)))

    elapsed_seconds = deque(maxlen=100000)
    elapsed_seconds.append(0.0)
    all_matching_scores = deque(maxlen=100000)
    all_matching_scores.append(0.0)
    for task_id, classes in enumerate(pairs):
        if verbose:
            print("=" * 50)
            print("| \t\t Classes ", classes, "\t\t |")
            print("=" * 50)
        if verbose:
            print("\033[94m building model \033[0m")
        models = []
        opts = []
        for ni in range(num_models):
            model = ConvClassifier(inputdim=input_dim, n_classes=num_classes, n_proj_latent_units=n_hidden_units,
                                   n_latent_units=n_latent_units, proj_act_fun=proj_act_fun, temperature=temperature,
                                   norm_fun=norm_fun, act_fun=act_fun, pool_fun=pool_fun,
                                   filter_factor=filter_factor, name="t_%d_e_%d" % (task_id, ni),
                                   dkl_weight=6.0 if ni < task_id else 0.0)
            models.append(model)
            opts.append(tf.keras.optimizers.Adam(lr*0.01 if ni < task_id else lr))  # , beta_1=0.9, beta_2=0.99, decay=1e-4))
        if verbose:
            print("\033[94m done building model \033[0m")

        if len(old_models):
            if verbose:
                print("\033[94m copying weights for task %d \033[0m" % task_id)
            for old_model, model in zip(old_models, models):
                copy_weights(old_model, model)
            del old_models
            if verbose:
                print("\033[94m done \033[0m")
        if verbose:
            print("\033[94m building data set \033[0m")
        orig_curr_x_train, orig_curr_y_train = extract_classes(classes, x_train, y_train, num_classes=num_classes)
        train_dataset = build_dataset(x=orig_curr_x_train, y=orig_curr_y_train, batch_size=batch_size,
                                      epochs=epochs)
        if verbose:
            print("\033[94m done building data set \033[0m")
            print("\033[94m %d samples in %d classes \033[0m" % (len(orig_curr_x_train), len(classes)))
            print("\033[94m aggregated into %d batches of size %d \033[0m" % (len(train_dataset), batch_size))
        centr_ls = [0.0]
        eval_accuracies = [0.0]
        curr_x_test, curr_y_test = extract_classes(classes, x_test, y_test, num_classes=num_classes)
        old_data.append([curr_x_test, curr_y_test])
        steps_per_epoch = len(train_dataset) / epochs
        eval_step_size = 1e9#int(steps_per_epoch * (epochs / 5))
        j = 0
        first_run = task_id == 0
        if verbose:
            print("\033[94m starting training \033[0m")
            print("\033[94m evaluating every %d steps \033[0m" % eval_step_size)
        sample_partition = [1 for _ in range(max_num_models)]
        unmatched_samples = []
        unmatched_targets = []
        similarity_threshold = np.percentile(all_matching_scores, 95)
        new_model_wait = int(steps_per_epoch * epochs)
        new_model_cooldown = 0
        task_mi = 0.0
        train_steps = 0
        all_contrs_ls = deque(maxlen=int(steps_per_epoch * 10))
        all_contrs_ls.append(0.0)
        all_class_ls = deque(maxlen=int(steps_per_epoch * 10))
        all_class_ls.append(0.0)
        uniformities = [0.0]
        for x, xxp, xxn, xxx, y in train_dataset:
            t_start = datetime.now()
            dispatched_inputs, dispatched_pos_pairs, dispatched_neg_pairs, dispatched_neg_pairs_2, \
            dispatched_outputs, idxs, matching_scores, _ = dispatch(models, x,
                                                                    xxp=xxp,
                                                                    xxn=xxn,
                                                                    xxx=xxx,
                                                                    y=y,
                                                                    first_run=first_run,
                                                                    max_num_models=max_num_models,
                                                                    similarity_threshold=similarity_threshold,
                                                                    new_model_cooldown=new_model_cooldown)
            if not first_run and len(dispatched_inputs) > len(models):
                unmatched_samples.extend(dispatched_inputs[-1])
                unmatched_targets.extend(dispatched_outputs[-1])
                if len(unmatched_samples) >= new_model_threshold and len(models) < max_num_models:
                    unmatched_samples = np.stack(unmatched_samples, axis=0)
                    if verbose:
                        print("\033[92m=" * 50)
                        print("| creating new model %d with %d samples \t|" % (num_models, len(unmatched_samples)))
                        print("=" * 50)
                    new_model = ConvClassifier(inputdim=input_dim, n_classes=num_classes,
                                               n_proj_latent_units=n_hidden_units,
                                               n_latent_units=n_latent_units, proj_act_fun=proj_act_fun,
                                               temperature=temperature,
                                               norm_fun=norm_fun, act_fun=act_fun, pool_fun=pool_fun,
                                               filter_factor=filter_factor, name="t_%d_e_%d" % (task_id, num_models+1),
                                               dkl_weight=0.0)
                    new_train_dataset = build_dataset(x=unmatched_samples, y=unmatched_targets, batch_size=batch_size,
                                                      epochs=100)
                    copy_weights(old_model=models[-1], new_model=new_model)
                    opts.append(tf.keras.optimizers.Adam(lr))
                    if verbose:
                        print("| training new model %d with %d batches \t |" % (num_models + 1, len(new_train_dataset)))
                    new_expert_contrastive_ls = []
                    new_expert_class_ls = []
                    new_expert_ctr_ls = []
                    new_train_itr = 1
                    for x, xxp, xxn, xxx, y in new_train_dataset:
                        if verbose:
                            print("batch %d of %d" % (new_train_itr, len(new_train_dataset)), end="\r")
                        class_l, dkl_loss, contr_l = new_model.update(inputs=x, pos_pair=xxp, neg_pair=xxn,
                                                                      neg_pair_2=xxx, train_y=y,
                                                                      opt=opts[num_models])
                        new_expert_contrastive_ls.append(np.mean(contr_l))
                        new_expert_class_ls.append(np.mean(class_l))
                        new_expert_ctr_ls.append(np.mean(dkl_loss))
                        new_train_itr += 1
                    models.append(new_model)
                    if verbose:
                        print("| done \t|")
                        print("| statistics: \t|")
                        print("| class l: %.3f \t|" % np.mean(new_expert_class_ls))
                        print("| ctr l: %.3f \t|" % np.mean(new_expert_contrastive_ls))
                        print("=" * 50 + "\033[0m")
                    num_models += 1
                    new_model_cooldown = new_model_wait
                    unmatched_samples = []
                    unmatched_targets = []
            if matching_scores is not None:
                all_matching_scores.extend(matching_scores)
            expert_contrastive_ls = []
            expert_class_ls = []
            expert_model_ls = []
            for expert_inputs, expert_pos_pairs, expert_neg_pairs, expert_neg_pairs_2, expert_outputs, model, opt, \
                mi in zip(dispatched_inputs, dispatched_pos_pairs,
                          dispatched_neg_pairs, dispatched_neg_pairs_2, dispatched_outputs,
                          models, opts, range(num_models)):
                if len(expert_inputs):# and mi >= task_id:
                    class_l, dkl_loss, contr_l = models[mi].update(inputs=expert_inputs,
                                                                   pos_pair=expert_pos_pairs,
                                                                   neg_pair=expert_neg_pairs,
                                                                   neg_pair_2=expert_neg_pairs_2,
                                                                   train_y=expert_outputs, opt=opt)
                    expert_contrastive_ls.append(np.mean(contr_l))
                    expert_class_ls.append(np.mean(class_l))
                    expert_model_ls.append(np.mean(dkl_loss))
                    sample_partition[mi] += 1.0 #np.shape(expert_inputs)[0]
            expert_contrastive_ls = np.mean(expert_contrastive_ls) if len(expert_contrastive_ls) else 0.0
            expert_class_ls = np.mean(expert_class_ls) if len(expert_class_ls) else 0.0
            expert_model_ls = np.mean(expert_model_ls) if len(expert_model_ls) else 0.0
            all_contrs_ls.append(expert_contrastive_ls)
            if verbose:
                print(
                    "task %d: %d of %d \t acc: %.2f \t train acc: %.3f  \t class l: %.3f \t mdl l: %.3f \t ctr contr l: %.3f \t task mi l: %.2f \t ETA: %s" % (
                        task_id + 1, j + 1, len(train_dataset),
                        np.mean(eval_accuracies),
                        eval_accuracies[-1],
                        np.mean(expert_class_ls),
                        np.mean(expert_model_ls),
                        np.mean(all_contrs_ls),
                        task_mi,
                        str(timedelta(seconds=np.mean(elapsed_seconds) * (len(train_dataset) - train_steps)))), end="\r")
            new_model_cooldown -= 1
            centr_ls.append(expert_contrastive_ls)
            j = j + 1
            if j % eval_step_size == 0:
                eval_accuracies = []
                const_class_accs = []
                num_evals = 1
                task_to_gate_matrix = np.ones(shape=(len(old_data), num_models))
                for task_i, (old_x_test, old_y_test) in enumerate(old_data):
                    old_train_dataset = build_dataset(x=old_x_test, y=old_y_test, batch_size=eval_batch_size, epochs=num_evals)
                    task_acc = 0.0
                    embeddings = [[] for _ in range(num_models)]
                    targets = [[] for _ in range(num_models)]
                    for xo, xxo, xxxo, xxxxo, yo in old_train_dataset:
                        dispatched_inputs, _, _, _, dispatched_outputs, _, _, const_class_acc = dispatch(models,
                                                                                                         x=xo,
                                                                                                         xxp=xxo,
                                                                                                         xxn=xxxo,
                                                                                                         xxx=xxxxo,
                                                                                                         y=yo,
                                                                                                         eval=True,
                                                                                                         task_id=task_id)
                        const_class_accs.append(const_class_acc)
                        acc, embds = eval_experts(models, dispatched_inputs, dispatched_outputs)
                        for ei, e in enumerate(embds):
                            embeddings[ei].extend(e)
                            targets[ei].extend(dispatched_outputs[ei])
                        task_acc += acc
                        for dmi, di in enumerate(dispatched_inputs):
                            task_to_gate_matrix[task_i, dmi] += dispatched_inputs[dmi].shape[0]
                    eval_accuracies.append(task_acc / len(old_train_dataset))
                    task_to_gate_matrix[task_i, :] = task_to_gate_matrix[task_i, :] / np.sum(
                        task_to_gate_matrix[task_i, :])
                    eval_const_task_acc = np.mean(const_class_accs, axis=0)
                task_mi = task_mi_from_mat(task_to_gate_matrix)

            t_end = datetime.now()
            t_delta = t_end - t_start
            elapsed_seconds.append(t_delta.total_seconds())
            train_steps = train_steps + 1
        # uniformities = uniformity(embeddings)
        # alignments = alignment(x=x_test, y=y_test, num_classes=(task_id+1)*2, models=models)
        if verbose:
            print("\n")
            print("Training Statistics:")
            print(
                "task %d: %d of %d \n acc: %.2f \n train acc: %.3f \n class l: %.3f \n ctr contr l: %.3f \n ctr rec l: %.2f \n task mi l: %.2f \n match score: %.2f " % (
                    task_id + 1, j + 1, len(train_dataset),
                    np.mean(eval_accuracies),
                    eval_accuracies[-1],
                    np.mean(all_class_ls),
                    np.mean(all_contrs_ls),
                    expert_model_ls,
                    task_mi,
                    similarity_threshold))
            # print("const task acc:", eval_const_task_acc)
            # print("uniformities:", uniformities, np.mean(uniformities))
            # print("alignment:", alignments, np.mean(alignments))
            print("sample partitions:", sample_partition)
        accuracies = []
        prior_accuracies = []
        if verbose:
            print("\033[94m done training \033[0m")
            print("\033[94m evaluation \033[0m")
        num_evals = 1
        task_to_gate_matrix = np.ones(shape=(len(old_data), num_models))
        for task_i, (old_x_test, old_y_test) in enumerate(old_data):
            old_train_dataset = build_dataset(x=old_x_test, y=old_y_test, batch_size=eval_batch_size, epochs=num_evals)
            task_acc = 0.0
            for xo, xxo, xxxo, xxxxo, yo in old_train_dataset:
                dispatched_inputs, _, _, _, dispatched_outputs, _, _, _ = dispatch(models,
                                                                                   x=xo,
                                                                                   xxp=xxo,
                                                                                   xxn=xxxo,
                                                                                   xxx=xxxxo,
                                                                                   y=yo,
                                                                                   eval=True,
                                                                                   task_id=task_id)
                acc, e = eval_experts(models, dispatched_inputs, dispatched_outputs)
                task_acc += acc
                for dmi, di in enumerate(dispatched_inputs):
                    task_to_gate_matrix[task_i, dmi] += dispatched_inputs[dmi].shape[0]
            mean_task_acc = task_acc / len(old_train_dataset)
            accuracy_matrix[task_i, task_id] = mean_task_acc
            accuracies.append(mean_task_acc)
            task_to_gate_matrix[task_i, :] = task_to_gate_matrix[task_i, :] / np.sum(
                task_to_gate_matrix[task_i, :])
        old_models = models
        del models
        if verbose:
            task_mi = task_mi_from_mat(task_to_gate_matrix)
            print("accuracies:", accuracies, np.mean(accuracies))
            print("prior accuracies:", prior_accuracies, np.mean(prior_accuracies))
            print("task parition: \n", task_to_gate_matrix)
            print("Task MI: %.4f" % task_mi)
            print("done")
    acc, avg_bwt, bwt = bwt_from_mat(accuracy_matrix)
    task_mi = task_mi_from_mat(task_to_gate_matrix)
    if verbose:
        print(accuracy_matrix)
        print("Avg. ACC: %.4f" % acc)
        print("BWT: %.4f" % bwt)
        print("Avg. BWT: %.4f" % avg_bwt)
        print("Avg. REM: %.4f" % (1.0 - np.abs(min(avg_bwt, 0.0))))
        print("Avg. PBWT: %.4f" % max(avg_bwt, 0.0))
        print("Task MI: %.4f" % task_mi)
    return acc, bwt, avg_bwt, (1.0 - np.abs(min(avg_bwt, 0.0))), max(avg_bwt, 0.0), task_mi, accuracy_matrix, task_to_gate_matrix


def mean_experiment(params):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    print("x_train shape:", x_train.shape)

    # Build the model.
    n_runs = 3
    mean_acc = 0.0
    acc, bwt, avg_bwt, rem, pwt, tmi, acc_mats, ttgms = [], [], [], [], [], [], [], []
    for i in range(n_runs):
        acci, bwti, avg_bwti, remi, pwti, tmii, acm, ttgm = run_experiment(params, x_train, y_train, x_test, y_test)
        acc.append(acci)
        bwt.append(bwti)
        avg_bwt.append(avg_bwti)
        rem.append(remi)
        pwt.append(pwti)
        tmi.append(tmii)
        acc_mats.append(acm)
        ttgms.append(ttgm)
        session.report({"training_iteration": i + 1, "mean_loss": mean_acc / (i + 1)})
    fname = "cifar10_data_%d_experts.pkl" % params["num_models"]
    dump(file=open(fname, "wb"), obj=[acc, bwt, avg_bwt, rem, pwt, tmi, acc_mats, ttgms, acc_mats, ttgms])
    acc, bwt, avg_bwt, rem, pwt, tmi, acc_mats, ttgms, acc_mats, ttgms = load(open(fname, "rb"))
    print("Avg. ACC: %.4f +/- %.2f" % (np.mean(acc), np.std(acc)))
    print("BWT: %.4f +/- %.2f" % (np.mean(bwt), np.std(bwt)))
    print("Avg. BWT: %.4f +/- %.2f" % (np.mean(avg_bwt), np.std(avg_bwt)))
    print("Avg. REM: %.4f +/- %.2f" % (np.mean(rem), np.std(rem)))
    print("Avg. PBWT: %.4f +/- %.2f" % (np.mean(pwt), np.std(pwt)))
    print("Task MI: %.4f +/- %.2f" % (np.mean(tmi), np.std(tmi)))
    acc_mats = np.array(acc_mats)
    ttgms = np.array(ttgms)
    print("Accuracy Matrix:")
    print(np.mean(acc_mats, axis=0))
    print(np.std(acc_mats, axis=0))
    print("Task To Gate Matrix:")
    print(np.mean(ttgms, axis=0))
    print(np.std(ttgms, axis=0))


# return mean_acc / n_runs


if __name__ == "__main__":
    optimize = False
    if optimize:
        scheduler = AsyncHyperBandScheduler(grace_period=1, max_t=3, time_attr="training_iteration")
        stopping_criteria = {"training_iteration": 3}
        tuner = tune.Tuner(
            tune.with_resources(mean_experiment, {"cpu": 2, "gpu": 0.3}),
            run_config=air.RunConfig(
                stop=stopping_criteria,
                name="hyper_search",
                verbose=1,
            ),
            tune_config=tune.TuneConfig(
                metric="mean_loss", mode="min", num_samples=100, scheduler=scheduler
            ),
            param_space={  # Hyperparameter space
                'batch_size': tune.choice([16, 32, 64, 128, 256, 512]),
                'n_hidden_units': tune.choice([16, 32, 64, 128, 256]),
                'n_latent_units': tune.choice([64, 128, 256]),
                'proj_act_fun': tf.nn.tanh,
                'act_fun': tf.keras.layers.LeakyReLU,
                'norm_fun': tf.keras.layers.LayerNormalization,
                'pool_fun': tf.keras.layers.GlobalMaxPooling2D,
                'epochs': 50,
                'lr': tune.loguniform(1e-4, 1e-2),
                'temperature': tune.loguniform(1e-3, 10.0),
                'filter_factor': tune.choice([1, 2, 3, 4]),
                'verbose': False,
                'dkl_weight': 1e-2
            },
        )
        results = tuner.fit()
        print("Best hyperparameters found were: ", results.get_best_result().config)
    else:
        params = {'batch_size': 64,
                  'n_hidden_units': 128,
                  'n_latent_units': 128,
                  'proj_act_fun': tf.nn.tanh,
                  'act_fun': tf.keras.layers.LeakyReLU,
                  'norm_fun': tf.keras.layers.LayerNormalization,
                  'pool_fun': tf.keras.layers.GlobalMaxPooling2D,
                  'epochs': 50,
                  'lr': 0.001,
                  'temperature': 0.1,
                  'filter_factor': 4,
                  'dkl_weight': 1e-2,
                  'num_models': 1,
                  'verbose': True}
        print("Mean Accuracy:", mean_experiment(params))
