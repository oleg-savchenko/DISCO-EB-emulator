#!/usr/bin/env python

import numpy as np
import jax.numpy as jnp
import jax
import optax
from jax import jit, vmap, value_and_grad, random, grad
from jax.nn import sigmoid
from tqdm import tqdm, trange
from sklearn.metrics import r2_score
from functools import partial
import pickle

model_parameters = ['omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s', 'z'] 

def init_fn(Parameters = None,
            modes = None,
            parameters_mean = None,
            parameters_std = None,
            features_mean = None,
            features_std = None,
            n_hidden = [512, 512, 512], 
            restore = False, 
            restore_filename = None,
            verbose = False,
            optimiser = optax,
            seed = random.PRNGKey(5)):

    '''Constructor'''

    #restore from file or train new emulators
    if restore is True:
        restore(restore_filename)
    else:
        #for saving
        global parameters
        global n_layers
        global architecture
        global n_parameters
        global n_modes
        global N_hidden
        global Modes
        Modes = modes
        parameters = Parameters
        N_hidden = n_hidden
        n_parameters = len(Parameters)
        n_modes = len(modes)
        architecture = [n_parameters] + n_hidden + [n_modes]
        n_layers = len(architecture) - 1
    
        parameters_mean_ = parameters_mean if parameters_mean is not None else jnp.zeros(n_parameters)
        parameters_std_ = parameters_std if parameters_std is not None else jnp.ones(n_parameters)

        features_mean_ = features_mean if features_mean is not None else np.zeros(n_modes)
        features_std_ = features_std if features_std is not None else np.ones(n_modes)

    #initialise weights and hyper-parameters
    weights = []
    h_params = []
    for i in range(n_layers):
        W = random.normal(key = seed, shape = [architecture[i+1], architecture[i]])*1e-3
        b = jnp.zeros(shape = [architecture[i+1]])
        weights.append([W, b])
    for i in range(n_layers-1):
        alphas = random.normal(key = seed, shape = [architecture[i+1]])*1e-3
        betas = random.normal(key = seed, shape = [architecture[i+1]])*1e-3
        h_params.append([alphas, betas])
    
    if restore is True:
        for i in range(n_layers):
            weights[i].set(weights_[i])

        for i in range(n_layers-1):
            h_params[i].set(h_params_)

    if verbose:
        multiline_str = "\nInitialized cosmopower_NN model, \n" \
                        f"mapping {n_parameters} input parameters to {n_modes} output modes, \n" \
                        f"using {len(n_hidden)} hidden layers, \n" \
                        f"with {list(n_hidden)} nodes, respectively. \n"
        print(multiline_str)
    return weights, h_params, seed


def update_emulator_parameters_and_save(weights, h_params, filename):

    '''Update and save parameters'''
    weights_ = [weights[i] for i in range(n_layers)]
    h_params_ = [h_params[i] for i in range(n_layers-1)]

     # put mean and std parameters to JAX arrays
    parameters_mean_ = jnp.asarray(parameters_mean)
    parameters_std_ = jnp.asarray(parameters_std)
    features_mean_ = jnp.asarray(features_mean)
    features_std_ = jnp.asarray(features_std)

    attributes = [weights_, 
                  h_params_, 
                  parameters_mean_, 
                  parameters_std_,
                  features_mean_,
                  features_std_,
                  n_parameters,
                  parameters,
                  n_modes,
                  Modes,
                  N_hidden,
                  n_layers,
                  architecture]
            
    # save attributes to file
    with open(filename + ".pkl", 'wb') as f:
        pickle.dump(attributes, f)


def restore(filename):

    
    with open(filename + ".pkl", 'rb') as f:
        weights_, h_params_, \
        parameters_mean_, parameters_std_, \
        features_mean_, features_std_, \
        n_parameters, parameters, \
        n_modes, modes, \
        n_hidden, n_layers, architecture = pickle.load(f)

    
def dict_to_ordered_array(input_dict, parameters):
    '''Order  model parameters'''
    if parameters is not None:
        return jnp.stack([input_dict[k] for k in parameters], axis=1)
    else:
        return jnp.stack([input_dict[k] for k in input_dict], axis=1)

def activation(x, 
               a, 
               b):
    '''activation function'''
    return jnp.multiply(jnp.add(b, jnp.multiply(sigmoid(jnp.multiply(a, x)), jnp.subtract(1., b))), x)


def forward_pass(weights,
                 h_params,
                 input_vec):

    act = []
    layer_out = [(input_vec - parameters_mean)/parameters_std]

    for i in range(len(weights[:-1])):
        w, b = weights[i]
        alpha, beta = h_params[i]
        act.append(jnp.dot(layer_out[-1], w.T) + b)
        layer_out.append(activation(act[-1], alpha, beta))


    #final layer prediction (no activations)
    w, b = weights[-1]
    preds = jnp.dot(layer_out[-1], w.T) + b[-1]

    #rescale
    preds = preds * features_std + features_mean
    return preds.squeeze()

def predictions(weights, h_params,
                parameters_arr):
    '''make predictions'''
    #parameters_arr = dict_to_ordered_array(parameters_dict)
    return vmap(forward_pass, in_axes = (None, None, 0))(weights, h_params, parameters_arr)

def _derivative(weights, h_params,
                parameters_arr):
    '''compute the logarithmic derivative of the predictions w.r.t. the input log features'''
    # parameters_arr = (parameters_arr - parameters_mean)/parameters_std
    diff = np.log(10) * jax.jacfwd(forward_pass, argnums=2)(weights, h_params, parameters_arr)
    return diff

def derivative(weights, h_params,
                parameters_arr):
    '''compute the derivative of the predictions w.r.t. the input batch'''
    return vmap(_derivative, in_axes=(None, None, 0))(weights, h_params, parameters_arr)

def ten_to_predictions(weights, h_params, input_vec):
    '''predictions to the power of 10'''
    return 10.** predictions(weights, h_params, input_vec)

def MSE(weights, h_params, data):
    param_data, actual = data
    preds = forward_pass(weights, h_params, param_data)
    preds = preds.squeeze()
    return jnp.sqrt(jnp.power(actual - preds, 2).mean())

def MSE_Sobolev(weights, h_params, data, _lambda=0.0):
    param_data, actual, actual_diff = data
    preds = forward_pass(weights, h_params, param_data)
    preds = preds.squeeze()
    # preds_diff = derivative(weights, h_params, param_data)
    return jnp.sqrt(jnp.power(actual - preds, 2).mean()) #+ _lambda * jnp.sqrt(jnp.power(actual_diff - preds_diff, 2).mean())

def TrainModelInBatchesSobolev(training_parameters,
                        training_features,
                        diff_training_features,
                        filename_saved_model, 
                        n_hidden,
                        _lambda = 1.0,
                        modes = np.logspace(-4, 2, 420),
                        validation_split = 0.1, 
                        learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],  
                        batch_sizes=[1024, 1024, 1024, 1024, 1024], 
                        patience_values = [100, 100, 100, 100, 100], 
                        max_epochs = [1000, 1000, 1000, 1000, 1000],
                        seed = random.PRNGKey(5)):
    '''Training function'''
    weights, h_params, seed = init_fn(Parameters = model_parameters,
                                      n_hidden = n_hidden,
                                      modes = modes, #[0], # two outputs: r(z) and dzdr
                                      verbose = True, # useful to understand the different steps in initialisation and training
                                      seed = seed)
    optimiser = optax.inject_hyperparams(optax.adam)(learning_rates[0])
    opt_state = optimiser.init(weights)
    hyper_opt_state = optimiser.init(h_params)
    
    

    @jit
    def update(weights, h_params, opt_state, hyper_opt_state, data, learning_rate):
        #find losses and gradients
        loss, gradients = value_and_grad(MSE_Sobolev, argnums = 0)(weights, h_params, data, _lambda = _lambda)
        gradients1 = grad(MSE_Sobolev, argnums = 1)(weights, h_params, data, _lambda = _lambda)

        #optimizer update
        updates, opt_state = optimiser.update(gradients, opt_state)
        updates1, hyper_opt_state = optimiser.update(gradients1, hyper_opt_state)

        #weight/hyper_param update
        weights = optax.apply_updates(weights, updates)
        h_params = optax.apply_updates(h_params, updates1)
        return weights, h_params, loss, opt_state, hyper_opt_state



    def training_step(weights,
                      h_params,
                      epoch,
                      opt_state,
                      hyper_opt_state,
                      batches,
                      batch_size,
                      param_train1,
                      feature_train1,
                      learning_rate,
                      ):



        losses = []
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
            else:
                start, end = int(batch*batch_size), None

            #Single batch of data
            X_batch, Y_batch, diff_Y_batch = training_parameters[start:end], training_features[start:end], diff_training_features[start:end]

            data = X_batch, Y_batch, diff_Y_batch

            weights, h_params, loss, opt_state, hyper_opt_state = update(weights, h_params, opt_state, hyper_opt_state, data, learning_rate)


            losses.append(loss)

        return weights, h_params, losses, opt_state, hyper_opt_state


   # if verbose:
    #    multiline_str = "Starting cosmopower_NN training, \n" \
     #                   f"using {int(100*validation_split)} per cent of training samples for validation. \n" \
      #                  f"Performing {len(learning_rates)} learning steps, with \n" \
       ###                 f"{list(learning_rates)} learning rates \n" \
          #              f"{list(batch_sizes)} batch sizes \n" \
           #             f"{list(patience_values)} patience values \n" \
            #            f"{list(max_epochs)} max epochs \n"
        #print(multiline_str)

    #training_parameters = dict_to_ordered_array(training_parameters, model_parameters)

    training_parameters = jnp.asarray(training_parameters)

    global parameters_mean
    global parameters_std
    global features_mean
    global features_std
    global diff_features_mean
    global diff_features_std
    
    parameters_mean = jnp.mean(training_parameters, axis=0)
    parameters_std = jnp.std(training_parameters, axis=0)

    # training_features = np.log10(training_features + 1e-30)
    features_mean = jnp.mean(training_features, axis=0)
    features_std = jnp.std(training_features, axis=0)

    diff_features_mean = jnp.mean(diff_training_features, axis=0)
    diff_features_std = jnp.std(diff_training_features, axis=0)

    n_validation = int(training_parameters.shape[0] * validation_split)

    n_training = training_parameters.shape[0] - n_validation

    for i in range(len(learning_rates)):
        print('learning rate = ' + str(learning_rates[i]) + ', batch size = ' + str(batch_sizes[i]))

        #update learning rate
        learning_rate = learning_rates[i]  

        #'inject' new learning rate
        opt_state.hyperparams['learning_rate'] = learning_rate
        hyper_opt_state.hyperparams['learning_rate'] = learning_rate

        #new subkey for shuffling
        seed, subkey = random.split(seed)

        #create shuffler for data
        training_selection = random.permutation(key = subkey, x = jnp.asarray(([True] * n_training + [False] * n_validation)), independent = True)

        #shuffle data
        param_train1 = training_parameters[training_selection]
        feature_train1 = training_features[training_selection]
        param_val = training_parameters[~training_selection]
        feature_val = training_features[~training_selection]

        validation = param_val, feature_val

        #initialise outputs/boundaries
        validation_loss = [np.infty]
        best_loss = np.infty
        early_stopping_counter = 0

        #set batch size
        batch_size = batch_sizes[i]
        batches = jnp.arange((param_train1.shape[0]//batch_size)+1)

        with trange(max_epochs[i]) as t:

            for epoch in t:

                #training step
                weights, h_params, losses, opt_state, hyper_opt_state = training_step(weights, h_params, epoch,
                                                                                      opt_state, hyper_opt_state,
                                                                                      batches, batch_size, param_train1,
                                                                                      feature_train1, learning_rate)


                validation_loss.append(MSE(weights, h_params, validation))

                # update the progress bar
                t.set_postfix(loss=validation_loss[-1])

                #conditions for early stopping
                if validation_loss[-1] < best_loss:
                    best_loss = validation_loss[-1]
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1 

                if early_stopping_counter >= patience_values[i]:
                    print('Validation loss = ' + str(best_loss))
                    val_preds = forward_pass(weights,  h_params, param_val)
                    print('R2 score = ' + str(r2_score(feature_val, val_preds)))
                    break

            #final accuracy measurements
            update_emulator_parameters_and_save(weights, h_params, filename_saved_model)
            print('final validation loss:', MSE(weights, h_params, validation))
            print('final r2 score:', r2_score(feature_val, forward_pass(weights, h_params, param_val)))

            print('Reached max number of epochs. Validation loss = ' + str(best_loss))
            
    return weights, h_params
