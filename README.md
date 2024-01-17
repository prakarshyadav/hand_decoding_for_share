# hand_decoding_for_share
Some core functionality of hand gesture decoding. Missing some vital components until work is ready to be published.

Directories
gifs: The prompts presented to the participants. Should have a couple of GIFs and dict/list of all classes in the study
inference/recording: GUI for data recoding experiments and the various buttons/fields that can be used. These are not run directly and called through run_GUI.py
ML_training: Functions for running ML training on HDEMG data.
tmsi_dual_interface: Core modules for data recoding and running the TMSi SAGA/64 devices for HDEMG recording and online streaming.

Scripts:
[run_GUI.py](run_gui.py): The script to run the data recording framework and conduct experiment.
[run_model_eval.py](run_model_eval.py): Runs evaluation of trained model.
[run_model_finetune.py](run_model_finetune.py): Finetunes a self-supervised pre-trained model.
[run_model_pre_training.py](run_model_pre_training.py): Runs self-supervised framework with triplet loss or Barlow Twins paradigm.
[run_model_training.py](run_model_training.py): Run training of model using Cross entropy loss.

Important functions:
[inference.decoder_inference.background_inference()](inference/decoder_inference.py) : This function reads realtime data and processes it on the fly to predict class using a trained model
[ML_training.vanilla_training.train_model.run_train_loss_CH()](ML_training/Vanilla_training/model_train.py) : Runs model training with cross entropy loss
[ML_training.SSL_training.pre_train_triplet.run_pretrain_triplet()](ML_training/SSL_training/pre_train_triplet.py) : Runs self-supervised training with triplet loss and hard mining loss

Parameter file:
[vanilla_training_sweep.yaml](vanilla_training_sweep.yaml): The various parameteres that are explored by Weights&Biases sweep.
