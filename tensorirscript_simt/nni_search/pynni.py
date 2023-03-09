from nni.experiment import Experiment

# Path: nni_search/pynni.py
experiment = Experiment('local')
experiment.config.experiment_working_directory = '/workspace/v-leiwang3/nni-experiments/'
experiment.id
experiment.config.experiment_name = '_tensorize_program_search'
search_space = {
    "num_warps": {"_type": "choice", "_value": [1, 2, 4, 8, 16, 32]},
    "chunk": {"_type": "choice", "_value": [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128]},
}
experiment.config.search_space = search_space

experiment.config.trial_command = 'python3 ./gemv_general.py'
experiment.config.trial_code_directory = '.'
experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.tuner.class_args['population_size'] = 2048
experiment.config.max_trial_number = 10000
experiment.config.trial_concurrency = 12
experiment.config.trial_gpu_number = 1
experiment.config.tuner_gpu_indices = [3]
experiment.config.use_annotation = False
experiment.config.training_service.use_active_gpu = False
experiment.config.training_service.platform = 'local'
experiment.config.training_service.max_trial_number_per_gpu = 12
experiment.config.training_service.gpu_indices = [3]

experiment.run(8080, debug=True)
input('Press enter to quit')
experiment.stop()
