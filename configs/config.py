MODEL = {
    'RESUME': ''
}

TRAIN = {
    'EPOCHS': 101,
    'START_EPOCH': 0,
    'AUTO_RESUME': True,
    'BASE_LR': 1e-4,
    'WARMUP_EPOCHS': 5,
    'WEIGHT_DECAY': 0.05,
    'MIN_LR': 1e-6,
    'WARMUP_LR': 5e-7,
    'LR_SCHEDULER': {
        'NAME': 'cosine',
        'DECAY_EPOCHS': 10,
        'WARMUP_PREFIX': True,
        'MULTISTEPS': []
    },
    'OPTIMIZER': {
        'NAME': 'adamw',
        'EPS': 1e-8,
        'BETAS': (0.9, 0.999),
        'MOMENTUM': 0.9
    }}
OUTPUT = 'output'
