# Copyright (c) OpenMMLab. All rights reserved.
"""MMDetection provides 17 registry nodes to support using modules across
projects. Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mm.engine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mm.engine.registry import DATASETS as MMENGINE_DATASETS
from mm.engine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mm.engine.registry import HOOKS as MMENGINE_HOOKS
from mm.engine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mm.engine.registry import LOOPS as MMENGINE_LOOPS
from mm.engine.registry import METRICS as MMENGINE_METRICS
from mm.engine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mm.engine.registry import MODELS as MMENGINE_MODELS
from mm.engine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mm.engine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mm.engine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mm.engine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mm.engine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mm.engine.registry import RUNNERS as MMENGINE_RUNNERS
from mm.engine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mm.engine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mm.engine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mm.engine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mm.engine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mm.engine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner', parent=MMENGINE_RUNNERS, scope='det', locations=['mm.det.engine.runner'])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS, scope='det',
    locations=['mm.det.engine.runner'])
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop', parent=MMENGINE_LOOPS, scope='det', locations=['mm.det.engine.runner'])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, scope='det', locations=['mm.det.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, scope='det', locations=['mm.det.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS, scope='det',
    locations=['mm.det.datasets.samplers'])
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS, scope='det',
    locations=['mm.det.datasets.transforms'])

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS, scope='det', locations=['mm.det.models'])
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS, scope='det',
    locations=['mm.det.models'])
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS, scope='det',
    locations=['mm.det.models'])

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS, scope='det',
    locations=['mm.det.engine.optimizers'])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS, scope='det',
    locations=['mm.det.engine.optimizers'])
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS, scope='det',
    locations=['mm.det.engine.optimizers'])
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS, scope='det',
    locations=['mm.det.engine.schedulers'])
# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, scope='det', locations=['mm.det.evaluation'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=MMENGINE_EVALUATOR, scope='det', locations=['mm.det.evaluation'])

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', parent=MMENGINE_TASK_UTILS, scope='det', locations=['mm.det.models'])

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS, scope='det',
    locations=['mm.det.visualization'])
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS, scope='det',
    locations=['mm.det.visualization'])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS, scope='det',
    # TODO: update the location when mmdet has its own log processor
    locations=['mm.det.engine'])
