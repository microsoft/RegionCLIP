import numpy as np
import random

from ..imagenet.dictionary import IMAGENET_CLASSES_DICTIONARY
from .prompt_engineering import prompt_engineering

def knowledge_engineering(classnames):

    if isinstance(classnames, list):
        classname = random.choice(classnames)
    else:
        classname = classnames

    if classname in IMAGENET_CLASSES_DICTIONARY:
        return 'a photo of the {}. '.format(classname) + IMAGENET_CLASSES_DICTIONARY[classname]
    else:
        return prompt_engineering(classname)