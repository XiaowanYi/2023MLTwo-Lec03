#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 00:51:17 2023

@author: xyi
"""

import coremltools
#%% lots of paths

caffemodel_path = "/Users/xyi/Desktop/apple-dev-ml/2023MLUnitTwo/lec3/caffe-converter/finetune_flickr_style.caffemodel"
prototxt_path = "/Users/xyi/Desktop/apple-dev-ml/2023MLUnitTwo/lec3/caffe-converter/deploy.prototxt"
class_labels_path = "/Users/xyi/Desktop/apple-dev-ml/2023MLUnitTwo/lec3/caffe-converter/styles.txt"

#%%
coreml_model = coremltools.converters.caffe.convert(
    (caffemodel_path, prototxt_path),
    image_input_names = 'data',
    class_labels = class_labels_path
)

coreml_model.author = 'Paris BA'

coreml_model.license = 'None'

coreml_model.short_description = 'Flickr Style'

coreml_model.input_description['data'] = 'An image.'

coreml_model.output_description['prob'] = (
    'Probabilities for style type, for a given input.'
)

coreml_model.output_description['classLabel'] = (
    'The most likely style type for the given input.'
)

coreml_model.save('Style_fromCaffe.mlmodel')