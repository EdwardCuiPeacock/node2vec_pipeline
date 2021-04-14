#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:39:01 2021

@author: edwardcui
"""
import tfx
from tfx.dsl.component.experimental.annotations import OutputDict
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.types.standard_artifacts import Examples
from tfx.types.standard_artifacts import Model

from node2vec import sample_1_iteration_numpy

#sample_1_iteration_numpy(W, p, q, walk_length=80, symmetrify=True, seed=None)

@component
def RandomWalkGen(input_adj: InputArtifact[Examples],
                  sampled_series: OutputArtifact[Examples],
                  walk_length: Parameter[int],
                  repetitions: Parameter[int],
                  p: Parameter[float],
                  q: Parameter[float],
                  num_nodes: Parameter[int],
                  component_name: Parameter[str]
                 ) -> None:
    # Loadin ghte data file
    
        