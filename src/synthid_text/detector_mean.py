# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for Mean and Weighted Mean scoring functions."""

from typing import Optional
import torch


def mean_score(
    g_values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
  """Computes the Mean score.

  Args:
    g_values: g-values of shape [batch_size, seq_len, watermarking_depth].
    mask: A binary array shape [batch_size, seq_len] indicating which g-values
      should be used. g-values with mask value 0 are discarded.

  Returns:
    Mean scores, of shape [batch_size]. This is the mean of the unmasked
      g-values.
  """
  watermarking_depth = g_values.shape[-1]
  num_unmasked = mask.sum(dim=1)  # shape [batch_size]
  return (
          (g_values * mask.unsqueeze(2)).sum(dim=(1, 2)) /
          (watermarking_depth * num_unmasked)
  )



def weighted_mean_score(
    g_values: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """Computes the Weighted Mean score.

  Args:
    g_values: g-values of shape [batch_size, seq_len, watermarking_depth].
    mask: A binary array shape [batch_size, seq_len] indicating which g-values
      should be used. g-values with mask value 0 are discarded.
    weights: array of non-negative floats, shape [watermarking_depth]. The
      weights to be applied to the g-values. If not supplied, defaults to
      linearly decreasing weights from 10 to 1.

  Returns:
    Weighted Mean scores, of shape [batch_size]. This is the mean of the
      unmasked g-values, re-weighted using weights.
  """
  watermarking_depth = g_values.shape[-1]

  if weights is None:
    weights = torch.linspace(start=10, end=1, steps=watermarking_depth)

  # Normalise weights so they sum to watermarking_depth.
  weights *= watermarking_depth / weights.sum()

  # Apply weights to g-values.
  g_values *= weights.unsqueeze(0).unsqueeze(0)

  num_unmasked = mask.sum(dim=1)  # shape [batch_size]
  return (
          (g_values * mask.unsqueeze(2)).sum(dim=(1, 2)) /
          (watermarking_depth * num_unmasked)
  )

