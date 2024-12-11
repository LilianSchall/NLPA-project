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

"""Bayesian detector class."""

import abc
from collections.abc import Mapping, Sequence
import enum
import functools
import gc
from typing import Any, Optional, Union
import torch.nn as nn
import numpy as np
from sklearn import model_selection
import torch
import tqdm
from torch import optim
from torch.nn import init

from src.synthid_text import logits_processing


def pad_to_len(
    arr: torch.tensor,
    target_len: int,
    *,
    left_pad: bool,
    eos_token: int,
    device: torch.device,
) -> torch.tensor:
  """Pad or truncate array to given length."""
  if arr.shape[1] < target_len:
    shape_for_ones = list(arr.shape)
    shape_for_ones[1] = target_len - shape_for_ones[1]
    padded = (
        torch.ones(
            shape_for_ones,
            device=device,
            dtype=torch.long,
        )
        * eos_token
    )
    if not left_pad:
      return torch.concatenate((arr, padded), dim=1)
    else:
      return torch.concatenate((padded, arr), dim=1)
  else:
    return arr[:, :target_len]


def filter_and_truncate(
    outputs: torch.tensor,
    truncation_length: Optional[int],
    eos_token_mask: torch.tensor,
) -> torch.tensor:
  """Filter and truncate outputs to given length.

  Args:
   outputs: output tensor of shape [batch_size, output_len]
   truncation_length: Length to truncate the final output. If None, then no
     truncation is applied.
   eos_token_mask: EOS token mask of shape [batch_size, output_len]

  Returns:
   output tensor of shape [batch_size, truncation_length].
  """
  if truncation_length:
    outputs = outputs[:, :truncation_length]
    truncation_mask = torch.sum(eos_token_mask, dim=1) >= truncation_length
    return outputs[truncation_mask, :]
  return outputs


def process_outputs_for_training(
    all_outputs: Sequence[torch.LongTensor],
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    tokenizer: Any,
    *,
    pos_truncation_length: Optional[int],
    neg_truncation_length: Optional[int],
    max_length: int,
    is_cv: bool,
    is_pos: bool,
    torch_device: torch.device,
) -> tuple[Sequence[torch.tensor], Sequence[torch.tensor]]:
  """Process raw model outputs into format understandable by the detector.

  Args:
   all_outputs: sequence of outputs of shape [batch_size, output_len].
   logits_processor: logits processor used for watermarking.
   tokenizer: tokenizer used for the model.
   pos_truncation_length: Length to truncate the watermarked outputs. If None,
     then no truncation is applied.
   neg_truncation_length: Length to truncate the unwatermarked outputs. If None,
     then no truncation is applied.
   max_length: Length to pad truncated outputs so that all processed entries.
     have same shape.
   is_cv: Process given outputs for cross validation.
   is_pos: Process given outputs for positives.
   torch_device: torch device to use.

  Returns:
    Tuple of
      all_masks: list of masks of shape [batch_size, max_length].
      all_g_values: list of g_values of shape [batch_size, max_length, depth].
  """
  all_masks = []
  all_g_values = []
  for outputs in tqdm.tqdm(all_outputs):
    # outputs is of shape [batch_size, output_len].
    # output_len can differ from batch to batch.
    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=outputs,
        eos_token_id=tokenizer.eos_token_id,
    )
    if is_pos or is_cv:
      # filter with length for positives for both train and CV.
      # We also filter for length when CV negatives are processed.
      outputs = filter_and_truncate(
          outputs, pos_truncation_length, eos_token_mask
      )
    elif not is_pos and not is_cv:
      outputs = filter_and_truncate(
          outputs, neg_truncation_length, eos_token_mask
      )

    # If no filtered outputs skip this batch.
    if outputs.shape[0] == 0:
      continue

    # All outputs are padded to max-length with eos-tokens.
    outputs = pad_to_len(
        outputs,
        max_length,
        left_pad=False,
        eos_token=tokenizer.eos_token_id,
        device=torch_device,
    )
    # outputs shape [num_filtered_entries, max_length]

    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=outputs,
        eos_token_id=tokenizer.eos_token_id,
    )

    context_repetition_mask = logits_processor.compute_context_repetition_mask(
        input_ids=outputs,
    )

    # context_repetition_mask of shape [num_filtered_entries, max_length -
    # (ngram_len - 1)].
    context_repetition_mask = pad_to_len(
        context_repetition_mask,
        max_length,
        left_pad=True,
        eos_token=0,
        device=torch_device,
    )
    # We pad on left to get same max_length shape.
    # context_repetition_mask of shape [num_filtered_entries, max_length].
    combined_mask = context_repetition_mask * eos_token_mask

    g_values = logits_processor.compute_g_values(
        input_ids=outputs,
    )

    # g_values of shape [num_filtered_entries, max_length - (ngram_len - 1),
    # depth].
    g_values = pad_to_len(
        g_values, max_length, left_pad=True, eos_token=0, device=torch_device
    )

    # We pad on left to get same max_length shape.
    # g_values of shape [num_filtered_entries, max_length, depth].
    all_masks.append(combined_mask)
    all_g_values.append(g_values)
  return all_masks, all_g_values


@enum.unique
class ScoreType(enum.Enum):
  """Type of score returned by a WatermarkDetector.

  In all cases, larger score corresponds to watermarked text.
  """

  # Negative p-value where the p-value is the probability of observing equal or
  # stronger watermarking in unwatermarked text.
  NEGATIVE_P_VALUE = enum.auto()

  # Prob(watermarked | g-values).
  POSTERIOR = enum.auto()


class LikelihoodModel(abc.ABC):
  """Watermark likelihood model base class defining __call__ interface."""

  @abc.abstractmethod
  def __call__(self, g_values: torch.Tensor) -> torch.Tensor:
    """Computes likelihoods given g-values and a mask.

    Args:
      g_values: g-values (all are 0 or 1) of shape [batch_size, seq_len,
        watermarking_depth, ...].

    Returns:
      an array of shape [batch_size, seq_len, watermarking_depth] or
      [batch_size, seq_len, 1] corresponding to the likelihoods
      of the g-values given either the watermarked hypothesis or
      the unwatermarked hypothesis; i.e. either P(g|watermarked)
      or P(g|unwatermarked).
    """
    pass


class LikelihoodModelWatermarked(nn.Module, LikelihoodModel):
  """Watermarked likelihood model for binary-valued g-values.

  This takes in g-values and returns P(g_values|watermarked).
  """

  # watermarking_depth: int
  # params: Optional[Mapping[str, Mapping[str, Any]]] = None

  def __init__(self, watermarking_depth: int, params: Optional[Mapping[str, Any]]):
    """Initializes the model parameters."""
    super().__init__()
    self.watermarking_depth = watermarking_depth

    def noise(seed, shape):
      torch.manual_seed(seed)
      return torch.randn(shape)

    self.beta = nn.Parameter(
            -2.5 + 0.001 * noise(seed=0, shape=(1, 1, self.watermarking_depth))
    )
    self.delta = nn.Parameter(
            0.001
            * noise(
                seed=0,
                shape=(1, 1, self.watermarking_depth, self.watermarking_depth),
            )
    )

  def l2_loss(self) -> torch.Tensor:
    return torch.sum(self.delta ** 2)

  def _compute_latents(
      self, g_values: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the unique token probability distribution given g-values.

    Args:
      g_values: Pseudorandom function values of shape [batch_size, seq_len,
        watermarking_depth].

    Returns:
      p_one_unique_token and p_two_unique_tokens, both of shape
        [batch_size, seq_len, watermarking_depth]. p_one_unique_token[i,t,l]
        gives the probability of there being one unique token in a tournament
        match on layer l, on timestep t, for batch item i.
        p_one_unique_token[i,t,l] + p_two_unique_token[i,t,l] = 1.
    """
    # Tile g-values to produce feature vectors for predicting the latents
    # for each layer in the tournament; our model for the latents psi is a
    # logistic regression model psi = sigmoid(delta * x + beta).
    x = g_values.unsqueeze(-2).repeat(
        1, 1, self.watermarking_depth, 1
    )  # [batch_size, seq_len, watermarking_depth, watermarking_depth]

    x = torch.tril(
        x, diagonal=-1
    )  # mask all elements above -1 diagonal for autoregressive factorization

    logits = (
        torch.einsum("ijkl,ijkl->ijk", self.delta, x) + self.beta
    )  # [batch_size, seq_len, watermarking_depth]

    p_two_unique_tokens = torch.sigmoid(logits)
    p_one_unique_token = 1 - p_two_unique_tokens
    return p_one_unique_token, p_two_unique_tokens

  def forward(self, g_values: torch.Tensor) -> torch.Tensor:
    """Computes the likelihoods P(g_values|watermarked).

    Args:
      g_values: g-values (values 0 or 1) of shape [batch_size, seq_len,
        watermarking_depth]

    Returns:
      p(g_values|watermarked) of shape [batch_size, seq_len,
      watermarking_depth].
    """
    p_one_unique_token, p_two_unique_tokens = self._compute_latents(g_values)

    # P(g_tl | watermarked) is equal to
    # 0.5 * [ (g_tl+0.5) * p_two_unique_tokens + p_one_unique_token].
    return 0.5 * ((g_values + 0.5) * p_two_unique_tokens + p_one_unique_token)


class LikelihoodModelUnwatermarked(nn.Module, LikelihoodModel):
  """Unwatermarked likelihood model for binary-valued g-values.

  This takes in g-values and returns p(g_values | not watermarked).
  """

  def __init__(self):
    """Initializes the model parameters."""
    super().__init__()

  def forward(self, g_values: torch.Tensor) -> torch.Tensor:
    """Computes the likelihoods P(g-values|not watermarked).

    Args:
      g_values: g-values (0 or 1 values) of shape [batch_size, seq_len,
        watermarking_depth, ...].

    Returns:
      Likelihoods of g-values given text is unwatermarked --
      p(g_values | not watermarked) of shape [batch_size, seq_len,
      watermarking_depth].
    """
    return 0.5 * torch.ones_like(g_values)  # all g-values have prob 0.5.


def _compute_posterior(
    likelihoods_watermarked: torch.Tensor,
    likelihoods_unwatermarked: torch.Tensor,
    mask: torch.Tensor,
    prior: float,
) -> torch.Tensor:
  """Compute posterior P(w|g) given likelihoods, mask and prior.

  Args:
    likelihoods_watermarked: shape [batch, length, depth]. Likelihoods
      P(g_values|watermarked) of g-values under watermarked model.
    likelihoods_unwatermarked: shape [batch, length, depth]. Likelihoods
      P(g_values|unwatermarked) of g-values under unwatermarked model.
    mask: A binary array shape [batch, length] indicating which g-values should
      be used. g-values with mask value 0 are discarded.
    prior: Prior probability P(w) that the text is watermarked.

  Returns:
    Posterior probability P(watermarked|g_values), shape [batch].
  """
  mask = mask.unsqueeze(-1)
  prior = torch.clamp(torch.tensor(prior), min=1e-5, max=1 - 1e-5)
  log_likelihoods_watermarked = torch.log(
      torch.clamp(likelihoods_watermarked, min=1e-30)
  )
  log_likelihoods_unwatermarked = torch.log(
      torch.clamp(likelihoods_unwatermarked, min=1e-30)
  )
  log_odds = log_likelihoods_watermarked - log_likelihoods_unwatermarked

  # Sum relative surprisals (log odds) across all token positions and layers.
  relative_surprisal_likelihood = torch.einsum(
      "i...->i", log_odds * mask
  )  # [batch_size].

  relative_surprisal_prior = torch.log(prior) - torch.log(1 - prior)

  # Combine prior and likelihood.
  relative_surprisal = (
      relative_surprisal_prior + relative_surprisal_likelihood
  )  # [batch_size]

  # Compute the posterior probability P(w|g) = sigmoid(relative_surprisal).
  return torch.sigmoid(relative_surprisal)


class BayesianDetectorModule(nn.Module):
  """Bayesian classifier for watermark detection Flax Module.

  This detector uses Bayes' rule to compute a watermarking score, which is
  the posterior probability P(watermarked|g_values) that the text is
  watermarked, given its g_values.

  Note that this detector only works with Tournament-based watermarking using
  the Bernoulli(0.5) g-value distribution.
  """

  # watermarking_depth: int  # The number of tournament layers.
  # params: Optional[Mapping[str, Mapping[str, Any]]] = None
  # baserate: float = 0.5  # Prior probability P(w) that a text is watermarked.

  def __init__(self, watermarking_depth: int, params: Optional[Mapping[str, Any]] = None, baserate: float = 0.5):
    """Initializes the model parameters."""
    super().__init__()
    self.watermarking_depth = watermarking_depth
    self.params = params
    self.baserate = baserate

    def _fetch_params():
      return {"params:": self.params["params"]["likelihood_model_watermarked"]}

    self.likelihood_model_watermarked = LikelihoodModelWatermarked(
        watermarking_depth=self.watermarking_depth,
        params=_fetch_params() if self.params is not None else None,
    )
    self.likelihood_model_unwatermarked = LikelihoodModelUnwatermarked()
    self.prior = nn.Parameter(torch.tensor(baserate), requires_grad=False)

    if self.params is not None:
        self.load_state_dict(self.params["params"])
    else:
        self._initialize_parameters()

  def _initialize_parameters(self):
      """Initialize the parameters (equivalent to `init` in Flax)."""
      # Example: We initialize layers using Xavier for Linear layers
      for m in self.modules():
          if isinstance(m, nn.Linear):  # Check if it's a Linear layer
              init.xavier_uniform_(m.weight)  # Xavier initialization for weights
              if m.bias is not None:
                  init.zeros_(m.bias)  # Initialize bias to zero

  @property
  def score_type(self) -> ScoreType:
    return ScoreType.POSTERIOR

  def l2_loss(self) -> torch.Tensor:
    return self.likelihood_model_watermarked.l2_loss()

  def forward(
      self,
      g_values: torch.Tensor,
      mask: torch.Tensor,
  ) -> torch.Tensor:
    """Computes the watermarked posterior P(watermarked|g_values).

    Args:
      g_values: g-values (with values 0 or 1) of shape [batch_size, seq_len,
        watermarking_depth, ...]
      mask: A binary array shape [batch_size, seq_len] indicating which g-values
        should be used. g-values with mask value 0 are discarded.

    Returns:
      P(watermarked | g_values), of shape [batch_size].
    """

    likelihoods_watermarked = self.likelihood_model_watermarked(g_values)
    likelihoods_unwatermarked = self.likelihood_model_unwatermarked(g_values)
    return _compute_posterior(
        likelihoods_watermarked, likelihoods_unwatermarked, mask, self.prior.item()
    )

  def score(
      self,
      g_values: Union[torch.Tensor, Sequence[torch.Tensor]],
      mask: torch.Tensor,
  ) -> torch.Tensor:
    if self.params is None:
      raise ValueError("params must be set before calling score")
    return self.forward(g_values, mask)


def xentropy_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
  """Calculates cross entropy loss."""
  y_pred = torch.clamp(y_pred, min=1e-5, max=1 - 1e-5)
  return torch.mean(-(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)))


def loss_fn(
    params: Mapping[str, Any],
    detector_inputs: Any,
    w_true: torch.Tensor,
    l2_batch_weight: float,
    detector_module: BayesianDetectorModule,
) -> torch.Tensor:
  """Calculates loss for a batch of data given parameters."""
  w_pred = detector_module(*detector_inputs)

  unweighted_l2 = detector_module.l2_loss()

  l2_loss = l2_batch_weight * unweighted_l2
  return xentropy_loss(w_true, w_pred) + l2_loss


def tpr_at_fpr(
    params: Mapping[str, Any],
    detector_inputs: Any,
    w_true: torch.Tensor,
    minibatch_size,
    detector_module: BayesianDetectorModule,
    target_fpr: float = 0.01,
) -> torch.Tensor:
  """Calculates TPR at FPR=target_fpr."""
  positive_idxs = w_true == 1
  negative_idxs = w_true == 0

  inds = torch.arange(0, len(detector_inputs[0]), minibatch_size)

  w_preds = []
  for start in inds:
    end = start + minibatch_size
    detector_inputs_ = (
        detector_inputs[0][start:end],
        detector_inputs[1][start:end],
    )
    w_pred = detector_module(*detector_inputs_)
    w_preds.append(w_pred)

  w_pred = torch.cat(w_preds, dim=0)

  positive_scores = w_pred[positive_idxs]
  negative_scores = w_pred[negative_idxs]
  
  fpr_threshold = torch.quantile(negative_scores, 1 - target_fpr)

  return torch.mean(positive_scores >= fpr_threshold)


@enum.unique
class ValidationMetric(enum.Enum):
  """Direction along the z-axis."""

  TPR_AT_FPR = "tpr_at_fpr"
  CROSS_ENTROPY = "cross_entropy"

  """Trains a Bayesian detector model.

  Args:
    detector_module: The detector module to train in-place.
    g_values: g-values of shape [num_train, seq_len, watermarking_depth].
    mask: A binary array shape [num_train, seq_len] indicating which g-values
      should be used. g-values with mask value 0 are discarded.
    watermarked: A binary array of shape [num_train] indicating whether the
      example is watermarked (0: unwatermarked, 1: watermarked).
    epochs: Number of epochs to train for.
    learning_rate: Learning rate for optimizer.
    minibatch_size: Minibatch size for training. Note that a minibatch requires
      ~ 32 * minibatch_size * seq_len * watermarked_depth * watermarked_depth
      bits of memory.
    seed: Seed for parameter initialization.
    l2_weight: Weight to apply to L2 regularization for delta parameters.
    shuffle: Whether to shuffle before training.
    g_values_val: Validation g-values of shape [num_val, seq_len,
      watermarking_depth].
    mask_val: Validation mask of shape [num_val, seq_len].
    watermarked_val: Validation watermark labels of shape [num_val].
    verbose: Boolean indicating verbosity of training. If true, the loss will be
      printed. Defaulted to False.
    validation_metric: validation metric to use.

  Returns:
    Tuple of
      training_history: Training history keyed by epoch number where the
      values are
        dictionaries containing the loss, validation loss, and model
        parameters,
        keyed by
        'loss', 'val_loss', and 'params', respectively.
      min_val_loss: Minimum validation loss achieved during training.
  """

def train(
    *,
    detector_module: BayesianDetectorModule,
    g_values: torch.Tensor,
    mask: torch.Tensor,
    watermarked: torch.Tensor,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    minibatch_size: int = 64,
    seed: int = 0,
    l2_weight: float = 0.0,
    shuffle: bool = True,
    g_values_val: Optional[torch.Tensor] = None,
    mask_val: Optional[torch.Tensor] = None,
    watermarked_val: Optional[torch.Tensor] = None,
    verbose: bool = False,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
) -> tuple[Mapping[int, Mapping[str, Any]], float]:

  minibatch_inds = torch.arange(0, len(g_values), minibatch_size)
  minibatch_inds_val = None
  if g_values_val is not None:
    minibatch_inds_val = torch.arange(0, len(g_values_val), minibatch_size)

  torch.manual_seed(seed)

  # param_rng, shuffle_rng = jax.random.split(rng)

  def coshuffle(*args):
    return [x[torch.randperm(x.size(0))] for x in args]

  if shuffle:
    g_values, mask, watermarked = coshuffle(g_values, mask, watermarked)

  def update_fn_if_fpr_tpr(params):
    """Loss function for negative TPR@FPR=1% as the validation loss."""
    tpr_ = tpr_at_fpr(
        params=params,
        detector_inputs=(g_values_val, mask_val),
        w_true=watermarked_val,
        minibatch_size=minibatch_size,
        detector_module=detector_module,
    )
    return -tpr_

  n_minibatches = len(g_values) // minibatch_size
  l2_batch_weight_train = l2_weight / n_minibatches
  l2_batch_weight_val = 0.0
  loss_fn_train = functools.partial(
      loss_fn,
      l2_batch_weight=l2_batch_weight_train,
      detector_module=detector_module,
  )

  loss_fn_jitted_val = functools.partial(
          loss_fn,
          l2_batch_weight=l2_batch_weight_val,
          detector_module=detector_module,
  )

  def update(gvalues, masks, labels, params, optimizer):
    loss_fn_partialed = functools.partial(
        loss_fn_train,
        detector_inputs=(gvalues, masks),
        w_true=labels,
    )

    loss = loss_fn_partialed(params)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), params, optimizer

  def update_with_minibatches(gvalues, masks, labels, inds, params, optimizer):
    """Update params iff opt_state is not None and always returns the loss."""
    losses = []
    for start in inds:
      end = start + minibatch_size
      loss, params, optimizer = update(
          gvalues[start:end],
          masks[start:end],
          labels[start:end],
          params,
          optimizer,
      )
      losses.append(loss)
    loss = torch.mean(torch.tensor(losses))
    return loss.item(), params, optimizer

  def validate_with_minibatches(gvalues, masks, labels, inds, params):
    """Update params iff opt_state is not None and always returns the loss."""
    losses = []
    for start in inds:
      end = start + minibatch_size
      loss = loss_fn_jitted_val(
          params,
          detector_inputs=(gvalues[start:end], masks[start:end]),
          w_true=labels[start:end],
      )
      losses.append(loss.item())
    return torch.mean(torch.tensor(losses)).item()

  def update_fn(opt_state, params):
    """Updates the model parameters and returns the loss."""
    loss, params, opt_state = update_with_minibatches(
        g_values, mask, watermarked, minibatch_inds, params, opt_state
    )
    val_loss = None
    if g_values_val is not None:
      if validation_metric == ValidationMetric.TPR_AT_FPR:
        val_loss = update_fn_if_fpr_tpr(params)
      else:
        val_loss = validate_with_minibatches(
            g_values_val,
            mask_val,
            watermarked_val,
            minibatch_inds_val,
            params,
        )

    return opt_state, params, loss, val_loss

  params = detector_module.params
  if params is None:
      detector_module._initialize_parameters()
      params = detector_module.params

  optimizer = optim.Adam(params.values(), lr=learning_rate)

  history = {}
  epochs_completed = 0

  while epochs_completed < epochs:
    optimizer, params, loss, val_loss = update_fn(optimizer, params)
    epochs_completed += 1

    history[epochs_completed] = {
        "loss": loss,
        "val_loss": val_loss,
        "params": params["params"],
    }

    if verbose:
      if val_loss is not None:
        print(
            f"Epoch {epochs_completed}: loss {loss} (train), {val_loss} (val)"
        )
      else:
        print(f"Epoch {epochs_completed}: loss {loss} (train)")

  val_loss = np.squeeze(
      np.array([history[epoch]["val_loss"] for epoch in range(1, epochs + 1)])
  )
  best_val_epoch = np.argmin(val_loss) + 1
  min_val_loss = val_loss[best_val_epoch - 1]

  print(f"Best val Epoch: {best_val_epoch}, min_val_loss: {min_val_loss}")
 
  detector_module.params = {"params": history[best_val_epoch]["params"]}

  return history, min_val_loss


class BayesianDetector:
  """Bayesian detector class used for watermark detection."""

  detector_module: BayesianDetectorModule
  tokenizer: Any
  logits_processor: logits_processing.SynthIDLogitsProcessor

  def __init__(
      self,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      tokenizer: Any,
      params: Mapping[str, Mapping[str, Any]],
  ):
    self.detector_module = BayesianDetectorModule(
        watermarking_depth=len(logits_processor.keys),
        params=params,
    )
    self.logits_processor = logits_processor
    self.tokenizer = tokenizer

  def score(self, outputs: torch.LongTensor) -> torch.Tensor:
    """Score the model output for possibility of being watermarked.

    Score is within [0, 1] where 0 is not watermarked and 1 is watermarked.

    Args:
      outputs: model output of shape [batch_size, output_len]

    Returns:
      scores of shape [batch_size]
    """
    # eos mask is computed, skip first ngram_len - 1 tokens
    # eos_mask will be of shape [batch_size, output_len]
    eos_token_mask = self.logits_processor.compute_eos_token_mask(
        input_ids=outputs,
        eos_token_id=self.tokenizer.eos_token_id,
    )[:, self.logits_processor.ngram_len - 1 :]

    # context repetition mask is computed
    context_repetition_mask = (
        self.logits_processor.compute_context_repetition_mask(
            input_ids=outputs,
        )
    )
    # context repetition mask shape [batch_size, output_len - (ngram_len - 1)]

    combined_mask = context_repetition_mask * eos_token_mask

    g_values = self.logits_processor.compute_g_values(
        input_ids=outputs,
    )
    # g values shape [batch_size, output_len - (ngram_len - 1), depth]
    return self.detector_module.score(
        g_values=g_values, mask=combined_mask
    )

  @classmethod
  def process_raw_model_outputs(
      cls,
      *,
      tokenized_wm_outputs: Union[Sequence[np.ndarray], np.ndarray],
      tokenized_uwm_outputs: Union[Sequence[np.ndarray], np.ndarray],
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      tokenizer: Any,
      torch_device: torch.device,
      test_size: float = 0.3,
      pos_truncation_length: Optional[int] = 200,
      neg_truncation_length: Optional[int] = 100,
      max_padded_length: int = 2300,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process raw models outputs into inputs we can train.

    Args:
      tokenized_wm_outputs: tokenized outputs of watermarked data.
      tokenized_uwm_outputs: tokenized outputs of unwatermarked data.
      logits_processor: logits processor used for watermarking.
      tokenizer: tokenizer used for the model.
      torch_device: torch device to use.
      test_size: test size to use for train-test split.
      pos_truncation_length: Length to truncate wm outputs. If None, no
        truncation is applied.
      neg_truncation_length: Length to truncate uwm outputs. If None, no
        truncation is applied.
      max_padded_length: Length to pad truncated outputs so that all processed
        entries have same shape.

    Returns:
      Tuple of train_g_values, train_masks, train_labels, cv_g_values, cv_masks,
        cv_labels
    """
    # Split data into train and CV
    train_wm_outputs, cv_wm_outputs = model_selection.train_test_split(
        tokenized_wm_outputs, test_size=test_size
    )

    train_uwm_outputs, cv_uwm_outputs = model_selection.train_test_split(
        tokenized_uwm_outputs, test_size=test_size
    )

    # Process both train and CV data for training
    wm_masks_train, wm_g_values_train = process_outputs_for_training(
        [
            torch.tensor(outputs, device=torch_device, dtype=torch.long)
            for outputs in train_wm_outputs
        ],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        pos_truncation_length=pos_truncation_length,
        neg_truncation_length=neg_truncation_length,
        max_length=max_padded_length,
        is_pos=True,
        is_cv=False,
        torch_device=torch_device,
    )
    wm_masks_cv, wm_g_values_cv = process_outputs_for_training(
        [
            torch.tensor(outputs, device=torch_device, dtype=torch.long)
            for outputs in cv_wm_outputs
        ],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        pos_truncation_length=pos_truncation_length,
        neg_truncation_length=neg_truncation_length,
        max_length=max_padded_length,
        is_pos=True,
        is_cv=True,
        torch_device=torch_device,
    )
    uwm_masks_train, uwm_g_values_train = process_outputs_for_training(
        [
            torch.tensor(outputs, device=torch_device, dtype=torch.long)
            for outputs in train_uwm_outputs
        ],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        pos_truncation_length=pos_truncation_length,
        neg_truncation_length=neg_truncation_length,
        max_length=max_padded_length,
        is_pos=False,
        is_cv=False,
        torch_device=torch_device,
    )
    uwm_masks_cv, uwm_g_values_cv = process_outputs_for_training(
        [
            torch.tensor(outputs, device=torch_device, dtype=torch.long)
            for outputs in cv_uwm_outputs
        ],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        pos_truncation_length=pos_truncation_length,
        neg_truncation_length=neg_truncation_length,
        max_length=max_padded_length,
        is_pos=False,
        is_cv=True,
        torch_device=torch_device,
    )

    # We get list of data; here we concat all together to be passed to the
    # detector.
    wm_masks_train = torch.cat(wm_masks_train, dim=0)
    wm_g_values_train = torch.cat(wm_g_values_train, dim=0)
    wm_labels_train = torch.ones((wm_masks_train.shape[0],), dtype=torch.bool)
    wm_masks_cv = torch.cat(wm_masks_cv, dim=0)
    wm_g_values_cv = torch.cat(wm_g_values_cv, dim=0)
    wm_labels_cv = torch.ones((wm_masks_cv.shape[0],), dtype=torch.bool)

    uwm_masks_train = torch.cat(uwm_masks_train, dim=0)
    uwm_g_values_train = torch.cat(uwm_g_values_train, dim=0)
    uwm_labels_train = torch.zeros(
        (uwm_masks_train.shape[0],), dtype=torch.bool
    )
    uwm_masks_cv = torch.cat(uwm_masks_cv, dim=0)
    uwm_g_values_cv = torch.cat(uwm_g_values_cv, dim=0)
    uwm_labels_cv = torch.zeros((uwm_masks_cv.shape[0],), dtype=torch.bool)

    # Concat pos and negatives data together.
    train_g_values = torch.cat((wm_g_values_train, uwm_g_values_train), dim=0)
    train_labels = torch.cat((wm_labels_train, uwm_labels_train), axis=0)
    train_masks = torch.cat((wm_masks_train, uwm_masks_train), axis=0)

    cv_g_values = torch.cat((wm_g_values_cv, uwm_g_values_cv), axis=0)
    cv_labels = torch.cat((wm_labels_cv, uwm_labels_cv), axis=0)
    cv_masks = torch.cat((wm_masks_cv, uwm_masks_cv), axis=0)

    # Free up GPU memory.
    del (
        wm_g_values_train,
        wm_labels_train,
        wm_masks_train,
        wm_g_values_cv,
        wm_labels_cv,
        wm_masks_cv,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # Shuffle data.
    train_g_values = train_g_values.squeeze()
    train_labels = train_labels.squeeze()
    train_masks = train_masks.squeeze()

    cv_g_values = cv_g_values.squeeze()
    cv_labels = cv_labels.squeeze()
    cv_masks = cv_masks.squeeze()

    shuffled_idx = list(range(train_g_values.shape[0]))
    shuffled_idx = np.array(shuffled_idx)
    np.random.shuffle(shuffled_idx)
    train_g_values = train_g_values[shuffled_idx]
    train_labels = train_labels[shuffled_idx]
    train_masks = train_masks[shuffled_idx]

    shuffled_idx = list(range(cv_g_values.shape[0]))
    shuffled_idx = np.array(shuffled_idx)
    np.random.shuffle(shuffled_idx)
    cv_g_values = cv_g_values[shuffled_idx]
    cv_labels = cv_labels[shuffled_idx]
    cv_masks = cv_masks[shuffled_idx]

    return (
        train_g_values,
        train_masks,
        train_labels,
        cv_g_values,
        cv_masks,
        cv_labels,
    )

  @classmethod
  def train_best_detector_given_g_values(
      cls,
      *,
      train_g_values: torch.Tensor,
      train_masks: torch.Tensor,
      train_labels: torch.Tensor,
      cv_g_values: torch.Tensor,
      cv_masks: torch.Tensor,
      cv_labels: torch.Tensor,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      tokenizer: Any,
      n_epochs: int = 50,
      learning_rate: float = 2.1e-2,
      l2_weights: torch.Tensor = torch.logspace(-3, -2, steps=4),
      verbose: bool = False,
  ) -> tuple["BayesianDetector", float]:
    """Train best detector given g_values, mask and labels."""
    best_detector = None
    lowest_loss = float("inf")
    val_losses = []
    for l2_weight in l2_weights:
      detector_module = BayesianDetectorModule(
          watermarking_depth=len(logits_processor.keys),
      )

      _, min_val_loss = train(
          detector_module=detector_module,
          g_values=train_g_values,
          mask=train_masks,
          watermarked=train_labels,
          g_values_val=cv_g_values,
          mask_val=cv_masks,
          watermarked_val=cv_labels,
          learning_rate=learning_rate,
          l2_weight=l2_weight.item(),
          epochs=n_epochs,
          verbose=verbose,
      )

      val_losses.append(min_val_loss)
      if min_val_loss < lowest_loss:
        lowest_loss = min_val_loss
        best_detector = detector_module

    return cls(logits_processor, tokenizer, best_detector.params), lowest_loss

  @classmethod
  def train_best_detector(
      cls,
      *,
      tokenized_wm_outputs: Union[Sequence[torch.Tensor], torch.Tensor],
      tokenized_uwm_outputs: Union[Sequence[torch.Tensor], torch.Tensor],
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      tokenizer: Any,
      torch_device: torch.device,
      test_size: float = 0.3,
      pos_truncation_length: Optional[int] = 200,
      neg_truncation_length: Optional[int] = 100,
      max_padded_length: int = 2300,
      n_epochs: int = 50,
      learning_rate: float = 2.1e-2,
      l2_weights: torch.Tensor = torch.logspace(-3, -2, steps=4),
      verbose: bool = False,
  ) -> tuple["BayesianDetector", float]:
    """Construct, train and return the best detector based on wm and uwm data.

    In practice, we have found that tuning pos_truncation_length,
    neg_truncation_length, n_epochs, learning_rate and l2_weights can help
    improve the performance of the detector. We recommend tuning these
    parameters for your data.

    Args:
      tokenized_wm_outputs: tokenized outputs of watermarked data.
      tokenized_uwm_outputs: tokenized outputs of unwatermarked data.
      logits_processor: logits processor used for watermarking.
      tokenizer: tokenizer used for the model.
      torch_device: torch device to use.
      test_size: test size to use for train-test split.
      pos_truncation_length: Length to truncate wm outputs. If None, no
        truncation is applied.
      neg_truncation_length: Length to truncate uwm outputs. If None, no
        truncation is done.
      max_padded_length: Length to pad truncated outputs so that all processed
        entries have same shape.
      n_epochs: Number of epochs to train the detector.
      learning_rate: Learning rate to use for training the detector.
      l2_weights: L2 weights to use for training the detector.
      verbose: Whether to print training progress.

    Returns:
      Tuple of trained detector and loss achieved on CV data.
    """
    if torch_device.type in ("cuda", "tpu"):
      raise ValueError(
          "We have found the training unstable on CPUs; we are working on"
          " a fix. Use GPU or TPU for training."
      )
    (
        train_g_values,
        train_masks,
        train_labels,
        cv_g_values,
        cv_masks,
        cv_labels,
    ) = cls.process_raw_model_outputs(
        tokenized_wm_outputs=tokenized_wm_outputs,
        tokenized_uwm_outputs=tokenized_uwm_outputs,
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        torch_device=torch_device,
        test_size=test_size,
        pos_truncation_length=pos_truncation_length,
        neg_truncation_length=neg_truncation_length,
        max_padded_length=max_padded_length,
    )

    train_g_values = torch.tensor(train_g_values, device=torch_device)
    train_masks = torch.tensor(train_masks, device=torch_device)
    train_labels = torch.tensor(train_labels, device=torch_device)
    cv_g_values = torch.tensor(cv_g_values, device=torch_device)
    cv_masks = torch.tensor(cv_masks, device=torch_device)
    cv_labels = torch.tensor(cv_labels, device=torch_device)

    return cls.train_best_detector_given_g_values(
        train_g_values=train_g_values,
        train_masks=train_masks,
        train_labels=train_labels,
        cv_g_values=cv_g_values,
        cv_masks=cv_masks,
        cv_labels=cv_labels,
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        verbose=verbose,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        l2_weights=l2_weights,
    )
