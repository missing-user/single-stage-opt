import numpy as np
from simsopt import mhd
from simsopt._core.util import ObjectiveFailure
import logging

class SpecBackoff(mhd.Spec):
    def __init__(self, *args, **kwargs):
      """
      A wrapper for the simsopt.mhd.Spec class, that implements a simple backoff strategy with maximum attempts to ensure convergence.
      Since SPEC only converges locally, a large step by the optimizer may cause issues. If the simulation fails to converge, 
      it will retry by splitting the step-intervall in half and running two consecutive simulations, using the previous result 
      as the initial guess for the next. This is recursively repeated until a maximum retry count is reached.

      params:
          max_attempts: Maximum number of attempts to run the simulation (default: 10)

      """
      # Done this way to keep the signature of the original class
      self.max_attempts = kwargs.pop("max_attempts", 10)
      super().__init__(*args, **kwargs)
      self.checkpoint_x = None

    def run(self, *args, **kwargs):
      target_x = self.x.copy()
      # DOFs have changed, reset checkpoint
      if np.shape(target_x) != np.shape(self.checkpoint_x):
        self.checkpoint_x = None
      self._run(target_x, *args, **kwargs)
      # logging.info(f"self.checkpoint_x = {self.checkpoint_x}")
      self.checkpoint_x = target_x.copy()

    def _run(self, target_x, *args, **kwargs):
      step_size = 1.0
      current_progress = 0.0
      for attempt in range(self.max_attempts):
        # Linear interpolation between last successful and target x
        current_progress = min(1.0, current_progress + step_size)

        if self.checkpoint_x is None:
          # This should only happen at the very first run
          x = target_x
        else:
          assert np.shape(target_x) == np.shape(self.checkpoint_x)
          x = self.checkpoint_x + current_progress * (target_x - self.checkpoint_x)

        # logging.info(f"Overriding x = \n{self.x} with a change of \n{x-self.x}") 
        self.x = x
        try:
          super().run(*args, **kwargs)
          logging.info(f"Attempt {attempt+1}/{self.max_attempts} successful with step: {current_progress}")
        except ObjectiveFailure as e:
          logging.warning(f"Attempt {attempt+1}/{self.max_attempts} Failed to converge, retrying with half the step-size. Failure reason: {e}")
          current_progress -= step_size
          step_size /= 2
          # Not enough steps left to reach the target, so we need to hurry up
          if (1.0 - current_progress)/step_size > (self.max_attempts-attempt):
            proposed_step_size = (self.max_attempts-attempt)/(1.0 - current_progress)
            # This proposal only makes sense if the step size is smaller than the one that just failed
            if proposed_step_size < 2*step_size:
              step_size = proposed_step_size
        # Converged at target
        if current_progress >= 1.0:
          return
      if current_progress < 1.0:
        raise ObjectiveFailure(f"SpecBackoff failed to converge after {self.max_attempts} attempts")