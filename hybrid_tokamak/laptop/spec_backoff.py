import numpy as np
from simsopt import mhd
from simsopt._core.util import ObjectiveFailure
import logging

class SpecBackoff(mhd.Spec):
    """
    A wrapper for the simsopt.mhd.Spec class, that implements a simple backoff strategy with maximum attempts to ensure convergence.
    Since SPEC only converges locally, a large step by the optimizer may cause issues. If the simulation fails to converge, 
    it will retry by splitting the step-intervall in half and running two consecutive simulations, using the previous result 
    as the initial guess for the next. This is recursively repeated until a maximum retry count is reached.

    params:
        max_attempts: Maximum number of attempts to run the simulation (default: 10)
    """

    def __init__(self, *args, **kwargs):
      # Done this way to keep the signature of the original class
      self.max_attempts = kwargs.pop("max_attempts", 10)
      super().__init__(*args, **kwargs)
      self.checkpoint_x = None

    def run(self, *args, **kwargs):
      target_x = self.x.copy()
      self._run(target_x, *args, **kwargs)
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
          x = self.checkpoint_x + current_progress * (target_x - self.checkpoint_x)
          assert np.shape(target_x) == np.shape(self.checkpoint_x)

        self.x = x
        try:
          logging.info(f"Attempt {attempt+1}/{self.max_attempts}, progress: {current_progress}")
          super().run(*args, **kwargs)
          return
        except ObjectiveFailure as e:
          logging.warning(f"Failed to converge, retrying with half the step-size. Failure reason:", e)
          current_progress -= step_size
          step_size /= 2
          # Not enough steps left to reach the target, so we need to hurry up
          if (1.0 - current_progress)/step_size > (self.max_attempts-attempt):
            proposed_step_size = (self.max_attempts-attempt)/(1.0 - current_progress)
            # This proposal only makes sense if the step size is smaller than the one that just failed
            if proposed_step_size < 2*step_size:
              step_size = proposed_step_size