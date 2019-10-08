import os
import shutil
import tensorflow as tf
from pathlib import Path, PurePath


# TODO make BEST_CHECKPOINTS_PATH as parameter that can be parsed
class BestCheckpointsExporter(tf.estimator.BestExporter):
    def export(
        self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export
    ):
        if self._best_eval_result is None or self._compare_fn(
            self._best_eval_result, eval_result
        ):
            tf.compat.v1.logging.info(
                "Exporting a better model ({} instead of {})...".format(
                    eval_result, self._best_eval_result
                )
            )
            BEST_CHECKPOINTS_PATH = "/app/models/best_checkpoint"
            # Delete checkpoints before copy
            for filepath in Path(BEST_CHECKPOINTS_PATH).glob("*"):
                if filepath.name not in ["label_encoder.npy"]:
                    filepath.unlink()
            # Copy the checkpoints files *.meta *.index, *.data* each time there is a better result
            outpath = Path(checkpoint_path)
            for filepath in outpath.parents[0].glob("{}.*".format(outpath.name)):
                shutil.copy(
                    filepath, PurePath(BEST_CHECKPOINTS_PATH).joinpath(filepath.name)
                )
            # Save the checkpoint file for the estimator api
            with open(os.path.join(BEST_CHECKPOINTS_PATH, "checkpoint"), "w") as f:
                f.write(
                    'model_checkpoint_path: "{}"'.format(
                        os.path.basename(checkpoint_path)
                    )
                )
            self._best_eval_result = eval_result
        else:
            tf.compat.v1.logging.info(
                "Keeping the current best model ({} instead of {}).".format(
                    self._best_eval_result, eval_result
                )
            )
