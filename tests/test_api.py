import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData

from circa import CIRCA, CIRCAModel
from circa.modules.callbacks import StopAfterLRReductions


class APITest(unittest.TestCase):
    def test_public_names_and_legacy_imports(self):
        from circa.api import CIRCARun
        from circa.model.CIRCA import CIRCA as LegacyModel
        self.assertIs(CIRCARun, CIRCA)
        self.assertIs(LegacyModel, CIRCAModel)
        self.assertEqual(CIRCAModel.__name__, 'CIRCAModel')
        self.assertEqual(CIRCA.__name__, 'CIRCA')

    def test_lr_reductions_ignore_warmup_and_restore(self):
        callback = StopAfterLRReductions(2)
        trainer = SimpleNamespace(optimizers=[SimpleNamespace(param_groups=[{'lr': 1e-5}])], should_stop=False)
        callback.on_train_start(trainer, None)
        for rate in (1e-4, 1e-3, 5e-4):
            trainer.optimizers[0].param_groups[0]['lr'] = rate
            callback.on_train_epoch_start(trainer, None)
        self.assertFalse(trainer.should_stop)
        restored = StopAfterLRReductions(2)
        restored.load_state_dict(callback.state_dict())
        trainer.optimizers[0].param_groups[0]['lr'] = 2.5e-4
        restored.on_train_epoch_start(trainer, None)
        self.assertTrue(trainer.should_stop)

    def test_best_checkpoint_multimodal_and_reference_coverage(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(9)
        adata = AnnData(sp.csr_matrix(rng.poisson(2, (64, 8)).astype(np.float32)))
        adata.obs['slide'] = np.repeat(['a', 'b'], 32)
        adata.obs['amyloid'] = rng.normal(size=64)
        adata.obsm['spatial'] = rng.random((64, 2))
        for extra in (False, True):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                run = CIRCA(adata, latent_dim=3, n_neighbors=2, batch_size=8,
                    slide_key='slide', reference_indices=[0, 32],
                    feature_groups={'amyloid': ['amyloid']} if extra else None,
                    model_kwargs=dict(hidden_dims=[8, 8], proj_hidden_dims=[8], psi_hidden_dims=[8],
                                      phi_type='gauss-abs', warmup_steps=1),
                    data_kwargs={'use_slide_batch_sampler': False, 'val_fraction': .25})
                run.fit(directory, max_epochs=2, accelerator='cpu',
                        trainer_kwargs=dict(logger=False, enable_progress_bar=False,
                                            enable_model_summary=False, limit_train_batches=2))
                self.assertIsInstance(run.model, CIRCAModel)
                self.assertTrue(np.isfinite(run.best_validation_loss))
                self.assertEqual(run.embeddings['cell'].shape, (64, 3))
                self.assertTrue(np.isfinite(run.embeddings['cell']).all())
                if extra:
                    self.assertEqual(run.embeddings['extra_amyloid'].shape, (64, 2))
                best = CIRCAModel.load_from_checkpoint(run.best_checkpoint_path, weights_only=False)
                for k, value in best.state_dict().items():
                    torch.testing.assert_close(value, run.model.state_dict()[k])
                run.datamodule.setup('predict')
                batch = next(iter(run.datamodule.predict_dataloader()))
                best.eval()
                with torch.inference_mode():
                    expected = best.predict_step(batch, 0)
                np.testing.assert_allclose(run.embeddings['cell'][expected['adata_idx']], expected['cell_emb'], atol=1e-6)
                self.assertNotIn('panel', adata.obs)


if __name__ == '__main__':
    unittest.main()
