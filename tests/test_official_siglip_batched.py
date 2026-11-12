"""CPU-only deterministic check for equivalence to the original full-T path."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np
from agcofl.official_siglip_batched import encode_camera_batched, validate_batch_frames, IMAGE_GRIDS


class FakeEinops:
    def rearrange(self, a, rule, **kw):
        if rule == 'k v h w c -> (k v) h w c':
            return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])
        if rule == '(k v) h w c -> k v h w c':
            return a.reshape(kw['k'],kw['v'], *a.shape[1:])
        raise ValueError(rule)


class FakeImageTools:
    def resize_with_pad(self, x, h, w):
        assert (h,w) == (224,224)
        return x


class FakeJax:
    def device_get(self,x):
        return x


class FakeAuthorEncoder:
    def __init__(self, max_T):
        self._jnp=np
        self._einops=FakeEinops()
        self._image_tools=FakeImageTools()
        self._jax=FakeJax()
        self._max_T=max_T
        self._pe={n:np.arange(max_T*t*768, dtype=np.float32).reshape(max_T,t,768) for n,t in IMAGE_GRIDS}
    def _siglip_call(self,x):
        # Frame-independent, exactly as real SigLIP inference without temporal pooling.
        mean=x.mean(axis=(2,3,4),keepdims=True).astype(np.float32)
        return np.broadcast_to(mean.reshape(x.shape[0],1,1,1),
                               (x.shape[0],1,256,2048)).copy()
    def _pool(self,tokens,t):
        n,v,p,d=tokens.shape
        return tokens.reshape(n,v,t,p//t,d).mean(axis=3)
    def _atomic_save(self,p,b):
        tmp=p.with_name('.'+p.stem+'.tmp.npy')
        np.save(tmp,b,allow_pickle=True)
        tmp.replace(p)


def full_reference(enc,rgb):
    t=len(rgb)
    x=(rgb.astype(np.float32)/255.0)*2.0-1.0
    x=enc._jnp.array(x[:,None])
    flat=enc._einops.rearrange(x,'k v h w c -> (k v) h w c')
    resized=enc._image_tools.resize_with_pad(flat,224,224)
    tokens=enc._siglip_call(enc._einops.rearrange(resized,'(k v) h w c -> k v h w c',k=t,v=1))
    result={}
    for n,p in IMAGE_GRIDS:
        result['image_emb_'+n]=np.asarray(enc._pool(tokens,p))[:,0]
        result['pos_emb_'+n]=enc._pe[n][:t]
    result['state_emb']=np.zeros((t,8),dtype=np.float32)
    result['timesteps']=np.arange(t,dtype=np.int32)
    return result


class BatchTest(unittest.TestCase):
    def test_equivalence_all_shapes_partial_last_batch(self):
        rng=np.random.default_rng(123)
        rgb=rng.integers(0,256,(5,4,7,3),dtype=np.uint8)
        enc=FakeAuthorEncoder(6)
        reference=full_reference(enc,rgb)
        for batch in (1,2,4):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as td:
                seen=[]
                text=encode_camera_batched(enc, Path(td),rgb,batch_frames=batch,
                                             progress=lambda a,b:seen.append((a,b)))
                actual=np.load(Path(td)/'front.npy',allow_pickle=True).item()
                self.assertEqual(set(actual),set(reference))
                for key,val in reference.items():
                    self.assertTrue(np.array_equal(actual[key],val),key)
                    self.assertEqual(actual[key].dtype,val.dtype)
                self.assertEqual(seen[-1],(5,5))
                self.assertIn('author-schema',text)

    def test_reject_overwrite_and_input_too_long(self):
        with tempfile.TemporaryDirectory() as td:
            rgb=np.zeros((3,4,5,3),dtype=np.uint8)
            encoder=FakeAuthorEncoder(4)
            encode_camera_batched(encoder,Path(td),rgb)
            with self.assertRaises(FileExistsError):
                encode_camera_batched(encoder,Path(td),rgb)
            with self.assertRaises(ValueError):
                encode_camera_batched(encoder,Path(td)/'new',np.zeros((5,4,5,3),dtype=np.uint8))

    def test_reject_bad_batch_sizes_without_allocating_gpu(self):
        for val in (0,9,-1,1.5,None,True):
            with self.subTest(value=val), self.assertRaises(ValueError):
                validate_batch_frames(val)

    def test_never_accept_unnatural_or_non_front_sources(self):
        enc=FakeAuthorEncoder(10)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                encode_camera_batched(enc,Path(td),np.zeros((3,2,2,3),dtype=np.float32))
            with self.assertRaises(ValueError):
                encode_camera_batched(enc,Path(td),np.zeros((3,2,2,3),dtype=np.uint8),camera='left')


if __name__=='__main__':unittest.main()
