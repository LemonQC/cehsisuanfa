# Historical v0.1.1 headless fix

**Superseded by v0.2.0.** Please use [HEADLESS_GUIDE.md](HEADLESS_GUIDE.md).

All `agcofl.cli` commands now set `render_backend='none'` even if `--video` is supplied. The flag produces a 2D animation of recorded cube coordinates; it **does not** initialize the RGB camera renderer or provide a robot RGB video. Start with `python -m agcofl.cli doctor`.