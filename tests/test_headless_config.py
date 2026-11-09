"""Guard against accidental GPU/Vulkan initialization from --video."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_all_cli_envs_disable_graphics_even_with_video():
    text=(ROOT/'agcofl'/'cli.py').read_text()
    tree=ast.parse(text)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_env')
    src=ast.get_source_segment(text,fn)
    assert "'render_backend':'none'" in src
    assert "kwargs['render_mode'] = None" in src
    assert 'if video else' not in src


def test_collection_and_policy_never_render_rgb_in_headless_mode():
    for file in ('cli.py','learning.py','teacher.py'):
        assert '.render_rgb_array(' not in (ROOT/'agcofl'/file).read_text()
