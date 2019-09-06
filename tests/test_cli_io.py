import os
from os import getcwd

import pytest

from hangar import Repository
from hangar.cli import cli

import numpy as np
from click.testing import CliRunner

@pytest.mark.parametrize("backend", ['00', '10'])
@pytest.mark.parametrize("plug", ['pil', 'matplotlib'])
def test_import_images(backend, plug, generate_3_images):

    im1, im2, im3 = generate_3_images
    imgDir = os.path.split(im1[0])[0]

    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        repo = Repository(P, exists=False)
        repo.init('foo', 'bar')
        dummyData = np.ones_like(im1[1])
        dummyData[:] = 0
        co1 = repo.checkout(write=True, branch='master')
        co1.arraysets.init_arrayset(
            name='dummy', prototype=dummyData, named_samples=True, backend=backend)
        co1.arraysets['dummy']['arr1.jpg'] = dummyData
        co1.close()

        res = runner.invoke(cli.import_data, ['--plugin', plug, 'dummy', imgDir], obj=repo)
        assert res.exit_code == 0

        co1b = repo.checkout(write=True)
        assert np.allclose(co1b.arraysets['dummy']['arr1.jpg'], dummyData)
        assert np.allclose(co1b.arraysets['dummy']['arr2.jpg'], im2[1])
        assert np.allclose(co1b.arraysets['dummy']['arr3.jpg'], im3[1])
        co1b.close()

        res = runner.invoke(cli.import_data, ['--plugin', plug, '--overwrite', 'dummy', imgDir], obj=repo)
        assert res.exit_code == 0

        co1c = repo.checkout(write=True)
        assert np.allclose(co1c.arraysets['dummy']['arr1.jpg'], im1[1])
        assert np.allclose(co1c.arraysets['dummy']['arr2.jpg'], im2[1])
        assert np.allclose(co1c.arraysets['dummy']['arr3.jpg'], im3[1])
        co1c.close()


@pytest.mark.parametrize("backend", ['00', '10'])
@pytest.mark.parametrize("in_commands,expected_fnames", [
    (['-o', '.', '-s', 'lol.jpg', 'master', 'dummy'], ['lol.jpg']),
    (['-o', '.', '-f', '.png', 'master', 'dummy'], ['lol.jpg.png', 'arr1.jpg.png', 'arr2.jpg.png', 'arr3.jpg.png']),
    (['-o', '.', '-f', '.jpg', 'master', 'dummy'], ['lol.jpg', 'arr1.jpg', 'arr2.jpg', 'arr3.jpg'])])
def test_export_images(backend, in_commands, expected_fnames, generate_3_images):

    im1, im2, im3 = generate_3_images
    imgDir = os.path.split(im1[0])[0]
    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        repo = Repository(P, exists=False)
        repo.init('foo', 'bar')
        dummyData = np.ones_like(im1[1])
        co1 = repo.checkout(write=True, branch='master')
        co1.arraysets.init_arrayset(
            name='dummy', prototype=dummyData, named_samples=True, backend=backend)
        co1.arraysets['dummy']['lol.jpg'] = dummyData
        co1.close()

        res = runner.invoke(cli.import_data, ['dummy', imgDir], obj=repo)
        assert res.exit_code == 0

        co1b = repo.checkout(write=True)
        co1b.commit('hi')

        res = runner.invoke(cli.export_data, in_commands, obj=repo)
        assert res.exit_code == 0
        for fn in expected_fnames:
            assert os.path.isfile(os.path.join(P, fn))
        co1b.close()


@pytest.mark.parametrize("backend", ['00', '10'])
def test_view_images(monkeypatch, backend, generate_3_images):

    im1, im2, im3 = generate_3_images
    imgDir = os.path.split(im1[0])[0]
    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        repo = Repository(P, exists=False)
        repo.init('foo', 'bar')
        dummyData = np.ones_like(im1[1])
        co1 = repo.checkout(write=True, branch='master')
        co1.arraysets.init_arrayset(
            name='dummy', prototype=dummyData, named_samples=True, backend=backend)
        co1.arraysets['dummy']['lol.jpg'] = dummyData
        co1.close()

        res = runner.invoke(cli.import_data, ['dummy', imgDir], obj=repo)
        assert res.exit_code == 0

        co1b = repo.checkout(write=True)
        co1b.commit('hi')

        from hangar.cli import io
        def mock_show(*args, **kwargs):
            return True
        monkeypatch.setattr(io, 'show', mock_show)

        res = runner.invoke(cli.view_data, ['master', 'dummy', 'arr1.jpg'], obj=repo)

        assert res.exit_code == 0