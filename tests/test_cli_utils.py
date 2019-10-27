"""Original code author tests for the click-plugin package.

__version__ = '1.1.1'
__author__ = 'Kevin Wurster, Sean Gillies'
__email__ = 'wursterk@gmail.com, sean.gillies@gmail.com'
__source__ = 'https://github.com/click-contrib/click-plugins'
__license__ = '''
New BSD License
Copyright (c) 2015-2019, Kevin D. Wurster, Sean C. Gillies
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither click-plugins nor the names of its contributors may not be used to
  endorse or promote products derived from this software without specific prior
  written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
"""
import pytest
from pkg_resources import EntryPoint
from pkg_resources import iter_entry_points
from pkg_resources import working_set

import click
from click.testing import CliRunner
from hangar.cli.utils import with_plugins


@pytest.fixture(scope='function')
def runner(request):
    return CliRunner()


# Create a few CLI commands for testing
@click.command()
@click.argument('arg')
def cmd1(arg):
    """Test command 1"""
    click.echo('passed')

@click.command()
@click.argument('arg')
def cmd2(arg):
    """Test command 2"""
    click.echo('passed')


# Manually register plugins in an entry point and put broken plugins in a
# different entry point.

# The `DistStub()` class gets around an exception that is raised when
# `entry_point.load()` is called.  By default `load()` has `requires=True`
# which calls `dist.requires()` and the `click.group()` decorator
# doesn't allow us to change this.  Because we are manually registering these
# plugins the `dist` attribute is `None` so we can just create a stub that
# always returns an empty list since we don't have any requirements.  A full
# `pkg_resources.Distribution()` instance is not needed because there isn't
# a package installed anywhere.
class DistStub(object):
    def requires(self, *args):
        return []


working_set.by_key['click']._ep_map = {
    '_test_hangar_plugins.test_cli_utils': {
        'cmd1': EntryPoint.parse(
            'cmd1=tests.test_cli_utils:cmd1', dist=DistStub()),
        'cmd2': EntryPoint.parse(
            'cmd2=tests.test_cli_utils:cmd2', dist=DistStub())
    },
    '_test_hangar_plugins.broken_plugins': {
        'before': EntryPoint.parse(
            'before=tests.broken_plugins:before', dist=DistStub()),
        'after': EntryPoint.parse(
            'after=tests.broken_plugins:after', dist=DistStub()),
        'do_not_exist': EntryPoint.parse(
            'do_not_exist=tests.broken_plugins:do_not_exist', dist=DistStub())
    }
}


# Main CLI groups - one with good plugins attached and the other broken
@with_plugins(iter_entry_points('_test_hangar_plugins.test_cli_utils'))
@click.group()
def good_cli():
    """Good CLI group."""
    pass

@with_plugins(iter_entry_points('_test_hangar_plugins.broken_plugins'))
@click.group()
def broken_cli():
    """Broken CLI group."""
    pass


def test_registered():
    # Make sure the plugins are properly registered.  If this test fails it
    # means that some of the for loops in other tests may not be executing.
    assert len([ep for ep in iter_entry_points('_test_hangar_plugins.test_cli_utils')]) > 1
    assert len([ep for ep in iter_entry_points('_test_hangar_plugins.broken_plugins')]) > 1


def test_register_and_run(runner):

    result = runner.invoke(good_cli)
    assert result.exit_code == 0

    for ep in iter_entry_points('_test_hangar_plugins'):
        cmd_result = runner.invoke(good_cli, [ep.name, 'something'])
        assert cmd_result.exit_code == 0
        assert cmd_result.output.strip() == 'passed'


def test_broken_register_and_run(runner):

    result = runner.invoke(broken_cli)
    assert result.exit_code == 0
    assert u'\U0001F4A9' in result.output or u'\u2020' in result.output

    for ep in iter_entry_points('_test_hangar_plugins.broken_plugins'):
        cmd_result = runner.invoke(broken_cli, [ep.name])
        assert cmd_result.exit_code != 0
        assert 'Traceback' in cmd_result.output


def test_group_chain(runner):


    # Attach a sub-group to a CLI and get execute it without arguments to make
    # sure both the sub-group and all the parent group's commands are present
    @good_cli.group()
    def sub_cli():
        """Sub CLI."""
        pass

    result = runner.invoke(good_cli)
    assert result.exit_code == 0
    assert sub_cli.name in result.output
    for ep in iter_entry_points('_test_hangar_plugins.test_cli_utils'):
        assert ep.name in result.output

    # Same as above but the sub-group has plugins
    @with_plugins(plugins=iter_entry_points('_test_hangar_plugins.test_cli_utils'))
    @good_cli.group(name='sub-cli-plugins')
    def sub_cli_plugins():
        """Sub CLI with plugins."""
        pass

    @sub_cli_plugins.command()
    @click.argument('arg')
    def cmd1(arg):
        """Test command 1"""
        click.echo('passed')

    result = runner.invoke(good_cli, ['sub-cli-plugins'])
    assert result.exit_code == 0
    for ep in iter_entry_points('_test_hangar_plugins.test_cli_utils'):
        assert ep.name in result.output

    # Execute one of the sub-group's commands
    result = runner.invoke(good_cli, ['sub-cli-plugins', 'cmd1', 'something'])
    assert result.exit_code == 0
    assert result.output.strip() == 'passed'


def test_exception():
    # Decorating something that isn't a click.Group() should fail
    with pytest.raises(TypeError):
        @with_plugins([])
        @click.command()
        def cli():
            """Whatever"""


def test_broken_register_and_run_with_help(runner):
    result = runner.invoke(broken_cli)
    assert result.exit_code == 0
    assert u'\U0001F4A9' in result.output or u'\u2020' in result.output

    for ep in iter_entry_points('_test_hangar_plugins.broken_plugins'):
        cmd_result = runner.invoke(broken_cli, [ep.name, "--help"])
        assert cmd_result.exit_code != 0
        assert 'Traceback' in cmd_result.output


def test_broken_register_and_run_with_args(runner):
    result = runner.invoke(broken_cli)
    assert result.exit_code == 0
    assert u'\U0001F4A9' in result.output or u'\u2020' in result.output

    for ep in iter_entry_points('_test_hangar_plugins.broken_plugins'):
        cmd_result = runner.invoke(broken_cli, [ep.name, "-a", "b"])
        assert cmd_result.exit_code != 0
        assert 'Traceback' in cmd_result.output