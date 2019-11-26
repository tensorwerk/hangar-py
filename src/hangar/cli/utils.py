"""
Core components for click_plugins


An extension module for click to enable registering CLI commands via setuptools
entry-points.
    from pkg_resources import iter_entry_points
    import click
    from click_plugins import with_plugins
    @with_plugins(iter_entry_points('entry_point.name'))
    @click.group()
    def cli():
        '''Commandline interface for something.'''
    @cli.command()
    @click.argument('arg')
    def subcommand(arg):
        '''A subcommand for something else'''

from click_plugins.core import with_plugins

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
import click

import os
import sys
import traceback


class StrOrIntType(click.ParamType):
    """Custom type for click to parse the sample name
    argument to integer or string
    """

    def convert(self, value, param, ctx):
        if not value:
            return None

        try:
            stype, sample = value.split(':') if ':' in value else ('str', value)
        except ValueError:
            self.fail(f"Sample name {value} not formatted properly", param, ctx)
        try:
            if stype not in ('str', 'int'):
                self.fail(f"type {stype} is not allowed", param, ctx)
            return int(sample) if stype == 'int' else str(sample)
        except (ValueError, TypeError):
            self.fail(f"{sample} is not a valid {stype}", param, ctx)


def parse_custom_arguments(click_args: list) -> dict:
    """
    Parse all the unknown arguments from click for downstream tasks. Used in
    user plugins for custom command line arguments.

    Parameters
    ----------
    click_args : list
        Unknown arguments from click

    Returns
    -------
    parsed : dict
        Parsed arguments stored as key value pair

    Note
    -----
    Unknown arguments must be long arguments i.e should start with --
    """
    parsed = {}
    for i in range(0, len(click_args), 2):
        key = click_args[i]
        val = click_args[i + 1]
        if not key.startswith('--'):
            raise RuntimeError(f"Could not parse argument {key}. It should be prefixed with `--`")
        parsed[key[2:]] = val
    return parsed


def with_plugins(plugins):

    """
    A decorator to register external CLI commands to an instance of
    `click.Group()`.
    Parameters
    ----------
    plugins : iter
        An iterable producing one `pkg_resources.EntryPoint()` per iteration.
    attrs : **kwargs, optional
        Additional keyword arguments for instantiating `click.Group()`.
    Returns
    -------
    click.Group()
    """

    def decorator(group):
        if not isinstance(group, click.Group):
            raise TypeError("Plugins can only be attached to an instance of click.Group()")

        for entry_point in plugins or ():
            try:
                group.add_command(entry_point.load())
            except Exception:
                # Catch this so a busted plugin doesn't take down the CLI.
                # Handled by registering a dummy command that does nothing
                # other than explain the error.
                group.add_command(BrokenCommand(entry_point.name))

        return group

    return decorator


class BrokenCommand(click.Command):

    """
    Rather than completely crash the CLI when a broken plugin is loaded, this
    class provides a modified help message informing the user that the plugin is
    broken and they should contact the owner.  If the user executes the plugin
    or specifies `--help` a traceback is reported showing the exception the
    plugin loader encountered.
    """

    def __init__(self, name):

        """
        Define the special help messages after instantiating a `click.Command()`.
        """

        click.Command.__init__(self, name)

        util_name = os.path.basename(sys.argv and sys.argv[0] or __file__)

        if os.environ.get('CLICK_PLUGINS_HONESTLY'):  # pragma no cover
            icon = u'\U0001F4A9'
        else:
            icon = u'\u2020'

        self.help = (
            "\nWarning: entry point could not be loaded. Contact "
            "its author for help.\n\n\b\n"
            + traceback.format_exc())
        self.short_help = (
            icon + " Warning: could not load plugin. See `%s %s --help`."
            % (util_name, self.name))

    def invoke(self, ctx):

        """
        Print the traceback instead of doing nothing.
        """

        click.echo(self.help, color=ctx.color)
        ctx.exit(1)

    def parse_args(self, ctx, args):
        return args
