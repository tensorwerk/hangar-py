# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx_click.ext',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'recommonmark',
    'IPython.sphinxext.ipython_console_highlighting',
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

nbsphinx_execute = 'never'

autodoc_mock_imports = ['torch', 'tensorflow']
autosummary_generate = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
master_doc = 'index'
project = 'Hangar'
year = '2019-2020'
author = 'Richard Izzo'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.5.1'

pygments_style = 'default'
pygments_lexer = 'PythonConsoleLexer'
highlight_options = {
    'python3': True
}
templates_path = ['.']
exclude_patterns = ['_build', '**.ipynb_checkpoints']
extlinks = {
    'issue': ('https://github.com/tensorwerk/hangar-py/issues/%s', '#'),
    'pr': ('https://github.com/tensorwerk/hangar-py/pull/%s', 'PR #'),
}
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/master', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
}

# Regular expressions that match URIs that should not be checked
# when doing a linkcheck build
linkcheck_ignore = [
    r'http://localhost:\d+/?', 'http://localhost/',
    'https://github.com/tensorwerk/hangar-py',
    r'https://github.com/tensorwerk/hangar-py/.*',
]
linkcheck_retries = 3

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# if not on_rtd:  # only set the theme if we're building docs locally
html_theme = 'sphinx_rtd_theme'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_include_init_with_doc = True

add_module_names = False
doctest_test_doctest_blocks = None
autoclass_content = 'class'

html_theme_options = {
    'style_nav_header_background': 'orange'
}
