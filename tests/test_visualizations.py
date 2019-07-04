import pytest
import numpy as np


def verify_out(capfd, expected):
    out, _ = capfd.readouterr()
    print(out)
    assert expected == out


three_way_log_contents = {
    'head': '074f81d6b9fa5fa856175d47c7cc95cc4a839965',
    'ancestors': {
        '074f81d6b9fa5fa856175d47c7cc95cc4a839965': ['e5ea58dd9c7ffacd45fb128ddc00aced08d13889', '3c9530ac0da1106c0acbe1201900c51548bbcdd9'],
        'e5ea58dd9c7ffacd45fb128ddc00aced08d13889': ['0ff3f2ec156ab8e1026b5271630ccae4556cc260'],
        '0ff3f2ec156ab8e1026b5271630ccae4556cc260': [''],
        '3c9530ac0da1106c0acbe1201900c51548bbcdd9': ['fed88489ab6e59913aee935169b15fe68755d82c'],
        'fed88489ab6e59913aee935169b15fe68755d82c': ['0ff3f2ec156ab8e1026b5271630ccae4556cc260']},
    'specs': {
        '074f81d6b9fa5fa856175d47c7cc95cc4a839965': {
            'commit_message': 'adding in the new testing datasets',
            'commit_time': 1562203830.775428, 'commit_user': 'Foo User', 'commit_email': 'foo@bar.com'},
        'e5ea58dd9c7ffacd45fb128ddc00aced08d13889': {
            'commit_message': 'commit adding validation images and labels',
            'commit_time': 1562203787.320624, 'commit_user': 'Foo User', 'commit_email': 'foo@bar.com'},
        '0ff3f2ec156ab8e1026b5271630ccae4556cc260': {
            'commit_message': 'first commit adding training images and labels',
            'commit_time': 1562203787.257128, 'commit_user': 'Foo User', 'commit_email': 'foo@bar.com'},
        '3c9530ac0da1106c0acbe1201900c51548bbcdd9': {
            'commit_message': 'added testing labels only',
            'commit_time': 1562203787.388417, 'commit_user': 'Foo User', 'commit_email': 'foo@bar.com'},
        'fed88489ab6e59913aee935169b15fe68755d82c': {
            'commit_message': 'added testing images only',
            'commit_time': 1562203787.372292, 'commit_user': 'Foo User', 'commit_email': 'foo@bar.com'}},
    'order': ['074f81d6b9fa5fa856175d47c7cc95cc4a839965',
              '3c9530ac0da1106c0acbe1201900c51548bbcdd9',
              'fed88489ab6e59913aee935169b15fe68755d82c',
              'e5ea58dd9c7ffacd45fb128ddc00aced08d13889',
              '0ff3f2ec156ab8e1026b5271630ccae4556cc260']}

three_way_hash_branch_map = {
    '3c9530ac0da1106c0acbe1201900c51548bbcdd9': ['add-test'],
    'e5ea58dd9c7ffacd45fb128ddc00aced08d13889': ['add-validation'],
    '074f81d6b9fa5fa856175d47c7cc95cc4a839965': ['master'],
    '0ff3f2ec156ab8e1026b5271630ccae4556cc260': ['untouched-live-demo-branch']}


def test_three_way_merge_graph(capfd):
    from hangar.diagnostics import Graph

    g = Graph(use_color=False)
    g.show_nodes(
        dag=three_way_log_contents['ancestors'],
        spec=three_way_log_contents['specs'],
        branch=three_way_hash_branch_map,
        start=three_way_log_contents['head'],
        order=three_way_log_contents['order'],
        show_time=False,
        show_user=False)

    real = '*   074f81d6b9fa5fa856175d47c7cc95cc4a839965 (master) : adding in the new testing datasets\n'\
           '|\  \n'\
           '| * 3c9530ac0da1106c0acbe1201900c51548bbcdd9 (add-test) : added testing labels only\n'\
           '| * fed88489ab6e59913aee935169b15fe68755d82c : added testing images only\n'\
           '* | e5ea58dd9c7ffacd45fb128ddc00aced08d13889 (add-validation) : commit adding validation images and labels\n'\
           '|/  \n'\
           '* 0ff3f2ec156ab8e1026b5271630ccae4556cc260 (untouched-live-demo-branch) : first commit adding training images and labels\n'

    verify_out(capfd, real)


flat_log_contents = {
    'head': '3c9530ac0da1106c0acbe1201900c51548bbcdd9',
    'ancestors': {
        '0ff3f2ec156ab8e1026b5271630ccae4556cc260': [''],
        '3c9530ac0da1106c0acbe1201900c51548bbcdd9': ['fed88489ab6e59913aee935169b15fe68755d82c'],
        'fed88489ab6e59913aee935169b15fe68755d82c': ['0ff3f2ec156ab8e1026b5271630ccae4556cc260']},
    'specs': {
        '0ff3f2ec156ab8e1026b5271630ccae4556cc260': {
            'commit_message': 'first commit adding training images and labels',
            'commit_time': 1562203787.257128, 'commit_user': 'Foo User', 'commit_email': 'foo@bar.com'},
        '3c9530ac0da1106c0acbe1201900c51548bbcdd9': {
            'commit_message': 'added testing labels only',
            'commit_time': 1562203787.388417, 'commit_user': 'Foo User', 'commit_email': 'foo@bar.com'},
        'fed88489ab6e59913aee935169b15fe68755d82c': {
            'commit_message': 'added testing images only',
            'commit_time': 1562203787.372292, 'commit_user': 'Foo User', 'commit_email': 'foo@bar.com'}},
    'order': ['3c9530ac0da1106c0acbe1201900c51548bbcdd9',
              'fed88489ab6e59913aee935169b15fe68755d82c',
              '0ff3f2ec156ab8e1026b5271630ccae4556cc260']}

flat_hash_branch_map = {
    '3c9530ac0da1106c0acbe1201900c51548bbcdd9': ['add-test'],
    '0ff3f2ec156ab8e1026b5271630ccae4556cc260': ['untouched-live-demo-branch']}


def test_flat_merge_graph(capfd):
    from hangar.diagnostics import Graph

    g = Graph(use_color=False)
    g.show_nodes(
        dag=flat_log_contents['ancestors'],
        spec=flat_log_contents['specs'],
        branch=flat_hash_branch_map,
        start=flat_log_contents['head'],
        order=flat_log_contents['order'],
        show_time=False,
        show_user=False)

    expected = '* 3c9530ac0da1106c0acbe1201900c51548bbcdd9 (add-test) : added testing labels only\n'\
               '* fed88489ab6e59913aee935169b15fe68755d82c : added testing images only\n'\
               '* 0ff3f2ec156ab8e1026b5271630ccae4556cc260 (untouched-live-demo-branch) : first commit adding training images and labels\n'

    verify_out(capfd, expected)
