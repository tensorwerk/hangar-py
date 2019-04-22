import numpy as np
from hangar.repository import Repository


def gen_3way_merge_with_mutations_and_meta(repo_pth):

    repo = Repository(path=repo_pth)
    repo.init(remove_old=True)
    d = repo.checkout(write=True)

    dset1n = 'dset_1'
    dset1arr = np.random.randint(0, 10, size=(10), dtype=np.int)
    d.init_dataset(name=dset1n, prototype=dset1arr)

    # commit initial data 0:5
    for i in range(5):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.datasets['dset_1'].add(arr, arrn)

    d.metadata.add('firstc', 'hello')
    d.metadata.add('firsttoremove', 'remove me')
    d.commit('first commit')

    # commit data on test branch 10:15
    repo.create_branch('testbranch')
    d.close()
    d = repo.checkout(write=True, branch_name='testbranch')
    for i in range(10, 15):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.datasets['dset_1'].add(arr, arrn)

    # mutate parts of data on test branch
    m1 = d.datasets['dset_1'].get('arr_1_0')
    m2 = d.datasets['dset_1'].get('arr_1_1')
    m1[0:5] = 15
    m2[0:5] = 15
    d.datasets['dset_1'].add(m1, 'arr_1_0')
    d.datasets['dset_1'].add(m2, f'arr_1_1')
    d.metadata.add('testc', 'world testbranch')
    d.commit('commit on test branch')

    # commit data on test branch 15:25
    for i in range(15, 25):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.datasets['dset_1'].add(arr, arrn)

    # remove data on test branch
    d.datasets['dset_1'].remove('arr_1_3')
    d.commit('second commit on test branch')

    # commit data on master 5:10 and add new subset on master.
    d.close()
    d = repo.checkout(write=True, branch_name='master')
    dset2n = 'dset_2'
    dset2arr = np.random.randint(0, 10, size=(20), dtype=np.int)
    dset2 = d.init_dataset(dset2n, prototype=dset2arr)

    for i in range(5, 10):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.datasets['dset_1'].add(arr, arrn)

        arrn = f'arr_2_{i}'
        arr = np.zeros_like(dset2arr)
        arr[:] = i
        dset2.add(arr, arrn)

    mutated = np.zeros_like(dset2arr)
    mutated[:] = 2
    dset2.add(mutated, 'arr_2_0')

    d.metadata.add('masterc', 'world master')
    d.metadata.remove('firsttoremove')
    d.commit('commit on master')
    d.close()

    return True


'''
def gen_name_conflict_data():

    invariant_init(remove_old='y')
    d = vset()

    dset1n = 'dset_1'
    dset1arr = np.random.randint(0, 10, size=(10), dtype=np.int)
    d.create_dset(dset_name=dset1n, sample_input=dset1arr)

    # commit initial data 0:5
    for i in range(5):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.add(dset_name=dset1n, data_name=arrn, data=arr)
    d.commit('first commit')

    # commit data on test branch 10:15
    invariant_create_branch('testbranch')
    d.checkout_branch('testbranch')
    for i in range(5, 10):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.add(dset_name=dset1n, data_name=arrn, data=arr)
    d.commit('commit on test branch')

    # commit data on master 5:10 and add new subset on master.
    d.checkout_branch('master')
    for i in range(9, 15):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i ** 2
        d.add(dset_name=dset1n, data_name=arrn, data=arr)
    d.commit('commit on master')

    print('complete')
    return True
'''


def gen_ff_merge_with_mutations_and_meta(repo_pth):

    repo = Repository(path=repo_pth)
    repo.init(remove_old=True)
    d = repo.checkout(write=True)

    dset1n = 'dset_1'
    dset1arr = np.random.randint(0, 10, size=(10), dtype=np.int)
    d.init_dataset(dset1n, prototype=dset1arr)

    # commit initial data 0:5
    for i in range(5):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.add(dset_name=dset1n, data_name=arrn, data=arr)

    d.add('firstc', 'hello')
    d.add('firsttoremove', 'remove me')
    d.commit('first commit')

    for i in range(10, 15):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.add(dset_name=dset1n, data_name=arrn, data=arr)

    # mutate parts of data on master
    mutated = np.zeros_like(dset1arr)
    mutated[:] = 23
    d.add(dset_name=dset1n, data_name=f'arr_1_0', data=mutated)
    d.add('testc', 'world master')
    d.commit('second commit on master')
    d.close()

    # commit data on test branch 10:15
    repo.create_branch('testbranch')
    d = repo.checkout(write=True, branch_name='testbranch')
    # commit data on test branch 15:25
    for i in range(15, 25):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.add(dset_name=dset1n, data_name=arrn, data=arr)

    # remove data on test branch
    d.remove(dset1n, 'arr_1_3')
    d.commit('first commit on test branch')
    d.close()

    return True


'''
def gen_3way_merge_schema_conflict():

    invariant_init(remove_old='y')
    d = vset()

    dset1n = 'dset_1'
    dset1arr = np.random.randint(0, 10, size=(10), dtype=np.int)
    d.create_dset(dset_name=dset1n, sample_input=dset1arr)

    # commit initial data 0:5
    for i in range(5):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        d.add(dset_name=dset1n, data_name=arrn, data=arr)
    d.commit('first commit')

    # add first subset with conflicting name.
    invariant_create_branch('testbranch')
    d.checkout_branch('testbranch')
    dset2n = 'dset_2'
    dset2arr = np.random.randint(0, 10, size=(20), dtype=np.int)
    d.create_dset(dset_name=dset2n, sample_input=dset2arr)
    for i in range(0, 3):
        arrn = f'arr_2_{i}'
        arr = np.zeros_like(dset2arr)
        arr[:] = i ** 2
        d.add(dset_name=dset2n, data_name=arrn, data=arr)
    d.commit('commit on test branch')

    # add another subset with conflicting name on different branch.
    d.checkout_branch('master')
    dset2n = 'dset_2'
    dset2arr = np.random.randint(0, 10, size=(30), dtype=np.int)
    d.create_dset(dset_name=dset2n, sample_input=dset2arr)
    for i in range(0, 5):
        arrn = f'arr_2_{i}'
        arr = np.zeros_like(dset2arr)
        arr[:] = i
        d.add(dset_name=dset2n, data_name=arrn, data=arr)
    d.commit('second commit on master')
    print('complete')
    return True


if __name__ == '__main__':
    gen_3way_merge_with_mutations_and_meta()
'''