*******************************
Checkouts, Branching, & Merging
*******************************

This section deals with navigating repository history, creating &
merging branches, and understanding conflicts

The Hangar Workflow
===================

The hangar workflow is intended to mimic common ``git`` workflows in which small
incremental changes are made and committed on dedicated ``topic`` branches.
After the ``topic`` has been adequatly set, ``topic`` branch is ``merged`` into
a seperate branch (commonly refered to as ``master``, though it need not be the
actual branch named ``"master"``), where well vetted and more permenant changes
are kept.

::

   Create Branch -> Checkout Branch -> Make Changes -> Commit

Making the Initial Commit
-------------------------

Let’s initialize a new repository and see how branching works in Hangar

.. code:: python

   >>> from hangar import Repository
   >>> import numpy as np

   >>> repo = Repository(path='/foo/bar/path')
   >>> repo.init(user_name='Rick Izzo', user_email='hangar.info@tensorwerk.com', remove_old=True)
   Hangar Repo initialized at: /foo/bar/path/__hangar


When a repository is first initialized, it has no history, no commits.

.. code:: python

   >>> repo.log() # -> returns None
   None

Though the repository is essentially empty at this point in time, there is one
thing which is present: A branch with the name: ``"master"``.

.. code:: python

   >>> repo.list_branches()
   ['master']


This ``"master"`` is the branch we make our first commit on; until we do, the
repository is in a semi-unstable state; with no history or contents, most of the
functionality of a repository (to store, retrieve, and work with versions of
data across time) just isn't possible. A significant potion of otherwise
standard operations will generally flat out refuse to to execute (ie. read-only
checkouts, log, push, etc.) until the first commit is made.

One of the only options available at this point in time is to create a
write-enabled checkout on the ``"master"`` branch and begin to add data so we
can make a commit. let’s do that now:

.. code:: python

   >>> co = repo.checkout(write=True)

As expected, there are no datasets or metadata samples recorded in the checkout

.. code:: python

   >>> print(f'number of metadata keys: {len(co.metadata)}')
   number of metadata keys: 0

   >>> print(f'number of datasets: {len(co.datasets)}')
   number of datasets: 0


Let’s add a dummy array just to put something in the repository history to
commit. We'll then close the checkout so we can explore some useful tools which
depend on having at least on historical record (commit) in the repo.

.. code:: python

   >>> dummy = np.arange(10, dtype=np.uint16)
   >>> dset = co.datasets.init_dataset(name='dummy_dataset', prototype=dummy)
   Dataset Initialized: `dummy_dataset`
   >>> dset['0'] = dummy
   >>> initialCommitHash = co.commit('first commit with a single sample added to a dummy dataset')
   Commit completed. Commit hash: b21ebbeeece723bf7aa2157eb2e8742a043df7d0
   >>> co.close()
   writer checkout of master closed

If we check the history now, we can see our first commit hash, and that it is
labeled with the branch name ``"master"``

.. code:: python

   >>> repo.log()
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 (master) : first commit with a single sample added to a dummy dataset


So now our repository contains: - A commit: a fully independent description of
the entire repository state as it existed at some point in time. A commit is
identified by a ``commit_hash`` - A branch: a label pointing to a particular
``commit`` / ``commit_hash``

Once committed, it is not possible to remove, modify, or otherwise tamper with
the contents of a commit in any way. It is a permenant record, which Hangar has
no method to change once written to disk.

In addition, as a ``commit_hash`` is not only calculated from the ``commit``\ ’s
contents, but from the ``commit_hash`` of its parents (more on this to follow),
knowing a single top-level ``commit_hash`` allows us to verify the integrity of
the entire repository history. This fundumental behavior holds even in cases of
disk-corruption or malicious use.

Working with Checkouts & Branches
=================================

As mentioned in the first tutorial, we work with the data in a repository though
a ``checkout``. There are two types of checkouts (each of which have different
uses and abilities):

**Checking out a branch/commit for reading:** is the process of retriving
records describing repository state at some point in time, and setting up access
to the referenced data.

-  Any number of read checkout processes can operate on a repository (on
   any number of commits) at the same time.

**Checking out a branch for writing:** is the process of setting up a (mutable)
``staging area`` to temporarily gather record references / data before all
changes have been made and staging area contents are ``committed`` in a new
permenant record of history (a ``commit``)

-  Only one write-enabled checkout can ever be operating in a repository
   at a time
-  When initially creating the checkout, the ``staging area`` is not
   actually “empty”. Instead, it has the full contents of the last ``commit``
   referenced by a branch’s ``HEAD``. These records can be removed/mutated/added
   to in any way to form the next ``commit``. The new ``commit`` retains a
   permenant reference identifying the previous ``HEAD`` ``commit`` was used as
   it’s base ``staging area``
-  On commit, the branch which was checked out has it’s ``HEAD`` pointer
   value updated to the new ``commit``\ ’s ``commit_hash``. A write-enabled
   checkout starting from the same branch will now use that ``commit``\ ’s
   record content as the base for it’s ``staging area``.

Creating Branches
-----------------

A branch is an individual series of changes/commits which diverge from the main
history of the repository at some point in time. All changes made along a branch
are completly isolated from those on other branches. After some point in time,
changes made in a disparate branches can be unified through an automatic
``merge`` process (described in detail later in this tutorial). In general, the
``Hangar`` branching model is semantically identical ``Git``; Hangar branches
also have the same lightweight and performant properties which make working with
``Git`` branches so appealing.

In hangar, branch must always have a ``name`` and a ``base_commit``. However, If
no ``base_commit`` is specified, the current writer branch ``HEAD`` ``commit``
is used as the ``base_commit`` hash for the branch automatically.

.. code:: python

   >>> branch_1 = repo.create_branch(name='testbranch')
   >>> branch_1
   'testbranch'

viewing the log, we see that a new branch named: ``testbranch`` is pointing to
our initial commit

.. code:: python

   >>> print(f'branch names: {repo.list_branches()} \n')
   branch names: ['master', 'testbranch']

   >>> repo.log()
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 (master) (testbranch) : first commit with a single sample added to a dummy dataset


If instead, we do actually specify the base commit (with a different branch
name) we see we do actually get a third branch. pointing to the same commit as
``"master"`` and ``"testbranch"``

.. code:: python

   >>> branch_2 = repo.create_branch(name='new', base_commit=initialCommitHash)
   >>> branch_2
   'new'

   >>> repo.log()
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 (master) (new) (testbranch) : first commit with a single sample added to a dummy dataset


Making changes on a branch
--------------------------

Let’s make some changes on the ``"new"`` branch to see how things work. We can
see that the data we added previously is still here (``dummy`` dataset containing
one sample labeled ``0``)

.. code:: python

   >>> co = repo.checkout(write=True, branch='new')
   >>> co.datasets
    Hangar Datasets
        Writeable: True
        Dataset Names:
          - dummy_dataset

   >>> co.datasets['dummy_dataset']
    Hangar DatasetDataWriter
       Dataset Name     : dummy_dataset
       Schema UUID      : d82cddc07e0211e9a08a8c859047adef
       Schema Hash      : 43edf7aa314c
       Variable Shape   : False
       (max) Shape      : (10,)
       Datatype         : <class 'numpy.uint16'>
       Named Samples    : True
       Access Mode      : a
       Num Samples      : 1

   >>> co.datasets['dummy_dataset']['0']
   array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint16)

Let’s add another sample to the ``dummy_dataset`` called ``1``

.. code:: python

   >>> arr = np.arange(10, dtype=np.uint16)
   >>> # let's increment values so that `0` and `1` aren't set to the same thing
   >>> arr += 1
   >>> co.datasets['dummy_dataset']['1'] = arr

We can see that in this checkout, there are indeed, two samples in the
``dummy_dataset``

.. code:: python

   >>> len(co.datasets['dummy_dataset'])
   2

That’s all the changes we'll make for now, let’s commit this and be done with
that branch.

.. code:: python

   >>> co.commit('commit on `new` branch adding a sample to dummy_dataset')
   Commit completed. Commit hash: 0cdd8c833f654d18ddc2b089fabee93c32c9c155
   >>> co.close()
   writer checkout of new closed

How do changes appear when made on a branch?
--------------------------------------------

If we look at the log, we see that the branch we were on (``new``) is a commit
ahead of ``master`` and ``testbranch``

.. code:: python

   >>> repo.log()
   * 0cdd8c833f654d18ddc2b089fabee93c32c9c155 (new) : commit on `new` branch adding a sample to dummy_dataset
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 (master) (testbranch) : first commit with a single sample added to a dummy dataset

The meaning is exactally what one would intuit. we made some changes, they were
reflected on the ``new`` branch, but the ``master`` and ``testbranch`` branches
were not impacted at all, nor were any of the commits!

Merging (Part 1) Fast-Forward Merges
====================================

Say we like the changes we made on the ``new`` branch so much that we want them
to be included into our ``master`` branch! How do we make this happen for this
scenario??

Well, the history between the ``HEAD`` of the ``"new"`` and the ``HEAD`` of the
``"master"`` branch is perfectly linear. In fact, when we began making changes
on ``"new"``, our staging area was *identical* to what the ``"master"`` ``HEAD``
commit references are right now!

If you’ll remember that a branch is just a pointer which assigns some ``name``
to a ``commit_hash``, it becomes apparent that a merge in this case really
doesn’t involve any work at all. With a linear history between ``"master"`` and
``"new"``, any ``commits`` exsting along the path between the ``HEAD`` of
``"new"`` and ``"master"`` are the only changes which are introduced, and we can
be sure that this is the only view of the data records which can exist!

What this means in practice is that for this type of merge, we can just update
the ``HEAD`` of ``"master"`` to point to the ``"HEAD"`` of ``"new"``, and the
merge is complete.

This situation is reffered to as a **Fast Forward (FF) Merge**. A FF merge is
safe to perform any time a linear history lies between the ``"HEAD"`` of some
``topic`` and ``base`` branch, regardless of how many commits or changes which
were introduced.

For other situations, a more complicated **Three Way Merge** is required. This
merge method will be explained a bit more later in this tutorail

.. code:: python

   >>> co = repo.checkout(write=True, branch='master')

Performing the Merge
--------------------

In practice, you’ll never need to know the details of the merge theory explained
above (or even remember it exists). Hangar automatically figures out which merge
algorithms should be used and then performes whatever calculations are needed to
compute the results.

As a user, merging in Hangar is a one-liner!

.. code:: python

   >>> digest = co.merge(message='message for commit (not used for FF merge)', dev_branch='new')
   Selected Fast-Forward Merge Stratagy
   removing all stage hash records

   >>> print(f'new commit digest: {digest}')
   new commit digest: 0cdd8c833f654d18ddc2b089fabee93c32c9c155

Let’s check the log!

.. code:: python

   >>> repo.log()
   * 0cdd8c833f654d18ddc2b089fabee93c32c9c155 (master) (new) : commit on `new` branch adding a sample to dummy_dataset
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 (testbranch) : first commit with a single sample added to a dummy dataset

   >>> co.branch_name
   master
   >>> co.commit_hash
   0cdd8c833f654d18ddc2b089fabee93c32c9c155

   >>> co.datasets['dummy_dataset']
    Hangar DatasetDataWriter
       Dataset Name     : dummy_dataset
       Schema UUID      : d82cddc07e0211e9a08a8c859047adef
       Schema Hash      : 43edf7aa314c
       Variable Shape   : False
       (max) Shape      : (10,)
       Datatype         : <class 'numpy.uint16'>
       Named Samples    : True
       Access Mode      : a
       Num Samples      : 2

   >>> co.close()
   writer checkout of master closed

As you can see, everything is as it should be!


Making a changes to introduce diverged histories
------------------------------------------------

Let’s now go back to our ``"testbranch"`` branch and make some changes there so
we can see what happens when changes don’t follow a linear history.

.. code:: python

   >>> co = repo.checkout(write=True, branch='testbranch')
   >>> co.datasets
    Hangar Datasets
        Writeable: True
        Dataset Names:
          - dummy_dataset

   >>> co.datasets['dummy_dataset']
    Hangar DatasetDataWriter
       Dataset Name     : dummy_dataset
       Schema UUID      : d82cddc07e0211e9a08a8c859047adef
       Schema Hash      : 43edf7aa314c
       Variable Shape   : False
       (max) Shape      : (10,)
       Datatype         : <class 'numpy.uint16'>
       Named Samples    : True
       Access Mode      : a
       Num Samples      : 1

We will start by mutating sample ``0`` in ``dummy_dataset`` to a different value

.. code:: python

   >>> dummy_dset = co.datasets['dummy_dataset']
   >>> old_arr = dummy_dset['0']
   >>> new_arr = old_arr + 50
   >>> new_arr
   array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59], dtype=uint16)

   >>> dummy_dset['0'] = new_arr

let’s make a commit here, then add some metadata and make a new commit (all on
the ``testbranch`` branch)

.. code:: python

   >>> digest = co.commit('mutated sample `0` of `dummy_dataset` to new value')
   Commit operation requested with message: mutated sample `0` of `dummy_dataset` to new value
   (288, 222, 288)
   removing all stage hash records
   Commit completed. Commit hash: 4fdb96afed4ec62e9fc80328abccae6bf6774fea
   >>> print(digest)
   4fdb96afed4ec62e9fc80328abccae6bf6774fea

   >>> repo.log()
   * 4fdb96afed4ec62e9fc80328abccae6bf6774fea (testbranch) : mutated sample `0` of `dummy_dataset` to new value
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 : first commit with a single sample added to a dummy dataset

   >>> co.metadata['hello'] = 'world'

   >>> digest = co.commit('added hellow world metadata')
   Commit operation requested with message: added hellow world metadata
   (348, 260, 348)
   removing all stage hash records
   Commit completed. Commit hash: ce8a9198d638b8fd89a175486d21d2bb2efabc91

   >>> print(digest)
   ce8a9198d638b8fd89a175486d21d2bb2efabc91
   >>> co.close()
   writer checkout of testbranch closed

Looking at our history how, we see that none of the original branches reference
our first commit anymore

.. code:: python

   >>> repo.log()
   * ce8a9198d638b8fd89a175486d21d2bb2efabc91 (testbranch) : added hellow world metadata
   * 4fdb96afed4ec62e9fc80328abccae6bf6774fea : mutated sample `0` of `dummy_dataset` to new value
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 : first commit with a single sample added to a dummy dataset

We can check the history of the ``"master"`` branch by specifying it as
an argument to the ``log()`` method

.. code:: python

   >>> repo.log('master')
   * 0cdd8c833f654d18ddc2b089fabee93c32c9c155 (master) (new) : commit on `new` branch adding a sample to dummy_dataset
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 : first commit with a single sample added to a dummy dataset


Merging (Part 2) Three Way Merge
================================

If we now want to merge the changes on ``"testbranch"`` into ``"master"``, we
can’t just follow a simple linear history; **the branches have diverged**.

For this case, Hangar implements a **Three Way Merge** algorithm which does the
following: - Find the most recent common ancestor ``commit`` present in both the
``"testbranch"`` and ``"master"`` branches - Compute what changed between the
common ancestor and each branch’s ``HEAD`` commit - Check if any of the changes
conflict with eachother (more on this in a later tutorial) - If no conflicts are
present, compute the results of the merge between the two sets of changes -
Create a new ``commit`` containing the merge results reference both branch
``HEAD``\ s as parents of the new ``commit``, and update the ``base`` branch
``HEAD`` to that new ``commit``\ ’s ``commit_hash``

.. code:: python

   >>> co = repo.checkout(write=True, branch='master')

Once again, as a user, the details are completly irrelevent, and the operation
occurs from the same one-liner call we used before for the FF Merge.

.. code:: python

   >>> co.merge(message='merge of testbranch into master', dev_branch='testbranch')
   Selected 3-Way Merge Strategy
   (410, 293, 410)
   removing all stage hash records
   'dea1aa627933b3efffa03c743c201ee1b41142c8'

If we now look at the log, we see that this has a much different look then
before. The three way merge results in a history which references changes made
in both diverged branches, and unifies them in a single ``commit``

.. code:: python

   >>> repo.log()
   *  dea1aa627933b3efffa03c743c201ee1b41142c8 (master) : merge of testbranch into master
   |\
   | * ce8a9198d638b8fd89a175486d21d2bb2efabc91 (testbranch) : added hellow world metadata
   | * 4fdb96afed4ec62e9fc80328abccae6bf6774fea : mutated sample `0` of `dummy_dataset` to new value
   * | 0cdd8c833f654d18ddc2b089fabee93c32c9c155 (new) : commit on `new` branch adding a sample to dummy_dataset
   |/
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 : first commit with a single sample added to a dummy dataset


Manually inspecting the merge results
-------------------------------------

``dummy_dataset`` should contain two arrays, key ``1`` was set in the previous
commit originally made in ``"new"`` and merged into ``"master"``. Key ``0`` was
mutated in ``"testbranch"`` and unchanged in ``"master"``, so the update from
``"testbranch"`` is kept.

There should be one metadata sample with they key ``"hello"`` and the value
``"world"``

.. code:: python

   >>> co.datasets
    Hangar Datasets
        Writeable: True
        Dataset Names:
          - dummy_dataset

   >>> co.datasets['dummy_dataset']
    Hangar DatasetDataWriter
       Dataset Name     : dummy_dataset
       Schema UUID      : d82cddc07e0211e9a08a8c859047adef
       Schema Hash      : 43edf7aa314c
       Variable Shape   : False
       (max) Shape      : (10,)
       Datatype         : <class 'numpy.uint16'>
       Named Samples    : True
       Access Mode      : a
       Num Samples      : 2

   >>> co.datasets['dummy_dataset']['0']
   array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59], dtype=uint16)
   >>> co.datasets['dummy_dataset']['1']
   array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=uint16)

   >>> co.metadata
    Hangar Metadata
        Writeable: True
        Number of Keys: 1
   >>> co.metadata['hello']
   'world'

   >>> co.close()
   writer checkout of master closed

**The Merge was a success!**


Conflicts
=========

Now that we’ve seen merging in action, the next step is to talk about conflicts.

How Are Conflicts Detected?
---------------------------

Any merge conflicts can be identified and addressed ahead of running a ``merge``
command by using the built in ``diff`` tools. When diffing commits, Hangar will
provide a list of conflicts which it identifies. In general these fall into 4
catagories:

1. **Additions** in both branches which created new keys (samples /
   datasets / metadata) with non-compatible values. For samples &
   metadata, the hash of the data is compared, for datasets, the schema
   specification is checked for compatibility in a method custom to the
   internal workings of Hangar.
2. **Removal** in ``Master Commit/Branch`` **& Mutation** in ``Dev Commit /
   Branch``. Applies for samples, datasets, and metadata
   identically.
3. **Mutation** in ``Dev Commit/Branch`` **& Removal** in ``Master Commit /
   Branch``. Applies for samples, datasets, and metadata
   identically.
4. **Mutations** on keys both branches to non-compatible values. For
   samples & metadata, the hash of the data is compared, for datasets, the
   schema specification is checked for compatibility in a method custom to the
   internal workings of Hangar.

Let’s make a merge conflict
---------------------------

To force a conflict, we are going to checkout the ``"new"`` branch and set the
metadata key ``"hello"`` to the value ``"foo conflict... BOO!"``. If we then try
to merge this into the ``"testbranch"`` branch (which set ``"hello"`` to a value
of ``"world"``) we see how hangar will identify the conflict and halt without
making any changes.

Automated conflict resolution will be introduced in a future version of Hangar,
for now it is up to the user to manually resolve conflicts by making any
necessary changes in each branch before reattempting a merge operation.

.. code:: python

   >>> co = repo.checkout(write=True, branch='new')
   >>> co.metadata['hello'] = 'foo conflict... BOO!'
   >>> co.commit ('commit on new branch to hello metadata key so we can demonstrate a conflict')
   Commit operation requested with message: commit on new branch to hello metadata key so we can demonstrate a conflict
   (410, 294, 410)
   removing all stage hash records
   Commit completed. Commit hash: 5e76faba059c156bc9ed181446e104765cb471c3
   '5e76faba059c156bc9ed181446e104765cb471c3'

   >>> repo.log()
   * 5e76faba059c156bc9ed181446e104765cb471c3 (new) : commit on new branch to hello metadata key so we can demonstrate a conflict
   * 0cdd8c833f654d18ddc2b089fabee93c32c9c155 : commit on `new` branch adding a sample to dummy_dataset
   * b21ebbeeece723bf7aa2157eb2e8742a043df7d0 : first commit with a single sample added to a dummy dataset


**When we attempt the merge, an exception is thrown telling us there is a conflict**

.. code:: python

   >>> co.merge(message='this merge should not happen', dev_branch='testbranch')
   Selected 3-Way Merge Strategy
   --------------------------------------------------------------------------------------------
   ValueError: HANGAR VALUE ERROR:: Merge ABORTED with conflict:
   {'dset': ConflictRecords(t1=(), t21=(), t22=(), t3=(), conflict=False),
    'meta': ConflictRecords(t1=('hello',), t21=(), t22=(), t3=(), conflict=True),
    'sample': {'dummy_dataset': ConflictRecords(t1=(), t21=(), t22=(), t3=(), conflict=False)},
    'conflict_found': True}

Checking for Conflicts
----------------------

Alternatively, use the diff methods on a checkout to test for conflicts before attempting a merge

.. code:: python

   >>> merge_results, conflicts_found = co.diff.branch('testbranch')
   >>> print(conflicts_found)
   {'dset': ConflictRecords(t1=(), t21=(), t22=(), t3=(), conflict=False),
    'meta': ConflictRecords(t1=('hello',), t21=(), t22=(), t3=(), conflict=True),
    'sample': {'dummy_dataset': ConflictRecords(t1=(), t21=(), t22=(), t3=(), conflict=False)},
    'conflict_found': True}

   >>> conflicts_found['meta']
   ConflictRecords(t1=('hello',), t21=(), t22=(), t3=(), conflict=True)

The type codes for a ``ConflictRecords`` ``namedtuple`` such as the one we saw:

::

   ConflictRecords(t1=('hello',), t21=(), t22=(), t3=(), conflict=True)

are as follow:

-  ``t1``: Addition of key in master AND dev with different values.
-  ``t21``: Removed key in master, mutated value in dev.
-  ``t22``: Removed key in dev, mutated value in master.
-  ``t3``: Mutated key in both master AND dev to different values.
-  ``conflict``: Bool indicating if any type of conflict is present.

Remove the Conflict Manually to Resolve Merging
-----------------------------------------------

.. code:: python

   >>> del co.metadata['hello']
   >>> co.metadata['resolved'] = 'conflict by removing hello key'
   >>> co.commit('commit which removes conflicting metadata key')
   Commit operation requested with message: commit which removes conflicting metadata key
   (413, 296, 413)
   removing all stage hash records
   Commit completed. Commit hash: 4f312b10775c2b0ac51b5f284d2f94e9a8548868
   '4f312b10775c2b0ac51b5f284d2f94e9a8548868'

   >>> co.merge(message='this merge succeeds as it no longer has a conflict', dev_branch='testbranch')
   Selected 3-Way Merge Strategy
   (465, 331, 465)
   removing all stage hash records
   '3550984bd91afe39d9462f7299c2542e7d45444d'

We can verify that history looks as we would expect via the log!

.. code:: python

   >>> repo.log()
   *  3550984bd91afe39d9462f7299c2542e7d45444d (new) : this merge succeeds as it no longer has a conflict
   |\
   * | 4f312b10775c2b0ac51b5f284d2f94e9a8548868 : commit which removes conflicting metadata key
   * | 5e76faba059c156bc9ed181446e104765cb471c3 : commit on new branch to hello metadata key so we can demonstrate a conflict
   | * ce8a9198d638b8fd89a175486d21d2bb2efabc91 (testbranch) : added hellow world metadata
   | * 4fdb96afed4ec62e9fc80328abccae6bf6774fea : mutated sample `0` of `dummy_dataset` to new value
   * | 0cdd8c833f654d18ddc2b089fabee93c32c9c155 : commit on `new` branch adding a sample to dummy_dataset
   |/
   *  b21ebbeeece723bf7aa2157eb2e8742a043df7d0 : first commit with a single sample added to a dummy dataset
