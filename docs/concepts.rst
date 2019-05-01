####################
Hangar Core Concepts
####################

This document provides a high level overview of the problems hangar is designed
to solve and introduces the core concepts for begining to use Hangar.

***************
What Is Hangar?
***************

At it's core hangar is designed to solve many of the same problems faced by
traditional code version control system (ie. ``Git``), just adapted for
numerical data:

* Time travel through the historical evolution of a dataset
* Zero-cost Branching to enable exploratory analysis and collaboration
* Cheap Merging to build datasets over time (with multiple collaborators)
* Completely abstracted organization and management of data files on disk
* Ability to only retrieve a small portion of the data (as needed) while still
  maintaining complete historical record
* Ability to push and pull changes directly to collaborators or a central server
  (ie a truley distributed version control system)

The ability of version control systems to perform these tasks for codebases is
largely taken for granted by almost every developer today; However, we are
in-fact standing on the shoulders of giants, with decades of engineering which
has resulted in these phenomenlly useful tools. Now that a new era of
"Data-Defined software" is taking hold, we find there is a strong need for
analogous version control systems which are designed to handle numerical data at
large scale... Welcome to Hangar!

***********
Inspiration
***********

The design of hangar was heavily influenced by the `Git <https://git-scm.org>`_
source-code version control system. As a Hangar user, many of the fundumental
building blocks and commands can be thought of as interchangeable:

* checkout
* commit
* branch
* merge
* diff
* push
* pull/fetch
* log

Emulating the high level the git syntax has allowed us to create a user
experience which should be familiar in many ways to Hangar users; a goal of the
project is to enable many of the same VCS workflows developers use for code
while working with their data!

There are, however, many fundumental differences in how humans/programs
interpret and use text in source files vs. numerical data which raise many
questions Hangar needs to uniquely solve:

* How do we connect some piece of "Data" with a meaning in the real world?
* How do we diff and merge large collections of data samples?
* How can we resolve conflicts?
* How do we make data access (reading and writing) convienent for both
  user-driven exploratory analyses and high performance production systems
  operating without supervision?
* How can we enable people to work on huge datasets in a local (laptop grade)
  development environment?

We will show how hangar solves these questions in a high-level guide below.
For a deep dive into the Hangar internals, we invite you to check out the
:ref:`ref-hangar-under-the-hood` page.

****************************
How Hangar Thinks About Data
****************************

From this point forward, **when we talk about "Data" we are actually talking
about n-dimensional arrays of numeric information. To Hangar, "Data" is just a
collection of numbers being passed into and out of it.** Data does not have a
file type, it does not have a file-extension, it does not mean anything to
Hangar itself - it is just numbers. This theory of "Data" is nearly as simple as
it gets, and this simplicity is what enables us to be unconstrained as we build
abstractions and utilities to operate on it.

Abstraction 1: What is a Repository?
====================================

A "Repository" consists of an historically ordered mapping of "Commits" over time
by various "Committers" across any number of "Branches".
Though there are many conceptual similarities in what a Git repo and a Hangar Repository
achieve, Hangar is designed with the express purpose of dealing with numeric data.
As such, when you read/write to/from a Repository, the main way of interaction with
information will be through (an arbitrary number of) Datasets in each Commit. A simple
key/value store is also included to store metadata, but as it is a minor point is will
largely be ignored for the rest of this post.

History exists at the Repository level, Information exists at the Commit level.

Abstraction 2: What is a Dataset?
=================================

Let's get philosophical and talk about what a "Dataset" is. The word "Dataset"
invokes some some meaning to humans; A dataset may a canonical name (like
"MNIST" or "CoCo"), it will have a source where it comes from, (ideally) it has a
purpose for some real-world task, it will have people who build, aggregate, and
nurture it, and most importantly a Dataset always contains pieces of some type
of information type which describes "something".

It's an abstract definition, but it is only us, the humans behind the machine, which
associate "Data" with some meaning in the real world; it is in the same vein
which we associate a group of Data in a "Dataset" with some real world meaning.

Our first abstraction is therefore the "Dataset": a grouping of some similar Data
pieces. To define a "Dataset" in Hangar, we need only provide:

* a name
* a type
* a shape

Abstraction 3: What Makes up a Dataset?
=======================================

The invividual pieces of information ("Data") which are grouped together in a
"Dataset" are called "Samples" in the Hangar vernacular. According to the
specification set by our definition of a Dataset, all samples must be numeric
arrays with each having:

1) Same type as defined in the Dataset specification
2) A shape with each dimension size <= the shape (``max shape``) set in the
   dataset.

Additionally, samples in a dataset can either be named, or unnamed (depending on
how you interpret what the information contained in the Dataset actually
represents).

Effective use of Hangar relies on having an understanding of what exactally a
"Sample" is in a particular Dataset. The most effective way to find out is to
ask: "What is the smallest piece of data which has a useful meaning to 'me' (or
'my' downstream processes". In the MNIST dataset, this would be a single digit
image (a 28x28 array); for a medical dataset it might be an entire (512x320x320)
MRI volume scan for a particular patient; while for the NASDAQ Stock Ticker it
might be an hours worth of price data points (or less, or more!) The point is
that when you think about what a sample is, it should typically be the smallest
atomic unit of useful information.

******************************************
Implications of the Hangar Data Philosophy
******************************************

The Domain-Specific File Format Problem
=======================================

Though it may first seem counterintuitive at first, there is an incredible
amount of freedom (and power) that is gained when "you" (the user) start to
decouple some information container from the data which it actually holds. At
the end of the day, the algorithms and systems you use to produce insight from
data are just mathematical operations; math does not operate on a specific file
type, math operates on numbers.

Human & Computational Cost
--------------------------

It seems strange that organizations & projects commonly rely on storing data on
disk in some domain-specific - or custom built - binary format (ie. a `.jpg`
image, `.nii` neuroimaging informatics study, `.cvs` tabular data, etc.), and
just deal with the hastle of maintaining all the infrastructure around reading,
writing, transforming, and preprocessing these files into useable numerical data
every time they want to interact with their Datasets. Even disregarding the
computational cost/overhead of preprocessing & transforming the data on every
read/write, these schemes require significant amounts of human capital
(developer time) to be spent on building, testing, and upkeep/maintenance; all
while adding significant complexity for users. Oh, and they also have a stragely
high inclination to degenerate into horrible complexity which essentialy becomes
"magic" after the original creators move on.

The Hangar system is quite different in this regards. First, **we trust that you
know what your data is and what it should be best represented as**. When writing
to a Hangar repository, you process the data into n-dimensional arrays once.
Then when you retrieve it you are provided with the same array, in the same
shape and datatype (unless you ask for a particular subarray-slice), already
initialized in memory and ready to compute on instantly.

High Performance From Simplicity
--------------------------------

Because Hangar is designed to deal (almost exclusively) with numerical arrays,
we are able to "stand on the shoulders of giants" once again by utilizing many
of the well validated, highly optimized, and community validated numerical array
data management utilities developed by the High Performance Computing community
over the past few decades.

In a sense, the backend of Hangar servers two functions:

1) Bookeeping: recording information about about datasets, samples, commits, etc.
2) Data Storage: highly optimized interfaces which store and retrieve data from
   from disk through it's backend utility.

The details are explained much more thorougly in :ref:`ref-hangar-under-the-hood`.

Because Hangar only considers data to be numbers, the choice of backend to store
data is (in a sense) completly arbitrary so long as `Data In == Data Out`.
**This fact has massive implications for the system**; instead of being tied to
a single backend (each of which will have significant performance tradeoffs for
arrays of particular datatypes, shapes, and access patterns), we simulatneously
store different data pieces in the backend which is most suited to it. A great
deal of care has been taken to optimize parameters in the backend interface
which affecting performance and compression of data samples.

The choice of backend to store a piece of data is selected automatically from
heuristics based on the dataset specification, system details, and context of
the storage service internal to Hangar. **As a user, this is completly
transparent to you** in all steps of interacting with the repository. It does
not require (or even accept) user specified configuration.

At the time of writing, Hangar has the following backends implemented (with
plans to potentially support more as needs arise):

1) `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
2) `Memmapped Arrays <https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html>`_
3) `TileDb <https://tiledb.io/>`_ (in development)


Open Source Software Style Collaboration in Dataset Curation
============================================================

Specialized Domain Knowledge is A Scarce Resource
-------------------------------------------------

A common side effect of the `The Domain-Specific File Format Problem`_ is that
anyone who wants to work with an organization's/project's data needs to not only
have some domain expertise (so they can do useful things with the data), but
they also need to have a non-trivial understanding of the projects dataset, file
format, and access conventions / transformation pipelines. *In a world where
highly specialized talent is already scarce, this phenonenom shrinks the pool of
available collatorators dramatically.*

Given this situation, it's understandable why when most organizations spend
massive amounts of money and time to build a team, collect & anotate data, and
build an infrastructure around that information, they hold it for their private
use with little reagards for how the world could use it together. Buisnesses
rely on proprietary information to stay ahead of their competitors, and because
this information is so difficult (and expensive) to generate, it's completly
reaonable that they should be the ones to benefit from all that work.

    **A Thought Experiment**

    Imagine that `Git` and `GitHub` didn't take over the world. Imagine that the
    `Diff` and `Patch` Unix tools never existed. Instead, imagine we were to live in
    a world where every software project had very different version control systems
    (largely homeade by non VCS experts, & not validated by a community over many
    years of use). Even worse, most of these tools don't allow users to easily
    branch, make changes, and automatically merge them back. It shouldn't be
    difficult to imagine how dramatically such a world would contrast to ours today.
    Open source software as we know it would hardly exist, and any efforts would
    probably be massively fragmented across the web (if there would even be a 'web'
    that we would recognize in this strange world).

    Without a way to collaborate in the open, open source software would largely not
    exist, and we would all be worse off for it.

    Doesn't this hypothetical sound quite a bit like the state of open source data
    collaboration in todays world?

The impetus for developing a tool like Hangar is the belief that if it is
simple for anyone with domain knowledge to collaborativly curate datasets
containing information they care about, then they will.* Open source software
development benefits everyone, we believe open source dataset curation can do
the same.

How To Overcome The "Size" Problem
----------------------------------

Even if the greatest tool imaginable existed to version, branch, and merge
datasets, it would face one massive problem which if it didn't solve would kill
the project: *The size of data can very easily exceeds what can fit on (most)
contributors laptops or personal workstations*. This section explains how Hangar
can handle working with datasets which are prohibitivly large to download or
store on a single machine.

As mentioned in `High Performance From Simplicity`_, under the hood Hangar deals
with "Data" and "Bookeeping" completly seperatly. We've previously covered what
exactally we mean by Data in `How Hangar Thinks About Data`_, so we'll briefly
cover the second major component of Hangar here. In short "Bookeeping" describes
everything about the repository. By everything, we do mean that the Bookeeping
records describe everything: all commits, parents, branches, datasets, samples,
data descriptors, schemas, commit message, etc. Though complete, these records
are fairly small (tens of MB in size for decently sized repositories with decent
history), and are highly compressed for fast transfer between a Hangar
client/server.

    **A brief technical interlude**

    There is one very important (and rather complex) property which gives Hangar
    Bookeeping massive power: **Existence of some data piece is always known to
    Hangar and stored immutably once committed. However, the access patern, backend,
    and locating information for this data piece may (and over time, will) be unique
    in every hangar repository instance**.

    Though the details of how this works is well beyond the scope of this document,
    the following example may provide some insight into the implications of this
    property:

        If you `clone` some hangar repository, Bookeeping says that "some number
        of data piece exist" and they should retrieved from the server. However,
        the bookeeping records transfered in a `fetch` / `push` / `clone`
        operation do not include information about where that piece of data
        existed on the client (or server) computer. Two synced repositories can
        use completly different backends to store the data, in completly
        different locations, and it does not matter - Hangar only guarrentees
        that when collaborators ask for a data sample in some checkout, that
        they will be provided with identical arrays, not that they will come
        from the same place or be stored in the same way. Only when data is
        actually retrieved is the "locating information" set for that repository
        instance.

Because Hangar makes no assumptions about how/where it should retrieve some
piece of data, or even an assumption that it exists on the local machine, and
because records are small and completly describe history, once a machine has the
Bookeeping, it can decide what data it actually wants to materialize on it's
local disk! These `partial fetch`/`partial clone` operations can materialize any
desired data, wheather it be for a few records at the head branch, for all data
in a commit, or for the entire historical data. A future release will even
include the ability to stream data directly to a hangar checkout and materialize
the data in memory without having to save it to disk at all!

More importantly: **Since Bookeeping describes all history, merging can be
performed between branches which may contain partial (or even no) actual data**.
Aka. You don't need data on disk to merge changes into it. It's an odd concept
which will be explained more in depth in the future.

.. note::

    The features described in this section are in active development for a
    future release. This is an exciting feature which we hope to do much more
    with; Time is our main constraint right now. Hangar is a young project, and
    is rapidly evolving. Current progress can be tracked in the `GitHub
    Repository <https://github.com/tensorwerk/hangar-py>`_

What Does it Mean to "Merge" Data?
----------------------------------

We'll start this section, once again, with a comparison to source code version
control systems. When dealing with source code text, merging is performed in
order to take a set of changes made to a document, and logically insert the
changes into some other version of the document. The goal is to generate a new
version of the document with all changes made to it in a fashion which conforms
to the "change author's" intentions. Simply put: the new version is valid and
what is expected by the authors.

This concept of what it means to merge text does not generally map well to
changes made in a dataset we'll explore why through this section, but look back
to the philosophy of Data outlined in `How Hangar Thinks About Data`_ for
inspiration as we begin. Remember, in the Hangar design a Sample is the smallest
array which contains useful information. As any smaller selection of the sample
array is meaningless, Hangar does not support subarray-slicing or per-index
updates *when writing* data. (subarray-slice queries are permitted for read
operations, though regular use is discouraged and may indicate that your samples
are larger than they should be).

Diffing Hangar Checkouts
^^^^^^^^^^^^^^^^^^^^^^^^

To understand merge logic, we first need to understand diffing, and the actors
operations which can occur.

:Addition:

    An operation which creates a dataset, sample, or some metadata which
    did not previously exist in the relevant branch history.

:Removal:

    An operation which removes some dataset, a sample, or some metadata which
    existed in the parent of the commit under consideration. (Note: removing a
    dataset also removes all samples contained in it)

:Mutation:

    An operation which sets: data to a sample, the value of some metadata key,
    or a dataset schema, to a different value than what it had previously been
    created with (Note: a dataset schema mutation is observed when a dataset is
    removed, and a new dataset with the same name is created with a different
    dtype/shape, all in the same commit)

Diffing additions and removals between branches is trivial, and performs
exactally as one would expect from a text diff. Where things diverge from text
is when we consider
mutations.

Say we have some sample in commit A, a branch is created, the sample is updated,
and commit C is created. At the same time, someone else checks out branch whose
HEAD is at commit A, and commits a change to the sample as well. If these
changes are identical, they are compatible, but what if they are not? In the
following example, we diff and merge each element of the sample array like we
would text:

::
                                                   Merge ??
      commit A          commit B            Does combining mean anything?

    [[0, 1, 2],        [[0, 1, 2],               [[1, 1, 1],
     [0, 1, 2], ----->  [2, 2, 2], ------------>  [2, 2, 2],
     [0, 1, 2]]         [3, 3, 3]]      /         [3, 3, 3]]
          \                            /
           \            commit C      /
            \                        /
             \          [[1, 1, 1], /
              ------->   [0, 1, 2],
                         [0, 1, 2]]

We see that a result can be generated, and can agree if this was a piece of
text, the result would be correct. Don't be fooled, this is an abomination and
utterly wrong/meaningless. Remember we said earlier ``"the result of a merge
should conform to the intentions of each author"``. This merge result conforms to
neither author's intention. The value of an array element is not isolated, every
value affects how the entire sample is understood. The values at Commit B or
commit C may be fine on their own, but if two samples are mutated independently
with non-identical updates, it is a conflict that needs to be handeled by the
authors.

This is the actual behavior of Hangar.

::
      commit A          commit B

    [[0, 1, 2],        [[0, 1, 2],
     [0, 1, 2], ----->  [2, 2, 2], ----- MERGE CONFLICT
     [0, 1, 2]]         [3, 3, 3]]      /
          \                            /
           \            commit C      /
            \                        /
             \          [[1, 1, 1], /
              ------->   [0, 1, 2],
                         [0, 1, 2]]

    * Datasets: Adding / Removing / Mutating (ie. changing dtype or shape)
    * Samples: Adding / Removing / Mutating (replacing contents of named sample with a different array)