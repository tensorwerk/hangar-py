====================
Hangar Core Concepts
====================

This document provides a high level overview of the problems hangar is designed
to solve and introduces the core concepts for begining to use Hangar.

What Is Hangar?
===============

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


Inspiration
===========

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


How Hangar Thinks About Data
============================

From this point forward, **when we talk about "Data" we are actually talking
about n-dimensional arrays of numeric information. To Hangar, "Data" is just a
collection of numbers being passed into and out of it.** Data does not have a
file type, it does not have a file-extension, it does not mean anything to
Hangar itself - it is just numbers. This theory of "Data" is nearly as simple as
it gets, and this simplicity is what enables us to be unconstrained as we build
abstractions and utilities to operate on it.


Abstraction 1: What is a Repository?
------------------------------------

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
---------------------------------

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
---------------------------------------

The invividual pieces of information ("Data") which are grouped together in a "Dataset"
are called "Samples" in the Hangar vernacular. According to the specification set by
our definition of a Dataset, all samples must be numeric arrays with each having:

1) Same type as defined in the Dataset specification
2) A shape with each dimension size <= the shape (``max shape``) set in the dataset.

Additionally, samples in a dataset can either be named, or unnamed (depending on
how you interpret what the information contained in the Dataset actually represents).

Effective use of Hangar relies on having an understanding of what exactally a
"Sample" is in a particular Dataset. The most effective way to find out is to
ask: "What is the smallest piece of data which has a useful meaning to 'me' (or
'my' downstream processes". In the MNIST dataset, this would be a single digit
image (a 28x28 array); for a medical dataset it might be an entire (512x320x320)
MRI volume scan for a particular patient; while for the NASDAQ Stock Ticker it
might be an hours worth of price data points (or less, or more!) The point is
that when you think about what a sample is, it should typically be the smallest
atomic unit of useful information.
