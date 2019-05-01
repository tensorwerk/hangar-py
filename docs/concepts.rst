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

* Time travel through the historical evolution a dataset
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
