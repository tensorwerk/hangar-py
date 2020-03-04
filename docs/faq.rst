.. _ref-faq:

==========================
Frequently Asked Questions
==========================

The following documentation are taken from questions and comments on the
`Hangar User Group Slack Channel <https://hangarusergroup.slack.com>`_
and over various Github issues.


How can I get an Invite to the Hangar User Group?
==================================================

Just click on `This Signup Link
<https://join.slack.com/t/hangarusergroup/shared_invite/enQtNjQ0NzM5ODQ1NjY1LWZlYmIzNTQ0ODZmOTAwMmNmOTgzZTAzM2NhMWE2MTNlMTRhMzNhN2Y3YmJmMjcwZDgxNDIyMDM1MzVhYzk4MjU>`_
to get started.


Data Integrity
==============

   Being a young project did you encounter some situations where the disaster
   was not a compilation error but dataset corruption? This is the most fearing
   aspect of using young projects but every project will start from a phase
   before becoming mature and production ready.

An absolute requirement of a system right this is to protect user data at all
costs (I’ll refer to this as preserving data "integrity" from here). During our
initial design of the system, we made the decision that preserving integrity
comes above all other system parameters: including performance, disk size,
complexity of the Hangar core, and even features should we not be able to make
them absolutely safe for the user. And to be honest, the very first versions of
Hangar were quite slow and difficult to use as a result of this.

The initial versions of Hangar (which we put together in ~2 weeks) had
essentially most of the features we have today. We’ve improved the API, made
things clearer, and added some visualization/reporting utilities, but not much
has changed. Essentially the entire development effort has been addressing
issues stemming from a fundamental need to protect user data at all costs. That
work has been very successful, and performance is extremely promising (and
improving all the time).

To get into the details here: There have been only 3 instances in the entire
time I’ve developed Hangar where we lost data irrecoverably:

1. We used to move data around between folders with some regularity (as a
   convenient way to mark some files as containing data which have been
   “committed”, and can no longer be opened in anything but read-only mode).
   There was a bug (which never made it past a local dev version) at one point
   where I accidentally called ``shutil.rmtree(path)`` with a directory one
   level too high… that wasn’t great.

   Just to be clear, we don’t do this anymore (since disk IO costs are way too
   high), but remnants of it’s intention are still very much alive and well.
   Once data has been added to the repository, and is “committed”, the file
   containing that data will never be opened in anything but read-only mode
   again. This reduces the chance of disk corruption massively from the start.

----

2. When I was implementing the numpy memmap array storage backend, I was
   totally surprised during an early test when I:

   .. code:: text

      - opened a write-enabled checkout
      - added some data
      - without committing, retrieved the same data again via the user facing API
      - overwrote some slice of the return array with new data and did some processing
      - asked Hangar for that same array key again, and instead of returning
        the contents got a fatal RuntimeError raised by Hangar with the
        code/message indicating "'DATA CORRUPTION ERROR: Checksum {cksum} !=
        recorded for {hashVal}"

   What had happened was that when opening a ``numpy.memmap`` array on disk in
   ``w+`` mode, the default behavior when returning a subarray is to return a
   subclass of ``np.ndarray`` of type ``np.memmap``. Though the numpy docs
   state: "The memmap object can be used anywhere an ndarray is accepted. Given
   a ``memmap fp``, ``isinstance(fp, numpy.ndarray)`` returns ``True``". I did
   not anticipate that updates to the subarray slice would also update the
   memmap on disk. A simple mistake to make; this has since been remedied by
   manually instantiating a new ``np.ndarray`` instance from the ``np.memmap``
   subarray slice buffer.

   However, the nice part is that this was a real world proof that our system
   design worked (and not just in tests). When you add data to a Hangar
   checkout (or receive it on a fetch/clone operation) we calculate a hash
   digest of the data via ``blake2b`` (a cryptographically secure algorithm in the
   python standard library). While this allows us to cryptographically verify full
   integrity checks and history immutability, cryptographic hashes are slow by
   design. When we want to read local data (which we’ve already ensured was
   correct when it was placed on disk) it would be prohibitively slow to do a
   full cryptographic verification on every read. However, since its NOT
   acceptable to provide no integrity verification (even for local writes) we
   compromise with a much faster (though non cryptographic) hash
   digest/checksum. This operation occurs on EVERY read of data from disk.

   The theory here is that even though Hangar makes every effort to guarantee
   safe operations itself, in the real world we have to deal with systems which
   break. We’ve planned for cases where some OS induced disk corruption occurs,
   or where some malicious actor modifies the file contents manually; we can’t
   stop that from happening, but Hangar can make sure that you will know about
   it when it happens!

----

3. Before we got smart with the HDF5 backend low level details, it was an issue
   for us to have a write-enabled checkout attempt to write an array to disk
   and immediately read it back in. I’ll gloss over the details for the sake of
   simplicity here, but basically I was presented with an CRC32 Checksum
   Verification Failed error in some edge cases. The interesting bit was that
   if I closed the checkout, and reopened it, it data was secure and intact on
   disk, but for immediate reads after writes, we weren’t propagating changes
   to the HDF5 chunk metadata cache to ``rw`` operations appropriately.

   This was fixed very early on by taking advantage of a new feature in HDF5
   1.10.4 referred to as Single Writer Multiple Reader (SWMR). The long and
   short is that by being careful to handle the order in which a new HDF5 file
   is created on disk and opened in w and r mode with SWMR enabled, the HDF5
   core guarantees the integrity of the metadata chunk cache at all times. Even
   if a fatal system crash occurs in the middle of a write, the data will be
   preserved. This solved this issue completely for us

   There are many many many more details which I could cover here, but the long
   and short of it is that in order to ensure data integrity, Hangar is
   designed to not let the user do anything they aren’t allowed to at any time

      -  Read checkouts have no ability to modify contents on disk via any
         method. It’s not possible for them to actually delete or overwrite
         anything in any way.
      -  Write checkouts can only ever write data. The only way to remove the
         actual contents of written data from disk is if changes have been made
         in the staging area (but not committed) and the
         ``reset_staging_area()`` method is called. And even this has no
         ability to remove any data which had previously existed in some commit
         in the repo’s history

   In addition, a Hangar checkout object is not what it appears to be (at first
   glance, use, or even during common introspection operations). If you try to
   operate on it after closing the checkout, or holding it while another
   checkout is started, you won’t be able to (there’s a whole lot of invisible
   “magic” going on with ``weakrefs``, ``objectproxies``, and instance
   attributes).  I would encourage you to do the following:

   .. code:: pycon

      >>> co = repo.checkout(write=True)
      >>> co.metadata['hello'] = 'world'
      >>> # try to hold a reference to the metadata object:
      >>> mRef = co.metadata
      >>> mRef['hello']
      'world'
      >>> co.commit('first commit')
      >>> co.close()
      >>> # what happens when you try to access the `co` or `mRef` object?
      >>> mRef['hello']
      ReferenceError: weakly-referenced object no longer exists
      >>> print(co)  # or any other operation
      PermissionError: Unable to operate on past checkout objects which have been closed. No operation occurred. Please use a new checkout.

   The last bit I’ll leave you with is a note on context managers and performance
   (how we handle record data safety and effectively

   .. seealso::

      - :ref:`ref-tutorial` (Part 1, In section: "performance")
      - :ref:`ref-hangar-under-the-hood`


How Can a Hangar Repository be Backed Up?
=========================================

Two strategies exist:

1. Use a remote server and Hangar’s built in ability to just push data to a
   remote! (tutorial coming soon, see :ref:`ref-api` for more details.

2. A Hangar repository is self contained in it’s .hangar directory. To back
   up the data, just copy/paste or rsync it to another machine! (edited)


On Determining ``Column`` Schema Sizes
=======================================

   Say I have a data group that specifies a data array with one dimension,
   three elements (say height, width, num channels) and later on I want to add
   bit depth. Can I do that, or do I need to make a new data group? Should it
   have been three scalar data groups from the start?

So right now it’s not possible to change the schema (shape, dtype) of a
column. I’ve thought about such a feature for a while now, and while it will
require a new user facing API option, its (almost) trivial to make it work in
the core. It just hasn’t seemed like a priority yet...

And no, I wouldn’t specify each of those as scalar data groups, they are a
related piece of information, and generally would want to be accessed together

Access patterns should generally dictate how much info is placed in a column


Is there a performance/space penalty for having lots of small data groups?
--------------------------------------------------------------------------

As far as a performance / space penalty, this is where it gets good :)

- Using fewer columns means that there are fewer records (the internal
  locating info, kind-of like a git tree) to store, since each record points to
  a sample containing more information.

- Using more columns means that the likelihood of samples having the same
  value increases, meaning fewer pieces of data are actually stored on disk
  (remember it’s a content addressable file store)

However, since the size of a record (40 bytes or so before compression, and we
generally see compression ratios around 15-30% of the original size once the
records are committed) is generally negligible compared to the size of data on
disk, optimizing for number of records is just way overkill. For this case, it
really doesn’t matter. **Optimize for ease of use**
