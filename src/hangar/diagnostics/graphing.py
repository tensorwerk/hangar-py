# -*- coding: utf-8 -*-
"""
Portions of this code have been taken and modified from the "asciidag" project.

URL:      https://github.com/sambrightman/asciidag/
File:     asciidag/graph.py
Commit:   7c1eefe3895630dc3906bbe9d553e0169202756a
Accessed: 25 MAR 2019

asciidag License
-------------------------------------------------------------------------------
License: Mozilla Public License 2.0
URL:     https://github.com/sambrightman/asciidag/blob/7c1eefe3895630dc3906bbe9d553e0169202756a/LICENSE
"""

import sys
import time
from enum import Enum

__all__ = ('Graph',)

COLOR_NORMAL = ""
COLOR_RESET = "\033[m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"
COLOR_BOLD_RED = "\033[1;31m"
COLOR_BOLD_GREEN = "\033[1;32m"
COLOR_BOLD_YELLOW = "\033[1;33m"
COLOR_BOLD_BLUE = "\033[1;34m"
COLOR_BOLD_MAGENTA = "\033[1;35m"
COLOR_BOLD_CYAN = "\033[1;36m"
COLOR_BG_RED = "\033[41m"
COLOR_BG_GREEN = "\033[42m"
COLOR_BG_YELLOW = "\033[43m"
COLOR_BG_BLUE = "\033[44m"
COLOR_BG_MAGENTA = "\033[45m"
COLOR_BG_CYAN = "\033[46m"

COLUMN_COLORS_ANSI = [
    COLOR_BOLD_RED,
    COLOR_BOLD_GREEN,
    COLOR_BOLD_YELLOW,
    COLOR_BOLD_BLUE,
    COLOR_BOLD_MAGENTA,
    COLOR_BOLD_CYAN,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_BLUE,
    COLOR_MAGENTA,
    COLOR_CYAN,
    COLOR_RESET,
]


class Column(object):  # pylint: disable=too-few-public-methods
    """A single column of output.

    Attributes:
        commit -- The parent commit of this column.
        color  -- The color to (optionally) print this column in.
                  This is an index into column_colors.

    """

    def __init__(self, commit, color):
        self.commit = commit
        self.color = color


class GraphState(Enum):  # pylint: disable=too-few-public-methods
    PADDING = 0
    SKIP = 1
    PRE_COMMIT = 2
    COMMIT = 3
    POST_MERGE = 4
    COLLAPSING = 5


class Graph(object):  # pragma: no cover
    """
    The commit currently being processed
        struct commit *commit

    The number of interesting parents that this commit has. Note that this is
    not the same as the actual number of parents. This count excludes parents
    that won't be printed in the graph output, as determined by
    is_interesting().
        int num_parents

    The width of the graph output for this commit. All rows for this commit are
    padded to this width, so that messages printed after the graph output are
    aligned.
        int width

    The next expansion row to print when state is GraphState.PRE_COMMIT
        int expansion_row

    The current output state. This tells us what kind of line next_line()
    should output.
        enum graph_state state

    The output state for the previous line of output. This is primarily used to
    determine how the first merge line should appear, based on the last line of
    the previous commit.
        enum graph_state prev_state

    The index of the column that refers to this commit. If none of the incoming
    columns refer to this commit, this will be equal to num_columns.
        int commit_index

    The commit_index for the previously displayed commit. This is used to
    determine how the first line of a merge graph output should appear, based
    on the last line of the previous commit.
        int prev_commit_index

    The maximum number of columns that can be stored in the columns and
    new_columns arrays. This is also half the number of entries that can be
    stored in the mapping and new_mapping arrays.
        int column_capacity

    The number of columns (also called "branch lines" in some places)
        int num_columns

    The number of columns in the new_columns array
        int num_new_columns

    The number of entries in the mapping array
        int mapping_size

    The column state before we output the current commit.
        struct column *columns

    The new column state after we output the current commit. Only valid when
    state is GraphState.COLLAPSING.
        struct column *new_columns

    An array that tracks the current state of each character in the output line
    during state GraphState.COLLAPSING. Each entry is -1 if this character is
    empty, or a non-negative integer if the character contains a branch line.
    The value of the integer indicates the target position for this branch
    line. (I.e., this array maps the current column positions to their desired
    positions.)

    The maximum capacity of this array is always sizeof(int) * 2 *
    column_capacity.
        int *mapping

    A temporary array for computing the next mapping state while we are
    outputting a mapping line. This is stored as part of the git_graph simply
    so we don't have to allocate a new temporary array each time we have to
    output a collapsing line.
        int *new_mapping

    The current default column color being used. This is stored as an index
    into the array column_colors.
        unsigned short default_column_color
    """
    def __init__(self,
                 fh=None,
                 first_parent_only=False,
                 use_color=True,
                 column_colors=None):
        """State machine for processing DAG nodes into ASCII graphs.

        show_nodes() deals with sorting the nodes from tips down into
        topological order. It then displays them line-by-line.

        """
        self.commit = None
        self.buf = ''

        if fh is None:
            self.outfile = sys.stdout
        else:
            self.outfile = fh
        self.first_parent_only = first_parent_only
        self.use_color = use_color
        if column_colors is None:
            self.column_colors = COLUMN_COLORS_ANSI
        else:
            self.column_colors = column_colors

        self.num_parents = 0
        self.width = 0
        self.expansion_row = 0
        self.state = GraphState.PADDING
        self.prev_state = GraphState.PADDING
        self.commit_index = 0
        self.prev_commit_index = 0
        self.num_columns = 0
        self.num_new_columns = 0
        self.mapping_size = 0
        # Start the column color at the maximum value, since we'll always
        # increment it for the first commit we output. This way we start at 0
        # for the first commit.
        self.default_column_color = len(self.column_colors) - 1

        self.columns = {}
        self.new_columns = {}
        self.mapping = {}
        self.new_mapping = {}

    def show_nodes(self, dag, spec, branch, start, order, stop='',
                   *, show_time=True, show_user=True):
        """Printing function that displays a DAG representing the commit history

        Print a revision history alongside a revision graph drawn with ASCII
        characters. Nodes printed as an * character are parents of the working
        directory. Any unreachable (but referenced nodes) are displayed at +

        Parameters
        ----------
        dag : dict
            directed acyclic graph of nodes and connections in commits. No more than
            2 connections per node
        spec: dict
            dictionary of commit specification (user name, email, message, etc).
        branch : dict
            dict of commit hash -> list of branch names whose HEAD commit is at
            that key.
        start : string
            commit hash to act as the top of the topological sort.
        order: list
            time based ordering of commit hashs
        stop : str, optional
            commit hash to stop generating the graph at if the DAG contains more
            history than is needed (the default is '', which is the "parent" of
            the initial repository commit.)
        """
        if start == stop:
            return

        fmtSpec = {}
        for cmt, cmtspec in spec.items():
            if show_time:
                t = f"({time.strftime('%d%b%Y %H:%M:%S', time.gmtime(cmtspec['commit_time']))})"
            else:
                t = ''
            if show_user:
                u = f"({cmtspec['commit_user']})"
            else:
                u = ''
            m = cmtspec['commit_message']
            br = ' '
            if cmt in branch:
                for branchName in branch[cmt]:
                    if self.use_color is True:
                        br = f'{br}({COLOR_BOLD_RED}{branchName}{COLOR_RESET}) '
                    else:
                        br = f'{br}({branchName}) '
            fmtSpec[cmt] = f'{cmt}{br}{t}{u}: {m}'

        for rev in order:
            parents = dag[rev]
            self._update(rev, parents)
            self._show_commit()
            self.outfile.write(fmtSpec[rev])
            if not self._is_commit_finished():
                self.outfile.write('\n')
                self._show_remainder()
            self.outfile.write('\n')

    def _write_column(self, col, col_char):
        if col.color is not None:
            self.buf += self.column_colors[col.color]
        self.buf += col_char
        if col.color is not None:
            self.buf += self.column_colors[-1]

    def _update_state(self, state):
        self.prev_state = self.state
        self.state = state

    def _interesting_parents(self):
        for parent in self.commit_parents:
            yield parent
            if self.first_parent_only:
                break

    def _get_current_column_color(self):
        if not self.use_color:
            return None
        return self.default_column_color

    def _increment_column_color(self):
        self.default_column_color = ((self.default_column_color + 1)
                                     % len(self.column_colors))

    def _find_commit_color(self, commit):
        for i in range(self.num_columns):
            if self.columns[i].commit == commit:
                return self.columns[i].color
        return self._get_current_column_color()

    def _insert_into_new_columns(self, commit, mapping_index):
        """
        If the commit is already in the new_columns list, we don't need to add
        it. Just update the mapping correctly.
        """
        for i in range(self.num_new_columns):
            if self.new_columns[i].commit == commit:
                self.mapping[mapping_index] = i
                return mapping_index + 2

        # This commit isn't already in new_columns. Add it.
        column = Column(commit, self._find_commit_color(commit))
        self.new_columns[self.num_new_columns] = column
        self.mapping[mapping_index] = self.num_new_columns
        self.num_new_columns += 1
        return mapping_index + 2

    def _update_width(self, is_commit_in_existing_columns):
        """
        Compute the width needed to display the graph for this commit. This is
        the maximum width needed for any row. All other rows will be padded to
        this width.

        Compute the number of columns in the widest row: Count each existing
        column (self.num_columns), and each new column added by this commit.
        """
        max_cols = self.num_columns + self.num_parents

        # Even if the current commit has no parents to be printed, it still
        # takes up a column for itself.
        if self.num_parents < 1:
            max_cols += 1

        # We added a column for the current commit as part of self.num_parents.
        # If the current commit was already in self.columns, then we have double
        # counted it.
        if is_commit_in_existing_columns:
            max_cols -= 1

        # Each column takes up 2 spaces
        self.width = max_cols * 2

    def _update_columns(self):
        """
        Swap self.columns with self.new_columns self.columns contains the state
        for the previous commit, and new_columns now contains the state for our
        commit.

        We'll re-use the old columns array as storage to compute the new columns
        list for the commit after this one.
        """
        self.columns, self.new_columns = self.new_columns, self.columns
        self.num_columns = self.num_new_columns
        self.num_new_columns = 0

        # Now update new_columns and mapping with the information for the commit
        # after this one.
        #
        # First, make sure we have enough room. At most, there will be
        # self.num_columns + self.num_parents columns for the next commit.
        max_new_columns = self.num_columns + self.num_parents

        # Clear out self.mapping
        self.mapping_size = 2 * max_new_columns
        for i in range(self.mapping_size):
            self.mapping[i] = -1

        # Populate self.new_columns and self.mapping
        #
        # Some of the parents of this commit may already be in self.columns. If
        # so, self.new_columns should only contain a single entry for each such
        # commit. self.mapping should contain information about where each
        # current branch line is supposed to end up after the collapsing is
        # performed.
        seen_this = False
        mapping_idx = 0
        is_commit_in_columns = True
        for i in range(self.num_columns + 1):
            if i == self.num_columns:
                if seen_this:
                    break
                is_commit_in_columns = False
                col_commit = self.commit
            else:
                col_commit = self.columns[i].commit

            if col_commit == self.commit:
                old_mapping_idx = mapping_idx
                seen_this = True
                self.commit_index = i
                for parent in self._interesting_parents():
                    # If this is a merge, or the start of a new childless
                    # column, increment the current color.
                    if self.num_parents > 1 or not is_commit_in_columns:
                        self._increment_column_color()
                    mapping_idx = self._insert_into_new_columns(
                        parent,
                        mapping_idx)
                # We always need to increment mapping_idx by at least 2, even if
                # it has no interesting parents. The current commit always takes
                # up at least 2 spaces.
                if mapping_idx == old_mapping_idx:
                    mapping_idx += 2
            else:
                mapping_idx = self._insert_into_new_columns(col_commit,
                                                            mapping_idx)

        # Shrink mapping_size to be the minimum necessary
        while (self.mapping_size > 1 and
               self.mapping[self.mapping_size - 1] < 0):
            self.mapping_size -= 1

        # Compute self.width for this commit
        self._update_width(is_commit_in_columns)

    def _update(self, commit, parents):
        self.commit = commit
        self.commit_parents = parents
        self.num_parents = len(list(self._interesting_parents()))

        # Store the old commit_index in prev_commit_index.
        # update_columns() will update self.commit_index for this commit.
        self.prev_commit_index = self.commit_index

        # Call update_columns() to update
        # columns, new_columns, and mapping.
        self._update_columns()
        self.expansion_row = 0

        # Update self.state.
        # Note that we don't call update_state() here, since we don't want to
        # update self.prev_state. No line for self.state was ever printed.
        #
        # If the previous commit didn't get to the GraphState.PADDING state, it
        # never finished its output. Goto GraphState.SKIP, to print out a line
        # to indicate that portion of the graph is missing.
        #
        # If there are 3 or more parents, we may need to print extra rows before
        # the commit, to expand the branch lines around it and make room for it.
        # We need to do this only if there is a branch row (or more) to the
        # right of this commit.
        #
        # If less than 3 parents, we can immediately print the commit line.
        if self.state != GraphState.PADDING:
            self.state = GraphState.SKIP
        elif (self.num_parents >= 3 and
              self.commit_index < (self.num_columns - 1)):
            self.state = GraphState.PRE_COMMIT  # noqa: E501 pylint: disable=redefined-variable-type
        else:
            self.state = GraphState.COMMIT

    def _is_mapping_correct(self):
        """
        The mapping is up to date if each entry is at its target, or is 1
        greater than its target. (If it is 1 greater than the target, '/' will
        be printed, so it will look correct on the next row.)
        """
        for i in range(self.mapping_size):
            target = self.mapping[i]
            if target < 0:
                continue
            if target == i // 2:
                continue
            return False
        return True

    def _pad_horizontally(self, chars_written):
        """Add spaces to string end so all lines of a commit have the same width.

        This way, fields printed to the right of the graph will remain aligned
        for the entire commit.
        """
        if chars_written >= self.width:
            return

        extra = self.width - chars_written
        self.buf += ' ' * extra

    def _output_padding_line(self):
        """Output a padding row, that leaves all branch lines unchanged
        """
        for i in range(self.num_new_columns):
            self._write_column(self.new_columns[i], '|')
            self.buf += ' '

        self._pad_horizontally(self.num_new_columns * 2)

    def _output_skip_line(self):
        """Output an ellipsis to indicate that a portion of the graph is missing.
        """
        self.buf += '...'
        self._pad_horizontally(3)

        if self.num_parents >= 3 and self.commit_index < self.num_columns - 1:
            self._update_state(GraphState.PRE_COMMIT)
        else:
            self._update_state(GraphState.COMMIT)

    def _output_pre_commit_line(self):
        """Formats a row with increased space around a commit with multiple parents.

        This is done in order to make room for the commit. It should only be
        called when there are 3 or more parents. We need 2 extra rows for every
        parent over 2.
        """
        assert self.num_parents >= 3, 'not enough parents to add expansion row'
        num_expansion_rows = (self.num_parents - 2) * 2

        # self.expansion_row tracks the current expansion row we are on.
        # It should be in the range [0, num_expansion_rows - 1]
        assert (0 <= self.expansion_row < num_expansion_rows), \
            'wrong number of expansion rows'

        # Output the row
        seen_this = False
        chars_written = 0
        for i in range(self.num_columns):
            col = self.columns[i]
            if col.commit == self.commit:
                seen_this = True
                self._write_column(col, '|')
                self.buf += ' ' * self.expansion_row
                chars_written += 1 + self.expansion_row
            elif seen_this and (self.expansion_row == 0):
                # This is the first line of the pre-commit output. If the
                # previous commit was a merge commit and ended in the
                # GraphState.POST_MERGE state, all branch lines after
                # self.prev_commit_index were printed as "\" on the previous
                # line. Continue to print them as "\" on this line. Otherwise,
                # print the branch lines as "|".
                if (self.prev_state == GraphState.POST_MERGE and
                        self.prev_commit_index < i):
                    self._write_column(col, '\\')
                else:
                    self._write_column(col, '|')
                chars_written += 1
            elif seen_this and (self.expansion_row > 0):
                self._write_column(col, '\\')
                chars_written += 1
            else:
                self._write_column(col, '|')
                chars_written += 1
            self.buf += ' '
            chars_written += 1

        self._pad_horizontally(chars_written)

        # Increment self.expansion_row, and move to state GraphState.COMMIT if
        # necessary
        self.expansion_row += 1
        if self.expansion_row >= num_expansion_rows:
            self._update_state(GraphState.COMMIT)

    # Draw an octopus merge and return the number of characters written.
    def _draw_octopus_merge(self):
        """
        Here dashless_commits represents the number of parents which don't
        need to have dashes (because their edges fit neatly under the commit).
        """
        dashless_commits = 2
        num_dashes = ((self.num_parents - dashless_commits) * 2) - 1
        for i in range(num_dashes):
            col_num = i // 2 + dashless_commits + self.commit_index
            self._write_column(self.new_columns[col_num], '-')
        col_num = num_dashes // 2 + dashless_commits + self.commit_index
        self._write_column(self.new_columns[col_num], '.')
        return num_dashes + 1

    def _output_commit_line(self):  # noqa: C901, E501 pylint: disable=too-many-branches
        """
        Output the row containing this commit Iterate up to and including
        self.num_columns, since the current commit may not be in any of the
        existing columns. (This happens when the current commit doesn't have
        any children that we have already processed.)
        """
        seen_this = False
        chars_written = 0
        for i in range(self.num_columns + 1):
            if i == self.num_columns:
                if seen_this:
                    break
                col_commit = self.commit
            else:
                col = self.columns[i]
                col_commit = self.columns[i].commit

            if col_commit == self.commit:
                seen_this = True
                self.buf += '*'
                chars_written += 1

                if self.num_parents > 2:
                    chars_written += self._draw_octopus_merge()
            elif seen_this and self.num_parents > 2:
                self._write_column(col, '\\')
                chars_written += 1
            elif seen_this and self.num_parents == 2:
                # This is a 2-way merge commit. There is no
                # GraphState.PRE_COMMIT stage for 2-way merges, so this is the
                # first line of output for this commit. Check to see what the
                # previous line of output was.
                #
                # If it was GraphState.POST_MERGE, the branch line coming into
                # this commit may have been '\', and not '|' or '/'. If so,
                # output the branch line as '\' on this line, instead of '|'.
                # This makes the output look nicer.
                if (self.prev_state == GraphState.POST_MERGE and
                        self.prev_commit_index < i):
                    self._write_column(col, '\\')
                else:
                    self._write_column(col, '|')
                chars_written += 1
            else:
                self._write_column(col, '|')
                chars_written += 1
            self.buf += ' '
            chars_written += 1

        self._pad_horizontally(chars_written)
        if self.num_parents > 1:
            self._update_state(GraphState.POST_MERGE)
        elif self._is_mapping_correct():
            self._update_state(GraphState.PADDING)
        else:
            self._update_state(GraphState.COLLAPSING)

    def _find_new_column_by_commit(self, commit):
        for i in range(self.num_new_columns):
            if self.new_columns[i].commit == commit:
                return self.new_columns[i]
        return None

    def _output_post_merge_line(self):
        seen_this = False
        chars_written = 0
        for i in range(self.num_columns + 1):
            if i == self.num_columns:
                if seen_this:
                    break
                col_commit = self.commit
            else:
                col = self.columns[i]
                col_commit = col.commit

            if col_commit == self.commit:
                # Since the current commit is a merge find the columns for the
                # parent commits in new_columns and use those to format the
                # edges.
                seen_this = True
                parents = self._interesting_parents()
                assert parents, 'merge has no parents'
                par_column = self._find_new_column_by_commit(next(parents))
                assert par_column, 'parent column not found'
                self._write_column(par_column, '|')
                chars_written += 1
                for parent in parents:
                    assert parent, 'parent is not valid'
                    par_column = self._find_new_column_by_commit(parent)
                    assert par_column, 'parent column not found'
                    self._write_column(par_column, '\\')
                    self.buf += ' '
                chars_written += (self.num_parents - 1) * 2
            elif seen_this:
                self._write_column(col, '\\')
                self.buf += ' '
                chars_written += 2
            else:
                self._write_column(col, '|')
                self.buf += ' '
                chars_written += 2

        self._pad_horizontally(chars_written)

        if self._is_mapping_correct():
            self._update_state(GraphState.PADDING)
        else:
            self._update_state(GraphState.COLLAPSING)

    def _output_collapsing_line(self):  # noqa: C901, E501 pylint: disable=too-many-branches
        used_horizontal = False
        horizontal_edge = -1
        horizontal_edge_target = -1

        # Clear out the new_mapping array
        for i in range(self.mapping_size):
            self.new_mapping[i] = -1

        for i in range(self.mapping_size):
            target = self.mapping[i]
            if target < 0:
                continue

            # Since update_columns() always inserts the leftmost column first,
            # each branch's target location should always be either its current
            # location or to the left of its current location.
            #
            # We never have to move branches to the right. This makes the graph
            # much more legible, since whenever branches cross, only one is
            # moving directions.
            assert target * 2 <= i, \
                'position {} targetting column {}'.format(i, target * 2)

            if target * 2 == i:
                # This column is already in the correct place
                assert self.new_mapping[i] == -1
                self.new_mapping[i] = target
            elif self.new_mapping[i - 1] < 0:
                # Nothing is to the left. Move to the left by one.
                self.new_mapping[i - 1] = target
                # If there isn't already an edge moving horizontally select this one.
                if horizontal_edge == -1:
                    horizontal_edge = i
                    horizontal_edge_target = target
                    # The variable target is the index of the graph column, and
                    # therefore target * 2 + 3 is the actual screen column of
                    # the first horizontal line.
                    for j in range((target * 2) + 3, i - 2, 2):
                        self.new_mapping[j] = target
            elif self.new_mapping[i - 1] == target:
                # There is a branch line to our left already, and it is our
                # target. We combine with this line, since we share the same
                # parent commit.
                #
                # We don't have to add anything to the output or new_mapping,
                # since the existing branch line has already taken care of it.
                pass
            else:
                # There is a branch line to our left, but it isn't our target.
                # We need to cross over it.
                #
                # The space just to the left of this branch should always be empty.
                #
                # The branch to the left of that space should be our eventual target.
                assert self.new_mapping[i - 1] > target
                assert self.new_mapping[i - 2] < 0
                assert self.new_mapping[i - 3] == target
                self.new_mapping[i - 2] = target
                # Mark this branch as the horizontal edge to prevent any other
                # edges from moving horizontally.
                if horizontal_edge == -1:
                    horizontal_edge = i

        # The new mapping may be 1 smaller than the old mapping
        if self.new_mapping[self.mapping_size - 1] < 0:
            self.mapping_size -= 1

        # Output a line based on the new mapping info
        for i in range(self.mapping_size):
            target = self.new_mapping[i]
            if target < 0:
                self.buf += ' '
            elif target * 2 == i:
                self._write_column(self.new_columns[target], '|')
            elif target == horizontal_edge_target and i != horizontal_edge - 1:
                # Set the mappings for all but the first segment to -1 so that
                # they won't continue into the next line.
                if i != (target * 2) + 3:
                    self.new_mapping[i] = -1
                used_horizontal = True
                self._write_column(self.new_columns[target], '_')
            else:
                if used_horizontal and i < horizontal_edge:
                    self.new_mapping[i] = -1
                self._write_column(self.new_columns[target], '/')

        self._pad_horizontally(self.mapping_size)
        self.mapping, self.new_mapping = self.new_mapping, self.mapping

        # If self.mapping indicates that all of the branch lines are already in
        # the correct positions, we are done. Otherwise, we need to collapse
        # some branch lines together.
        if self._is_mapping_correct():
            self._update_state(GraphState.PADDING)

    def _next_line(self):  # pylint: disable=too-many-return-statements
        if self.state == GraphState.PADDING:
            self._output_padding_line()
            return False
        elif self.state == GraphState.SKIP:
            self._output_skip_line()
            return False
        elif self.state == GraphState.PRE_COMMIT:
            self._output_pre_commit_line()
            return False
        elif self.state == GraphState.COMMIT:
            self._output_commit_line()
            return True
        elif self.state == GraphState.POST_MERGE:
            self._output_post_merge_line()
            return False
        elif self.state == GraphState.COLLAPSING:
            self._output_collapsing_line()
            return False
        else:
            return False

    def _padding_line(self):
        """Output a padding line in the graph.

        This is similar to next_line(). However, it is guaranteed to never print
        the current commit line. Instead, if the commit line is next, it will
        simply output a line of vertical padding, extending the branch lines
        downwards, but leaving them otherwise unchanged.
        """
        if self.state != GraphState.COMMIT:
            self._next_line()
            return

        # Output the row containing this commit
        # Iterate up to and including self.num_columns, since the current commit
        # may not be in any of the existing columns. (This happens when the
        # current commit doesn't have any children that we have already
        # processed.)
        for i in range(self.num_columns):
            col = self.columns[i]
            self._write_column(col, '|')
            if col.commit == self.commit and self.num_parents > 2:
                self.buf += ' ' * (self.num_parents - 2) * 2
            else:
                self.buf += ' '

        self._pad_horizontally(self.num_columns)

        # Update self.prev_state since we have output a padding line
        self.prev_state = GraphState.PADDING

    def _is_commit_finished(self):
        return self.state == GraphState.PADDING

    def _show_commit(self):
        shown_commit_line = False

        # When showing a diff of a merge against each of its parents, we are
        # called once for each parent without update having been called. In this
        # case, simply output a single padding line.
        if self._is_commit_finished():
            self._show_padding()
            shown_commit_line = True

        while not shown_commit_line and not self._is_commit_finished():
            shown_commit_line = self._next_line()
            self.outfile.write(self.buf)
            if not shown_commit_line:
                self.outfile.write('\n')
            self.buf = ''

    def _show_padding(self):
        self._padding_line()
        self.outfile.write(self.buf)
        self.buf = ''

    def _show_remainder(self):
        shown = False

        if self._is_commit_finished():
            return False

        while True:
            self._next_line()
            self.outfile.write(self.buf)
            self.buf = ''
            shown = True

            if not self._is_commit_finished():
                self.outfile.write('\n')
            else:
                break

        return shown
