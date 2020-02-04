from typing import Iterable, Union, Tuple
import lmdb


class CursorRangeIterator:

    @staticmethod
    def cursor_range_iterator(datatxn: lmdb.Transaction, startRangeKey: bytes, keys: bool, values: bool
                              ) -> Iterable[Union[Tuple[bytes], Tuple[bytes, bytes]]]:
        """Common method used to implement cursor range iterators

        Parameters
        ----------
        datatxn : lmdb.Transaction
            open database transaction to read values from
        startRangeKey : bytes
            range in which to iterate cursor over until end of db or out of
            lexicographic range.
        keys : bool, optional
            If True, yield metadata keys encountered, if False only values
            are returned. By default, True.
        values : bool, optional
            If True, yield metadata hash values encountered, if False only
            keys are returned. By default, True.

        Yields
        ------
        Iterable[Union[Tuple[bytes], Tuple[bytes, bytes]]]:
            db keys or key/value tuple
        """
        len_RangeKey = len(startRangeKey)
        with datatxn.cursor() as cursor:
            rangeItemsExist = cursor.set_range(startRangeKey)
            if not rangeItemsExist:
                # break out prematurely in the case where no matching items exist.
                # Important to not disrupt callers who may expect to recieves some
                # iterable for processing.
                return iter([])

            # divide loop into returned type sections as perf optimization
            # (rather then if/else checking on every iteration of loop)
            if keys and not values:
                while rangeItemsExist:
                    recKey = cursor.key()
                    if recKey[:len_RangeKey] == startRangeKey:
                        yield recKey
                        rangeItemsExist = cursor.next()
                        continue
                    else:
                        rangeItemsExist = False
            elif values and not keys:
                while rangeItemsExist:
                    recKey, recVal = cursor.item()
                    if recKey[:len_RangeKey] == startRangeKey:
                        yield recVal
                        rangeItemsExist = cursor.next()
                        continue
                    else:
                        rangeItemsExist = False
            elif keys and values:
                while rangeItemsExist:
                    recKey, recVal = cursor.item()
                    if recKey[:len_RangeKey] == startRangeKey:
                        yield (recKey, recVal)
                        rangeItemsExist = cursor.next()
                        continue
                    else:
                        rangeItemsExist = False
            else:  # pragma: no cover
                raise RuntimeError(f'Internal hangar error while iterating cursor records for '
                                   f' {startRangeKey}. one of [`keys`, `values`] must be True.')
