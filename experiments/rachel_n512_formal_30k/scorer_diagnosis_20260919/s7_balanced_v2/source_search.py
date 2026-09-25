"""Bounded rejection sampling; never relax geometry or source strata.

At a fixed non-Partial slot, scale and the pre-weather length gate are
deterministic for a source Pair. Retrying a source rejected by that gate cannot
help. Remove only such proven-infeasible IDs from the sampling pool. All other
failures still consume the original damage-attempt budget.
"""


class SourceSearch:
    def __init__(self, original, alternatives, rng, budget, enabled):
        if budget <= 0 or not alternatives:
            raise ValueError('positive attempt budget and nonempty source pool required')
        if any(e['source_stratum'] != original['source_stratum'] for e in alternatives):
            raise ValueError('source search cannot change structural stratum')
        self.original, self.pool, self.rng = original, list(alternatives), rng
        self.budget, self.enabled = budget, enabled
        self.draws = self.damage_attempts = 0
        self.excluded = set()

    def draw(self):
        if self.damage_attempts >= self.budget or not self.pool:
            return None
        index = self.draws
        source = (self.original if index < 4 and self.original['pair_id'] not in self.excluded
                  else self.pool[int(self.rng.integers(len(self.pool)))])
        self.draws += 1
        return index, source

    def rejected(self, source, static_length=False):
        if self.enabled and static_length:
            pid = source['pair_id']
            if pid in self.excluded:
                raise AssertionError('a proven-infeasible source was retried')
            self.excluded.add(pid)
            # Retain multiplicity/weights of still-eligible native sources.
            self.pool = [e for e in self.pool if e['pair_id'] != pid]
        else:
            self.damage_attempts += 1

