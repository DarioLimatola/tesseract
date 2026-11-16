//! The attention head: match-and-count over the context.
//!
//! The last token of the context is the query. Every other token is a
//! candidate. The score of a candidate is the number of bits on which it
//! agrees with the query, counted over the head's match mask and over the
//! positions both tokens know. The best candidate is selected and the bits
//! of it chosen by the head's value mask are added to the state as votes.
//! The head retrieves and binds; it does not score a next token.
//!
//! What is learned:
//!
//! - the **match mask**: which positions take part in the comparison. It
//!   starts full and learning prunes it;
//! - the **value mask**: which positions of the selected token are copied.
//!   It starts empty, so a new head writes nothing.
//!
//! How the selection is trained (hard selection has no smooth error signal):
//! the error at the output says which candidate would have reduced it most
//! if selected. When that candidate was not the one selected, every match
//! connection on which the two candidates differ in agreement with the query
//! gets one unit of evidence, in the direction that favours the useful one.
//! This is the Boolean sensitivity of the comparison between two scores.
//! Integer counters, counting and comparing only.

use crate::counters::{Automaton, Growth, accumulate, grow, prune};
use crate::entity::{BITS, Entity, WORDS};
use crate::state::{State, Token};
use crate::structure::{Census, StructurePolicy, StructureReport, count_connections};

/// Settings of an attention head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionParams {
    /// Automaton of the match connections (no sign).
    pub matching: Automaton,
    /// Automaton of the value connections (no sign).
    pub value: Automaton,
    /// Initial evidence of the match connections, which start live.
    pub match_init: i16,
    /// A candidate is selected only with at least this score.
    pub min_score: u32,
    /// A right selection that wins by less than this still learns.
    pub select_margin: u32,
    /// The state must agree with the target by at least this much.
    pub output_margin: i8,
    /// Positions the match mask may use.
    pub match_allowed: Entity,
    /// Positions the value mask may use.
    pub value_allowed: Entity,
}

impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            matching: Automaton {
                live_at: 24,
                dormant_below: 8,
                limit: 64,
                protect_at: 65,
                signed: false,
            },
            value: Automaton {
                live_at: 12,
                dormant_below: 4,
                limit: 64,
                protect_at: 65,
                signed: false,
            },
            match_init: 24,
            min_score: 1,
            select_margin: 32,
            output_margin: 2,
            match_allowed: Entity::ONES,
            value_allowed: Entity::ONES,
        }
    }
}

/// What one learning step did to a head.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeadUpdate {
    /// Candidate that was selected, if any.
    pub selected: Option<usize>,
    /// Candidate that would have reduced the error most, if any did.
    pub useful: Option<usize>,
    /// True when the match mask received evidence.
    pub taught: bool,
    /// Connection changes (match and value masks).
    pub growth: Growth,
}

/// One attention head with a binary match mask and a binary value mask.
#[derive(Clone, Debug)]
pub struct AttentionHead {
    params: AttentionParams,
    match_mask: Entity,
    match_protected: Entity,
    value_mask: Entity,
    value_protected: Entity,
    usage: u64,
    match_counters: Vec<i16>,
    value_counters: Vec<i16>,
}

/// Bits on which query and candidate agree, where both are known.
#[inline]
fn agreement(query: &Token, candidate: &Token) -> Entity {
    let mut out = Entity::ZERO;
    for w in 0..WORDS {
        out.0[w] = !(query.bits.0[w] ^ candidate.bits.0[w]) & query.known.0[w] & candidate.known.0[w];
    }
    out
}

impl AttentionHead {
    /// A new head: it matches on every allowed position and writes nothing.
    pub fn new(params: AttentionParams) -> Self {
        let mut match_counters = vec![0i16; BITS];
        for (p, c) in match_counters.iter_mut().enumerate() {
            if params.match_allowed.get(p) {
                *c = params.match_init;
            }
        }
        Self {
            params,
            match_mask: params.match_allowed,
            match_protected: Entity::ZERO,
            value_mask: Entity::ZERO,
            value_protected: Entity::ZERO,
            usage: 0,
            match_counters,
            value_counters: vec![0; BITS],
        }
    }

    /// The match mask.
    pub fn match_mask(&self) -> &Entity {
        &self.match_mask
    }

    /// The value mask.
    pub fn value_mask(&self) -> &Entity {
        &self.value_mask
    }

    /// The counters of the match mask.
    pub fn match_counters(&self) -> &[i16] {
        &self.match_counters
    }

    /// Protected match connections.
    pub fn match_protected(&self) -> &Entity {
        &self.match_protected
    }

    /// Marks match connections as protected: they are never pruned.
    pub fn protect_match(&mut self, positions: &Entity) {
        self.match_protected = self.match_protected | (*positions & self.match_mask);
    }

    /// Times the head selected a candidate while learning.
    pub fn usage(&self) -> u64 {
        self.usage
    }

    /// Score of one candidate: agreeing bits under the match mask, where
    /// both tokens are known.
    #[inline]
    pub fn score(&self, query: &Token, candidate: &Token) -> u32 {
        let mut n = 0;
        for w in 0..WORDS {
            n += (!(query.bits.0[w] ^ candidate.bits.0[w])
                & query.known.0[w]
                & candidate.known.0[w]
                & self.match_mask.0[w])
                .count_ones();
        }
        n
    }

    /// Reference for [`AttentionHead::score`]: one bit at a time.
    pub fn score_reference(&self, query: &Token, candidate: &Token) -> u32 {
        (0..BITS)
            .filter(|&p| {
                self.match_mask.get(p)
                    && query.known.get(p)
                    && candidate.known.get(p)
                    && query.bits.get(p) == candidate.bits.get(p)
            })
            .count() as u32
    }

    /// Scores of all candidates against the query.
    pub fn scores(&self, query: &Token, candidates: &[Token], out: &mut Vec<u32>) {
        out.clear();
        out.extend(candidates.iter().map(|c| self.score(query, c)));
    }

    /// The selected candidate: the best score, if it reaches `min_score`.
    /// Ties go to the earliest candidate.
    pub fn select(&self, scores: &[u32]) -> Option<usize> {
        let mut best: Option<(u32, usize)> = None;
        for (j, &s) in scores.iter().enumerate() {
            if s >= self.params.min_score && best.is_none_or(|(bs, _)| s > bs) {
                best = Some((s, j));
            }
        }
        best.map(|(_, j)| j)
    }

    /// Votes a candidate would write: `(plus, minus)`.
    #[inline]
    fn votes(&self, candidate: &Token) -> (Entity, Entity) {
        let writable = candidate.known & self.value_mask;
        (candidate.bits & writable, writable.and_not(&candidate.bits))
    }

    /// Forward pass. The last token of `context` is the query; the state
    /// starts from it and receives the votes of the selected candidate.
    /// An empty context gives an empty state.
    pub fn forward(&self, context: &[Token]) -> State {
        let Some((query, candidates)) = context.split_last() else {
            return State::zero();
        };
        let mut state = State::from_token(query, 1);
        let mut scores = Vec::with_capacity(candidates.len());
        self.scores(query, candidates, &mut scores);
        if let Some(j) = self.select(&scores) {
            let (plus, minus) = self.votes(&candidates[j]);
            state.add_votes(&plus, &minus);
        }
        state
    }

    /// Error-driven learning from one example: the context (query last)
    /// and the wanted bits of the output state.
    pub fn learn(&mut self, context: &[Token], target: &Entity, target_mask: &Entity) -> HeadUpdate {
        let mut update = HeadUpdate::default();
        let Some((query, candidates)) = context.split_last() else {
            return update;
        };
        let mut scores = Vec::with_capacity(candidates.len());
        self.scores(query, candidates, &mut scores);
        let selected = self.select(&scores);
        update.selected = selected;
        let mut state = State::from_token(query, 1);
        if let Some(j) = selected {
            self.usage += 1;
            let (plus, minus) = self.votes(&candidates[j]);
            state.add_votes(&plus, &minus);
        }
        let (up, down) = state.error(target, target_mask, self.params.output_margin);
        if (up | down).is_zero() {
            return update;
        }

        // Usefulness of each candidate: votes with the wanted direction
        // minus votes against it, through the current value mask.
        let useful: Vec<i32> = candidates
            .iter()
            .map(|c| {
                let (plus, minus) = self.votes(c);
                (plus & up).count_ones() as i32 + (minus & down).count_ones() as i32
                    - (plus & down).count_ones() as i32
                    - (minus & up).count_ones() as i32
            })
            .collect();
        let best_use = useful.iter().copied().max().unwrap_or(0);
        // Among the most useful candidates, the one closest to being selected.
        let teacher = (0..candidates.len())
            .filter(|&j| useful[j] == best_use && best_use > 0)
            .max_by_key(|&j| (scores[j], std::cmp::Reverse(j)));
        update.useful = teacher;

        // Match mask: move the comparison between two candidates.
        let pair = match (selected, teacher) {
            (Some(s), Some(t)) if useful[s] < best_use => Some((Some(s), t)),
            (None, Some(t)) => Some((None, t)),
            (Some(s), Some(_)) => {
                // Right selection: learn only if a worse candidate is close.
                let rival = (0..candidates.len())
                    .filter(|&j| useful[j] < useful[s])
                    .max_by_key(|&j| (scores[j], std::cmp::Reverse(j)));
                match rival {
                    Some(r) if scores[s] < scores[r] + self.params.select_margin => {
                        Some((Some(r), s))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((loser, winner)) = pair {
            let allowed = self.params.match_allowed;
            let win = agreement(query, &candidates[winner]) & allowed;
            let lose = loser.map_or(Entity::ZERO, |l| agreement(query, &candidates[l]) & allowed);
            let plus = win.and_not(&lose);
            let minus = lose.and_not(&win);
            accumulate(&mut self.match_counters, &plus, &minus, self.params.matching.limit);
            let mut sign = Entity::ONES;
            update.growth.add(grow(
                &self.match_counters,
                &mut sign,
                &mut self.match_mask,
                &mut self.match_protected,
                &plus,
                &self.params.matching,
            ));
            update.taught = true;
        }

        // Value mask: would copying this position of the selected candidate
        // have moved the state the wanted way?
        if let Some(j) = selected {
            let c = &candidates[j];
            let writable = c.known & self.params.value_allowed;
            let ones = c.bits & writable;
            let zeros = writable.and_not(&c.bits);
            let plus = (ones & up) | (zeros & down);
            let minus = (ones & down) | (zeros & up);
            accumulate(&mut self.value_counters, &plus, &minus, self.params.value.limit);
            let mut sign = Entity::ONES;
            update.growth.add(grow(
                &self.value_counters,
                &mut sign,
                &mut self.value_mask,
                &mut self.value_protected,
                &plus,
                &self.params.value,
            ));
        }
        update
    }

    /// One structure step: weak, unprotected connections of both masks
    /// become dormant, on a budget. Does nothing while the error is rising.
    pub fn structure_step(
        &mut self,
        policy: &StructurePolicy,
        error_rising: bool,
    ) -> StructureReport {
        let mut report = StructureReport::default();
        if error_rising {
            report.skipped = true;
            return report;
        }
        report.connections_pruned += prune(
            &self.match_counters,
            &mut self.match_mask,
            &self.match_protected,
            &self.params.matching,
            policy.connection_budget,
            policy.min_live_inputs,
        );
        report.connections_pruned += prune(
            &self.value_counters,
            &mut self.value_mask,
            &self.value_protected,
            &self.params.value,
            policy.connection_budget,
            0,
        );
        report
    }

    /// Counts of connections. The head counts as one active unit.
    pub fn census(&self) -> Census {
        let mut c = Census {
            units_active: 1,
            ..Census::default()
        };
        count_connections(
            &mut c,
            &self.match_mask,
            &self.match_protected,
            &self.match_counters,
            true,
        );
        count_connections(
            &mut c,
            &self.value_mask,
            &self.value_protected,
            &self.value_counters,
            true,
        );
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn new_head_writes_nothing() {
        let mut rng = Rng::new(41);
        let head = AttentionHead::new(AttentionParams::default());
        let context: Vec<Token> = (0..5).map(|_| Token::full(rng.entity())).collect();
        let out = head.forward(&context);
        assert_eq!(out, State::from_token(&context[4], 1));
        assert_eq!(head.forward(&[]), State::zero());
    }

    #[test]
    fn score_matches_reference() {
        let mut rng = Rng::new(42);
        let mut head = AttentionHead::new(AttentionParams::default());
        for _ in 0..30 {
            head.match_mask = rng.entity();
            let q = Token::partial(rng.entity(), rng.entity());
            let c = Token::partial(rng.entity(), rng.entity());
            assert_eq!(head.score(&q, &c), head.score_reference(&q, &c));
        }
    }

    #[test]
    fn selects_the_closest_candidate_and_copies_through_the_value_mask() {
        let mut rng = Rng::new(43);
        let mut head = AttentionHead::new(AttentionParams::default());
        head.value_mask = Entity::domain_mask(1);
        let key_domain = Entity::domain_mask(0);
        let records: Vec<Entity> = (0..6).map(|_| rng.entity()).collect();
        let mut context: Vec<Token> = records.iter().map(|r| Token::full(*r)).collect();
        // The query knows only the key domain of record 3.
        context.push(Token::partial(records[3], key_domain));
        let out = head.forward(&context).token();
        assert_eq!(out.known, key_domain | Entity::domain_mask(1));
        assert_eq!(
            out.bits & Entity::domain_mask(1),
            records[3] & Entity::domain_mask(1)
        );
    }

    #[test]
    fn min_score_can_select_nothing() {
        let mut rng = Rng::new(44);
        let mut params = AttentionParams::default();
        params.min_score = 5000;
        let head = AttentionHead::new(params);
        let scores = vec![10, 4000, 4096];
        assert_eq!(head.select(&scores), None);
        let head = AttentionHead::new(AttentionParams::default());
        assert_eq!(head.select(&scores), Some(2));
        assert_eq!(head.select(&[7, 7, 3]), Some(0));
        let _ = rng.next_u64();
    }
}
