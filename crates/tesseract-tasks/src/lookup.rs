//! Test 3: attention lookup.
//!
//! A sequence holds several records and then a query. Each record is one
//! entity with a key in one domain and a value in another. The query has
//! the key of one record and no value; the answer is that record's value.
//!
//! To make the selection something that must be learned, the other domains
//! mislead: in each decoy domain the query carries the code of a *wrong*
//! record, and the remaining domains are random bits. A head that matches
//! on everything is pulled towards the wrong records; it has to learn which
//! domain is the key and which domain to copy.
//!
//! Shows: one attention head with binary keys can learn what to select.
//! Baselines: chance, "the value of the last record", and "the value of
//! the record closest to the query over all domains" (what an untrained
//! head that matches on everything would answer).

use tesseract_codecs::{ByteCodec, Codec};
use tesseract_core::{DOMAINS, Entity, Rng, Token};

use crate::{Baseline, Example, Task};

/// Settings of the lookup task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupSettings {
    /// Records per sequence.
    pub records: usize,
    /// Number of possible values (at most 256).
    pub values: usize,
    /// Number of possible keys (at most 256, at least `records`).
    pub keys: usize,
    /// Number of decoy domains (0 to 6).
    pub decoys: usize,
    /// Domain that holds the key.
    pub key_domain: usize,
    /// Domain that holds the value.
    pub value_domain: usize,
    /// Number of held-out sequences.
    pub held_out: usize,
    /// Seed of the task.
    pub seed: u64,
}

impl Default for LookupSettings {
    fn default() -> Self {
        Self {
            records: 8,
            values: 16,
            keys: 64,
            decoys: 3,
            key_domain: 2,
            value_domain: 5,
            held_out: 2000,
            seed: 1,
        }
    }
}

/// The lookup task.
#[derive(Clone, Debug)]
pub struct Lookup {
    settings: LookupSettings,
    key_codec: ByteCodec,
    value_codec: ByteCodec,
    decoy_codecs: Vec<ByteCodec>,
    noise_domains: Vec<usize>,
    stream: Rng,
    held_out: Vec<Example>,
    candidates: Vec<Entity>,
    last_value_score: f64,
    nearest_record_score: f64,
}

impl Lookup {
    /// Builds the codes and the held-out sequences.
    pub fn new(settings: LookupSettings) -> Self {
        let mut s = settings;
        s.key_domain %= DOMAINS;
        s.value_domain %= DOMAINS;
        if s.value_domain == s.key_domain {
            s.value_domain = (s.key_domain + 1) % DOMAINS;
        }
        s.records = s.records.clamp(2, 64);
        s.keys = s.keys.clamp(s.records, 256);
        s.values = s.values.clamp(2, 256);
        s.decoys = s.decoys.min(DOMAINS - 2);
        let root = Rng::new(s.seed);
        let free: Vec<usize> = (0..DOMAINS)
            .filter(|d| *d != s.key_domain && *d != s.value_domain)
            .collect();
        let decoy_codecs: Vec<ByteCodec> = free[..s.decoys]
            .iter()
            .map(|&d| ByteCodec::new(d, s.seed.wrapping_add(1000 + d as u64)))
            .collect();
        let value_codec = ByteCodec::new(s.value_domain, s.seed.wrapping_add(2));
        let candidates = (0..s.values).map(|v| value_codec.entity(v as u8)).collect();
        let mut task = Self {
            settings: s,
            key_codec: ByteCodec::new(s.key_domain, s.seed.wrapping_add(1)),
            value_codec,
            decoy_codecs,
            noise_domains: free[s.decoys..].to_vec(),
            stream: root.fork(2),
            held_out: Vec::new(),
            candidates,
            last_value_score: 0.0,
            nearest_record_score: 0.0,
        };
        let mut held = root.fork(3);
        let mut last_right = 0;
        for _ in 0..s.held_out {
            let (example, last_value) = task.sequence(&mut held);
            last_right += usize::from(last_value == example.answer);
            task.held_out.push(example);
        }
        task.last_value_score = last_right as f64 / s.held_out.max(1) as f64;
        let nearest_right = task
            .held_out
            .iter()
            .filter(|e| task.nearest_record_answer(e) == Some(e.answer))
            .count();
        task.nearest_record_score = nearest_right as f64 / s.held_out.max(1) as f64;
        task
    }

    /// The settings after clamping.
    pub fn settings(&self) -> &LookupSettings {
        &self.settings
    }

    /// Answer of the "closest record over all domains" baseline.
    fn nearest_record_answer(&self, example: &Example) -> Option<usize> {
        let (query, records) = example.context.split_last()?;
        let best = records
            .iter()
            .enumerate()
            .max_by_key(|(i, r)| {
                (
                    query.bits.agreement(&r.bits, &(query.known & r.known)),
                    std::cmp::Reverse(*i),
                )
            })
            .map(|(_, r)| r)?;
        self.value_codec.decode(best).map(usize::from)
    }

    /// One sequence, and the value of its last record (for the baseline).
    fn sequence(&self, rng: &mut Rng) -> (Example, usize) {
        let s = &self.settings;
        let n = s.records;
        // Distinct keys, and in every decoy domain distinct decoy codes.
        let mut keys: Vec<u8> = (0..s.keys as u16).map(|k| k as u8).collect();
        rng.shuffle(&mut keys);
        let values: Vec<usize> = (0..n).map(|_| rng.below(s.values as u64) as usize).collect();
        let decoy_ids: Vec<Vec<u8>> = self
            .decoy_codecs
            .iter()
            .map(|_| {
                let mut ids: Vec<u8> = (0..=255).collect();
                rng.shuffle(&mut ids);
                ids.truncate(n);
                ids
            })
            .collect();
        let asked = rng.below(n as u64) as usize;

        let mut context = Vec::with_capacity(n + 1);
        for i in 0..n {
            let mut t = Token::default();
            self.key_codec.encode(&keys[i], &mut t);
            self.value_codec.encode(&(values[i] as u8), &mut t);
            for (codec, ids) in self.decoy_codecs.iter().zip(&decoy_ids) {
                codec.encode(&ids[i], &mut t);
            }
            self.fill_noise(&mut t, rng);
            context.push(t);
        }
        // The query: the asked key, no value, and in each decoy domain the
        // code of a wrong record.
        let mut q = Token::default();
        self.key_codec.encode(&keys[asked], &mut q);
        for (codec, ids) in self.decoy_codecs.iter().zip(&decoy_ids) {
            let wrong = (asked + 1 + rng.below(n as u64 - 1) as usize) % n;
            codec.encode(&ids[wrong], &mut q);
        }
        self.fill_noise(&mut q, rng);
        context.push(q);

        let example = Example {
            context,
            target: self.candidates[values[asked]],
            answer: values[asked],
        };
        (example, values[n - 1])
    }

    fn fill_noise(&self, token: &mut Token, rng: &mut Rng) {
        let random = rng.entity();
        for &d in &self.noise_domains {
            token.bits.set_domain(d, random.domain(d));
            token.known = token.known | Entity::domain_mask(d);
        }
    }
}

impl Task for Lookup {
    fn name(&self) -> &'static str {
        "lookup"
    }

    fn axes(&self) -> &'static str {
        "batch, time"
    }

    fn next_example(&mut self) -> Example {
        let mut rng = self.stream.clone();
        let (e, _) = self.sequence(&mut rng);
        self.stream = rng;
        e
    }

    fn held_out(&self) -> &[Example] {
        &self.held_out
    }

    fn candidates(&self) -> &[Entity] {
        &self.candidates
    }

    fn target_mask(&self) -> Entity {
        Entity::domain_mask(self.settings.value_domain)
    }

    fn baselines(&self) -> Vec<Baseline> {
        vec![
            Baseline {
                name: "chance".into(),
                score: 1.0 / self.settings.values as f64,
            },
            Baseline {
                name: "last_value".into(),
                score: self.last_value_score,
            },
            Baseline {
                name: "nearest_record".into(),
                score: self.nearest_record_score,
            },
        ]
    }

    fn pass_threshold(&self) -> f64 {
        0.95
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequences_have_the_documented_shape() {
        let task = Lookup::new(LookupSettings {
            held_out: 200,
            ..LookupSettings::default()
        });
        let s = *task.settings();
        let key = Entity::domain_mask(s.key_domain);
        let value = Entity::domain_mask(s.value_domain);
        for e in task.held_out() {
            assert_eq!(e.context.len(), 9);
            let (q, records) = e.context.split_last().unwrap();
            assert!((q.known & value).is_zero(), "the query has no value");
            assert_eq!(q.known | value, Entity::ONES);
            // Exactly one record has the query's key, and its value is the answer.
            let matching: Vec<&Token> = records
                .iter()
                .filter(|r| r.bits & key == q.bits & key)
                .collect();
            assert_eq!(matching.len(), 1);
            assert_eq!(matching[0].bits & value, e.target);
            assert_eq!(task.candidates()[e.answer], e.target);
            // In every decoy domain the query matches exactly one wrong record.
            for codec in &task.decoy_codecs {
                let d = Entity::domain_mask(codec.domain());
                let hits: Vec<&Token> = records
                    .iter()
                    .filter(|r| r.bits & d == q.bits & d)
                    .collect();
                assert_eq!(hits.len(), 1);
                assert_ne!(hits[0].bits & key, q.bits & key, "decoys point to wrong records");
            }
        }
        let b = task.baselines();
        assert_eq!(b[0].score, 1.0 / 16.0);
        assert!(b[1].score > 0.05 && b[1].score < 0.35, "last value: {}", b[1].score);
        // Matching on everything is misled by the decoys.
        assert!(b[2].score > 0.1 && b[2].score < 0.6, "nearest record: {}", b[2].score);
    }

    #[test]
    fn settings_are_clamped() {
        let task = Lookup::new(LookupSettings {
            records: 1,
            values: 1000,
            keys: 0,
            decoys: 20,
            key_domain: 3,
            value_domain: 3,
            held_out: 1,
            seed: 0,
        });
        let s = task.settings();
        assert_eq!((s.records, s.values, s.keys, s.decoys), (2, 256, 2, 6));
        assert_ne!(s.key_domain, s.value_domain);
    }
}
