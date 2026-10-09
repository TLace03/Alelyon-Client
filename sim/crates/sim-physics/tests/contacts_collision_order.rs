//! The units of both passes of the collision step are independent (phase 1c-ii review).
//!
//! The spec's GPU mapping is "pass 1 is one independent unit per candidate, pass 2 a prefix sum
//! over the per-candidate counts". The first implementation compacted the pre-contacts IN PLACE in
//! pass 2, which is safe only when the candidates run one after another in list order: the
//! destination range of one candidate overlaps the source range of an earlier one (in every one
//! of the pile's eight golden states), so a parallel scatter would race. Pass 2 is now out of
//! place (`Data::pre_*` are separate from the contact arrays) and each unit reads and writes only
//! its own ranges.
//!
//! The tests run the passes in other orders through `sim_physics::faults::collide_in_order` (pass 1
//! and pass 2 each in the list order, reversed, shuffled by a fixed generator, and pass 1 grouped
//! by collider, as a GPU dispatch per collider type would run it) on every golden state of every
//! model, with the contact and pre-contact arrays first filled with garbage, and require the
//! contacts to be the same BITS as the list order's. The positive control shows that the test has
//! power: the old in-place compaction, simulated on the pile's slot layout, gives another result
//! when pass 2 runs in reverse order.

mod common;

use common::contacts::*;
use sim_physics::faults::collide_in_order;
use sim_physics::{Data, Model, Real};

/// The bits of every contact array of `d` up to `ncon`, and the counts.
fn fingerprint<R: Real>(d: &Data<R>) -> Vec<u64> {
    let n = d.ncon;
    let mut out = vec![d.ncon as u64];
    let mut put = |a: &[R]| out.extend(a.iter().map(|x| x.to_f64().to_bits()));
    put(&d.contact_dist[..n]);
    put(&d.contact_pos[..3 * n]);
    put(&d.contact_frame[..9 * n]);
    put(&d.contact_includemargin[..n]);
    put(&d.contact_friction[..5 * n]);
    put(&d.contact_solref[..2 * n]);
    put(&d.contact_solreffriction[..2 * n]);
    put(&d.contact_solimp[..5 * n]);
    put(&d.contact_mu[..n]);
    put(&d.contact_h[..36 * n]);
    out.extend(d.contact_dim[..n].iter().map(|&x| x as u64));
    out.extend(d.contact_geom[..2 * n].iter().map(|&x| x as u64));
    out.extend(d.contact_exclude[..n].iter().map(|&x| x as u64));
    out.extend(d.contact_efc_address[..n].iter().map(|&x| x as u64));
    out.extend(d.cand_ncon.iter().map(|&x| x as u64));
    out.extend(d.cand_start.iter().map(|&x| x as u64));
    out
}

/// A permutation of `0..n` from a fixed xorshift generator (Fisher-Yates).
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        p.swap(i, (next() % (i as u64 + 1)) as usize);
    }
    p
}

/// The candidates grouped by collider (stable), as a dispatch per collider type would run them.
fn by_collider<R: Real>(m: &Model<R>) -> Vec<usize> {
    let mut p: Vec<usize> = (0..m.candidates.len()).collect();
    p.sort_by_key(|&k| format!("{:?}", m.candidates[k].collider));
    p
}

/// Fills every array the collision step writes with garbage, so that nothing can be left over from
/// an earlier run.
fn poison<R: Real>(d: &mut Data<R>) {
    let nan = R::from_f64(f64::NAN);
    for a in [
        &mut d.contact_dist,
        &mut d.contact_pos,
        &mut d.contact_frame,
        &mut d.contact_includemargin,
        &mut d.contact_friction,
        &mut d.contact_solref,
        &mut d.contact_solreffriction,
        &mut d.contact_solimp,
        &mut d.contact_mu,
        &mut d.contact_h,
        &mut d.pre_dist,
        &mut d.pre_pos,
        &mut d.pre_frame,
    ] {
        a.fill(nan);
    }
    d.contact_dim.fill(77);
    d.contact_geom.fill(77);
    d.contact_exclude.fill(77);
    d.contact_efc_address.fill(77);
    d.cand_ncon.fill(77);
    d.cand_start.fill(77);
    d.cand_overflow.fill(true);
}

/// One golden state of one model in precision `R`: the list order's contacts and the contacts of
/// every other order, equal to the bit. Returns how many (state, order) comparisons it made.
fn check_state<R: Real>(m: &Model<R>, s: &serde_json::Value, label: &str) -> usize {
    let mut d = tdata(m, s);
    sim_physics::kinematics(m, &mut d);
    sim_physics::collide(m, &mut d);
    let reference = fingerprint(&d);
    let n = m.candidates.len();
    let identity: Vec<usize> = (0..n).collect();
    let reverse: Vec<usize> = (0..n).rev().collect();
    let (shuf_a, shuf_b) = (shuffled(n, 0x9E37_79B9_7F4A_7C15), shuffled(n, 0xD1B5_4A32));
    let grouped = by_collider(m);
    let orders: [(&str, &Vec<usize>, &Vec<usize>); 6] = [
        ("list order", &identity, &identity),
        ("pass 2 reversed", &identity, &reverse),
        ("pass 1 reversed", &reverse, &identity),
        ("both reversed", &reverse, &reverse),
        ("both shuffled", &shuf_a, &shuf_b),
        (
            "pass 1 grouped by collider, pass 2 shuffled",
            &grouped,
            &shuf_a,
        ),
    ];
    for (name, p1, p2) in orders {
        let mut e = tdata(m, s);
        sim_physics::kinematics(m, &mut e);
        poison(&mut e);
        collide_in_order(m, &mut e, p1, p2);
        assert_eq!(
            fingerprint(&e),
            reference,
            "{label}: the contacts differ when the passes run in this order: {name}"
        );
        assert_eq!(e.warning_collision_overflow, 0);
    }
    orders.len()
}

fn gate(which: TWhich) {
    let mut compared = 0usize;
    for v in CONES {
        let g = tgolden(which, v);
        let c64 = tcompile::<f64>(which, v);
        let c32 = tcompile::<f32>(which, v);
        for (k, s) in tstates(&g).iter().enumerate() {
            compared += check_state(
                &c64.model,
                s,
                &format!("{} {} state {k} (f64)", which.name(), v.name()),
            );
            compared += check_state(
                &c32.model,
                s,
                &format!("{} {} state {k} (f32)", which.name(), v.name()),
            );
        }
    }
    println!(
        "MEASURED collision order {}: {compared} (state, order) comparisons in f64 and f32, every one equal to the bit to the list order's contacts",
        which.name()
    );
}

macro_rules! order_tests {
    ($($name:ident: $which:expr;)*) => {
        $(
            #[test]
            fn $name() {
                gate($which);
            }
        )*
    };
}

order_tests! {
    the_sphere_scenes_collision_units_are_order_independent: TWhich::Sphere;
    the_box_scenes_collision_units_are_order_independent: TWhich::Box;
    the_stacks_collision_units_are_order_independent: TWhich::Stack;
    the_capsules_collision_units_are_order_independent: TWhich::Capsules;
    the_piles_collision_units_are_order_independent: TWhich::Pile;
    the_humanoids_collision_units_are_order_independent: TWhich::Humanoid;
    the_zoos_collision_units_are_order_independent: TWhich::Zoo;
}

/// The positive control: the first implementation's in-place compaction (`contact[dst] =
/// pre[src]` into the same array, `dst` the running count, `src` the candidate's slot) run with
/// pass 2 in reverse order gives another list than in list order on the pile's golden states,
/// because the destination range of a later candidate overlaps the source range of an earlier
/// one, which is the hazard a parallel scatter would hit; the out-of-place pass 2 gives the same
/// list in both. (A statement about the layout and the algorithm, on integers: slot ids.)
#[test]
fn the_old_in_place_compaction_would_have_failed_this_test() {
    let (mut hazards, mut states) = (0usize, 0usize);
    for v in CONES {
        let g = tgolden(TWhich::Pile, v);
        let c = tcompile::<f64>(TWhich::Pile, v);
        let m = &c.model;
        for s in tstates(&g) {
            let mut d = tdata(m, s);
            sim_physics::kinematics(m, &mut d);
            sim_physics::collide(m, &mut d);
            states += 1;
            // slot ids: slot i holds the number i + 1 (0 = empty)
            let slots = m.ncon_max;
            let compact = |order: &[usize], in_place: bool| -> Vec<usize> {
                let pre: Vec<usize> = (0..slots).map(|i| i + 1).collect();
                let mut contact = if in_place {
                    pre.clone()
                } else {
                    vec![0; slots]
                };
                for &k in order {
                    let c = &m.candidates[k];
                    for t in 0..d.cand_ncon[k] {
                        let (src, dst) = (c.slot_offset + t, d.cand_start[k] + t);
                        let value = if in_place { contact[src] } else { pre[src] };
                        contact[dst] = value;
                    }
                }
                contact.truncate(d.ncon);
                contact
            };
            let list: Vec<usize> = (0..m.candidates.len()).collect();
            let reverse: Vec<usize> = list.iter().rev().copied().collect();
            // out of place: the same in any order
            assert_eq!(compact(&list, false), compact(&reverse, false));
            // in place: the list order works (the first implementation), the reverse does not
            assert_eq!(compact(&list, true), compact(&list, false));
            hazards += usize::from(compact(&reverse, true) != compact(&list, false));
        }
    }
    println!(
        "MEASURED collision order: the old in-place compaction in reverse order gives a different contact list in {hazards} of {states} pile states (both cones); the out-of-place pass 2 in none"
    );
    assert!(
        hazards > 0,
        "the layout has no overlap: the control has no power"
    );
}
