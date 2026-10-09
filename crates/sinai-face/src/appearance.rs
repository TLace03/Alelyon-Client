//! How a person has shaped their Sinai: the creator's controls, the values set
//! on them, the colours chosen, and the file all of it is kept in.
//!
//! The controls are data (`assets/sinai_controls.json`), not code: each names
//! the MakeHuman targets it drives and what its ends were measured to do. This
//! module reads that catalog, holds the values, and turns them into target
//! weights for `body.rs`. It draws nothing, so it is tested without a window.
//!
//! The saved file is read forgivingly where the dock's layout is read strictly.
//! A layout that is half right puts a panel somewhere nobody can reach; an
//! appearance that is half right is still that person's Sinai. So a control
//! this build does not know is set aside with a notice rather than refusing the
//! file, and a value out of range is brought into range. A file that is not an
//! appearance at all is refused, and left on disk untouched until the person
//! saves over it.

use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const CATALOG_JSON: &str = include_str!("../assets/sinai_controls.json");

const SCHEMA_ID: &str = "sinai-appearance";
const SCHEMA_VERSION: u64 = 1;
const SHARE_PREFIX: &str = "SINAI1:";

#[derive(Clone, Debug)]
pub struct Category {
    pub id: String,
    pub label: String,
    /// Where the creator's camera goes to show this part of Sinai.
    pub focus: String,
}

#[derive(Clone, Debug)]
pub enum Kind {
    /// From -1 to 1: below zero drives `min`, above zero `max`, at up to `max_limit`.
    Slider {
        min: String,
        max: String,
        ends: [String; 2],
        max_limit: f32,
    },
    /// From 0 to 1, one shape blended in.
    Shape { target: String },
}

#[derive(Clone, Debug)]
pub struct Control {
    pub id: String,
    pub category: String,
    pub label: String,
    /// Set per side, Sinai's left and right, linked unless a person unlinks them.
    pub sided: bool,
    pub kind: Kind,
    /// Why the slider stops short of its target's full weight, when it does.
    pub why: Option<String>,
    /// Where the camera should look while this control moves, when that is
    /// not where its category looks: the shoulders, for a control in a part
    /// of the creator that otherwise shows the face.
    pub focus: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    fn suffix(self) -> &'static str {
        match self {
            Side::Left => "l",
            Side::Right => "r",
        }
    }
}

impl Control {
    pub fn range(&self) -> (f32, f32) {
        match self.kind {
            Kind::Slider { .. } => (-1.0, 1.0),
            Kind::Shape { .. } => (0.0, 1.0),
        }
    }

    /// Where its value is kept: the id, or one key per side.
    pub fn key(&self, side: Option<Side>) -> String {
        match (self.sided, side) {
            (true, Some(s)) => format!("{}.{}", self.id, s.suffix()),
            (true, None) => format!("{}.l", self.id),
            (false, _) => self.id.clone(),
        }
    }

    pub fn keys(&self) -> Vec<String> {
        if self.sided {
            vec![self.key(Some(Side::Left)), self.key(Some(Side::Right))]
        } else {
            vec![self.id.clone()]
        }
    }

    /// The target names this control drives on one side, with their weights per unit of value.
    fn targets(&self, side: Option<Side>) -> Vec<(String, bool, f32)> {
        let name = |t: &str| match side {
            Some(s) => t.replace("{s}", s.suffix()),
            None => t.to_string(),
        };
        match &self.kind {
            Kind::Slider { min, max, max_limit, .. } => vec![(name(min), false, 1.0), (name(max), true, *max_limit)],
            Kind::Shape { target } => vec![(name(target), true, 1.0)],
        }
    }

    /// Every target this control can move, both ends and both sides: what the
    /// creator lights up while the control is hovered.
    pub fn all_targets(&self) -> Vec<String> {
        let sides: Vec<Option<Side>> = if self.sided {
            vec![Some(Side::Left), Some(Side::Right)]
        } else {
            vec![None]
        };
        sides.into_iter().flat_map(|s| self.targets(s).into_iter().map(|t| t.0)).collect()
    }
}

pub struct Catalog {
    pub categories: Vec<Category>,
    pub controls: Vec<Control>,
    /// Starting points for the creator's Looks menu, by label: shapes only.
    pub looks: Vec<(String, Appearance)>,
    by_id: HashMap<String, usize>,
}

fn text(v: &serde_json::Value, key: &str, what: &str) -> Result<String, String> {
    v.get(key)
        .and_then(|s| s.as_str())
        .map(str::to_string)
        .ok_or(format!("{what} has no `{key}`"))
}

impl Catalog {
    /// The catalog this build ships with.
    pub fn builtin() -> &'static Catalog {
        static CATALOG: std::sync::OnceLock<Catalog> = std::sync::OnceLock::new();
        CATALOG.get_or_init(|| match Catalog::parse(CATALOG_JSON) {
            Ok(c) => c,
            Err(why) => panic!("the control catalog that ships with this build is unreadable: {why}"),
        })
    }

    pub fn parse(json: &str) -> Result<Catalog, String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("not JSON: {e}"))?;
        if v.get("schema").and_then(|s| s.as_str()) != Some("sinai-controls") {
            return Err("not a Sinai control catalog".into());
        }
        let mut categories = Vec::new();
        for c in v.get("categories").and_then(|c| c.as_array()).ok_or("no categories")? {
            categories.push(Category {
                id: text(c, "id", "a category")?,
                label: text(c, "label", "a category")?,
                focus: text(c, "focus", "a category")?,
            });
        }
        let mut controls = Vec::new();
        let mut by_id = HashMap::new();
        for c in v.get("controls").and_then(|c| c.as_array()).ok_or("no controls")? {
            let id = text(c, "id", "a control")?;
            let category = text(c, "category", &id)?;
            if !categories.iter().any(|k| k.id == category) {
                return Err(format!("{id} is in category {category}, which is not listed"));
            }
            let kind = match c.get("target").and_then(|t| t.as_str()) {
                Some(t) => Kind::Shape { target: t.to_string() },
                None => {
                    let ends = c.get("ends").and_then(|e| e.as_array()).ok_or(format!("{id} has no ends"))?;
                    if ends.len() != 2 {
                        return Err(format!("{id} needs two ends"));
                    }
                    let end = |k: usize| ends[k].as_str().map(str::to_string).ok_or(format!("{id}: an end is not text"));
                    let max_limit = c.get("max_limit").and_then(|m| m.as_f64()).unwrap_or(1.0) as f32;
                    if !(max_limit > 0.0 && max_limit <= 1.0) {
                        return Err(format!("{id} has max_limit {max_limit}"));
                    }
                    Kind::Slider {
                        min: text(c, "min", &id)?,
                        max: text(c, "max", &id)?,
                        ends: [end(0)?, end(1)?],
                        max_limit,
                    }
                }
            };
            if by_id.insert(id.clone(), controls.len()).is_some() {
                return Err(format!("{id} appears twice"));
            }
            controls.push(Control {
                id,
                category,
                label: text(c, "label", "a control")?,
                sided: c.get("sided").and_then(|s| s.as_bool()).unwrap_or(false),
                kind,
                why: c.get("why").and_then(|s| s.as_str()).map(str::to_string),
                focus: c.get("focus").and_then(|s| s.as_str()).map(str::to_string),
            });
        }
        let mut catalog = Catalog {
            categories,
            controls,
            looks: Vec::new(),
            by_id,
        };
        // Read with the forgiving reader and then held to the strict rule: a
        // look that ships with the build must set nothing the build lacks.
        let mut looks = Vec::new();
        for look in v.get("looks").and_then(|l| l.as_array()).into_iter().flatten() {
            let label = text(look, "label", "a look")?;
            let (a, notes) = Appearance::from_json_value(look, &catalog);
            if !notes.is_empty() {
                return Err(format!("look {label}: {}", notes.join("; ")));
            }
            looks.push((label, a));
        }
        catalog.looks = looks;
        Ok(catalog)
    }

    pub fn control(&self, id: &str) -> Option<&Control> {
        self.by_id.get(id).map(|&k| &self.controls[k])
    }

    pub fn in_category<'a>(&'a self, category: &'a str) -> impl Iterator<Item = &'a Control> + 'a {
        self.controls.iter().filter(move |c| c.category == category)
    }

    /// The control and side a stored key belongs to.
    fn resolve(&self, key: &str) -> Option<(&Control, Option<Side>)> {
        if let Some(c) = self.control(key) {
            return (!c.sided).then_some((c, None));
        }
        let (id, suffix) = key.rsplit_once('.')?;
        let side = match suffix {
            "l" => Side::Left,
            "r" => Side::Right,
            _ => return None,
        };
        let c = self.control(id)?;
        c.sided.then_some((c, Some(side)))
    }
}

/// The brand look, gold on carbon. Kept as exact values rather than as colour
/// codes, because an 8-bit code rounds them and the look is the look.
pub mod brand {
    pub const LATTICE: [f32; 3] = [0.851, 0.706, 0.357];
    pub const FILL: [f32; 3] = [0.075, 0.061, 0.046];
    pub const SHADOW: [f32; 3] = [0.020, 0.017, 0.020];
    pub const GLOW: [f32; 3] = [0.980, 0.835, 0.500];
}

/// What one colour slot draws with: the person's choice, or the brand's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Colours {
    pub lattice: [f32; 3],
    pub fill: [f32; 3],
    pub shadow: [f32; 3],
    pub glow: [f32; 3],
    /// The irises' colour and how much of it shows: none in the brand look,
    /// where the eye is drawn in the lattice's own colour.
    pub iris: [f32; 4],
}

fn unit(c: [u8; 3]) -> [f32; 3] {
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn parse_hex(s: &str) -> Option<[u8; 3]> {
    let s = s.strip_prefix('#')?;
    if s.len() != 6 || !s.is_ascii() {
        return None;
    }
    let byte = |k: usize| u8::from_str_radix(&s[k..k + 2], 16).ok();
    Some([byte(0)?, byte(2)?, byte(4)?])
}

/// The colour slots a person can set. Unset means the brand's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Palette {
    pub lattice: Option<[u8; 3]>,
    pub fill: Option<[u8; 3]>,
    pub glow: Option<[u8; 3]>,
    pub iris: Option<[u8; 3]>,
}

impl Palette {
    pub const SLOTS: [&'static str; 4] = ["lattice", "fill", "glow", "iris"];

    pub fn get(&self, slot: &str) -> Option<[u8; 3]> {
        match slot {
            "lattice" => self.lattice,
            "fill" => self.fill,
            "glow" => self.glow,
            "iris" => self.iris,
            _ => None,
        }
    }

    pub fn set(&mut self, slot: &str, c: Option<[u8; 3]>) {
        match slot {
            "lattice" => self.lattice = c,
            "fill" => self.fill = c,
            "glow" => self.glow = c,
            "iris" => self.iris = c,
            _ => {}
        }
    }

    /// What the shader draws with.
    pub fn resolve(&self) -> Colours {
        let fill = self.fill.map_or(brand::FILL, unit);
        Colours {
            lattice: self.lattice.map_or(brand::LATTICE, unit),
            fill,
            // A chosen body colour darkens into its own shadow, about as far as the brand's does.
            shadow: if self.fill.is_some() { fill.map(|v| v * 0.27) } else { brand::SHADOW },
            glow: self.glow.map_or(brand::GLOW, unit),
            iris: match self.iris {
                Some(c) => {
                    let u = unit(c);
                    [u[0], u[1], u[2], 1.0]
                }
                None => [brand::LATTICE[0], brand::LATTICE[1], brand::LATTICE[2], 0.0],
            },
        }
    }

    /// A slot's colour as the picker shows it.
    pub fn shown(&self, slot: &str) -> [u8; 3] {
        let c = self.resolve();
        let f = |v: [f32; 3]| v.map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8);
        self.get(slot).unwrap_or_else(|| match slot {
            "fill" => f(c.fill),
            "glow" => f(c.glow),
            _ => f(c.lattice),
        })
    }
}

/// One person's Sinai.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Appearance {
    /// Only values that are not zero are kept; a missing key is zero.
    values: BTreeMap<String, f32>,
    /// Sided controls whose two sides are set separately.
    unlinked: BTreeSet<String>,
    pub palette: Palette,
}

impl Appearance {
    pub fn value(&self, key: &str) -> f32 {
        self.values.get(key).copied().unwrap_or(0.0)
    }

    pub fn get(&self, c: &Control, side: Option<Side>) -> f32 {
        self.value(&c.key(side))
    }

    pub fn is_linked(&self, c: &Control) -> bool {
        c.sided && !self.unlinked.contains(&c.id)
    }

    fn put(&mut self, key: String, v: f32) {
        if v == 0.0 {
            self.values.remove(&key);
        } else {
            self.values.insert(key, v);
        }
    }

    /// Set a control. A linked control sets both sides whichever side is named.
    pub fn set(&mut self, c: &Control, side: Option<Side>, v: f32) {
        let (lo, hi) = c.range();
        let v = if v.is_finite() { v.clamp(lo, hi) } else { 0.0 };
        if c.sided && (self.is_linked(c) || side.is_none()) {
            self.put(c.key(Some(Side::Left)), v);
            self.put(c.key(Some(Side::Right)), v);
        } else {
            self.put(c.key(side), v);
        }
    }

    /// Set the two sides separately from now on.
    pub fn unlink(&mut self, c: &Control) {
        if c.sided {
            self.unlinked.insert(c.id.clone());
        }
    }

    /// Set them together again. Sinai's right takes its left's value: the two
    /// must agree, and silently averaging would give a value nobody set.
    pub fn link(&mut self, c: &Control) {
        if c.sided && self.unlinked.remove(&c.id) {
            let v = self.get(c, Some(Side::Left));
            self.put(c.key(Some(Side::Right)), v);
        }
    }

    /// Back to zero, and linked.
    pub fn reset(&mut self, c: &Control) {
        for k in c.keys() {
            self.values.remove(&k);
        }
        self.unlinked.remove(&c.id);
    }

    pub fn reset_category(&mut self, catalog: &Catalog, category: &str) {
        for c in catalog.in_category(category) {
            self.reset(c);
        }
    }

    pub fn is_default_shape(&self) -> bool {
        self.values.is_empty()
    }

    /// The target weights this appearance asks for, by target index.
    pub fn weights(&self, catalog: &Catalog, find: impl Fn(&str) -> Option<usize>) -> Vec<(usize, f32)> {
        let mut out = Vec::new();
        for c in &catalog.controls {
            let sides: Vec<Option<Side>> = if c.sided {
                vec![Some(Side::Left), Some(Side::Right)]
            } else {
                vec![None]
            };
            for side in sides {
                let v = self.get(c, side);
                if v == 0.0 {
                    continue;
                }
                for (name, positive, scale) in c.targets(side) {
                    let w = if positive { v.max(0.0) } else { (-v).max(0.0) } * scale;
                    if w > 0.0 {
                        if let Some(t) = find(&name) {
                            out.push((t, w));
                        }
                    }
                }
            }
        }
        out
    }

    pub fn to_json_value(&self) -> serde_json::Value {
        let mut colours = serde_json::Map::new();
        for slot in Palette::SLOTS {
            if let Some(c) = self.palette.get(slot) {
                colours.insert(slot.into(), hex(c).into());
            }
        }
        serde_json::json!({
            "values": self.values,
            "unlinked": self.unlinked,
            "colours": colours,
        })
    }

    /// Read one back. Returns the appearance and what was set aside, in words.
    pub fn from_json_value(v: &serde_json::Value, catalog: &Catalog) -> (Appearance, Vec<String>) {
        let mut a = Appearance::default();
        let mut notes = Vec::new();
        if let Some(values) = v.get("values").and_then(|x| x.as_object()) {
            let mut unknown = Vec::new();
            for (key, value) in values {
                match (catalog.resolve(key), value.as_f64()) {
                    (Some((c, _)), Some(x)) => {
                        let (lo, hi) = c.range();
                        let x = x as f32;
                        if !x.is_finite() || x < lo || x > hi {
                            notes.push(format!("{key} was {x}, brought into {lo}..{hi}"));
                        }
                        a.put(key.clone(), if x.is_finite() { x.clamp(lo, hi) } else { 0.0 });
                    }
                    (Some(_), None) => notes.push(format!("{key} is not a number and was left at zero")),
                    (None, _) => unknown.push(key.clone()),
                }
            }
            if !unknown.is_empty() {
                notes.push(format!(
                    "{} setting{} this version does not have {} set aside: {}",
                    unknown.len(),
                    if unknown.len() == 1 { "" } else { "s" },
                    if unknown.len() == 1 { "was" } else { "were" },
                    unknown.join(", ")
                ));
            }
        }
        if let Some(list) = v.get("unlinked").and_then(|x| x.as_array()) {
            for id in list.iter().filter_map(|x| x.as_str()) {
                if catalog.control(id).map_or(false, |c| c.sided) {
                    a.unlinked.insert(id.to_string());
                }
            }
        }
        // A control saved linked whose sides disagree was edited by hand; the
        // left side wins, as it does when a person links them.
        let linked: Vec<&Control> = catalog.controls.iter().filter(|c| c.sided && !a.unlinked.contains(&c.id)).collect();
        for c in linked {
            let (l, r) = (a.get(c, Some(Side::Left)), a.get(c, Some(Side::Right)));
            if l != r {
                a.put(c.key(Some(Side::Right)), l);
            }
        }
        if let Some(colours) = v.get("colours").and_then(|x| x.as_object()) {
            for slot in Palette::SLOTS {
                if let Some(s) = colours.get(slot).and_then(|x| x.as_str()) {
                    match parse_hex(s) {
                        Some(c) => a.palette.set(slot, Some(c)),
                        None => notes.push(format!("the {slot} colour {s:?} is not a colour; the brand's is used")),
                    }
                }
            }
        }
        (a, notes)
    }

    /// A short text that carries this appearance to someone else.
    ///
    /// Each value goes as a hash of its key and a byte, so a code made by a
    /// build with more controls still opens in this one, minus what it lacks.
    /// A byte is 1/127 of the travel: a value can come back up to 0.004 off.
    pub fn share_code(&self) -> String {
        let mut bytes = vec![1u8];
        let mut flags = 0u8;
        let mut colours = Vec::new();
        for (bit, slot) in Palette::SLOTS.iter().enumerate() {
            if let Some(c) = self.palette.get(slot) {
                flags |= 1 << bit;
                colours.extend_from_slice(&c);
            }
        }
        bytes.push(flags);
        bytes.extend(colours);
        let n = self.values.len().min(u16::MAX as usize) as u16;
        bytes.extend_from_slice(&n.to_le_bytes());
        for (key, &v) in self.values.iter().take(n as usize) {
            bytes.extend_from_slice(&fnv1a(key.as_bytes()).to_le_bytes());
            bytes.push(((v * 127.0).round().clamp(-127.0, 127.0) as i8) as u8);
        }
        let check = (fnv1a(&bytes) & 0xffff) as u16;
        bytes.extend_from_slice(&check.to_le_bytes());
        use base64::Engine;
        format!("{SHARE_PREFIX}{}", base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
    }

    pub fn from_share_code(code: &str, catalog: &Catalog) -> Result<(Appearance, Vec<String>), String> {
        let body = code.trim().strip_prefix(SHARE_PREFIX).ok_or("not a Sinai appearance code")?;
        use base64::Engine;
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(body.trim())
            .map_err(|_| "the code is damaged (not valid text for a code)".to_string())?;
        if bytes.len() < 6 {
            return Err("the code is too short to be one".into());
        }
        let (payload, check) = bytes.split_at(bytes.len() - 2);
        if (fnv1a(payload) & 0xffff) as u16 != u16::from_le_bytes([check[0], check[1]]) {
            return Err("the code is damaged (its check does not match); copy it again".into());
        }
        if payload[0] != 1 {
            return Err(format!("a version {} code, which this build does not read", payload[0]));
        }
        let mut at = 2;
        let mut a = Appearance::default();
        for (bit, slot) in Palette::SLOTS.iter().enumerate() {
            if payload[1] & (1 << bit) != 0 {
                let c = payload.get(at..at + 3).ok_or("the code is cut short")?;
                a.palette.set(slot, Some([c[0], c[1], c[2]]));
                at += 3;
            }
        }
        let n = payload.get(at..at + 2).ok_or("the code is cut short")?;
        let n = u16::from_le_bytes([n[0], n[1]]) as usize;
        at += 2;
        let keys: HashMap<u32, String> = catalog
            .controls
            .iter()
            .flat_map(|c| c.keys())
            .map(|k| (fnv1a(k.as_bytes()), k))
            .collect();
        let mut unknown = 0;
        for _ in 0..n {
            let e = payload.get(at..at + 5).ok_or("the code is cut short")?;
            at += 5;
            let h = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
            let v = (e[4] as i8) as f32 / 127.0;
            match keys.get(&h).and_then(|k| catalog.resolve(k).map(|(c, _)| (k, c))) {
                Some((k, c)) => {
                    let (lo, hi) = c.range();
                    a.put(k.clone(), v.clamp(lo, hi));
                }
                None => unknown += 1,
            }
        }
        if at != payload.len() {
            return Err("the code has more in it than it says".into());
        }
        // Sides that differ were set apart on purpose.
        for c in catalog.controls.iter().filter(|c| c.sided) {
            if a.get(c, Some(Side::Left)) != a.get(c, Some(Side::Right)) {
                a.unlinked.insert(c.id.clone());
            }
        }
        let mut notes = Vec::new();
        if unknown > 0 {
            notes.push(format!("{unknown} setting(s) in the code are not in this version and were left out"));
        }
        Ok((a, notes))
    }

    /// A random Sinai, from a seed, within ranges that stay a plausible face:
    /// about half the controls moved, by up to half their travel, asymmetry kept
    /// slight, and one or two head shapes blended in.
    pub fn random(catalog: &Catalog, seed: u64, only: Option<&str>) -> Appearance {
        let mut rng = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut a = Appearance::default();
        let shapes: Vec<&Control> = catalog
            .controls
            .iter()
            .filter(|c| matches!(c.kind, Kind::Shape { .. }) && only.map_or(true, |o| c.category == o))
            .collect();
        for c in catalog.controls.iter().filter(|c| only.map_or(true, |o| c.category == o)) {
            if !matches!(c.kind, Kind::Slider { .. }) || rng.unit() > 0.5 {
                continue;
            }
            let spread = if c.category == "asym" { 0.15 } else { 0.5 };
            // Triangular, so most values are small and the ends are rare.
            let v = (rng.unit() + rng.unit() - 1.0) * spread;
            a.set(c, None, v);
        }
        if !shapes.is_empty() {
            let count = 1 + (rng.unit() * 2.0) as usize;
            for _ in 0..count.min(shapes.len()) {
                let c = shapes[(rng.unit() * shapes.len() as f32) as usize % shapes.len()];
                a.set(c, None, 0.15 + 0.3 * rng.unit());
            }
        }
        a
    }

    /// Keep the shape of `other` for one category, everything else from self.
    pub fn with_category_from(&self, other: &Appearance, catalog: &Catalog, category: &str) -> Appearance {
        let mut out = self.clone();
        out.reset_category(catalog, category);
        for c in catalog.in_category(category) {
            for k in c.keys() {
                if let Some(&v) = other.values.get(&k) {
                    out.values.insert(k, v);
                }
            }
            if other.unlinked.contains(&c.id) {
                out.unlinked.insert(c.id.clone());
            }
        }
        out
    }
}

/// FNV-1a, 32 bits: small, stable across builds and platforms, and all a share
/// code needs to tell its keys apart.
fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// splitmix64: a seedable generator in four lines, so a randomised Sinai can be
/// made again from its seed and no dependency is needed for it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The saved file: the appearance in use, and the looks a person has kept.
#[derive(Clone, Debug, Default)]
pub struct Saved {
    pub current: Appearance,
    pub looks: Vec<(String, Appearance)>,
    pub notice: String,
}

impl Saved {
    pub fn to_json(&self) -> String {
        let looks: Vec<serde_json::Value> = self
            .looks
            .iter()
            .map(|(name, a)| {
                let mut v = a.to_json_value();
                v["name"] = name.clone().into();
                v
            })
            .collect();
        let mut v = self.current.to_json_value();
        v["schema"] = SCHEMA_ID.into();
        v["version"] = SCHEMA_VERSION.into();
        v["looks"] = looks.into();
        serde_json::to_string_pretty(&v).unwrap_or_default()
    }

    pub fn from_json(text: &str, catalog: &Catalog) -> Result<Saved, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?;
        if v.get("schema").and_then(|s| s.as_str()) != Some(SCHEMA_ID) {
            return Err("not a Sinai appearance".into());
        }
        if v.get("version").and_then(|s| s.as_u64()) != Some(SCHEMA_VERSION) {
            return Err("a version of the appearance this build does not read".into());
        }
        let (current, mut notes) = Appearance::from_json_value(&v, catalog);
        let mut looks = Vec::new();
        for look in v.get("looks").and_then(|l| l.as_array()).into_iter().flatten() {
            let name = look.get("name").and_then(|n| n.as_str()).unwrap_or("Unnamed").to_string();
            let (a, more) = Appearance::from_json_value(look, catalog);
            notes.extend(more.into_iter().map(|m| format!("look {name}: {m}")));
            looks.push((name, a));
        }
        Ok(Saved {
            current,
            looks,
            notice: notes.join("; "),
        })
    }

    /// The appearance from disk, or the default, saying which.
    pub fn restored(catalog: &Catalog) -> Saved {
        let mut saved = Saved::read_saved(catalog);
        // A move into the state home that failed this start is said beside the appearance's
        // other notices, not only in the log.
        if let Some(why) = crate::state_home::notice(crate::state_home::File::Appearance) {
            saved.notice = if saved.notice.is_empty() { why } else { format!("{why}; {}", saved.notice) };
        }
        saved
    }

    fn read_saved(catalog: &Catalog) -> Saved {
        let Some(path) = saved_at() else {
            return Saved::default();
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Saved::default(),
            Err(e) => {
                return Saved {
                    notice: format!("Saved appearance could not be read: {e}"),
                    ..Saved::default()
                }
            }
        };
        match Saved::from_json(&text, catalog) {
            Ok(saved) => {
                if !saved.notice.is_empty() {
                    eprintln!("[angel] appearance at {}: {}", path.display(), saved.notice);
                }
                saved
            }
            Err(why) => {
                eprintln!("[angel] the saved appearance at {} was not used: {why}", path.display());
                Saved {
                    notice: format!("Saved appearance could not be read ({why}); it is untouched until you save"),
                    ..Saved::default()
                }
            }
        }
    }

    pub fn save(&mut self) {
        let Some(path) = saved_at() else {
            self.notice = "No settings directory is available; the appearance is not saved".into();
            return;
        };
        self.notice = match self.save_to(&path) {
            Ok(()) => String::new(),
            Err(why) => format!("Appearance not saved: {why}"),
        };
    }

    /// Written beside itself and renamed into place, so a crash mid-write
    /// leaves the last good appearance rather than half of a new one.
    pub fn save_to(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        let mut created = false;
        let result = (|| {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|e| e.to_string())?;
            created = true;
            file.write_all(self.to_json().as_bytes()).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            drop(file);
            std::fs::rename(&temporary, path).map_err(|e| e.to_string())
        })();
        if result.is_err() && created {
            // Only the sibling created for this save; no recursive removal.
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }
}

/// Where the appearance lives: beside the dock's layout, in the window's state
/// home, `~/.alelyon/angel` (`state_home.rs`). ANGEL_APPEARANCE points it
/// elsewhere, for tests and for anyone keeping more than one.
pub fn saved_at() -> Option<std::path::PathBuf> {
    crate::state_home::saved_at(crate::state_home::File::Appearance)
}

/// Undo and redo over whole appearances. Small enough to copy: a few hundred
/// numbers each.
#[derive(Default)]
pub struct History {
    past: Vec<Appearance>,
    future: Vec<Appearance>,
}

impl History {
    const DEPTH: usize = 200;

    /// Remember `before` as the state to return to, and forget anything redone.
    pub fn record(&mut self, before: &Appearance) {
        if self.past.last() == Some(before) {
            return;
        }
        self.past.push(before.clone());
        if self.past.len() > Self::DEPTH {
            self.past.remove(0);
        }
        self.future.clear();
    }

    pub fn undo(&mut self, now: &Appearance) -> Option<Appearance> {
        let back = self.past.pop()?;
        self.future.push(now.clone());
        Some(back)
    }

    pub fn redo(&mut self, now: &Appearance) -> Option<Appearance> {
        let forward = self.future.pop()?;
        self.past.push(now.clone());
        Some(forward)
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }
}

#[cfg(test)]
#[path = "appearance_tests.rs"]
mod tests;
