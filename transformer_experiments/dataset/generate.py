
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Noun:
    singular: str
    plural: str
    semtype: str
    animate: bool = False

    def form(self, number: str) -> str:
        return self.singular if number == "singular" else self.plural


@dataclass(frozen=True)
class Verb:
    singular: str
    plural: str
    subj_types: frozenset[str]
    obj_types: frozenset[str] = frozenset()  # empty => intransitive

    @property
    def transitive(self) -> bool:
        return len(self.obj_types) > 0

    def form(self, number: str) -> str:
        return self.singular if number == "singular" else self.plural


@dataclass(frozen=True)
class Adjective:
    form: str
    modifies: frozenset[str]
    category: Optional[str] = None  # at most one adjective per category per NP


@dataclass(frozen=True)
class PPFrame:
    """Licensed (head_semtype, preposition, complement_semtype) triple."""
    head_type: str
    prep: str
    comp_type: str


# ---- real lexicon ---------------------------------------------------------

NOUNS: list[Noun] = [
    # animals
    Noun("cat", "cats", "animal", animate=True),
    Noun("dog", "dogs", "animal", animate=True),
    Noun("squirrel", "squirrels", "animal", animate=True),
    Noun("mouse", "mice", "animal", animate=True),
    Noun("donkey", "donkeys", "animal", animate=True),
    Noun("goose", "geese", "animal", animate=True),
    Noun("ox", "oxen", "animal", animate=True),
    # people
    Noun("man", "men", "person", animate=True),
    Noun("woman", "women", "person", animate=True),
    Noun("worker", "workers", "person", animate=True),
    Noun("child", "children", "person", animate=True),
    Noun("policeman", "policemen", "person", animate=True),
    Noun("teacher", "teachers", "person", animate=True),
    Noun("farmer", "farmers", "person", animate=True),
    # vehicles
    Noun("car", "cars", "vehicle"),
    Noun("van", "vans", "vehicle"),
    Noun("helicopter", "helicopters", "vehicle"),
    Noun("truck", "trucks", "vehicle"),
    Noun("boat", "boats", "vehicle"),
    Noun("plane", "planes", "vehicle"),
    Noun("spaceship", "spaceships", "vehicle"),
    # furniture
    Noun("table", "tables", "furniture"),
    Noun("stove", "stoves", "furniture"),
    Noun("bed", "beds", "furniture"),
    Noun("chair", "chairs", "furniture"),
    Noun("sofa", "sofas", "furniture"),
    Noun("shelf", "shelves", "furniture"),
    Noun("fridge", "fridges", "furniture"),
    # container
    Noun("bucket", "buckets", "container"),
    Noun("backpack", "backpacks", "container"),
    Noun("box", "boxes", "container"),
    Noun("bag", "bags", "container"),
    Noun("basket", "baskets", "container"),
    Noun("suitcase", "suitcases", "container"),
    Noun("bottle", "bottles", "container"),
    # small objects
    Noun("phone", "phones", "object"),
    Noun("pen", "pens", "object"),
    Noun("spoon", "spoons", "object"),
    Noun("crystal", "crystals", "object"),
    Noun("ring", "rings", "object"),
    Noun("cup", "cups", "object"),
    Noun("coin", "coins", "object"),
    # locations
    Noun("garden", "gardens", "location"),
    Noun("house", "houses", "location"),
    Noun("field", "fields", "location"),
    Noun("park", "parks", "location"),
    Noun("forest", "forests", "location"),
    Noun("building", "buildings", "location"),
    Noun("mountain", "mountains", "location"),
]

VERBS: list[Verb] = [
    # intransitive
    Verb("runs", "run", frozenset({"animal", "person"})),
    Verb("jumps", "jump", frozenset({"animal", "person"})),
    Verb("sleeps", "sleep", frozenset({"animal", "person"})),
    Verb("arrives", "arrive", frozenset({"animal", "person", "vehicle"})),
    Verb("stops", "stop", frozenset({"animal", "person", "vehicle"})),
    Verb("travels", "travel", frozenset({"animal", "person", "vehicle"})),
    Verb("appears", "appear", frozenset({"animal", "person", "vehicle"})),
    Verb("shines", "shine", frozenset({"object"})),
    Verb("collapses", "collapse", frozenset({"furniture"})),
    Verb("breaks", "break", frozenset({"furniture"})),
    Verb("shakes", "shake", frozenset({"furniture"})),
    Verb("remains", "remain", frozenset({"object", "location"})),
    Verb("burns", "burn", frozenset({"location"})),
    Verb("exists", "exist", frozenset({"animal", "person", "vehicle", "furniture", "container", "object", "location"})),
    Verb("falls", "fall", frozenset({"animal", "person", "vehicle", "furniture", "container", "object"})),
    # transitive
    Verb("chases", "chase",
        frozenset({"animal", "person"}),
        frozenset({"animal", "person", "vehicle"})),
    Verb("meets", "meet",
        frozenset({"animal", "person"}),
        frozenset({"animal", "person"})),
    Verb("bites", "bite",
        frozenset({"animal"}),
        frozenset({"animal", "person"})),
    Verb("rides", "ride",
        frozenset({"animal", "person"}),
        frozenset({"vehicle"})),
    Verb("searches", "search",
        frozenset({"animal", "person"}),
        frozenset({"vehicle", "container", "location"})),
    Verb("pushes", "push",
        frozenset({"animal", "person"}),
        frozenset({"animal", "person", "vehicle", "furniture", "container", "object"})),
    Verb("pulls", "pull",
        frozenset({"animal", "person", "vehicle"}),
        frozenset({"vehicle", "furniture", "container"})),
    Verb("operates", "operate",
        frozenset({"person"}),
        frozenset({"vehicle"})),
    Verb("moves", "move",
        frozenset({"animal", "person"}),
        frozenset({"furniture", "container", "object"})),
    Verb("sees", "see",
        frozenset({"animal", "person"}),
        frozenset({"animal", "person", "vehicle", "furniture", "container", "object", "location"})),
    Verb("finds", "find",
        frozenset({"animal", "person"}),
        frozenset({"animal", "person", "vehicle", "furniture", "container", "object", "location"})),
    Verb("destroys", "destroy",
        frozenset({"animal", "person"}),
        frozenset({"vehicle", "furniture", "container", "object"})),
    Verb("touches", "touch",
        frozenset({"animal", "person"}),
        frozenset({"vehicle", "furniture", "object"})),
    Verb("carries", "carry",
        frozenset({"person", "vehicle"}),
        frozenset({"furniture", "container", "object"})),
    Verb("explores", "explore",
        frozenset({"animal", "person"}),
        frozenset({"location"})),
    Verb("visits", "visit",
        frozenset({"animal", "person"}),
        frozenset({"location"})),
    Verb("enters", "enter",
        frozenset({"animal", "person"}),
        frozenset({"location"})),
    Verb("leaves", "leave",
        frozenset({"animal", "person"}),
        frozenset({"location"})),
    Verb("approaches", "approach",
        frozenset({"animal", "person", "vehicle"}),
        frozenset({"animal", "person", "vehicle", "location"})),
    Verb("follows", "follow",
        frozenset({"animal", "person", "vehicle"}),
        frozenset({"animal", "person", "vehicle"})),
    Verb("opens", "open",
        frozenset({"person"}),
        frozenset({"container"})),
    Verb("closes", "close",
        frozenset({"person"}),
        frozenset({"container"})),
    Verb("buys", "buy",
        frozenset({"person"}),
        frozenset({"vehicle", "furniture", "container", "object"})),
    Verb("holds", "hold",
        frozenset({"person", "container"}),
        frozenset({"object"})),
    Verb("transports", "transport",
        frozenset({"vehicle"}),
        frozenset({"person", "animal", "container", "object"})),
]

# Copula is handled separately from VERBS so it isn't sampled by non-copula
# templates (which would produce fragments like "the chair is").
COPULA = ["is", "are"]  # (singular, plural)

ADJECTIVES: list[Adjective] = [
    # size
    Adjective("small", frozenset({"animal", "person", "vehicle",
                                  "furniture", "container", "object"}), "size"),
    Adjective("large", frozenset({"animal", "person", "vehicle",
                                  "furniture", "container", "object", "location"}), "size"),
    # body state (for animates)
    Adjective("hungry", frozenset({"animal", "person"}), "body_state"),
    Adjective("tired", frozenset({"animal", "person"}), "body_state"),
    Adjective("wounded", frozenset({"animal", "person"}), "body_state"),
    Adjective("healthy", frozenset({"animal", "person"}), "body_state"),
    Adjective("sick", frozenset({"animal", "person"}), "body_state"),
    # mood (for animates)
    Adjective("angry", frozenset({"animal", "person"}), "mood"),
    Adjective("quiet", frozenset({"animal", "person", "location"}), "mood"),
    # age (animates)
    Adjective("young", frozenset({"animal", "person"}), "age"),
    Adjective("old", frozenset({"animal", "person"}), "age"),
    # strength (animates)
    Adjective("strong", frozenset({"animal", "person"}), "strength"),
    Adjective("weak", frozenset({"animal", "person"}), "strength"),
    # physical condition (objects/vehicles/etc.)
    Adjective("new", frozenset({"vehicle", "furniture", "container", "object"}), "condition"),
    Adjective("worn", frozenset({"vehicle", "furniture", "container", "object"}), "condition"),
    Adjective("broken", frozenset({"vehicle", "furniture", "container", "object"}), "condition"),
    Adjective("sturdy", frozenset({"furniture", "container"}), "condition"),
    Adjective("fragile", frozenset({"furniture", "container", "object"}), "condition"),
    # surface
    Adjective("shiny", frozenset({"object"}), "surface"),
    Adjective("wet", frozenset({"animal", "person", "vehicle", "furniture",
                                "container", "object"}), "surface"),
    # fill
    Adjective("empty", frozenset({"vehicle", "container"}), "fill"),
    Adjective("filled", frozenset({"container"}), "fill"),
    # temperature
    Adjective("frosted", frozenset({"vehicle"}), "temperature"),
    Adjective("frosty", frozenset({"location"}), "temperature"),
    Adjective("scorched", frozenset({"vehicle", "location"}), "temperature"),
    # color
    Adjective("blue", frozenset({"vehicle", "furniture", "container", "object"}), "color"),
    Adjective("red", frozenset({"vehicle", "furniture", "container", "object"}), "color"),
    Adjective("yellow", frozenset({"vehicle", "furniture", "container", "object"}), "color"),
]

DETERMINERS = ["the"]  # keep stable to avoid number-marking confounds (a/an)

PP_FRAMES: list[PPFrame] = [
    PPFrame("animal", "in", "vehicle"),
    PPFrame("animal", "in", "location"),
    PPFrame("animal", "near", "furniture"),
    PPFrame("person", "in", "vehicle"),
    PPFrame("person", "in", "location"),
    PPFrame("person", "near", "furniture"),
    PPFrame("vehicle", "near", "location"),
    PPFrame("furniture", "in", "vehicle"),
    PPFrame("furniture", "in", "location"),
    PPFrame("container", "in", "vehicle"),
    PPFrame("container", "in", "location"),
    PPFrame("container", "near", "furniture"),
    PPFrame("object", "on", "furniture"),
    PPFrame("object", "near", "furniture"),
    PPFrame("object", "in", "container"),
    # same-class "near" frames: two distinct nouns of the same class
    PPFrame("animal", "near", "animal"),
    PPFrame("person", "near", "person"),
    PPFrame("vehicle", "near", "vehicle"),
    PPFrame("furniture", "near", "furniture"),
    PPFrame("container", "near", "container"),
    PPFrame("object", "near", "object"),
    PPFrame("location", "near", "location"),
]

COMPLEMENTIZER = "that"

nouns = []
for noun in NOUNS:
    nouns.append(noun.singular)
    nouns.append(noun.plural)

verbs = []
for verb in VERBS:
    verbs.append(verb.singular)
    verbs.append(verb.plural)

adjectives = [adj.form for adj in ADJECTIVES]

prepositions = sorted({frame.prep for frame in PP_FRAMES})

VOCABULARY = sorted(nouns + verbs + adjectives + COPULA + [COMPLEMENTIZER] + DETERMINERS + prepositions)

# ---- indexes --------------------------------------------------------------

def index_lexicon():
    nouns_by_type: dict[str, list[Noun]] = defaultdict(list)
    for n in NOUNS:
        nouns_by_type[n.semtype].append(n)

    verbs_by_subj: dict[str, list[Verb]] = defaultdict(list)
    for v in VERBS:
        for t in v.subj_types:
            verbs_by_subj[t].append(v)

    adjs_by_mod: dict[str, list[Adjective]] = defaultdict(list)
    for a in ADJECTIVES:
        for t in a.modifies:
            adjs_by_mod[t].append(a)

    pp_by_head: dict[str, list[PPFrame]] = defaultdict(list)
    for f in PP_FRAMES:
        pp_by_head[f.head_type].append(f)

    return nouns_by_type, verbs_by_subj, adjs_by_mod, pp_by_head


NOUNS_BY_TYPE, VERBS_BY_SUBJ, ADJS_BY_MOD, PP_BY_HEAD = index_lexicon()


# Form -> lemma object maps, for recovering POS/number/transitivity info from
# a surface token (e.g. during nonce substitution).
NOUN_BY_FORM: dict[str, Noun] = {}
for _n in NOUNS:
    NOUN_BY_FORM[_n.singular] = _n
    NOUN_BY_FORM[_n.plural] = _n
VERB_BY_FORM: dict[tuple[str, str], Verb] = {}  # (form, number) -> Verb
for _v in VERBS:
    VERB_BY_FORM[(_v.singular, "singular")] = _v
    VERB_BY_FORM[(_v.plural, "plural")] = _v
ADJ_BY_FORM: dict[str, Adjective] = {a.form: a for a in ADJECTIVES}

# Nouns indexed by their number form (every noun appears in both lists under
# its singular surface and its plural surface). Convenient for nonce swaps.
NOUNS_BY_NUMBER: dict[str, list[Noun]] = {
    "singular": list(NOUNS),
    "plural": list(NOUNS),
}

# Verbs indexed by number form and transitivity, for nonce swaps that preserve
# both agreement form and sentence structure.
VERBS_BY_NUMBER_TRANS: dict[tuple[str, bool], list[Verb]] = defaultdict(list)
for _v in VERBS:
    VERBS_BY_NUMBER_TRANS[("singular", _v.transitive)].append(_v)
    VERBS_BY_NUMBER_TRANS[("plural", _v.transitive)].append(_v)


# ---------------------------------------------------------------------------
# Builder: accumulates tokens + metadata as the sentence is produced
# ---------------------------------------------------------------------------

@dataclass
class Builder:
    tokens: list[str] = field(default_factory=list)
    roles: dict[str, int] = field(default_factory=dict)  # role -> token index
    attractor_numbers: list[str] = field(default_factory=list)
    # One dict per content word (noun/verb/adj) in emission order:
    #   {"index": int, "pos": "noun"|"verb"|"adj",
    #    "number": "singular"|"plural"|None, "np_id": int|None}
    # np_id groups adjectives emitted inside the same NP; it is None for verbs
    # and for predicate adjectives emitted outside an NP builder.
    content_tags: list[dict] = field(default_factory=list)

    def emit(self, tok: str, role: Optional[str] = None) -> int:
        idx = len(self.tokens)
        self.tokens.append(tok)
        if role:
            self.roles[role] = idx
        return idx

    def emit_content(self, tok: str, pos: str,
                     number: Optional[str] = None,
                     role: Optional[str] = None,
                     np_id: Optional[int] = None) -> int:
        idx = self.emit(tok, role=role)
        self.content_tags.append({"index": idx, "pos": pos,
                                  "number": number, "np_id": np_id})
        return idx


# ---------------------------------------------------------------------------
# NP builder with optional adjectives
# ---------------------------------------------------------------------------

def pick_noun(semtype: Optional[str], rng: random.Random,
              exclude: "Optional[Noun | Iterable[Noun]]" = None) -> Noun:
    pool = NOUNS_BY_TYPE[semtype] if semtype else NOUNS
    if exclude is not None:
        ex: set[Noun] = {exclude} if isinstance(exclude, Noun) else set(exclude)
        filtered = [n for n in pool if n not in ex]
        if filtered:
            pool = filtered
    return rng.choice(pool)


def pick_adjective(semtype: str, rng: random.Random,
                   exclude_forms: Optional[set[str]] = None,
                   exclude_categories: Optional[set[str]] = None) -> Optional[Adjective]:
    pool = ADJS_BY_MOD.get(semtype, [])
    if exclude_forms:
        pool = [a for a in pool if a.form not in exclude_forms]
    if exclude_categories:
        pool = [a for a in pool
                if a.category is None or a.category not in exclude_categories]
    return rng.choice(pool) if pool else None


def build_np(b: Builder,
             number: str,
             semtype: Optional[str],
             rng: random.Random,
             max_adj: int = 2,
             role: Optional[str] = None,
             exclude: "Optional[Noun | Iterable[Noun]]" = None,
             collect_adjs: Optional[list[str]] = None,
             collect_adj_cats: Optional[list[str]] = None) -> Noun:
    """Emit 'Det (Adj)* N' and return the chosen noun.

    Within an NP, no adjective form repeats, and at most one adjective per
    category appears (so you never get "small large" or "blue red"). If
    `collect_adjs` / `collect_adj_cats` are supplied, emitted forms / category
    names are appended to them (used for cross-NP dedup, e.g. preventing a
    copula predicate from echoing a subject-NP color).
    """
    noun = pick_noun(semtype, rng, exclude=exclude)
    np_id = len(b.content_tags)  # a unique id for grouping this NP's content tags
    b.emit(rng.choice(DETERMINERS))
    n_adj = rng.randint(0, max_adj)
    used_forms: set[str] = set()
    used_cats: set[str] = set()
    for _ in range(n_adj):
        a = pick_adjective(noun.semtype, rng,
                           exclude_forms=used_forms,
                           exclude_categories=used_cats)
        if a is None:
            break
        b.emit_content(a.form, pos="adj", np_id=np_id)
        used_forms.add(a.form)
        if a.category is not None:
            used_cats.add(a.category)
        if collect_adjs is not None:
            collect_adjs.append(a.form)
        if collect_adj_cats is not None and a.category is not None:
            collect_adj_cats.append(a.category)
    b.emit_content(noun.form(number), pos="noun", number=number,
                   role=role, np_id=np_id)
    return noun


# ---------------------------------------------------------------------------
# Templates
#
# Each returns (metadata_dict). The Builder is mutated in place.
# ---------------------------------------------------------------------------

def tpl_simple(b: Builder, rng: random.Random, subj_number: str) -> dict:
    """Det (Adj)* N V."""
    subj = build_np(b, subj_number, None, rng, role="subject")
    verb = rng.choice(VERBS_BY_SUBJ[subj.semtype])
    b.emit_content(verb.form(subj_number), pos="verb",
                   number=subj_number, role="main_verb")
    if verb.transitive:
        obj_type = rng.choice(list(verb.obj_types))
        obj_number = rng.choice(["singular", "plural"])
        build_np(b, obj_number, obj_type, rng, role="object")
    return {
        "template": "simple",
        "intervening_phrase_type": "none",
        "n_attractors": 0,
        "verb_lemma_pair": (verb.singular, verb.plural),
    }


def tpl_pp(b: Builder, rng: random.Random, subj_number: str,
           n_pps: int = 1) -> dict:
    """Det N (PP)+ V with licensed PPs."""
    # subject semtype must license at least one PP
    head_types = [t for t in NOUNS_BY_TYPE if PP_BY_HEAD.get(t)]
    subj_type = rng.choice(head_types)
    subj = build_np(b, subj_number, subj_type, rng, max_adj=1, role="subject")

    current_head_type = subj.semtype
    chain_nouns: set[Noun] = {subj}
    attractor_numbers: list[str] = []
    for _ in range(n_pps):
        frames = PP_BY_HEAD.get(current_head_type, [])
        if not frames:
            break
        frame = rng.choice(frames)
        b.emit(frame.prep)
        attr_number = rng.choice(["singular", "plural"])
        attractor_numbers.append(attr_number)
        # Exclude every noun already in the chain, so same-class "near" (and
        # any other preposition that happens to re-pick a prior lexeme) never
        # yields "the fields near the houses near the field".
        attr = build_np(b, attr_number, frame.comp_type, rng, max_adj=0,
                        exclude=chain_nouns)
        current_head_type = attr.semtype
        chain_nouns.add(attr)

    verb = rng.choice(VERBS_BY_SUBJ[subj.semtype])
    b.emit_content(verb.form(subj_number), pos="verb",
                   number=subj_number, role="main_verb")
    if verb.transitive:
        obj_type = rng.choice(list(verb.obj_types))
        obj_number = rng.choice(["singular", "plural"])
        build_np(b, obj_number, obj_type, rng, max_adj=0, role="object",
                 exclude=chain_nouns)

    b.attractor_numbers = attractor_numbers
    return {
        "template": f"pp_x{n_pps}",
        "intervening_phrase_type": "PP",
        "n_attractors": len(attractor_numbers),
        "verb_lemma_pair": (verb.singular, verb.plural),
    }


def tpl_relative_clause(b: Builder, rng: random.Random,
                        subj_number: str) -> dict:
    """Det (Adj)* N [that V_emb (NP_obj)?] V (NP)?

    Subject-extracted RC: the embedded subject slot is a gap bound to the main
    NP, so no embedded subject is emitted. The embedded verb agrees with the
    main subject's number. The only possible attractor is the embedded object.
    """
    subj = build_np(b, subj_number, None, rng, max_adj=1, role="subject")
    b.emit(COMPLEMENTIZER)

    # Embedded verb must license main subject's semtype as its subject.
    emb_verb_pool = VERBS_BY_SUBJ[subj.semtype]
    embedded_verb = rng.choice(emb_verb_pool)
    b.emit_content(embedded_verb.form(subj_number), pos="verb",
                   number=subj_number)

    attractor_numbers: list[str] = []
    if embedded_verb.transitive:
        obj_type = rng.choice(list(embedded_verb.obj_types))
        obj_number = rng.choice(["singular", "plural"])
        attractor_numbers.append(obj_number)
        build_np(b, obj_number, obj_type, rng, max_adj=0)

    # Main verb may repeat the embedded verb's form; resample to avoid awkwardness.
    main_verb_pool = [v for v in emb_verb_pool if v is not embedded_verb] or emb_verb_pool
    main_verb = rng.choice(main_verb_pool)
    b.emit_content(main_verb.form(subj_number), pos="verb",
                   number=subj_number, role="main_verb")
    if main_verb.transitive:
        obj_type = rng.choice(list(main_verb.obj_types))
        obj_number = rng.choice(["singular", "plural"])
        build_np(b, obj_number, obj_type, rng, max_adj=0, role="object")

    b.attractor_numbers = attractor_numbers
    return {
        "template": "relative_clause",
        "intervening_phrase_type": "RC",
        "n_attractors": len(attractor_numbers),
        "verb_lemma_pair": (main_verb.singular, main_verb.plural),
    }


def _emit_copula_predicate(b: Builder, rng: random.Random,
                           subj: Noun,
                           exclude_adj_forms: Optional[set[str]] = None,
                           exclude_adj_cats: Optional[set[str]] = None,
                           exclude_nouns: Optional[set[Noun]] = None) -> None:
    """Emit an AP or licensed PP as a copula predicate.

    Adjective predicates will not echo any form in `exclude_adj_forms` or any
    category in `exclude_adj_cats` (typically the subject NP's adjectives and
    their categories, so "the blue phone is red" doesn't occur). A PP
    predicate's complement will not reuse any noun in `exclude_nouns`
    (typically the subject + its attractor chain, so pp_copula sentences don't
    end with "near the helicopters" when "helicopters" already appeared).
    """
    exclude_adj_forms = exclude_adj_forms or set()
    exclude_adj_cats = exclude_adj_cats or set()
    exclude_nouns = exclude_nouns or set()
    adj_pool = [a for a in ADJS_BY_MOD.get(subj.semtype, [])
                if a.form not in exclude_adj_forms
                and (a.category is None or a.category not in exclude_adj_cats)]
    pp_pool = PP_BY_HEAD.get(subj.semtype, [])
    choices = []
    if adj_pool: choices.append("adj")
    if pp_pool:  choices.append("pp")
    pred_kind = rng.choice(choices) if choices else "adj"

    if pred_kind == "adj" and adj_pool:
        b.emit_content(rng.choice(adj_pool).form, pos="adj")
    elif pred_kind == "pp" and pp_pool:
        frame = rng.choice(pp_pool)
        b.emit(frame.prep)
        comp_number = rng.choice(["singular", "plural"])
        build_np(b, comp_number, frame.comp_type, rng, max_adj=0,
                 exclude=exclude_nouns)


def tpl_object_rc(b: Builder, rng: random.Random,
                  subj_number: str) -> dict:
    """Det (Adj)* N [that Det N V_trans] V (NP)?

    Object-extracted RC: the embedded verb is transitive and its object slot is
    gapped — coreferential with the main subject. The embedded subject is the
    sole attractor, matching Linzen's hardest configuration.
    """
    # Main subject semtype must be a valid object for some transitive verb,
    # and must itself license at least one main verb.
    valid_subj_types = [
        t for t in NOUNS_BY_TYPE
        if VERBS_BY_SUBJ.get(t)
        and any(v.transitive and t in v.obj_types for v in VERBS)
    ]
    subj_type = rng.choice(valid_subj_types)
    subj = build_np(b, subj_number, subj_type, rng, max_adj=1, role="subject")

    b.emit(COMPLEMENTIZER)

    # Embedded verb: transitive, accepts the main subject's semtype as object.
    emb_verb_pool = [v for v in VERBS
                     if v.transitive and subj.semtype in v.obj_types]
    emb_verb = rng.choice(emb_verb_pool)

    # Embedded subject must license emb_verb as its subject, and must be a
    # different lexeme than the main subject when they share a semtype.
    emb_subj_type = rng.choice(list(emb_verb.subj_types))
    emb_subj_number = rng.choice(["singular", "plural"])
    exclude = subj if emb_subj_type == subj.semtype else None
    build_np(b, emb_subj_number, emb_subj_type, rng, max_adj=0, exclude=exclude)
    b.emit_content(emb_verb.form(emb_subj_number), pos="verb",
                   number=emb_subj_number)
    # No overt embedded object — the object slot is the gap bound to the main subject.

    main_verb = rng.choice(VERBS_BY_SUBJ[subj.semtype])
    b.emit_content(main_verb.form(subj_number), pos="verb",
                   number=subj_number, role="main_verb")
    if main_verb.transitive:
        obj_type = rng.choice(list(main_verb.obj_types))
        obj_number = rng.choice(["singular", "plural"])
        build_np(b, obj_number, obj_type, rng, max_adj=0, role="object")

    b.attractor_numbers = [emb_subj_number]
    return {
        "template": "object_rc",
        "intervening_phrase_type": "RC",
        "n_attractors": 1,
        "verb_lemma_pair": (main_verb.singular, main_verb.plural),
    }


def tpl_copula(b: Builder, rng: random.Random, subj_number: str) -> dict:
    """Det (Adj)* N COP PRED, where PRED is an AP or a licensed PP."""
    subj_adjs: list[str] = []
    subj_adj_cats: list[str] = []
    subj = build_np(b, subj_number, None, rng, max_adj=1, role="subject",
                    collect_adjs=subj_adjs, collect_adj_cats=subj_adj_cats)

    cop_sg, cop_pl = COPULA
    b.emit_content(cop_sg if subj_number == "singular" else cop_pl, pos="verb", number=subj_number, role="main_verb")
    _emit_copula_predicate(b, rng, subj,
                           exclude_adj_forms=set(subj_adjs),
                           exclude_adj_cats=set(subj_adj_cats),
                           exclude_nouns={subj})

    # Nothing between subject and verb, so no attractors.
    return {
        "template": "copula",
        "intervening_phrase_type": "none",
        "n_attractors": 0,
        "verb_lemma_pair": COPULA,
    }


def tpl_pp_copula(b: Builder, rng: random.Random, subj_number: str,
                  n_pps: int = 1) -> dict:
    """Det (Adj)* N (PP)+ COP PRED. The PP chain inserts attractors."""
    # Subject must license at least one PP, same constraint as tpl_pp.
    head_types = [t for t in NOUNS_BY_TYPE if PP_BY_HEAD.get(t)]
    subj_type = rng.choice(head_types)
    subj_adjs: list[str] = []
    subj_adj_cats: list[str] = []
    subj = build_np(b, subj_number, subj_type, rng, max_adj=1, role="subject",
                    collect_adjs=subj_adjs, collect_adj_cats=subj_adj_cats)

    current_head_type = subj.semtype
    chain_nouns: set[Noun] = {subj}
    attractor_numbers: list[str] = []
    for _ in range(n_pps):
        frames = PP_BY_HEAD.get(current_head_type, [])
        if not frames:
            break
        frame = rng.choice(frames)
        b.emit(frame.prep)
        attr_number = rng.choice(["singular", "plural"])
        attractor_numbers.append(attr_number)
        attr = build_np(b, attr_number, frame.comp_type, rng, max_adj=0,
                        exclude=chain_nouns)
        current_head_type = attr.semtype
        chain_nouns.add(attr)

    cop_sg, cop_pl = COPULA
    b.emit_content(cop_sg if subj_number == "singular" else cop_pl, pos="verb", number=subj_number, role="main_verb")
    _emit_copula_predicate(b, rng, subj,
                           exclude_adj_forms=set(subj_adjs),
                           exclude_adj_cats=set(subj_adj_cats),
                           exclude_nouns=chain_nouns)

    b.attractor_numbers = attractor_numbers
    return {
        "template": f"pp_copula_x{n_pps}",
        "intervening_phrase_type": "PP",
        "n_attractors": len(attractor_numbers),
        "verb_lemma_pair": COPULA,
    }


TEMPLATES = {
    "simple":          lambda b, r, n: tpl_simple(b, r, n),
    "pp_x1":           lambda b, r, n: tpl_pp(b, r, n, n_pps=1),
    "pp_x2":           lambda b, r, n: tpl_pp(b, r, n, n_pps=2),
    "relative_clause": lambda b, r, n: tpl_relative_clause(b, r, n),
    "object_rc":       lambda b, r, n: tpl_object_rc(b, r, n),
    "copula":          lambda b, r, n: tpl_copula(b, r, n),
    "pp_copula_x1":    lambda b, r, n: tpl_pp_copula(b, r, n, n_pps=1),
    "pp_copula_x2":    lambda b, r, n: tpl_pp_copula(b, r, n, n_pps=2),
}


# ---------------------------------------------------------------------------
# Sentence record
# ---------------------------------------------------------------------------

def generate_sentence(template_name: str, subj_number: str,
                      rng: random.Random) -> dict:
    b = Builder()
    tpl_meta = TEMPLATES[template_name](b, rng, subj_number)

    verb_sg, verb_pl = tpl_meta.pop("verb_lemma_pair")
    verb_correct = verb_sg if subj_number == "singular" else verb_pl
    verb_incorrect = verb_pl if subj_number == "singular" else verb_sg

    return {
        "tokens": b.tokens,
        "length": len(b.tokens),
        "subject_number": subj_number,
        "subject_position": b.roles.get("subject"),
        "verb_position": b.roles.get("main_verb"),
        "verb_correct": verb_correct,
        "verb_incorrect": verb_incorrect,
        "attractor_numbers": b.attractor_numbers,
        "last_attractor_mismatch": (
            b.attractor_numbers[-1] != subj_number if b.attractor_numbers else None
        ),
        "content_tags": b.content_tags,
        **tpl_meta,
    }


# ---------------------------------------------------------------------------
# Nonce substitution (Gulordava et al. 2018)
#
# Swap every content word (noun / verb / adj) for a random same-POS word that
# preserves the morphosyntactic features needed to keep the sentence
# grammatical, but is otherwise free of semantic constraints:
#   - noun: same number, DIFFERENT semtype
#   - verb: same number form, same transitivity (so the sentence shape stays
#           valid when a transitive verb's object NP is still present)
#   - adj:  any adjective form, with intra-NP form dedup via np_id
# ---------------------------------------------------------------------------

def make_nonce(record: dict, rng: random.Random) -> dict:
    """Return a nonce copy of `record` with content words substituted.

    Preserves grammatical structure and all metadata that depends only on
    structure (positions, numbers, attractor list). Rebuilds `verb_correct` /
    `verb_incorrect` from the new main verb's lemma pair.
    """
    tokens = list(record["tokens"])
    tags = record["content_tags"]
    # Track adjective forms already emitted in each NP so we don't produce
    # "small small" after a swap.
    used_adj_forms_per_np: dict[int, set[str]] = defaultdict(set)
    new_main_verb: Optional[Verb] = None

    for tag in tags:
        idx = tag["index"]
        original = tokens[idx]
        pos = tag["pos"]

        if pos == "noun":
            orig_noun = NOUN_BY_FORM.get(original)
            number = tag["number"]
            if orig_noun is None:
                continue
            pool = [n for n in NOUNS_BY_NUMBER[number]
                    if n.semtype != orig_noun.semtype]
            new_noun = rng.choice(pool) if pool else orig_noun
            tokens[idx] = new_noun.form(number)

        elif pos == "verb":
            number = tag["number"]
            orig_verb = VERB_BY_FORM.get((original, number))
            if orig_verb is None:
                continue
            pool = [v for v in VERBS_BY_NUMBER_TRANS[(number, orig_verb.transitive)]
                    if v is not orig_verb]
            new_verb = rng.choice(pool) if pool else orig_verb
            tokens[idx] = new_verb.form(number)
            if idx == record.get("verb_position"):
                new_main_verb = new_verb

        elif pos == "adj":
            np_id = tag.get("np_id")
            used = used_adj_forms_per_np[np_id] if np_id is not None else set()
            pool = [a for a in ADJECTIVES
                    if a.form != original and a.form not in used]
            if not pool:
                pool = [a for a in ADJECTIVES if a.form != original] or ADJECTIVES
            new_adj = rng.choice(pool)
            tokens[idx] = new_adj.form
            if np_id is not None:
                used_adj_forms_per_np[np_id].add(new_adj.form)

    nonce = dict(record)
    nonce["tokens"] = tokens
    nonce["is_nonce"] = True
    if new_main_verb is not None:
        nonce["verb_correct"] = new_main_verb.form(record["subject_number"])
        nonce["verb_incorrect"] = new_main_verb.form(
            "plural" if record["subject_number"] == "singular" else "singular"
        )
    return nonce


# ---------------------------------------------------------------------------
# CLI helper: sample and print sentences by template
# ---------------------------------------------------------------------------

def sample_sentences(n_per_template: int = 3,
                     templates: Optional[list[str]] = None,
                     seed: Optional[int] = None) -> None:
    """Print n_per_template sampled sentences for each template, with the
    Gulordava-style nonce substitution shown directly below each original."""
    rng = random.Random(seed)
    templates = templates or list(TEMPLATES)
    for tpl in templates:
        print(f"\n=== {tpl} ===")
        for _ in range(n_per_template):
            number = rng.choice(["singular", "plural"])
            rec = generate_sentence(tpl, number, rng)
            nonce = make_nonce(rec, rng)
            agr = sum(1 for a in rec["attractor_numbers"]
                      if a != rec["subject_number"])

            verify_record(rec)
            verify_record(nonce)

            header = (f"[subj={rec['subject_number']:8}]"
                      f" [n_attr={rec['n_attractors']}, agr_mismatch={agr}]")
            print(f"  {header}")
            print(f"    ORIG : {' '.join(rec['tokens'])}")
            print(f"    NONCE: {' '.join(nonce['tokens'])}")


def verify_record(rec: dict) -> None:
    """Assert structural invariants on a sentence record (original or nonce).

    These are properties of the record itself; nothing is re-generated.
    """
    verb_pos = rec["verb_position"]
    subj_pos = rec["subject_position"]
    assert subj_pos is not None and verb_pos is not None, \
        "subject and main verb positions must be set"
    assert subj_pos < verb_pos, \
        f"subject at {subj_pos} should precede verb at {verb_pos}"
    assert rec["tokens"][verb_pos] == rec["verb_correct"], \
        f"token at verb_position ({rec['tokens'][verb_pos]!r}) " \
        f"must equal verb_correct ({rec['verb_correct']!r})"
    assert rec["verb_correct"] != rec["verb_incorrect"], \
        "verb_correct and verb_incorrect must differ"
    assert len(rec["attractor_numbers"]) == rec["n_attractors"], \
        f"attractor_numbers length ({len(rec['attractor_numbers'])}) " \
        f"must equal n_attractors ({rec['n_attractors']})"
    assert rec["subject_number"] in ("singular", "plural")
    for a in rec["attractor_numbers"]:
        assert a in ("singular", "plural")


if __name__ == "__main__":
    sample_sentences(n_per_template=4, seed=0)
