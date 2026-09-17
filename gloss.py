"""ASL gloss → English — Project Veronica stage 7.

A sign recogniser outputs **gloss**: the sequence of signs that were made.
Gloss is not English, and rendering it as though it were is a mistranslation
rather than a rough edge.

    YESTERDAY ME GO STORE        ->  Yesterday I went to the store.
    YOU NAME WHAT                ->  What is your name?
    ME TIRED                     ->  I am tired.
    ME UNDERSTAND NOT            ->  I do not understand.

ASL differs from English in ways this module has to undo:

* **No copula.** `ME TIRED` has no word for "am"; English requires one.
* **Tense is a time marker, not an inflection.** `YESTERDAY ME GO` marks the
  whole utterance past; the verb itself does not change. English inflects.
* **Wh-words go at the end** (or both ends), where English fronts them.
* **Topic-comment order**, so the object is often fronted.
* **No articles.** English needs "the", "a".

Why rules and not a model
-------------------------
Over a closed vocabulary of ~68 signs, a template-based transformer is honest,
debuggable, and — the part that matters — **degrades gracefully**. When it
cannot parse something it returns the gloss, which is still readable: a Deaf
signer reading "ME GO STORE" loses nothing, while a fluent-sounding English
sentence that says the wrong thing is worse than no translation at all. A
learned translator needs a parallel corpus this project does not have and
fails in the opposite direction, by inventing fluent text.

What this deliberately does not attempt
---------------------------------------
Real ASL grammar is much richer than this, and pretending otherwise in a
thesis would be the same mistake as quoting a shuffled-split accuracy:

* **Aspect** — marked by modifying a verb's movement (repeated, continuous,
  habitual). Stage 3 records the movement but nothing maps it to meaning.
* **Spatial agreement** — verbs move between points in signing space that were
  assigned to referents earlier in the conversation.
* **Classifiers** — handshapes standing in for objects, with productive
  meaning from their movement.
* **Role shift** — a body shift marking reported speech.
* **Non-manual grammar** — stage 8; a brow raise makes a statement a question,
  so `render()` takes an explicit `question` flag for stage 8 to fill in.

Rendering these as plain declaratives is a known, documented limitation, not a
claim of coverage.

Standard library only, so it runs in CI. Spelling correction stays in
nlp_bridge.py, which needs SymSpell: grammar and spelling are separate jobs and
only one of them is testable without a third-party dictionary.
"""
import re

# ── lexicon ────────────────────────────────────────────────────────────────
# Whole utterances in themselves. Nothing is built around these; they are
# produced as-is and end the sentence.
FIXED = {
    "HELLO": "hello",
    "GOODBYE": "goodbye",
    "PLEASE": "please",
    "THANK-YOU": "thank you",
    "EXCUSE-ME": "excuse me",
    "NICE-TO-MEET-YOU": "nice to meet you",
    "YES": "yes",
    "NO": "no",
    "MAYBE": "maybe",
    "AGAIN": "again",
}

# HE-SHE renders as "they". The ASL sign is a point, which carries no gender:
# choosing "he" or "she" would invent information the signer did not give,
# every time. Singular "they" keeps exactly what was signed.
PRONOUN = {
    "ME": {"subject": "I", "object": "me", "possessive": "my",
           "be": "am", "past_be": "was", "third": False},
    "YOU": {"subject": "you", "object": "you", "possessive": "your",
            "be": "are", "past_be": "were", "third": False},
    "HE-SHE": {"subject": "they", "object": "them", "possessive": "their",
               "be": "are", "past_be": "were", "third": False},
    "WE": {"subject": "we", "object": "us", "possessive": "our",
           "be": "are", "past_be": "were", "third": False},
    "THEY": {"subject": "they", "object": "them", "possessive": "their",
             "be": "are", "past_be": "were", "third": False},
}

# Irregular forms written out rather than derived. Fifteen verbs is few enough
# that a table cannot be wrong in a way a suffix rule can.
VERB = {
    "GO": {"base": "go", "third": "goes", "past": "went", "ing": "going"},
    "COME": {"base": "come", "third": "comes", "past": "came", "ing": "coming"},
    "EAT": {"base": "eat", "third": "eats", "past": "ate", "ing": "eating"},
    "DRINK": {"base": "drink", "third": "drinks", "past": "drank", "ing": "drinking"},
    "WORK": {"base": "work", "third": "works", "past": "worked",
             "ing": "working", "noun": "work"},
    "WANT": {"base": "want", "third": "wants", "past": "wanted", "ing": "wanting"},
    "NEED": {"base": "need", "third": "needs", "past": "needed", "ing": "needing"},
    "HELP": {"base": "help", "third": "helps", "past": "helped",
             "ing": "helping", "noun": "help"},
    "STOP": {"base": "stop", "third": "stops", "past": "stopped", "ing": "stopping"},
    "WAIT": {"base": "wait", "third": "waits", "past": "waited", "ing": "waiting"},
    "FINISH": {"base": "finish", "third": "finishes", "past": "finished", "ing": "finishing"},
    "UNDERSTAND": {"base": "understand", "third": "understands",
                   "past": "understood", "ing": "understanding"},
    "LOVE": {"base": "love", "third": "loves", "past": "loved", "ing": "loving"},
    "LIKE": {"base": "like", "third": "likes", "past": "liked", "ing": "liking"},
    "HURT": {"base": "hurt", "third": "hurts", "past": "hurt", "ing": "hurting"},
}

ADJECTIVE = {
    # SORRY behaves like an adjective, not a set phrase: ASL's ME SORRY is
    # "I am sorry", and treating it as fixed produced "Sorry, I."
    "SORRY": "sorry",
    "GOOD": "good", "BAD": "bad", "HAPPY": "happy", "SAD": "sad",
    "TIRED": "tired", "SICK": "sick", "HUNGRY": "hungry",
    "THIRSTY": "thirsty", "DEAF": "deaf", "HEARING": "hearing",
    "SLOW": "slow", "FAST": "fast",
}

# `article` is what the noun takes as an object: "the store", "water" (mass),
# "my mother" (taken by the possessive instead). `motion` marks a destination,
# so GO/COME take "to" before it.
NOUN = {
    "MOTHER": {"en": "mother", "article": "", "kin": True},
    "FATHER": {"en": "father", "article": "", "kin": True},
    "FRIEND": {"en": "friend", "article": "a", "kin": True},
    "FAMILY": {"en": "family", "article": "the", "kin": True},
    "NAME": {"en": "name", "article": "the"},
    "SCHOOL": {"en": "school", "article": "", "motion": True},
    "HOME": {"en": "home", "article": "", "motion": True},
    "BATHROOM": {"en": "bathroom", "article": "the", "motion": True},
    "WATER": {"en": "water", "article": ""},
    "MONEY": {"en": "money", "article": ""},
}

# Tense comes from the time marker, because ASL puts it there and not on the
# verb. `front` is whether English reads better with it at the start.
TIME = {
    "NOW": {"en": "now", "tense": "present", "front": False},
    "TODAY": {"en": "today", "tense": "present", "front": True},
    "TOMORROW": {"en": "tomorrow", "tense": "future", "front": True},
    "YESTERDAY": {"en": "yesterday", "tense": "past", "front": True},
    "LATER": {"en": "later", "tense": "future", "front": False},
}

WH = {
    "WHAT": "what", "WHERE": "where", "WHEN": "when",
    "WHO": "who", "WHY": "why", "HOW": "how",
}

NEGATIVE = {"NOT", "NO", "DONT-KNOW", "DONT-UNDERSTAND"}

# Glosses that are a negated verb in one sign.
NEGATED_VERB = {
    "DONT-KNOW": "know",
    "DONT-UNDERSTAND": "understand",
}

MORE = "MORE"
REST = "_REST"


def is_letter(token):
    return len(token) == 1 and token.isalpha()


def _capitalize(sentence):
    return sentence[0].upper() + sentence[1:] if sentence else sentence


def _join(parts):
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


# ── fingerspelling ─────────────────────────────────────────────────────────
def group_tokens(tokens):
    """Collapse runs of single letters into one fingerspelled token.

    Fingerspelling is how names and unfamiliar words arrive in a real
    conversation, so a Veronica transcript is a mix of sign glosses and letter
    runs. Spelling correction is nlp_bridge.py's job -- this only decides where
    a spelled word starts and stops.
    """
    grouped = []
    letters = []
    for token in tokens:
        if is_letter(token):
            letters.append(token)
        else:
            if letters:
                grouped.append("".join(letters))
                letters = []
            grouped.append(token)
    if letters:
        grouped.append("".join(letters))
    return grouped


# ── parsing ────────────────────────────────────────────────────────────────
def _classify(token):
    if token in PRONOUN:
        return "pronoun"
    if token in VERB:
        return "verb"
    if token in ADJECTIVE:
        return "adjective"
    if token in NOUN:
        return "noun"
    if token in TIME:
        return "time"
    if token in WH:
        return "wh"
    if token in FIXED:
        return "fixed"
    if token in NEGATIVE:
        return "negative"
    return "unknown"


def parse(tokens):
    """Gloss tokens -> a shallow description of the utterance.

    Deliberately shallow. Anything it cannot place goes in `leftover`, and
    render() falls back to the gloss rather than guessing -- an utterance this
    does not understand should look untranslated, not look wrong.
    """
    parsed = {
        "time": None, "tense": "present", "subject": None, "verb": None,
        "adjective": None, "objects": [], "wh": None, "negated": False,
        "fixed": [], "spelled": [], "leftover": [], "intensified": False,
    }

    for token in group_tokens(tokens):
        if token == REST:
            continue
        kind = _classify(token)

        if kind == "time":
            parsed["time"] = token
            parsed["tense"] = TIME[token]["tense"]
        elif kind == "wh":
            parsed["wh"] = token
        elif kind == "negative":
            parsed["negated"] = True
            if token in NEGATED_VERB:
                parsed["verb"] = token
            elif token in FIXED and parsed["verb"] is None:
                parsed["fixed"].append(token)
        elif kind == "pronoun":
            if parsed["subject"] is None:
                parsed["subject"] = token
            else:
                parsed["objects"].append(token)
        elif kind == "verb":
            if parsed["verb"] is None:
                parsed["verb"] = token
            else:
                # A second verb is a complement, not a replacement:
                # ME NEED HELP is "I need help", not "I help". Overwriting
                # here silently dropped the governing verb.
                parsed["objects"].append(token)
        elif kind == "adjective":
            parsed["adjective"] = token
        elif kind == "noun":
            if parsed["subject"] is None and parsed["verb"] is None \
                    and parsed["adjective"] is None:
                parsed["subject"] = token
            else:
                parsed["objects"].append(token)
        elif kind == "fixed":
            parsed["fixed"].append(token)
        elif token == MORE:
            parsed["intensified"] = True
        elif len(token) > 1 and token.isalpha() and token.isupper() \
                and token not in FIXED:
            # A run of fingerspelled letters: a name or an unknown word.
            parsed["spelled"].append(token)
        else:
            parsed["leftover"].append(token)

    return parsed


# ── rendering ──────────────────────────────────────────────────────────────
def _subject_words(parsed):
    subject = parsed["subject"]
    if subject in PRONOUN:
        entry = PRONOUN[subject]
        return entry["subject"], entry["third"], entry
    if subject in NOUN:
        return _noun_phrase(subject, possessive=None), True, None
    if subject:
        return subject.title(), True, None
    return None, False, None


def _noun_phrase(token, possessive=None):
    entry = NOUN.get(token)
    if entry is None:
        # Not a noun at all. Callers should not reach here, but a renderer
        # that raises turns a mistranslation into a dead pipeline, and the
        # gloss is still readable.
        return f"{possessive} {token.lower()}" if possessive else token.lower()
    if possessive:
        return f"{possessive} {entry['en']}"
    article = entry.get("article")
    return f"{article} {entry['en']}".strip() if article else entry["en"]


def _conjugate(verb_token, tense, third, negated):
    verb = VERB.get(verb_token)
    if verb is None:
        # A one-sign negated verb (DONT-KNOW): there is no positive entry.
        base = NEGATED_VERB.get(verb_token, verb_token.lower())
        verb = {"base": base, "third": base + "s",
                "past": base + "ed", "ing": base + "ing"}

    if negated:
        if tense == "past":
            return f"did not {verb['base']}"
        if tense == "future":
            return f"will not {verb['base']}"
        return f"do{'es' if third else ''} not {verb['base']}"

    if tense == "past":
        return verb["past"]
    if tense == "future":
        return f"will {verb['base']}"
    return verb["third"] if third else verb["base"]


def _object_words(parsed, verb_token):
    """Objects, with the article and the 'to' that a destination needs."""
    words = []
    # A destination takes "to" whenever a motion verb governs it anywhere in
    # the clause, not only when it is the main verb: ME NEED GO BATHROOM has
    # GO sitting in object position as a complement of NEED.
    motion = (verb_token in ("GO", "COME")
              or any(o in ("GO", "COME") for o in parsed["objects"]))
    for token in parsed["objects"]:
        if token in PRONOUN:
            words.append(PRONOUN[token]["object"])
            continue
        if token in VERB:
            # A verb in object position is a complement. It takes its noun
            # form where it has one and an infinitive otherwise, so both
            # ME NEED HELP and ME WANT GO come out as English.
            words.append(VERB[token].get("noun") or f"to {VERB[token]['base']}")
            continue
        entry = NOUN.get(token)
        if entry is None:
            words.append(token.lower())
            continue
        possessive = None
        if entry.get("kin") and parsed["subject"] in PRONOUN:
            possessive = PRONOUN[parsed["subject"]]["possessive"]
        phrase = _noun_phrase(token, possessive)
        if motion and entry.get("motion") and not phrase.startswith("home"):
            phrase = f"to {phrase}"
        elif motion and entry.get("motion"):
            phrase = phrase                      # "go home", never "go to home"
        words.append(phrase)
    return words


def _render_question(parsed):
    """Wh-questions. ASL puts the wh-word at the end; English fronts it."""
    wh = WH[parsed["wh"]]
    subject_words, third, pronoun = _subject_words(parsed)

    # "YOU NAME WHAT" -- the wh-word is asking about a possessed noun, so the
    # subject is really the possessor. This is the single most common question
    # in a first conversation, which earns it a case of its own.
    if (parsed["wh"] == "WHAT" and pronoun and parsed["objects"]
            and parsed["verb"] is None and parsed["adjective"] is None):
        owned = _noun_phrase(parsed["objects"][0], pronoun["possessive"])
        be = "was" if parsed["tense"] == "past" else "is"
        return _join([wh, be, owned]) + "?"

    if parsed["verb"]:
        verb = VERB.get(parsed["verb"], {"base": parsed["verb"].lower()})
        auxiliary = {"past": "did", "future": "will"}.get(
            parsed["tense"], "does" if third else "do")
        return _join([wh, auxiliary, subject_words, verb["base"]]
                     + _object_words(parsed, parsed["verb"])) + "?"

    if parsed["adjective"]:
        be = _be(parsed, third, pronoun)
        return _join([wh, be, subject_words, ADJECTIVE[parsed["adjective"]]]) + "?"

    if subject_words:
        be = _be(parsed, third, pronoun)
        return _join([wh, be, subject_words]) + "?"
    return wh + "?"


def _be(parsed, third, pronoun):
    past = parsed["tense"] == "past"
    if pronoun:
        return pronoun["past_be"] if past else pronoun["be"]
    return ("was" if past else "is") if third else ("were" if past else "are")


def render(tokens, question=False):
    """Gloss tokens -> an English sentence.

    `question` is for stage 8: a brow raise turns a statement into a yes/no
    question with no change to the hands at all, so the flag exists now and
    the facial channel fills it in later.
    """
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""

    parsed = parse(tokens)

    # Nothing but set phrases and spelled words: say them and stop. Building a
    # clause around "HELLO" would invent structure that was not signed.
    has_clause = any(parsed[key] for key in ("subject", "verb", "adjective",
                                             "objects", "wh"))
    if not has_clause:
        pieces = [FIXED[t] for t in parsed["fixed"]]
        if parsed["time"]:
            pieces.append(TIME[parsed["time"]]["en"])
        pieces += [t.title() for t in parsed["spelled"]] + parsed["leftover"]
        return _capitalize(_join(pieces)) + ("?" if question else ".") \
            if pieces else ""

    if parsed["wh"]:
        sentence = _render_question(parsed)
    else:
        sentence = _render_statement(parsed, question)

    prefix = [FIXED[t] for t in parsed["fixed"]]
    if parsed["time"] and TIME[parsed["time"]]["front"]:
        prefix.append(TIME[parsed["time"]]["en"])

    if prefix:
        sentence = _join(prefix) + ", " + sentence
    return _capitalize(sentence)


def _render_statement(parsed, question):
    subject_words, third, pronoun = _subject_words(parsed)
    tail = []
    if parsed["time"] and not TIME[parsed["time"]]["front"]:
        tail.append(TIME[parsed["time"]]["en"])
    tail += [t.title() for t in parsed["spelled"]] + parsed["leftover"]

    if parsed["verb"]:
        verb = _conjugate(parsed["verb"], parsed["tense"], third,
                          parsed["negated"])
        body = _join([subject_words, verb]
                     + _object_words(parsed, parsed["verb"]) + tail)
        if question:
            # Yes/no question from a brow raise: invert rather than reorder,
            # which keeps the rest of the clause exactly as parsed.
            body = _invert(subject_words, parsed, third) or body
        return body + ("?" if question else ".")

    if parsed["adjective"]:
        adjective = ADJECTIVE[parsed["adjective"]]
        if parsed["intensified"]:
            adjective = f"very {adjective}"
        if not subject_words:
            return _join([adjective] + tail) + ("?" if question else ".")
        be = _be(parsed, third, pronoun)
        negation = " not" if parsed["negated"] else ""
        if question:
            return _join([be, subject_words, negation.strip(), adjective]
                         + tail) + "?"
        return _join([subject_words, be + negation, adjective] + tail) + \
            ("?" if question else ".")

    # Pronouns with no predicate to attach them to: English takes the object
    # forms side by side rather than "I you".
    if (pronoun and parsed["objects"]
            and all(o in PRONOUN for o in parsed["objects"])):
        names = [pronoun["object"]] + [PRONOUN[o]["object"]
                                       for o in parsed["objects"]]
        return ", ".join(names + tail) + ("?" if question else ".")

    # Subject and objects but no predicate -- most often a possessive phrase
    # ("YOU NAME"), so render the phrase rather than inventing a verb.
    if subject_words and parsed["objects"]:
        if pronoun and parsed["objects"][0] in NOUN:
            owned = " and ".join(
                _noun_phrase(o, pronoun["possessive"]) if o in NOUN
                else o.lower() for o in parsed["objects"])
            if tail:
                # "ME NAME O-M-A-R" -- a possessive phrase followed by a
                # fingerspelled word is an introduction, and English needs the
                # copula ASL does not have: "My name is Omar."
                be = "was" if parsed["tense"] == "past" else "is"
                return _join([owned, be] + tail) + ("?" if question else ".")
            return _join([owned] + tail) + ("?" if question else ".")
        return _join([subject_words] + _object_words(parsed, None) + tail) + \
            ("?" if question else ".")

    objects = _object_words(parsed, None)
    if not objects and pronoun and not tail:
        # Nothing to predicate: "Hello, me." rather than "Hello, I."
        subject_words = pronoun["object"]
    return _join([subject_words] + objects + tail) + ("?" if question else ".")


def _invert(subject_words, parsed, third):
    verb = VERB.get(parsed["verb"])
    if verb is None or not subject_words:
        return None
    auxiliary = {"past": "did", "future": "will"}.get(
        parsed["tense"], "does" if third else "do")
    negation = "not" if parsed["negated"] else ""
    return _join([auxiliary, subject_words, negation, verb["base"]]
                 + _object_words(parsed, parsed["verb"]))
