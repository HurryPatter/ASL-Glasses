"""Offline tests for ASL gloss -> English.

Standard library only, so it runs in CI.

The contract these protect is not "every sentence is perfect English" -- that
is not achievable by rules over a 68-sign vocabulary and claiming it would be
dishonest. It is:

  1. the grammatical differences that ASL *systematically* has are undone
     (no copula, tense as a time marker, wh-words at the end), and
  2. anything it cannot parse degrades to readable gloss rather than to
     confident nonsense.

(2) is the one that matters for a deployed translator. A fluent English
sentence that says the wrong thing is worse than no translation.

Run:  python -m unittest test_gloss -v
"""
import unittest

import gloss


def render(text, **kw):
    return gloss.render(text.split(), **kw)


class TestCopula(unittest.TestCase):
    """ASL has no word for 'is'. English requires one."""

    def test_pronoun_and_adjective(self):
        self.assertEqual(render("ME TIRED"), "I am tired.")
        self.assertEqual(render("YOU SICK"), "You are sick.")
        self.assertEqual(render("WE HAPPY"), "We are happy.")

    def test_noun_subject_takes_third_person(self):
        self.assertEqual(render("MOTHER SICK"), "Mother is sick.")
        self.assertEqual(render("FATHER GOOD"), "Father is good.")

    def test_past_tense_copula(self):
        self.assertEqual(render("YESTERDAY ME SICK"), "Yesterday, I was sick.")
        self.assertEqual(render("YESTERDAY YOU SAD"), "Yesterday, you were sad.")

    def test_introduction_gets_a_copula(self):
        self.assertEqual(render("ME NAME O M A R"), "My name is Omar.")
        self.assertEqual(render("HELLO ME NAME L A I L A"),
                         "Hello, my name is Laila.")


class TestTense(unittest.TestCase):
    """Tense is a time marker in ASL, not an inflection on the verb."""

    def test_the_time_marker_sets_the_tense(self):
        self.assertEqual(render("ME GO SCHOOL"), "I go to school.")
        self.assertEqual(render("YESTERDAY ME GO SCHOOL"),
                         "Yesterday, I went to school.")
        self.assertEqual(render("TOMORROW ME GO SCHOOL"),
                         "Tomorrow, I will go to school.")

    def test_irregular_pasts(self):
        self.assertEqual(render("YESTERDAY ME EAT"), "Yesterday, I ate.")
        self.assertEqual(render("YESTERDAY ME DRINK WATER"),
                         "Yesterday, I drank water.")
        self.assertEqual(render("YESTERDAY ME UNDERSTAND"),
                         "Yesterday, I understood.")

    def test_trailing_time_words_stay_at_the_end(self):
        # English puts "now" and "later" after the clause, unlike "yesterday".
        self.assertEqual(render("ME GO NOW"), "I go now.")
        self.assertEqual(render("ME EAT LATER"), "I will eat later.")

    def test_the_verb_itself_is_not_marked_in_the_gloss(self):
        # The point of the whole rule: GO is the same sign in all three.
        rendered = {render(f"{t} ME GO SCHOOL") for t in
                    ("YESTERDAY", "TODAY", "TOMORROW")}
        self.assertEqual(len(rendered), 3)


class TestQuestions(unittest.TestCase):
    """ASL puts the wh-word at the end; English fronts it."""

    def test_wh_word_moves_to_the_front(self):
        self.assertEqual(render("YOU NAME WHAT"), "What is your name?")
        self.assertTrue(render("YOU GO WHERE").startswith("Where"))
        self.assertTrue(render("YOU GO WHERE").endswith("?"))

    def test_wh_question_with_a_verb(self):
        self.assertEqual(render("YOU WANT WHAT"), "What do you want?")
        self.assertEqual(render("YESTERDAY YOU GO WHERE"),
                         "Yesterday, where did you go?")

    def test_wh_question_with_an_adjective(self):
        self.assertEqual(render("YOU WHY SAD"), "Why are you sad?")

    def test_a_bare_wh_word(self):
        self.assertEqual(render("WHY"), "Why?")


class TestNegation(unittest.TestCase):

    def test_single_sign_negated_verbs(self):
        self.assertEqual(render("ME DONT-UNDERSTAND"), "I do not understand.")
        self.assertEqual(render("ME DONT-KNOW"), "I do not know.")

    def test_negation_agrees_with_tense(self):
        self.assertEqual(render("YESTERDAY ME DONT-UNDERSTAND"),
                         "Yesterday, I did not understand.")


class TestVerbComplements(unittest.TestCase):
    """A second verb is a complement, not a replacement."""

    def test_the_governing_verb_survives(self):
        # This read "I help." before the fix -- the second verb overwrote the
        # first, silently changing what the signer said.
        self.assertEqual(render("ME NEED HELP"), "I need help.")

    def test_action_complements_become_infinitives(self):
        self.assertEqual(render("ME WANT GO"), "I want to go.")

    def test_a_destination_takes_to_through_a_complement(self):
        # GO is in object position here, so reading motion off the main verb
        # alone loses the "to".
        self.assertEqual(render("ME NEED GO BATHROOM"),
                         "I need to go to the bathroom.")

    def test_home_takes_no_preposition(self):
        self.assertEqual(render("ME GO HOME"), "I go home.")
        self.assertEqual(render("ME GO BATHROOM"), "I go to the bathroom.")


class TestPronounsAndPossessives(unittest.TestCase):

    def test_object_pronouns(self):
        self.assertEqual(render("ME LOVE YOU"), "I love you.")
        self.assertEqual(render("YOU HELP ME"), "You help me.")

    def test_kin_nouns_take_the_subjects_possessive(self):
        self.assertEqual(render("ME LOVE MOTHER"), "I love my mother.")
        self.assertEqual(render("YOU LOVE FATHER"), "You love your father.")

    def test_the_gender_neutral_point_stays_gender_neutral(self):
        # HE-SHE is a point, which carries no gender. Rendering it as "he" or
        # "she" would invent information the signer did not give, so it comes
        # out as singular "they".
        self.assertEqual(render("HE-SHE TIRED"), "They are tired.")
        self.assertEqual(render("ME HELP HE-SHE"), "I help them.")


class TestFingerspelling(unittest.TestCase):
    """Names arrive spelled, so a transcript mixes glosses and letter runs."""

    def test_letter_runs_are_grouped(self):
        self.assertEqual(gloss.group_tokens(list("OMAR")), ["OMAR"])
        self.assertEqual(gloss.group_tokens(["ME", "N", "A", "M", "E"]),
                         ["ME", "NAME"])

    def test_runs_are_split_by_signs_between_them(self):
        self.assertEqual(gloss.group_tokens(["A", "B", "GO", "C", "D"]),
                         ["AB", "GO", "CD"])

    def test_a_bare_name(self):
        self.assertEqual(render("O M A R"), "Omar.")


class TestFixedPhrases(unittest.TestCase):

    def test_set_phrases_pass_through(self):
        self.assertEqual(render("HELLO"), "Hello.")
        self.assertEqual(render("THANK-YOU"), "Thank you.")
        self.assertEqual(render("NICE-TO-MEET-YOU"), "Nice to meet you.")

    def test_no_clause_is_invented_around_them(self):
        # "Hello." and not "Hello is." -- building structure that was not
        # signed is the failure mode this module is most at risk of.
        self.assertEqual(render("HELLO"), "Hello.")

    def test_a_phrase_can_prefix_a_clause(self):
        self.assertEqual(render("HELLO ME TIRED"), "Hello, I am tired.")


class TestIntensifier(unittest.TestCase):

    def test_more_intensifies_an_adjective(self):
        self.assertEqual(render("ME HUNGRY MORE"), "I am very hungry.")


class TestNonManualFlag(unittest.TestCase):
    """Stage 8: a brow raise turns a statement into a yes/no question with no
    change to the hands, so the flag exists now for the face to fill in."""

    def test_verb_clauses_invert(self):
        self.assertEqual(render("YOU UNDERSTAND", question=True),
                         "Do you understand?")
        self.assertEqual(render("YOU WANT WATER", question=True),
                         "Do you want water?")

    def test_adjective_clauses_invert(self):
        self.assertEqual(render("YOU HUNGRY", question=True),
                         "Are you hungry?")

    def test_the_same_gloss_is_a_statement_without_the_flag(self):
        self.assertEqual(render("YOU HUNGRY"), "You are hungry.")


class TestGracefulDegradation(unittest.TestCase):
    """The contract that matters for a deployed translator.

    Unparseable input must come out as readable gloss, never as confident
    English that says something else.
    """

    def test_unknown_signs_survive_rather_than_vanishing(self):
        out = render("ME WANT ZEBRA")
        self.assertIn("zebra", out.lower())

    def test_nothing_in_nothing_out(self):
        self.assertEqual(gloss.render([]), "")
        self.assertEqual(gloss.render(["", ""]), "")

    def test_rest_tokens_are_dropped(self):
        self.assertEqual(render("_REST ME TIRED _REST"), "I am tired.")

    def test_every_vocabulary_sign_renders_without_raising(self):
        # Nothing in signset should be able to crash the renderer, including
        # signs this module has no lexicon entry for.
        import signset
        for sign in signset.SIGNS:
            self.assertIsInstance(gloss.render([sign]), str)
            self.assertIsInstance(gloss.render(["ME", sign]), str)

    def test_every_pair_of_signs_renders_without_raising(self):
        # This sweep is what caught the renderer crashing on ME YOU: the
        # possessive branch assumed its object was a noun. A translator that
        # raises on a sign combination nobody anticipated is worse than one
        # that renders it awkwardly, because it takes the whole pipeline down
        # mid-conversation.
        import itertools
        import signset

        for a, b in itertools.product(signset.SIGNS, repeat=2):
            out = gloss.render([a, b])
            self.assertIsInstance(out, str)
            if out:
                self.assertTrue(out[0].isupper(), (a, b, out))
                self.assertIn(out[-1], ".?", (a, b, out))

    def test_output_is_always_capitalised_and_terminated(self):
        import signset
        for sign in signset.SIGNS:
            out = gloss.render(["ME", sign])
            if out:
                self.assertTrue(out[0].isupper(), out)
                self.assertIn(out[-1], ".?", out)


if __name__ == "__main__":
    unittest.main()
