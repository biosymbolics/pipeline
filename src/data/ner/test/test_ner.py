import unittest

from data.ner import NerTagger


class TestNerUtils(unittest.TestCase):
    """
    from data.ner import NerTagger; tagger=NerTagger()
    t = tagger.extract([text], link=False)[0]
    [(t1[0], t1.start_char, t1.end_char) for t1 in t]
    """

    def setUp(self):
        self.tagger = NerTagger(
            entity_types=frozenset(["compounds", "diseases", "mechanisms"]),
            # rule_sets=[],
        )

    def test_ner(self):
        test_conditions = [
            {
                "text": """
                Bioenhanced formulations comprising eprosartan in oral solid dosage form.
                This invention relates to bioenhanced formulations comprising eprosartan or eprosartan mesylate in the amorphous form, a process for its production, compositions containing the compound and methods of using the compound to block angiotensin II receptors and to treat hypertension, congestive heart failure and renal failure.
                """,
                "expected_output": [
                    "eprosartan mesylate",
                    "hypertension",
                    "congestive heart failure",
                    "renal failure",
                ],
            },
            {
                "text": """
                Pharmaceutical composition in particular for preventing and treating mucositis induced by radiotherapy or chemotherapy.
                The invention concerns a pharmaceutical composition designed to adhere to a mucous membrane in particular for preventing or treating radiotherapy-related and chemotherapy-related mucositis, induced by radiotherapy or combined radiochemotherapy, comprising an efficient amount of an antiradical compound mixed with a vehicle which is liquid at room temperature and gels at the mucous membrane temperature and capable of adhering to the mucous membrane by its gelled state.
                """,
                "expected_output": [
                    "radiotherapy",
                    "chemotherapy",
                    "radiotherapy relate",
                    "chemotherapy related mucositis",
                    "radiotherapy",
                    "combined radiochemotherapy",
                    "antiradical compound",
                ],
            },
            {
                "text": """
                Novel aspartyl dipeptide ester derivatives and sweeteners.
                Novel aspartyl dipeptide ester derivatives (including salts thereof) such as N-[N-[3-(3-hydroxy-4-methoxyphenyl)propyl]-L-α-aspartyl]-L-(α-methyl)phenylalanine 1-methyl ester which are usable as sweeteners; and sweeteners, foods, etc. containing the same. These compounds are usable as low-caloric sweeteners being much superior in the degree of sweetness to the conventional ones.
                """,
                "expected_output": [
                    "novel aspartyl dipeptide ester derivative",
                    "aspartyl dipeptide ester",
                    "novel aspartyl dipeptide ester derivative",
                    "n[n[3(3 hydroxy 4 methoxyphenyl)propyl]-lα aspartyl]-l(α methyl)phenylalanine 1 methyl ester",
                ],
            },
            # inconsistent results
            # {
            #     "text": """
            #     Muscarinic antagonists
            #     Heterocyclic derivatives of di-N-substituted piperazine or 1,4 di-substituted piperidine compounds in accordance with formula (I) (including all isomers, salts and solvates), wherein one of Y and Z is -N- and the other is -N- or -CH-; X is -O-, -S-, -SO-, -SO2- or -CH2-; Q is (1), (2), (3); R is alkyl, cycloalkyl, optionally substituted aryl or heteroaryl; R?1, R2 and R3¿ are H or alkyl; R4 is alkyl, cyclolalkyl or (4); R5 is H, alkyl, -C(O)alkyl, arylcarbonyl, -SO¿2?alkyl, aryl-sulfonyl-C(O)Oalkyl, aryloxycarbonyl, -C(O)NH-alkyl or aryl-aminocarbonyl, wherein the aryl portion is optionally substituted; R?6¿ is H or alkyl; and R7 is H, alkyl, hydroxyalkyl or alkoxyalkyl; are muscarinic antagonists useful for treating cognitive disorders such as Alzheimer&#39;s disease. Pharmaceutical compositions and methods of treatment are also disclosed.
            #     """,
            #     "expected_output": [
            #         "muscarinic antagonists heterocyclic derivative",
            #         "din substituted piperazine heterocyclic derivative",
            #         "1,4 di substituted piperidine",
            #         "di substituted piperidine compound",
            #         "optionally substituted aryl",
            #         "heteroaryl",
            #         "r7 is h",
            #         "hydroxyalkyl or alkoxyalkyl; are muscarinic antagonist",
            #         "muscarinic antagonists useful",
            #         "alzheimer disease",
            #     ],
            # },
            {
                "text": """
                Method of treating meniere&#39;s disease and corresponding apparatus
                In a method of treating Ménière&#39;s disease intermittent air pressure pulse trains are administred to an outwardly sealed external ear volume bordering to a surgically perforated tympanic membrane. In a pulse train air pressure is increased from ambient (p0) to a first level (p1) and from there repeatedly to a second level (p2) and repeatedly decreased to the first level (p1), and finally decreased to ambient (p0). P1 is from 4 to 16 cm H2O, p2 is from 8 to 16 cm H2O, with the proviso that p1 &gt; p2, the pressure increase rate is from 0 to 4 mm H2O per millisecond, the pressure decrease rate is from 0 to 2 mm H2O per millisecond, the modulation frequency is from 3 to 9 Hz, the intermittent time period is from 3 to 10 seconds. Also disclosed is an apparatus for carrying out the method.
                """,
                "expected_output": [
                    "meniere disease",
                    "ménière disease intermittent air pressure pulse train",
                    "intermittent air pressure pulse train",
                    "surgically perforated tympanic membrane",
                    # the below are perhaps an indexing problem
                    # if no unescaping, we get:
                    # [..., 'h2o per millisecond']
                    "the 16 cm h2o,",  # TODO;  P1 is from 4 to 16 cm H2
                    "to 4 mm h2o per",  # TODO; the pressure increase rate is from 0 to 4 mm H2O per millisecond
                    "millisecond, the",  # TODO; the pressure increase rate is from 0 to 4 mm H2O per millisecond, the pressure decrease...
                ],
            },
            {
                "text": """
                Cox-2 inhibitors in combination with centrally acting analgesics
                A method of alleviating a pain state not associated with a cough condition is provided which comprises administering a cyclooxygenase-2 inhibitor and a centrally active analgesic selected from the group consisting of a narcotic analgesic selected from the group consisitng of codeine and hydrocodone; an agonist-antagonist analgesic and tramadol. A method and analgesic composition therefor is also provided for treating all pain states which comprises administering a cyclooxygenase-2 inhibitor and a centrally acting analgesic selected from the group consisting of a narcotic analgesic other than codeine and hydrocodone; an agonist-antagonist analgesic and tramadol.
                """,
                "expected_output": [
                    "cox2 inhibitor",
                    "centrally acting analgesic",
                    "pain state",
                    "cough condition",
                    "cyclooxygenase 2 inhibitor",
                    "narcotic analgesic",
                    "codeine",
                    "agonist antagonist analgesic",
                    "analgesic composition therefor",
                    "cyclooxygenase 2 inhibitor",
                    "than codeine and",  # TODO
                    "agonist antagonist analgesic",
                    # hydrocodone and tramadol # TODO
                ],
            },
        ]

        for condition in test_conditions:
            text = condition["text"]
            expected_output = condition["expected_output"]

            result = self.tagger.extract_strings([text], link=False)[0]

            if result != expected_output:
                print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)
