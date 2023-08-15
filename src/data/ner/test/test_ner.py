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
                    "to 4 mm h2o per",  # TODO
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
            {
                "text": """
                Biomarkers for oxidative stress
                This invention relates generally to methods of detecting and quantifying biomarkers of oxidative stress in proteins. The biomarker may be any amino acid that has undergone oxidation (or other modification, e.g. chloro-tyrosine, dityrosine). Emphasis is given herein on oxidized sulfur- or selenium-containing amino acids (SSAA). The biomarker of oxidative stress in proteins may be detected with an antibody that binds to oxidized amino acids, specifically oxidized sulfur- or selenium-containing amino acids. The antibody may be monoclonal or polyclonal. The presence of biomarker or amount of biomarker present in a sample may be used to aid in assessing the efficacy of environmental, nutritional and therapeutic interventions, among other uses.
                """,
                "expected_output": [
                    "oxidative stress",
                    "protein",
                    "dityrosine",
                    "selenium containing amino acid",
                    "protein",
                    "antibody",
                    "selenium containing amino acid",
                    "antibody",
                    "biomarker or",  # TODO
                    # antibody that binds to oxidized amino acids # TODO
                    # chloro-tyrosine todo
                    # oxidized sulfur- or selenium-containing amino acids (SSAA) # TODO
                ],
            },
            {
                "text": """
                Antagonistic peptide targeting il-2, il-9, and il-15 signaling for the treatment of cytokine-release syndrome and cytokine storm associated disorders
                The γc-family Interleukin-2 (IL-2), Interleukin-9 (IL-9), and Interleukin-15 (IL-15) cytokines are associated with important human diseases, such as cytokine-release syndrome and cytokine storm associated disorders. Compositions, methods, and kits to modulate signaling by at least one IL-2, IL-9, or IL-15 γc-cytokine family members for inhibiting, ameliorating, reducing a severity of, treating, delaying the onset of, or preventing at least one cytokine storm related disorder are described.
                """,
                "expected_output": [
                    "antagonistic peptide",  # TODO: targeting il-2, il-9, and il-15
                    "cytokine release syndrome",
                    "cytokine storm associated disorder",
                    "cytokine",
                    "cytokine release syndrome",
                    "cytokine storm",
                    "il15 γc cytokine family member",
                    "cytokine storm related disorder",
                    # γc-family Interleukin-2 (IL-2) # TODO
                ],
            },
            {
                "text": """
                Combination drug containing probucol and a tetrazolylalkoxy-dihydrocarbostyril derivative with superoxide supressant effects
                This invention relates to a combination drug comprising a combination of a tetrazolylalkoxy-dihydrocarbostyril derivative of the formula: wherein R is cycloalkyl, A is lower alkylene, and the bond between 3-and 4-positions of carbostyril nucleus is single bond or double bond, or a salt thereof and Probucol, which is useful for preventing and treating cerebral infarction including acute cerebral infarction and chronic cerebral infarction, arteriosclerosis, renal diseases (e.g. diabetic nephropathy, renal failure, nephritis), and diabetes owing to synergistic superoxide suppressant effects of the combination.
                """,
                "expected_output": [
                    "probucol",
                    "tetrazolylalkoxy dihydrocarbostyril derivative",
                    "superoxide supressant effect",
                    "tetrazolylalkoxy dihydrocarbostyril derivative",
                    "cerebral infarction",
                    "chronic cerebral infarction",
                    "renal disease",
                    "renal failure"
                    # arteriosclerosis, diabetic nephropathy, nephritis, diabetes # TODO
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
