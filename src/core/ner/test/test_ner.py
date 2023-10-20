import unittest
import pytest

from core.ner import NerTagger


# PROTEIN BINDING TO Akt2
# Acetylated hmgb1 protein
# Hla-a24-restricted cancer antigen peptide
# Smg-1-binding protein and method of screening substance controlling its activity
# Pablo, a polypeptide that interacts with bcl-xl, and uses related thereto
# Antibody capable of recognizing 8-nitroguanine
# Abc transporter-associated gene abcc13
# Use of poly-alpha2,8-sialic acid mimetic peptides to modulate ncam functions.
# Cyclin dependent kinase 5 phosphorylation of disabled 1 protein


@pytest.mark.skip(reason="Too stocastic to include in CI")
class TestNerUtils(unittest.TestCase):
    """
    from core.ner import NerTagger; tagger=NerTagger()
    t = tagger.extract([text])[0]
    [(t1[0], t1.start_char, t1.end_char) for t1 in t]
    """

    def setUp(self):
        self.tagger = NerTagger(
            entity_types=frozenset(["compounds", "diseases", "mechanisms"]),
            link=False
            # rule_sets=[],
        )

    def test_ner(self):
        test_conditions = [
            # {
            #     "text": """
            #     Bioenhanced formulations comprising eprosartan in oral solid dosage form.
            #     This invention relates to bioenhanced formulations comprising eprosartan or eprosartan mesylate in the amorphous form, a process for its production, compositions containing the compound and methods of using the compound to block angiotensin II receptors and to treat hypertension, congestive heart failure and renal failure.
            #     """,
            #     "expected_output": [
            #         "amorphous form",
            #         "bioenhanced formulation",
            #         "block angiotensin ii receptor",
            #         "congestive heart failure",
            #         "eprosartan mesylate",
            #         "hypertension",
            #         "renal failure",
            #     ],
            # },
            # {
            #     "text": [
            #         """
            #     Bioenhanced formulations comprising eprosartan in oral solid dosage form.
            #     This invention relates to bioenhanced formulations comprising eprosartan or eprosartan mesylate in the amorphous form, a process for its production, compositions containing the compound and methods of using the compound to block angiotensin II receptors and to treat hypertension, congestive heart failure and renal failure.
            #     """,
            #         "Cox-2 inhibitors in combination with centrally acting analgesics",
            #     ],
            #     "expected_output": [
            #         [
            #             "amorphous form",
            #             "bioenhanced formulation",
            #             "block angiotensin ii receptor",
            #             "congestive heart failure",
            #             "eprosartan mesylate",
            #             "hypertension",
            #             "renal failure",
            #         ],
            #         ["analgesic", "centrally acting analgesic", "cox2 inhibitor"],
            #     ],
            # },
            # {
            #     "text": """
            #     Pharmaceutical composition in particular for preventing and treating mucositis induced by radiotherapy or chemotherapy.
            #     The invention concerns a pharmaceutical composition designed to adhere to a mucous membrane in particular for preventing or treating radiotherapy-related and chemotherapy-related mucositis, induced by radiotherapy or combined radiochemotherapy, comprising an efficient amount of an antiradical compound mixed with a vehicle which is liquid at room temperature and gels at the mucous membrane temperature and capable of adhering to the mucous membrane by its gelled state.
            #     """,
            #     "expected_output": [
            #         "antiradical compound",
            #         "chemotherapy",
            #         "chemotherapy related mucositis",
            #         "combined radiochemotherapy",
            #         "radiotherapy",
            #         "radiotherapy relate",
            #     ],
            # },
            {
                "text": """
                Novel aspartyl dipeptide ester derivatives and sweeteners.
                Novel aspartyl dipeptide ester derivatives (including salts thereof) such as N-[N-[3-(3-hydroxy-4-methoxyphenyl)propyl]-L-α-aspartyl]-L-(α-methyl)phenylalanine 1-methyl ester which are usable as sweeteners; and sweeteners, foods, etc. containing the same. These compounds are usable as low-caloric sweeteners being much superior in the degree of sweetness to the conventional ones.
                """,
                "expected_output": [
                    "aspartyl dipeptide ester",
                    "n[n[33 hydroxy 4 methoxyphenylpropyl]-lα aspartyl]-l(α methyl)phenylalanine 1 methyl ester",
                    "novel aspartyl dipeptide ester derivative",
                ],
            },
            # {
            #     "text": """
            #     Muscarinic antagonists
            #     Heterocyclic derivatives of di-N-substituted piperazine or 1,4 di-substituted piperidine compounds in accordance with formula (I) (including all isomers, salts and solvates), wherein one of Y and Z is -N- and the other is -N- or -CH-; X is -O-, -S-, -SO-, -SO2- or -CH2-; Q is (1), (2), (3); R is alkyl, cycloalkyl, optionally substituted aryl or heteroaryl; R?1, R2 and R3¿ are H or alkyl; R4 is alkyl, cyclolalkyl or (4); R5 is H, alkyl, -C(O)alkyl, arylcarbonyl, -SO¿2?alkyl, aryl-sulfonyl-C(O)Oalkyl, aryloxycarbonyl, -C(O)NH-alkyl or aryl-aminocarbonyl, wherein the aryl portion is optionally substituted; R?6¿ is H or alkyl; and R7 is H, alkyl, hydroxyalkyl or alkoxyalkyl; are muscarinic antagonists useful for treating cognitive disorders such as Alzheimer&#39;s disease. Pharmaceutical compositions and methods of treatment are also disclosed.
            #     """,
            #     "expected_output": [
            #         "muscarinic antagonists heterocyclic derivative",
            #         "din substituted piperazine",
            #         "1,4 di substituted piperidine",
            #         "di substituted piperidine compound",
            #         "optionally substituted aryl",
            #         "arylcarbonyl",
            #         "aryloxycarbonyl",
            #         "aryl aminocarbonyl",
            #         "optionally",  # TODO
            #         "muscarinic antagonists useful",
            #         "alzheimer disease",
            #     ],
            # },
            # {
            #     "text": """
            #     Method of treating meniere&#39;s disease and corresponding apparatus
            #     In a method of treating Ménière&#39;s disease intermittent air pressure pulse trains are administred to an outwardly sealed external ear volume bordering to a surgically perforated tympanic membrane. In a pulse train air pressure is increased from ambient (p0) to a first level (p1) and from there repeatedly to a second level (p2) and repeatedly decreased to the first level (p1), and finally decreased to ambient (p0). P1 is from 4 to 16 cm H2O, p2 is from 8 to 16 cm H2O, with the proviso that p1 &gt; p2, the pressure increase rate is from 0 to 4 mm H2O per millisecond, the pressure decrease rate is from 0 to 2 mm H2O per millisecond, the modulation frequency is from 3 to 9 Hz, the intermittent time period is from 3 to 10 seconds. Also disclosed is an apparatus for carrying out the method.
            #     """,
            #     "expected_output": [
            #         "intermittent air pressure pulse train",
            #         "meniere disease",
            #         "ménière disease intermittent air pressure pulse train",
            #         "modulation frequency",
            #         "surgically perforated tympanic membrane",
            #     ],
            # },
            # {
            #     "text": """
            #     Cox-2 inhibitors in combination with centrally acting analgesics
            #     A method of alleviating a pain state not associated with a cough condition is provided which comprises administering a cyclooxygenase-2 inhibitor and a centrally active analgesic selected from the group consisting of a narcotic analgesic selected from the group consisitng of codeine and hydrocodone; an agonist-antagonist analgesic and tramadol. A method and analgesic composition therefor is also provided for treating all pain states which comprises administering a cyclooxygenase-2 inhibitor and a centrally acting analgesic selected from the group consisting of a narcotic analgesic other than codeine and hydrocodone; an agonist-antagonist analgesic and tramadol.
            #     """,
            #     "expected_output": [
            #         "active analgesic",
            #         "agonist antagonist analgesic",
            #         "analgesic",
            #         "analgesic composition therefor",  # ugh
            #         "centrally acting analgesic",
            #         "codeine",
            #         "cough condition",
            #         "cox2 inhibitor",
            #         "cyclooxygenase 2 inhibitor",
            #         "narcotic analgesic",
            #         "narcotic analgesic other",
            #         "pain state",
            #     ],
            # },
            {
                "text": """
                Biomarkers for oxidative stress
                This invention relates generally to methods of detecting and quantifying biomarkers of oxidative stress in proteins. The biomarker may be any amino acid that has undergone oxidation, or other modification such as chloro-tyrosine, dityrosin. Emphasis is given herein on oxidized sulfur- or selenium-containing amino acids (SSAA). The biomarker of oxidative stress in proteins may be detected with an antibody that binds to oxidized amino acids, specifically oxidized sulfur- or selenium-containing amino acids. The antibody may be monoclonal or polyclonal. The presence of biomarker or amount of biomarker present in a sample may be used to aid in assessing the efficacy of environmental, nutritional and therapeutic interventions, among other uses.
                """,
                "expected_output": [
                    "antibody",
                    "dityrosine)",
                    "monoclonal",
                    "oxidative stress",
                    "polyclonal",
                    "protein",
                    "quantifying biomarker",
                    "selenium containing amino acid"
                    # antibody that binds to oxidized amino acids # TODO
                    # chloro-tyrosine # TODO
                    # oxidized sulfur- or selenium-containing amino acids (SSAA) # TODO
                ],
            },
            # {
            #     "text": """
            #     Antagonistic peptide targeting il-2, il-9, and il-15 signaling for the treatment of cytokine-release syndrome and cytokine storm associated disorders
            #     The γc-family Interleukin-2 (IL-2), Interleukin-9 (IL-9), and Interleukin-15 (IL-15) cytokines are associated with important human diseases, such as cytokine-release syndrome and cytokine storm associated disorders. Compositions, methods, and kits to modulate signaling by at least one IL-2, IL-9, or IL-15 γc-cytokine family members for inhibiting, ameliorating, reducing a severity of, treating, delaying the onset of, or preventing at least one cytokine storm related disorder are described.
            #     """,
            #     "expected_output": [
            #         "antagonistic peptide targeting il2",
            #         "cytokine",
            #         "cytokine release syndrome",
            #         "cytokine storm",
            #         "cytokine storm associated disorder",
            #         "cytokine storm related disorder",
            #         "il15 γc cytokine family",
            #         "inhibiting",
            #         "modulate signal",
            #     ],
            # },
            {
                "text": """
                Combination drug containing probucol and a tetrazolylalkoxy-dihydrocarbostyril derivative with superoxide supressant effects
                This invention relates to a combination drug comprising a combination of a tetrazolylalkoxy-dihydrocarbostyril derivative of the formula: wherein R is cycloalkyl, A is lower alkylene, and the bond between 3-and 4-positions of carbostyril nucleus is single bond or double bond, or a salt thereof and Probucol, which is useful for preventing and treating cerebral infarction including acute cerebral infarction and chronic cerebral infarction, arteriosclerosis, renal diseases, e.g. diabetic nephropathy, renal failure, nephritis, and diabetes owing to synergistic superoxide suppressant effects of the combination.
                """,
                "expected_output": [
                    "arteriosclerosis",
                    "cerebral infarction",
                    "derivative",
                    "probucol",
                    "renal disease",
                    "renal failure",
                    "superoxide supressant effect",
                    "synergistic superoxide suppressant effect",
                    "tetrazolylalkoxy dihydrocarbostyril",
                ],
            },
            {
                "text": """
                    5-[2-(pyridin-2-ylamino)-1,3-thiazol-5-yl]-2,3-dihydro-1 h-isoindol-1 -one derivatives and their use as dual inhibitors of phosphatidylinositol 3-kinase delta &amp; gamma.
                    There are disclosed certain novel compounds (including pharmaceutically acceptable salts thereof) (I) that inhibit phosphatidylinositol 3-kinase gamma (PI3Kδ) and phosphatidylinositol 3-kinase gamma (ΡΙ3Κγ) activity, to their utility in treating and/or preventing clinical conditions including respiratory diseases, such as asthma and chronic obstructive pulmonary disease (COPD), to their use in therapy, to pharmaceutical compositions containing them and to processes for preparing such compounds.
                """,
                "expected_output": [
                    "asthma",
                    "chronic obstructive pulmonary disease",
                    "derivative",
                    "dual inhibitor",
                    "phosphatidylinositol 3 kinase delta & gamma",
                    "phosphatidylinositol 3 kinase gamma (pi3kδ",
                    "phosphatidylinositol 3 kinase gamma (ρι3κγ) activity",
                    "respiratory disease",
                ],
            },
            {
                "text": """
                    Compounds useful in the treatment of disorders responsive to the inhibition of apoptosis signal-regulating kinase 1 (ASK1)
                """,
                "expected_output": [
                    # "apoptosis signal regulating kinase 1",
                    # "disorders response to the inhibition of apoptosis signal regulating kinase 1",
                    "disorders responsive",
                    "apoptosis signal regulating kinase",
                    # "ask1 inhibitor"
                ],
            },
            {
                "text": "Compositions containing reduced amounts of daclizumab acidic isoforms and methods for preparing the same.",
                "expected_output": [
                    "daclizumab acidic isoform",
                ],
            },
            {
                "text": "The present invention relates to a method for PEGylating interferon beta.",
                "expected_output": [
                    # "pegylating interferon beta",
                ],
            },
            {
                "text": """
                    Bipolar pliers for microsurgery and coeliosurgery
                    The invention concerns prehensile, rotating and detachable bipolar electro-coagulating pliers for microsurgery and coeliosurgery. It concerns reusable pliers for microsurgery and coeliosurgery prehensile and ensuring bipolar coagulation without bonding the coagulated tissues, combining in a single instrument both functions, so as to save time and handling procedure during an operation. Said pliers consist of: a head called fixed part (1) internally and externally insulated except for the fixed coagulating jaw (1); a mobile jaw (2) articulated on a ceramic pin (3) provided like the fixed jaw (1) with a coating preventing the coagulated tissues from being bonded; an externally insulated tube (7) with its snake (6) likewise insulated, connected to the mobile blade (2) and the handle (12-11); a handle controlling (11-12) power supply electrode-holders (14-13) connected to the tube and snake assembly by a screw socket (9) and the movement controlling ball (10). The insulation between the various metal components is provided by the ceramic pin (3) and the hot-deposited insulating coating.
                """,
                "expected_output": [],
            },
            # virus-induced antigen expressed on the plasma membrane of virus-infected cells, or; e) a receptor molecule or the fragment thereof with an affinity to an epitope of the viral structural proteins
            # WARNING:core.ner.binder.utils:Overlap detected between sphingosine-1-phosphate synthesis inhibitors, and (iii) protein kinase C inhibitors. A preferred glucosyl ceramide synthesis inhibitor and [glucosylceramide synthesis inhibitors, (ii) sphingosine-1-phosphate synthesis inhibitors, and (iii) protein kinase C inhibitors. A preferred glucosyl ceramide synthesis inhibitor is 1-phenyl-2-palmitoylamino-3-morpholino-1-propanol. A preferred sphingosine-1-phosphate synthesis inhibitor, sphingosine-1-phosphate synthesis inhibitors, and (iii) protein kinase C inhibitors. A preferred glucosyl ceramide synthesis inhibitor is 1-phenyl-2-palmitoylamino-3-morpholino-1-propanol. A preferred sphingosine-1-phosphate synthesis inhibitor, protein kinase C inhibitors. A preferred glucosyl ceramide synthesis inhibitor is 1-phenyl-2-palmitoylamino-3-morpholino-1-propanol. A preferred sphingosine-1-phosphate synthesis inhibitor]
            # WARNING:core.ner.binder.utils:Overlap detected between schizophrenia and [disorders treatable and/or preventable with anti-convulsive agents, such as epilepsy including post-traumatic epilepsy, Parkinson's disease, psychosis, migraine, cerebral ischaemia, Alzheimer's disease and other degenerative diseases such as Huntingdon's chorea, schizophrenia, obsessive compulsive disorders (OCD)]
            # WARNING:core.ner.binder.utils:Overlap detected between anti-inflammatory agents and [anti-inflammatory agents
            # Bifunctional activation cascade antagonist]
            # WARNING:core.ner.binder.utils:Overlap detected between π-bound ligand and [monoanionic ligand bound to the metal, B is a π-bound ligand]
        ]

        for condition in test_conditions:
            text = condition["text"]
            expected_output = condition["expected_output"]

            result = self.tagger.extract_string_map([text])[text]

            if result != expected_output:
                print("Actual", result, "expected", expected_output)

            self.assertEqual(result, expected_output)  # results are too stochastic
