import unittest
from prisma.enums import OntologyLevel
import pytest

from clients.umls.types import EdgeRecord, NodeRecord
from data.etl.entity.biomedical_entity.umls import AncestorUmlsGraph
from typings import DocType
from utils.classes import overrides

pytest_plugins = ("pytest_asyncio",)

LEVEL_INSTANCE_THRESHOLD = 25
LEVEL_OVERRIDE_DELTA = 500
LEVEL_MIN_PREV_COUNT = 5

# if we ever want an auto-test for UMLS, use this example.
# g.get_count("C1333196") should be approx sum(list([g.get_count(id) or 0 for id in list(g.G.successors("C1333196"))]))
# = approx 19228


class MockAncestorUmlsGraph(AncestorUmlsGraph):
    @overrides(AncestorUmlsGraph)
    @classmethod
    async def create(
        cls,
        doc_type: DocType = DocType.patent,
    ) -> "AncestorUmlsGraph":
        aug = cls(
            doc_type,
            instance_threshold=LEVEL_INSTANCE_THRESHOLD,
            previous_threshold=LEVEL_MIN_PREV_COUNT,
            current_override_threshold=LEVEL_INSTANCE_THRESHOLD,
        )
        await aug.load()
        return aug

    @overrides(AncestorUmlsGraph)
    async def load_nodes(
        self,
    ) -> list[NodeRecord]:
        """
        Mock node generator
        """
        nodes = [
            NodeRecord(
                id="PARENT1_CHILD1",
                count=5,  # subinstance
            ),
            NodeRecord(
                id="PARENT1_CHILD2",
                count=1000,
            ),
            NodeRecord(
                id="PARENT2_CHILD1",
                count=1000,
            ),
            NodeRecord(
                id="PARENT2_CHILD2",
                count=1000,
            ),
            NodeRecord(
                id="PARENT1and2_CHILD1",
                count=1000,
            ),
            NodeRecord(
                id="PARENT1and4_CHILD1",
                count=1000,
            ),
            NodeRecord(
                id="PARENT3_ONLY_CHILD",
                count=5,  # subinstance
            ),
            NodeRecord(
                id="PARENT_WITH_COUNT_CHILD",
                count=2,  # subinstance
            ),
            NodeRecord(
                id="GRANDPARENT2_DIRECT_CHILD",
                count=1,  # tiny subinstance
            ),
            NodeRecord(id="PARENT1"),
            NodeRecord(id="PARENT2"),
            NodeRecord(id="PARENT3"),
            NodeRecord(id="PARENT4"),
            NodeRecord(id="PARENT_WITH_COUNT", count=5),
            NodeRecord(id="GRANDPARENT1"),
            # GRANDPARENT2_DIRECT_CHILD & PARENT3's parent
            NodeRecord(id="GRANDPARENT2"),
            NodeRecord(id="GRANDPARENT3"),
        ]

        return nodes

    @overrides(AncestorUmlsGraph)
    async def load_edges(
        self,
    ) -> list[EdgeRecord]:
        edges = [
            EdgeRecord(
                head="PARENT1",
                tail="PARENT1_CHILD1",
            ),
            EdgeRecord(
                head="PARENT1",
                tail="PARENT1_CHILD1",
            ),
            EdgeRecord(
                head="PARENT1",
                tail="PARENT1_CHILD2",
            ),
            EdgeRecord(
                head="PARENT2",
                tail="PARENT2_CHILD1",
            ),
            EdgeRecord(
                head="PARENT2",
                tail="PARENT2_CHILD2",
            ),
            EdgeRecord(
                head="PARENT3",
                tail="PARENT3_ONLY_CHILD",
            ),
            EdgeRecord(
                head="PARENT1",
                tail="PARENT1and2_CHILD1",
            ),
            EdgeRecord(
                head="PARENT2",
                tail="PARENT1and2_CHILD1",
            ),
            EdgeRecord(
                head="PARENT1",
                tail="PARENT1and4_CHILD1",
            ),
            EdgeRecord(
                head="PARENT4",
                tail="PARENT1and4_CHILD1",
            ),
            EdgeRecord(
                head="GRANDPARENT1",
                tail="PARENT1",
            ),
            EdgeRecord(
                head="GRANDPARENT1",
                tail="PARENT2",
            ),
            EdgeRecord(
                head="GRANDPARENT2",
                tail="PARENT3",
            ),
            EdgeRecord(
                head="GRANDPARENT2",
                tail="GRANDPARENT2_DIRECT_CHILD",
            ),
            EdgeRecord(
                head="PARENT_WITH_COUNT",
                tail="PARENT_WITH_COUNT_CHILD",
            ),
            EdgeRecord(
                head="GRANDPARENT3",
                tail="PARENT4",
            ),
        ]
        return edges


@pytest.mark.asyncio
async def test_ancestor_counts():
    tc = unittest.TestCase()
    graph = await MockAncestorUmlsGraph.create()
    p1_count = graph.get_count("PARENT1")
    p2_count = graph.get_count("PARENT2")
    p3_count = graph.get_count("PARENT3")
    p4_count = graph.get_count("PARENT4")
    p5_count = graph.get_count("PARENT_WITH_COUNT")
    gp1_count = graph.get_count("GRANDPARENT1")
    gp2_count = graph.get_count("GRANDPARENT2")
    gp3_count = graph.get_count("GRANDPARENT3")
    tc.assertEqual(p1_count, 3005)
    tc.assertEqual(p2_count, 3000)
    tc.assertEqual(p3_count, 5)
    tc.assertEqual(p4_count, 1000)
    tc.assertEqual(p5_count, 7)
    tc.assertEqual(gp1_count, 6005)  # should be 5005
    tc.assertEqual(gp2_count, 6)
    tc.assertEqual(gp3_count, 1000)

    print([(n["id"], n["level"]) for n in graph.nodes.values()])

    tc.assertEqual(
        graph.get_ontology_level("PARENT1_CHILD1"), OntologyLevel.SUBINSTANCE
    )
    tc.assertEqual(
        graph.get_ontology_level("PARENT3_ONLY_CHILD"), OntologyLevel.SUBINSTANCE
    )
    tc.assertEqual(
        graph.get_ontology_level("GRANDPARENT2_DIRECT_CHILD"), OntologyLevel.SUBINSTANCE
    )
    tc.assertEqual(graph.get_ontology_level("PARENT1_CHILD2"), OntologyLevel.INSTANCE)
    tc.assertEqual(graph.get_ontology_level("PARENT2_CHILD1"), OntologyLevel.INSTANCE)
    tc.assertEqual(graph.get_ontology_level("PARENT2_CHILD2"), OntologyLevel.INSTANCE)

    tc.assertEqual(graph.get_ontology_level("PARENT1"), OntologyLevel.L1_CATEGORY)
    tc.assertEqual(graph.get_ontology_level("PARENT2"), OntologyLevel.L1_CATEGORY)
    tc.assertEqual(graph.get_ontology_level("PARENT3"), OntologyLevel.NA)

    tc.assertEqual(graph.get_ontology_level("GRANDPARENT1"), OntologyLevel.L2_CATEGORY)
    tc.assertEqual(graph.get_ontology_level("GRANDPARENT2"), OntologyLevel.INSTANCE)
