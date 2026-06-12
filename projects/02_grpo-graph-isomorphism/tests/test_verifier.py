"""Comprehensive pytest suite for GraphIsomorphismVerifier."""

import sys
import os
import itertools
import pytest
import networkx as nx

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_isomorphism_env import (
    GraphIsomorphismVerifier,
    GraphIsomorphismEnv,
    Data,
    _edges_to_sorted_tuples,
    _generate_isomorphic_pair,
)



@pytest.fixture
def verifier():
    return GraphIsomorphismVerifier()


def _make_iso_data(G1, G2, perm_dict, n):
    """Build a Data object for an isomorphic pair."""
    perm_str = ", ".join(f"{k}->{v}" for k, v in sorted(perm_dict.items()))
    return Data(
        question="",
        answer=perm_str,
        metadata={
            "is_isomorphic": True,
            "n_nodes": n,
            "g1_edges": _edges_to_sorted_tuples(G1),
            "g2_edges": _edges_to_sorted_tuples(G2),
            "ground_truth_perm": perm_dict,
        },
    )


def _make_noniso_data(G1, G2, n):
    """Build a Data object for a non-isomorphic pair."""
    return Data(
        question="",
        answer="NOT ISOMORPHIC",
        metadata={
            "is_isomorphic": False,
            "n_nodes": n,
            "g1_edges": _edges_to_sorted_tuples(G1),
            "g2_edges": _edges_to_sorted_tuples(G2),
        },
    )


def _wrap(answer_text):
    """Wrap answer text in <think>/<answer> tags."""
    return f"<think>reasoning</think><answer>{answer_text}</answer>"




class TestAutomorphism:
    def test_petersen_multiple_valid_mappings(self, verifier):
        """Must accept multiple valid permutations."""
        from networkx.algorithms.isomorphism import GraphMatcher
        import random

        n = 10
        G1 = nx.relabel_nodes(nx.petersen_graph(), {i: i + 1 for i in range(n)})
        rng = random.Random(777)

        # Create G2 via one known permutation
        nodes = list(range(1, n + 1))
        shuffled = nodes.copy()
        rng.shuffle(shuffled)
        perm = dict(zip(nodes, shuffled))
        G2 = nx.relabel_nodes(G1, perm)

        data = _make_iso_data(G1, G2, perm, n)

        # Use GraphMatcher to enumerate actual valid isomorphisms
        matcher = GraphMatcher(G1, G2)
        accepted = set()
        for iso in matcher.isomorphisms_iter():
            perm_str = ", ".join(f"{k}->{v}" for k, v in sorted(iso.items()))
            assert verifier.verify(data, _wrap(perm_str)), (
                f"Valid isomorphism rejected: {iso}"
            )
            accepted.add(tuple(sorted(iso.items())))
            if len(accepted) >= 10:
                break  # enough to prove the point

        assert len(accepted) >= 5, (
            f"Expected >=5 accepted automorphisms for Petersen graph, got {len(accepted)}"
        )




class TestZeroIndexed:
    def test_0indexed_reindex(self):
        """{0:2, 1:0, 2:1} for n=3 must parse to {1:3, 2:1, 3:2}."""
        mapping = GraphIsomorphismVerifier._parse_mapping("{0: 2, 1: 0, 2: 1}", 3)
        assert mapping == {1: 3, 2: 1, 3: 2}, f"Got {mapping}"




class TestParsingStrategies:
    EXPECTED = {1: 3, 2: 1, 3: 2}

    def test_json_dict(self):
        mapping = GraphIsomorphismVerifier._parse_mapping(
            '{"1":"3", "2":"1", "3":"2"}', 3
        )
        assert mapping == self.EXPECTED, f"Got {mapping}"

    def test_json_dict_single_quotes(self):
        """LLMs sometimes use single quotes."""
        mapping = GraphIsomorphismVerifier._parse_mapping(
            "{'1': '3', '2': '1', '3': '2'}", 3
        )
        assert mapping == self.EXPECTED, f"Got {mapping}"

    def test_permutation_list(self):
        mapping = GraphIsomorphismVerifier._parse_mapping("[3, 1, 2]", 3)
        assert mapping == self.EXPECTED, f"Got {mapping}"

    def test_arrow_format(self):
        mapping = GraphIsomorphismVerifier._parse_mapping("1->3, 2->1, 3->2", 3)
        assert mapping == self.EXPECTED, f"Got {mapping}"

    def test_arrow_format_unicode(self):
        mapping = GraphIsomorphismVerifier._parse_mapping("1→3, 2→1, 3→2", 3)
        assert mapping == self.EXPECTED, f"Got {mapping}"

    def test_colon_format(self):
        mapping = GraphIsomorphismVerifier._parse_mapping("1:3, 2:1, 3:2", 3)
        assert mapping == self.EXPECTED, f"Got {mapping}"

    def test_pair_extraction_from_prose(self):
        """Extract from prose lacking standard delimiters."""
        text = "vertex 1 becomes 3 and vertex 2 becomes 1 and vertex 3 becomes 2"
        mapping = GraphIsomorphismVerifier._parse_mapping(text, 3)
        assert mapping == self.EXPECTED, f"Got {mapping}"




class TestEdgeCases:
    @pytest.fixture
    def iso_data_3(self):
        """Simple 3-node iso pair for tag-level tests."""
        G1 = nx.Graph()
        G1.add_nodes_from([1, 2, 3])
        G1.add_edges_from([(1, 2), (2, 3)])
        perm = {1: 2, 2: 3, 3: 1}
        G2 = nx.relabel_nodes(G1, perm)
        return _make_iso_data(G1, G2, perm, 3)

    # 4a — empty answer
    def test_empty_answer(self, verifier, iso_data_3):
        assert verifier.extract_answer("") == ""
        assert verifier.extract_answer(None) == ""
        assert verifier.verify(iso_data_3, "") is False

    # 4b — only <think>, no <answer>
    def test_think_only_no_answer(self, verifier, iso_data_3):
        sol = "<think>Let me think about this carefully...</think>"
        assert verifier.extract_answer(sol) == ""
        assert verifier.verify(iso_data_3, sol) is False

    # 4c — unclosed <answer> tag
    def test_unclosed_answer_tag(self, verifier, iso_data_3):
        perm = iso_data_3.metadata["ground_truth_perm"]
        perm_str = ", ".join(f"{k}->{v}" for k, v in sorted(perm.items()))
        sol = f"<think>ok</think><answer>{perm_str}"
        extracted = verifier.extract_answer(sol)
        assert perm_str in extracted or extracted  # extracted something
        assert verifier.verify(iso_data_3, sol) is True

    # 4d — multiple <answer> blocks → takes LAST
    def test_multiple_answer_blocks(self, verifier, iso_data_3):
        perm = iso_data_3.metadata["ground_truth_perm"]
        correct = ", ".join(f"{k}->{v}" for k, v in sorted(perm.items()))
        sol = (
            "<think>first try</think>"
            "<answer>1->1, 2->2, 3->3</answer>"  # wrong mapping
            "<think>wait, let me redo</think>"
            f"<answer>{correct}</answer>"  # correct mapping (last)
        )
        assert verifier.verify(iso_data_3, sol) is True

    def test_multiple_answer_blocks_last_wrong(self, verifier, iso_data_3):
        """If LAST block is wrong, verify should fail even if first was right."""
        perm = iso_data_3.metadata["ground_truth_perm"]
        correct = ", ".join(f"{k}->{v}" for k, v in sorted(perm.items()))
        sol = (
            f"<answer>{correct}</answer>"  # correct mapping
            "<answer>1->1, 2->2, 3->3</answer>"  # wrong — this is taken (last)
        )
        result = verifier.verify(iso_data_3, sol)
        assert result is False

    # 4e — "NOT ISOMORPHIC" for iso pair
    def test_not_iso_for_iso_pair(self, verifier, iso_data_3):
        sol = _wrap("NOT ISOMORPHIC")
        assert verifier.verify(iso_data_3, sol) is False

    # 4f — correct (identity) mapping for non-iso pair
    def test_identity_mapping_for_non_iso_pair(self, verifier):
        G1 = nx.Graph()
        G1.add_nodes_from([1, 2, 3])
        G1.add_edges_from([(1, 2), (2, 3)])
        # G2 has different edges
        G2 = nx.Graph()
        G2.add_nodes_from([1, 2, 3])
        G2.add_edges_from([(1, 2), (1, 3)])
        data = _make_noniso_data(G1, G2, 3)
        identity = ", ".join(f"{i}->{i}" for i in range(1, 4))
        sol = _wrap(identity)
        assert verifier.verify(data, sol) is False




class TestNotIsoPatterns:
    @pytest.mark.parametrize(
        "text",
        [
            "Not isomorphic",              # pattern 0
            "NOT ISOMORPHIC",              # pattern 0 (caps)
            "non-isomorphic",              # pattern 1
            "nonisomorphic",               # pattern 1 (no hyphen)
            "non isomorphic",              # pattern 1 (space)
            "no isomorphism",              # pattern 2
            "graphs are not isomorphic",   # pattern 3
            "the graph are not isomorphic",  # pattern 3 variant
            "NOT_ISOMORPHIC",              # pattern 4
            "false",                       # pattern 5
            "False",                       # pattern 5 (capitalized)
        ],
    )
    def test_pattern_matches(self, text):
        assert GraphIsomorphismVerifier._is_not_isomorphic_declaration(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "isomorphic",    # should NOT match — no "not"
            "mapping found",
            "1->2, 2->1",
        ],
    )
    def test_pattern_no_false_positives(self, text):
        assert GraphIsomorphismVerifier._is_not_isomorphic_declaration(text) is False

    def test_noniso_declaration_accepted_for_noniso_pair(self, verifier):
        """Verify non-isomorphic patterns."""
        G1 = nx.Graph()
        G1.add_nodes_from([1, 2, 3])
        G1.add_edges_from([(1, 2)])
        G2 = nx.Graph()
        G2.add_nodes_from([1, 2, 3])
        G2.add_edges_from([(1, 2), (2, 3)])
        data = _make_noniso_data(G1, G2, 3)

        for phrase in ["NOT ISOMORPHIC", "non-isomorphic", "no isomorphism",
                       "graphs are not isomorphic", "NOT_ISOMORPHIC", "false"]:
            sol = _wrap(phrase)
            assert verifier.verify(data, sol) is True, f"Failed for phrase: {phrase}"




class TestDisconnectedGraph:
    def test_graph_with_isolated_nodes(self, verifier):
        """Verify correct mapping on graph with isolated nodes."""
        G1 = nx.Graph()
        G1.add_nodes_from([1, 2, 3, 4])
        G1.add_edge(1, 2)

        # G2: same structure, permuted. perm: 1->3, 2->4, 3->1, 4->2
        perm = {1: 3, 2: 4, 3: 1, 4: 2}
        G2 = nx.relabel_nodes(G1, perm)

        data = _make_iso_data(G1, G2, perm, 4)
        perm_str = ", ".join(f"{k}->{v}" for k, v in sorted(perm.items()))
        sol = _wrap(perm_str)
        assert verifier.verify(data, sol) is True

    def test_disconnected_wrong_mapping(self, verifier):
        """Wrong mapping on disconnected graph should fail."""
        G1 = nx.Graph()
        G1.add_nodes_from([1, 2, 3, 4])
        G1.add_edge(1, 2)

        perm = {1: 3, 2: 4, 3: 1, 4: 2}
        G2 = nx.relabel_nodes(G1, perm)
        data = _make_iso_data(G1, G2, perm, 4)

        # Wrong: maps edge (1,2) -> (1,2) but G2 has edge (3,4)
        wrong = "1->1, 2->2, 3->3, 4->4"
        sol = _wrap(wrong)
        assert verifier.verify(data, sol) is False

    def test_fully_disconnected_graph(self, verifier):
        """No edges at all. Any bijection should be accepted."""
        G1 = nx.Graph()
        G1.add_nodes_from([1, 2, 3])
        G2 = G1.copy()
        # Any permutation works: edges are empty on both sides
        for perm_tuple in itertools.permutations([1, 2, 3]):
            perm = {i + 1: p for i, p in enumerate(perm_tuple)}
            data = _make_iso_data(G1, G2, perm, 3)
            perm_str = ", ".join(f"{k}->{v}" for k, v in sorted(perm.items()))
            sol = _wrap(perm_str)
            assert verifier.verify(data, sol) is True, f"Failed for perm {perm}"
