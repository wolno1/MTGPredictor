import json
import unittest

from logic import (
    CardDatabase,
    LEARNING_FILE,
    META_DECKS_DIR,
    analyze_battlefield,
    choose_best_target,
    choose_block_assignments,
    ensure_learning_file,
    recommend_blocks,
    recommend_combat_line,
    load_decks_from_folder,
    parse_decklist,
    rank_matching_decks,
    simulate_matchup,
    simulate_many_matchups,
    simulate_many_opening_hands,
    simulate_opening_hand,
    simulate_opening_turns,
    simulate_combat_exchange,
    simulate_stack_interaction,
    train_from_matchup_results,
    suggest_plays_from_hand,
    resolve_simple_trigger,
)


class DeckToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card_db = CardDatabase()
        cls.card_db.load()
        cls.meta_decks = load_decks_from_folder(META_DECKS_DIR, cls.card_db)

    def test_parse_arena_style_lines_and_sideboard(self):
        deck_text = """
        Deck
        2 The Book of Exalted Deeds (AFR) 4
        4 Giada, Font of Hope (SNC) 14

        Sideboard
        2 Duress
        """
        deck = parse_decklist(deck_text, deck_name="Sample", card_db=self.card_db)

        self.assertEqual(deck.total_cards, 6)
        self.assertEqual(sum(entry.quantity for entry in deck.sideboard), 2)
        self.assertEqual(deck.entries[0].name, "The Book of Exalted Deeds")
        self.assertTrue(all(entry.is_known for entry in deck.entries))

    def test_inference_prefers_angel_control(self):
        results = rank_matching_decks(
            ["Righteous Valkyrie", "Giada, Font of Hope", "Kayla's Reconstruction"],
            self.meta_decks,
            top_n=3,
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["deck_name"], "AngelControl")

    def test_advisor_prefers_castable_curve_play(self):
        results = suggest_plays_from_hand(
            ["Giada, Font of Hope", "Righteous Valkyrie", "Plains"],
            available_mana=2,
            card_db=self.card_db,
            lands_played_this_turn=True,
            opponent_pressure="medium",
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["name"], "Giada, Font of Hope")

    def test_advisor_prioritizes_removal_under_pressure(self):
        results = suggest_plays_from_hand(
            ["Lay Down Arms", "Speaker of the Heavens"],
            available_mana=1,
            card_db=self.card_db,
            lands_played_this_turn=True,
            opponent_pressure="high",
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["name"], "Lay Down Arms")

    def test_battlefield_analysis_detects_estimated_lethal(self):
        summary = analyze_battlefield(
            my_battlefield=["Giada, Font of Hope", "Righteous Valkyrie"],
            opponent_battlefield=["Speaker of the Heavens"],
            my_life=18,
            opponent_life=3,
            card_db=self.card_db,
        )

        self.assertTrue(summary["lethal_available"])
        self.assertIn("letal estimado", summary["combat_recommendation"].casefold())

    def test_battlefield_analysis_detects_crackback_risk(self):
        summary = analyze_battlefield(
            my_battlefield=["Speaker of the Heavens"],
            opponent_battlefield=["Righteous Valkyrie", "Giada, Font of Hope"],
            my_life=3,
            opponent_life=18,
            card_db=self.card_db,
        )

        self.assertEqual(summary["danger_level"], "high")
        self.assertIn("defens", summary["combat_recommendation"].casefold())

    def test_combat_line_recommends_attack_for_lethal(self):
        result = recommend_combat_line(
            my_battlefield=["Giada, Font of Hope", "Righteous Valkyrie"],
            opponent_battlefield=["Speaker of the Heavens"],
            my_life=18,
            opponent_life=3,
            card_db=self.card_db,
        )

        self.assertEqual(result["plan"], "attack")
        self.assertIn("ataca", result["summary"].casefold())

    def test_combat_line_recommends_hold_back_when_dead_on_crackback(self):
        result = recommend_combat_line(
            my_battlefield=["Speaker of the Heavens"],
            opponent_battlefield=["Righteous Valkyrie", "Giada, Font of Hope"],
            my_life=3,
            opponent_life=18,
            card_db=self.card_db,
        )

        self.assertEqual(result["plan"], "defend")
        self.assertIn("bloque", result["summary"].casefold())

    def test_opening_hand_simulation_keeps_good_curve(self):
        deck = next(deck for deck in self.meta_decks if deck.name == "AngelControl")
        result = simulate_opening_hand(deck, self.card_db, seed=7)

        self.assertEqual(len(result["hand"]), 7)
        self.assertIn(result["decision"], {"keep", "mulligan"})
        self.assertGreaterEqual(result["land_count"], 0)

    def test_opening_hand_simulation_flags_zero_land_hand(self):
        deck_text = """
        30 Giada, Font of Hope
        30 Righteous Valkyrie
        """
        deck = parse_decklist(deck_text, deck_name="NoLands", card_db=self.card_db)
        result = simulate_opening_hand(deck, self.card_db, seed=1)

        self.assertEqual(result["land_count"], 0)
        self.assertEqual(result["decision"], "mulligan")

    def test_many_opening_hands_returns_consistency_stats(self):
        deck = next(deck for deck in self.meta_decks if deck.name == "AngelControl")
        result = simulate_many_opening_hands(deck, self.card_db, simulations=25, seed=11)

        self.assertEqual(result["simulations"], 25)
        self.assertIn("keep_rate", result)
        self.assertIn("average_lands", result)
        self.assertGreaterEqual(result["keep_rate"], 0)
        self.assertLessEqual(result["keep_rate"], 100)

    def test_opening_turns_simulation_returns_turn_log(self):
        deck = next(deck for deck in self.meta_decks if deck.name == "AngelControl")
        result = simulate_opening_turns(deck, self.card_db, turns=3, seed=5)

        self.assertEqual(result["turns_simulated"], 3)
        self.assertEqual(len(result["turn_log"]), 3)
        self.assertIn("final_hand_size", result)
        self.assertIn("lands_in_play", result)

    def test_opening_turns_follow_mtg_phase_order(self):
        deck = next(deck for deck in self.meta_decks if deck.name == "AngelControl")
        result = simulate_opening_turns(deck, self.card_db, turns=1, seed=5)

        phases = [step["step"] for step in result["turn_log"][0]["phases"]]
        self.assertEqual(
            phases,
            [
                "untap",
                "upkeep",
                "draw",
                "main1",
                "begin_combat",
                "declare_attackers",
                "declare_blockers",
                "combat_damage",
                "end_combat",
                "main2",
                "end_step",
                "cleanup",
            ],
        )

    def test_matchup_simulation_returns_winner_and_scores(self):
        deck_a = next(deck for deck in self.meta_decks if deck.name == "AngelControl")
        deck_b = next(deck for deck in self.meta_decks if deck.name == "MonoBlack140426")
        result = simulate_matchup(deck_a, deck_b, self.card_db, turns=3, seed=13)

        self.assertEqual(result["turns_simulated"], 3)
        self.assertIn(result["winner"], {deck_a.name, deck_b.name, "draw"})
        self.assertIn("score_a", result)
        self.assertIn("score_b", result)
        self.assertIn("life_a", result)
        self.assertIn("life_b", result)
        self.assertEqual(len(result["turn_history"]), 3)

    def test_many_matchups_returns_winrate_report(self):
        deck_a = next(deck for deck in self.meta_decks if deck.name == "AngelControl")
        deck_b = next(deck for deck in self.meta_decks if deck.name == "MonoBlack140426")
        result = simulate_many_matchups(deck_a, deck_b, self.card_db, simulations=20, turns=3, seed=21)

        self.assertEqual(result["simulations"], 20)
        self.assertIn("winrate_a", result)
        self.assertIn("winrate_b", result)
        self.assertIn("draw_rate", result)

    def test_training_updates_learning_weights(self):
        ensure_learning_file(LEARNING_FILE)
        before = json.loads(LEARNING_FILE.read_text(encoding="utf-8"))

        result = train_from_matchup_results(
            {
                "winrate_a": 80.0,
                "winrate_b": 20.0,
                "average_score_a": 13.0,
                "average_score_b": 8.0,
            },
            learning_path=LEARNING_FILE,
        )

        after = json.loads(LEARNING_FILE.read_text(encoding="utf-8"))
        self.assertIn("weights", result)
        self.assertNotEqual(before["weights"], after["weights"])

    def test_stack_interaction_can_counter_or_remove(self):
        result = simulate_stack_interaction(
            acting_spell="Speaker of the Heavens",
            response_spell="Eaten Alive",
            card_db=self.card_db,
        )

        self.assertIn("stack", result)
        self.assertGreaterEqual(len(result["stack"]), 1)
        self.assertIn("resolution", result)

    def test_simple_trigger_resolution_returns_event(self):
        event = resolve_simple_trigger(
            source_name="Bishop of Wings",
            trigger_type="etb",
            controller_state={"life": 20, "library": [], "hand": [], "battlefield": [], "graveyard": []},
            opponent_state={"life": 20, "battlefield": [], "graveyard": []},
            card_db=self.card_db,
        )

        self.assertIn("triggered", event)
        self.assertIn("detail", event)

    def test_stack_interaction_supports_multiple_responses(self):
        result = simulate_stack_interaction(
            acting_spell="Speaker of the Heavens",
            response_spell="Eaten Alive",
            card_db=self.card_db,
            extra_responses=["Lay Down Arms"],
        )

        self.assertIn("stack", result)
        self.assertEqual(len(result["stack"]), 3)
        self.assertGreaterEqual(len(result["resolution"]), 2)

    def test_target_selection_prefers_more_dangerous_creature(self):
        target = choose_best_target(
            ["Speaker of the Heavens", "Righteous Valkyrie"],
            self.card_db,
        )

        self.assertEqual(target, "Righteous Valkyrie")

    def test_target_selection_avoids_ward_when_possible(self):
        target = choose_best_target(
            ["A-Syndicate Infiltrator", "Speaker of the Heavens"],
            self.card_db,
            source_name="Lay Down Arms",
        )

        self.assertEqual(target, "Speaker of the Heavens")

    def test_target_selection_respects_protection_from_color(self):
        target = choose_best_target(
            ["Abbey Gargoyles", "Speaker of the Heavens"],
            self.card_db,
            source_name="Shock",
        )

        self.assertEqual(target, "Speaker of the Heavens")

    def test_matchup_simulation_records_blocks_when_possible(self):
        deck_a = parse_decklist(
            """
            20 Plains
            20 Speaker of the Heavens
            20 Giada, Font of Hope
            """,
            deck_name="Attackers",
            card_db=self.card_db,
        )
        deck_b = parse_decklist(
            """
            20 Swamp
            20 Monoist Gravliner
            20 Susurian Voidborn
            """,
            deck_name="Blockers",
            card_db=self.card_db,
        )
        result = simulate_matchup(deck_a, deck_b, self.card_db, turns=4, seed=42)

        self.assertTrue(any("bloquea" in phase["detail"] for turn in result["turn_history"] for phase in turn["phases"] if phase["step"] == "declare_blockers"))

    def test_block_assignments_prefer_biggest_threat(self):
        assignments = choose_block_assignments(
            attackers=["Speaker of the Heavens", "Righteous Valkyrie"],
            blockers=["Monoist Gravliner", "Susurian Voidborn"],
            card_db=self.card_db,
            defender_life=8,
        )

        self.assertGreaterEqual(len(assignments), 1)
        primary_block = assignments[0]
        self.assertEqual(primary_block["attacker"], "Righteous Valkyrie")

    def test_recommend_blocks_returns_human_readable_plan(self):
        result = recommend_blocks(
            my_battlefield=["Monoist Gravliner", "Susurian Voidborn"],
            opponent_attackers=["Speaker of the Heavens", "Righteous Valkyrie"],
            my_life=8,
            card_db=self.card_db,
        )

        self.assertEqual(result["plan"], "block")
        self.assertTrue(any("Righteous Valkyrie" in item for item in result["assignments"]))
        self.assertIn("bloquea", result["summary"].casefold())

    def test_block_assignments_can_double_block_big_threat(self):
        assignments = choose_block_assignments(
            attackers=[{"name": "Big Threat", "power": 4, "toughness": 4}],
            blockers=[
                {"name": "Token A", "power": 2, "toughness": 2},
                {"name": "Token B", "power": 2, "toughness": 3},
            ],
            card_db=self.card_db,
            defender_life=6,
        )

        self.assertEqual(len(assignments), 2)
        self.assertTrue(all(pair["attacker_name"] == "Big Threat" for pair in assignments))

    def test_block_assignments_respect_flying_evasion(self):
        assignments = choose_block_assignments(
            attackers=[{"name": "Sky Drake", "power": 3, "toughness": 3, "text": "flying"}],
            blockers=[{"name": "Ground Bear", "power": 3, "toughness": 3}],
            card_db=self.card_db,
            defender_life=10,
        )

        self.assertEqual(assignments, [])

    def test_combat_exchange_applies_trample_damage(self):
        result = simulate_combat_exchange(
            attackers=[{"name": "Charging Beast", "power": 5, "toughness": 5, "text": "trample"}],
            blockers=[{"name": "Small Guard", "power": 2, "toughness": 2}],
        )

        self.assertEqual(result["damage_to_player"], 3)
        self.assertIn("arrolla", result["summary"].casefold())

    def test_first_strike_kills_before_regular_damage(self):
        result = simulate_combat_exchange(
            attackers=[{"name": "Swift Duelist", "power": 3, "toughness": 1, "text": "first strike"}],
            blockers=[{"name": "Big Guard", "power": 3, "toughness": 3}],
        )

        self.assertIn("Big Guard", result["defender_graveyard"])
        self.assertNotIn("Swift Duelist", result["attacker_graveyard"])
        self.assertIn("first strike", result["summary"].casefold())

    def test_deathtouch_needs_only_one_damage(self):
        result = simulate_combat_exchange(
            attackers=[{"name": "Poison Fang", "power": 1, "toughness": 1, "text": "deathtouch"}],
            blockers=[{"name": "Huge Wall", "power": 0, "toughness": 8}],
        )

        self.assertIn("Huge Wall", result["defender_graveyard"])

    def test_double_strike_hits_twice_when_unblocked(self):
        result = simulate_combat_exchange(
            attackers=[{"name": "Twinblade Adept", "power": 2, "toughness": 2, "text": "double strike"}],
            blockers=[],
        )

        self.assertEqual(result["damage_to_player"], 4)
        self.assertIn("double strike", result["summary"].casefold())

    def test_reach_can_block_flying(self):
        assignments = choose_block_assignments(
            attackers=[{"name": "Sky Drake", "power": 3, "toughness": 3, "text": "flying"}],
            blockers=[{"name": "Web Archer", "power": 2, "toughness": 4, "text": "reach"}],
            card_db=self.card_db,
            defender_life=10,
        )

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0]["blocker_name"], "Web Archer")

    def test_play_suggestions_can_use_custom_weights(self):
        results = suggest_plays_from_hand(
            ["Lay Down Arms", "Speaker of the Heavens"],
            available_mana=1,
            card_db=self.card_db,
            lands_played_this_turn=True,
            opponent_pressure="medium",
            learned_weights={
                "board_presence": 0.5,
                "card_advantage": 0.5,
                "mana_efficiency": 0.5,
                "lethal_pressure": 3.0,
            },
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["name"], "Lay Down Arms")


if __name__ == "__main__":
    unittest.main()
